import re
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Set, List, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

RTT_BASE = "https://api.rtt.io/api/v1"

WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
DOW_TO_IDX = {d: i for i, d in enumerate(WEEKDAYS)}  # MON=0 .. SUN=6


# ----------------------------
# Normalisation / parsing
# ----------------------------
def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def parse_ddmmyyyy(s: str) -> Optional[dt.date]:
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


def parse_hhmm(s: str) -> Optional[str]:
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return None


def hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def within_tolerance(expected: Optional[str], actual: Optional[str], tol_min: int) -> bool:
    if expected is None or actual is None:
        return False
    if tol_min <= 0:
        return expected == actual
    return abs(hhmm_to_minutes(expected) - hhmm_to_minutes(actual)) <= tol_min


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def dow_code(d: dt.date) -> str:
    return WEEKDAYS[d.weekday()]


def parse_days_run(value: str) -> Set[str]:
    """
    Convert common 'Diagram Days Run' formats into weekday set: {"MON",...}
    Supports:
      - MON-FRI, MON-SAT, SAT-SUN
      - MON,TUE,WED
      - DAILY / EVERY DAY
      - WEEKDAYS / WEEKENDS
      - 'MTWTFSS' style (heuristic)
      - 'SAT ONLY', etc.
    """
    if pd.isna(value):
        return set()
    raw = str(value).strip().upper()
    if not raw:
        return set()

    raw = raw.replace("&", " ").replace("/", " ")
    raw = re.sub(r"\s+", " ", raw)
    raw = raw.replace("ONLY", "").strip()

    if raw in {"DAILY", "EVERY DAY", "EVERYDAY"}:
        return set(WEEKDAYS)
    if raw in {"WEEKDAYS"}:
        return {"MON", "TUE", "WED", "THU", "FRI"}
    if raw in {"WEEKENDS"}:
        return {"SAT", "SUN"}

    # Range like MON-FRI
    m = re.match(r"^(MON|TUE|WED|THU|FRI|SAT|SUN)\s*-\s*(MON|TUE|WED|THU|FRI|SAT|SUN)$", raw)
    if m:
        a, b = m.group(1), m.group(2)
        ia, ib = DOW_TO_IDX[a], DOW_TO_IDX[b]
        if ia <= ib:
            return set(WEEKDAYS[ia:ib + 1])
        return set(WEEKDAYS[ia:] + WEEKDAYS[:ib + 1])

    # Comma/space list like MON,TUE,WED
    tokens = re.split(r"[,\s]+", raw)
    tokens = [t for t in tokens if t]
    if tokens and all(t in WEEKDAYS for t in tokens):
        return set(tokens)

    # Condensed letters heuristic e.g. MTWTFSS
    compact = re.sub(r"[^MTWFSU]", "", raw)
    if len(compact) >= 5:
        allowed = set()
        if "M" in compact:
            allowed.add("MON")
        if "W" in compact:
            allowed.add("WED")
        if "F" in compact:
            allowed.add("FRI")
        if "T" in compact:
            allowed.update({"TUE", "THU"})  # safe assumption
        if "S" in compact:
            allowed.update({"SAT", "SUN"})
        if "U" in compact:
            allowed.add("SUN")
        return allowed

    # Single day mention
    for d in WEEKDAYS:
        if d in raw:
            return {d}

    return set()


def find_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    """
    Find a column by "loose" matching:
    - ignores case
    - ignores spaces/underscores/punctuation
    """
    def key(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

    w = key(wanted)
    for c in df.columns:
        if key(c) == w:
            return c

    # fallback: contains-match
    for c in df.columns:
        if w in key(c):
            return c

    return None


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_pass_trips(uploaded_file) -> pd.DataFrame:
    # pass-trips.csv often has 3 metadata lines before the header row
    return pd.read_csv(uploaded_file, skiprows=3)


@st.cache_data
def load_rail_refs(uploaded_file) -> pd.DataFrame:
    # RailReferences.csv: TIPLOC, CRS, Description (no header)
    return pd.read_csv(uploaded_file, header=None, names=["tiploc", "crs", "description"])


def build_desc_to_crs(rail_refs: pd.DataFrame) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for _, r in rail_refs.iterrows():
        desc = norm(r["description"])
        crs = str(r["crs"]).strip() if not pd.isna(r["crs"]) else ""
        if desc and crs and desc not in d:
            d[desc] = crs

    # Helpful aliases (extend as needed)
    aliases = {
        "kings cross": "KGX",
        "london kings cross": "KGX",
        "kings x": "KGX",
        "edinburgh": "EDB",
        "edinburgh waverley": "EDB",
    }
    for k, v in aliases.items():
        d.setdefault(norm(k), v)
    return d


# ----------------------------
# RTT calls
# ----------------------------
def rtt_location_services(crs_or_tiploc: str, run_date: dt.date, auth: HTTPBasicAuth) -> dict:
    y = run_date.year
    m = f"{run_date.month:02d}"
    d = f"{run_date.day:02d}"
    url = f"{RTT_BASE}/json/search/{crs_or_tiploc}/{y}/{m}/{d}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def flatten_location_services(payload: dict) -> pd.DataFrame:
    services = payload.get("services", []) or []
    rows = []
    for s in services:
        ld = (s.get("locationDetail") or {})
        rows.append({
            "trainIdentity": (s.get("trainIdentity") or "").strip(),
            "serviceUid": s.get("serviceUid"),
            "runDate": s.get("runDate"),
            "atocName": s.get("atocName"),
            "plannedCancel": bool(s.get("plannedCancel", False)),
            "gbttBookedDeparture": ld.get("gbttBookedDeparture"),
        })
    return pd.DataFrame(rows)


def rtt_service_detail(service_uid: str, run_date_iso: str, auth: HTTPBasicAuth) -> dict:
    url = f"{RTT_BASE}/json/service/{service_uid}/{run_date_iso}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_origin_dest_times_from_detail(payload: dict) -> dict:
    locs = payload.get("locations", []) or []
    if not locs:
        return {"origin_name": None, "dest_name": None, "book_dep": None, "book_arr": None}

    origin = locs[0]
    dest = locs[-1]

    def booked_dep(loc):
        ld = loc.get("locationDetail") or {}
        return ld.get("gbttBookedDeparture") or ld.get("gbttBookedArrival")

    def booked_arr(loc):
        ld = loc.get("locationDetail") or {}
        return ld.get("gbttBookedArrival") or ld.get("gbttBookedDeparture")

    return {
        "origin_name": origin.get("description"),
        "dest_name": dest.get("description"),
        "book_dep": booked_dep(origin),
        "book_arr": booked_arr(dest),
    }


# ----------------------------
# Matching structures
# ----------------------------
@dataclass
class ExpectedService:
    headcode: str
    origin_name: str
    dest_name: str
    dep_hhmm: Optional[str]
    arr_hhmm: Optional[str]
    origin_crs: Optional[str]
    dest_crs: Optional[str]
    diagram_id: str


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="No-Brand RTT Checker", layout="wide")
st.title("No-Brand RTT Checker")
st.caption("Checks services where Brand is blank. Validates Headcode + From/To + booked dep/arr on eligible dates.")

with st.sidebar:
    st.header("1) Upload files")
    pass_trips_file = st.file_uploader("Upload pass-trips.csv", type=["csv"])
    rail_refs_file = st.file_uploader("Upload RailReferences.csv (optional but recommended)", type=["csv"])

    st.header("2) Select date range")
    today = dt.date.today()
    date_from = st.date_input("From", value=today)
    date_to = st.date_input("To", value=today + dt.timedelta(days=7))

    st.header("3) Rules")
    use_diagram_window = st.checkbox("Use Diagram Start/End as guardrail", value=True)
    tol = st.selectbox("Time tolerance (minutes)", options=[0, 1, 2, 3, 5], index=0)

    st.header("4) Run")
    run_btn = st.button("Run check", type="primary")

# Credentials from Streamlit secrets
rtt_user = st.secrets.get("RTT_USER", "")
rtt_pass = st.secrets.get("RTT_PASS", "")
auth = HTTPBasicAuth(rtt_user, rtt_pass) if rtt_user and rtt_pass else None

if not pass_trips_file:
    st.info("Upload pass-trips.csv to begin.")
    st.stop()

df = load_pass_trips(pass_trips_file)

with st.expander("Debug: detected columns from pass-trips.csv"):
    st.write(list(df.columns))

if rail_refs_file:
    rail_refs = load_rail_refs(rail_refs_file)
    desc_to_crs = build_desc_to_crs(rail_refs)
    st.success("Using uploaded RailReferences.csv")
else:
    desc_to_crs = {}
    st.warning("RailReferences.csv not uploaded. Origin CRS mapping may fail; upload it for best results.")

# Required columns (loose-match)
col_brand = find_col(df, "Brand")
col_headcode = find_col(df, "Headcode")
col_origin = find_col(df, "JourneyOrigin") or find_col(df, "Journey Origin") or find_col(df, "Origin")
col_dest = find_col(df, "JourneyDestination") or find_col(df, "Journey Destination") or find_col(df, "Destination")
col_dep = find_col(df, "JourneyDeparture") or find_col(df, "Journey Departure") or find_col(df, "Departure")
col_arr = find_col(df, "JourneyArrival") or find_col(df, "Journey Arrival") or find_col(df, "Arrival")
col_start = find_col(df, "DiagramStart Date") or find_col(df, "Diagram Start Date") or find_col(df, "Start Date")
col_end = find_col(df, "DiagramEnd Date") or find_col(df, "Diagram End Date") or find_col(df, "End Date")
col_days = find_col(df, "Diagram Days Run") or find_col(df, "Days Run") or find_col(df, "DaysRun")
col_diagram_id = find_col(df, "DiagramID") or find_col(df, "Diagram Id") or find_col(df, "Diagram ID")

missing = [name for name, col in {
    "Brand": col_brand,
    "Headcode": col_headcode,
    "Journey origin": col_origin,
    "Journey destination": col_dest,
    "Journey departure": col_dep,
    "Journey arrival": col_arr,
    "Diagram start date": col_start,
    "Diagram end date": col_end,
    "Diagram days run": col_days,
}.items() if col is None]

with st.expander("Debug: column mapping used by the app"):
    st.write({
        "Brand": col_brand,
        "Headcode": col_headcode,
        "JourneyOrigin": col_origin,
        "JourneyDestination": col_dest,
        "JourneyDeparture": col_dep,
        "JourneyArrival": col_arr,
        "DiagramStart Date": col_start,
        "DiagramEnd Date": col_end,
        "Diagram Days Run": col_days,
        "DiagramID": col_diagram_id,
    })

if missing:
    st.error(
        "I couldn't find these required columns in your pass-trips.csv: "
        + ", ".join(missing)
        + ".\n\nOpen the Debug columns expander above and tell me the closest column names."
    )
    st.stop()

# Filter: Brand blank
df["Brand_missing"] = df[col_brand].isna() | (df[col_brand].astype(str).str.strip() == "")
df_nb = df[df["Brand_missing"]].copy()

# Parse fields
df_nb["DiagramStart_dt"] = df_nb[col_start].apply(parse_ddmmyyyy)
df_nb["DiagramEnd_dt"] = df_nb[col_end].apply(parse_ddmmyyyy)
df_nb["dep_hhmm"] = df_nb[col_dep].apply(parse_hhmm)
df_nb["arr_hhmm"] = df_nb[col_arr].apply(parse_hhmm)
df_nb["days_set"] = df_nb[col_days].apply(parse_days_run)

df_nb["origin_crs"] = df_nb[col_origin].apply(lambda x: desc_to_crs.get(norm(x)) if desc_to_crs else None)
df_nb["dest_crs"] = df_nb[col_dest].apply(lambda x: desc_to_crs.get(norm(x)) if desc_to_crs else None)

st.subheader("Rows considered (Brand blank)")
st.write(f"Rows: **{len(df_nb)}** | Unique headcodes: **{df_nb[col_headcode].nunique()}**")

with st.expander("Preview filtered input rows"):
    preview_cols = [col_headcode, col_origin, col_dest, col_dep, col_arr, col_start, col_end, col_days]
    if col_diagram_id:
        preview_cols.append(col_diagram_id)
    extra = ["origin_crs", "dest_crs", "dep_hhmm", "arr_hhmm"]
    st.dataframe(df_nb[preview_cols + extra], use_container_width=True)

if not run_btn:
    st.stop()

if auth is None:
    st.error("RTT credentials not configured. Add RTT_USER and RTT_PASS in Streamlit Cloud Secrets.")
    st.stop()

# Build expected services per date within user-selected range
date_from = min(date_from, date_to)
date_to = max(date_from, date_to)

expected_by_date: Dict[dt.date, List[ExpectedService]] = {}
for d in daterange(date_from, date_to):
    dc = dow_code(d)
    subset = df_nb[df_nb["days_set"].apply(lambda s: dc in s if isinstance(s, set) else False)].copy()

    if use_diagram_window:
        subset = subset[
            subset.apply(
                lambda r: (r["DiagramStart_dt"] is None or d >= r["DiagramStart_dt"]) and
                          (r["DiagramEnd_dt"] is None or d <= r["DiagramEnd_dt"]),
                axis=1
            )
        ]

    exp_list: List[ExpectedService] = []
    for _, r in subset.iterrows():
        exp_list.append(
            ExpectedService(
                headcode=str(r[col_headcode]).strip(),
                origin_name=str(r[col_origin]).strip(),
                dest_name=str(r[col_dest]).strip(),
                dep_hhmm=r["dep_hhmm"],
                arr_hhmm=r["arr_hhmm"],
                origin_crs=r.get("origin_crs"),
                dest_crs=r.get("dest_crs"),
                diagram_id=str(r[col_diagram_id]).strip() if col_diagram_id else "",
            )
        )
    expected_by_date[d] = exp_list

total_expected = sum(len(v) for v in expected_by_date.values())

st.subheader("Expected checks in selected date range")
st.write(f"Date range: **{date_from} → {date_to}** | Expected service-checks: **{total_expected}**")

@st.cache_data(ttl=300)
def cached_location_search(origin: str, d: dt.date, user: str, pw: str) -> dict:
    a = HTTPBasicAuth(user, pw)
    return rtt_location_services(origin, d, a)

def check_one(expected: ExpectedService, d: dt.date, origin_services: pd.DataFrame) -> Tuple[bool, str, bool, str]:
    headcode = expected.headcode
    dep = expected.dep_hhmm

    candidates = origin_services[origin_services["trainIdentity"] == headcode].copy()
    if candidates.empty:
        return False, "Headcode not found at origin on date", False, ""

    # departure time at origin
    if dep:
        candidates["dep_ok"] = candidates["gbttBookedDeparture"].apply(lambda t: within_tolerance(dep, t, tol))
        candidates = candidates[candidates["dep_ok"] == True]
        if candidates.empty:
            return False, "Headcode found but departure time differs", False, ""

    cand = candidates.iloc[0]
    uid = cand.get("serviceUid")
    run_date_iso = cand.get("runDate")
    planned_cancel = bool(cand.get("plannedCancel", False))
    atoc_name = cand.get("atocName") or ""

    if not uid or not run_date_iso:
        return False, "Insufficient RTT data to verify destination/arrival", planned_cancel, atoc_name

    try:
        detail = rtt_service_detail(uid, run_date_iso, auth)
        det = extract_origin_dest_times_from_detail(detail)
    except Exception:
        return False, "RTT service detail lookup failed", planned_cancel, atoc_name

    # destination name check (soft but effective)
    if expected.dest_name and det.get("dest_name"):
        expn = norm(expected.dest_name)
        actn = norm(det["dest_name"])
        if expn not in actn and actn not in expn:
            return False, f"Destination differs (RTT: {det.get('dest_name')})", planned_cancel, atoc_name

    # arrival time check
    exp_arr = expected.arr_hhmm
    rtt_arr = det.get("book_arr")
    if exp_arr and rtt_arr:
        if not within_tolerance(exp_arr, rtt_arr, tol):
            return False, f"Arrival time differs (RTT: {rtt_arr})", planned_cancel, atoc_name

    return True, "Matched", planned_cancel, atoc_name

report_rows = []
done = 0
progress = st.progress(0) if total_expected > 0 else None

for d, exp_list in expected_by_date.items():
    if not exp_list:
        continue

    by_origin: Dict[str, List[ExpectedService]] = {}
    unmapped: List[ExpectedService] = []

    for e in exp_list:
        if e.origin_crs:
            by_origin.setdefault(e.origin_crs, []).append(e)
        else:
            unmapped.append(e)

    for e in unmapped:
        report_rows.append({
            "date": d.isoformat(),
            "headcode": e.headcode,
            "from": e.origin_name,
            "to": e.dest_name,
            "dep": e.dep_hhmm,
            "arr": e.arr_hhmm,
            "status": "NOT CHECKED",
            "reason": "Origin CRS not mapped (upload RailReferences.csv or add alias)",
            "planned_cancel": "",
            "operator": "",
            "diagram_id": e.diagram_id,
        })
        done += 1
        if progress:
            progress.progress(min(1.0, done / max(1, total_expected)))

    for origin_crs, services in by_origin.items():
        try:
            payload = cached_location_search(origin_crs, d, rtt_user, rtt_pass)
            origin_services = flatten_location_services(payload)
        except Exception as ex:
            for e in services:
                report_rows.append({
                    "date": d.isoformat(),
                    "headcode": e.headcode,
                    "from": e.origin_name,
                    "to": e.dest_name,
                    "dep": e.dep_hhmm,
                    "arr": e.arr_hhmm,
                    "status": "FAIL",
                    "reason": f"RTT location query failed: {ex}",
                    "planned_cancel": "",
                    "operator": "",
                    "diagram_id": e.diagram_id,
                })
                done += 1
                if progress:
                    progress.progress(min(1.0, done / max(1, total_expected)))
            continue

        for e in services:
            ok, reason, planned_cancel, atoc_name = check_one(e, d, origin_services)
            report_rows.append({
                "date": d.isoformat(),
                "headcode": e.headcode,
                "from": e.origin_name,
                "to": e.dest_name,
                "dep": e.dep_hhmm,
                "arr": e.arr_hhmm,
                "status": "OK" if ok else "FAIL",
                "reason": "" if ok else reason,
                "planned_cancel": "Y" if planned_cancel else "",
                "operator": atoc_name,
                "diagram_id": e.diagram_id,
            })
            done += 1
            if progress:
                progress.progress(min(1.0, done / max(1, total_expected)))

if progress:
    progress.empty()

report = pd.DataFrame(report_rows)

fails = report[report["status"] == "FAIL"]
oks = report[report["status"] == "OK"]
not_checked = report[report["status"] == "NOT CHECKED"]

st.subheader("Result summary")
st.write(f"OK: **{len(oks)}** | FAIL: **{len(fails)}** | Not checked: **{len(not_checked)}**")

def format_line(r) -> str:
    ddmmyyyy = dt.date.fromisoformat(r["date"]).strftime("%d/%m/%Y")
    return f"{r['headcode']}: {r['dep']} {r['from']} - {r['to']} {r['arr']} on {ddmmyyyy}"

st.markdown("### Daily-style report")
if len(fails) == 0 and len(not_checked) == 0:
    st.success("All services found to be running!")
    st.write("The following services were checked and running as expected:")
    for _, r in oks.sort_values(["date", "dep", "headcode"]).iterrows():
        st.write(f"- {format_line(r)}")
else:
    st.error("Some services did not match.")
    if len(oks) > 0:
        st.write("**Matched:**")
        for _, r in oks.sort_values(["date", "dep", "headcode"]).iterrows():
            st.write(f"- {format_line(r)}")
    if len(fails) > 0:
        st.write("**Not matched / differences:**")
        for _, r in fails.sort_values(["date", "dep", "headcode"]).iterrows():
            st.write(f"- {format_line(r)} — {r['reason']}")
    if len(not_checked) > 0:
        st.write("**Not checked:**")
        for _, r in not_checked.sort_values(["date", "dep", "headcode"]).iterrows():
            st.write(f"- {format_line(r)} — {r['reason']}")

st.subheader("Detailed table")
st.dataframe(report.sort_values(["date", "status", "dep", "headcode"]), use_container_width=True)

st.download_button(
    "Download report as CSV",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name=f"rtt_no_brand_report_{date_from}_{date_to}.csv",
    mime="text/csv",
)
