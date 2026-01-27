import re
import datetime as dt
from typing import Optional, Dict, Set, List

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

RTT_BASE = "https://api.rtt.io/api/v1"


# ----------------------------
# Small utilities
# ----------------------------
def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    cols = list(df.columns)
    km = {key(c): c for c in cols}

    for w in candidates:
        kw = key(w)
        if kw in km:
            return km[kw]

    for w in candidates:
        kw = key(w)
        for c in cols:
            if kw and kw in key(c):
                return c
    return None


def norm_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ----------------------------
# Time normalisation
# ----------------------------
def parse_pass_hhmm(x) -> Optional[str]:
    """PASS file times are GBTT/public in HH:MM (with colon)."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return None


def rtt_public_to_hhmm(x) -> Optional[str]:
    """
    RTT public timetable fields typically HHmm (e.g. 1503).
    Convert to HH:MM. If HHmmss appears, truncate to HH:MM.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() == "null":
        return None
    s = re.sub(r"\D", "", s)
    if len(s) == 4:      # HHmm
        return f"{s[:2]}:{s[2:]}"
    if len(s) == 6:      # HHmmss -> HH:MM
        return f"{s[:2]}:{s[2:4]}"
    return None


# ----------------------------
# Days run parsing (your timetable key)
# ----------------------------
def parse_journey_days_run(value) -> Set[int]:
    """
    Timetable key:
      M  Monday
      T  Tuesday
      W  Wednesday
      Th Thursday
      F  Friday
      S  Saturday
      Su Sunday

    Suffix rules:
      - ...O => runs ONLY on the day(s) preceding the O
      - ...X => runs on ALL days in the section EXCEPT the day(s) preceding the X

    Examples:
      MSX => except Monday & Saturday
      FSX => except Friday & Saturday
      TWO => only Tue & Wed
      SX  => except Saturday
      ThO => only Thursday
      SuO => only Sunday

    Also supports railway-style tokens as fallback:
      MO TO WO ThO FO SO Su (and combinations)
    Returns weekday indexes Mon=0..Sun=6.
    """
    if pd.isna(value):
        return set()
    raw = str(value).strip()
    if not raw:
        return set()

    # --- 1) Fallback: railway-style tokens (MO/TO/WO/ThO/FO/SO/Su) ---
    rail_map = {"MO": 0, "TO": 1, "WO": 2, "THO": 3, "FO": 4, "SO": 5, "SU": 6}
    tokens = re.split(r"[,\s;/]+", raw, flags=re.IGNORECASE)
    rail_days = set()
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        t_up = t.upper().replace("TH0", "THO")
        # normalise "Su" -> "SU"
        if t_up == "SU":
            rail_days.add(6)
        elif t_up in rail_map:
            rail_days.add(rail_map[t_up])
    if rail_days:
        return rail_days

    # --- 2) Timetable key codes: M/T/W/Th/F/S/Su + optional O/X suffix ---
    blob = re.sub(r"[\s,;/]+", "", raw)  # remove separators
    if not blob:
        return set()

    mode = None
    if blob[-1].upper() in ("O", "X"):
        mode = blob[-1].upper()
        blob = blob[:-1]  # day(s) preceding suffix

    # Parse day tokens in order, preferring 2-letter tokens first (Th, Su)
    i = 0
    days: Set[int] = set()
    while i < len(blob):
        part2 = blob[i:i+2]
        part1 = blob[i:i+1]

        if part2.lower() == "th":
            days.add(3)
            i += 2
            continue
        if part2.lower() == "su":
            days.add(6)
            i += 2
            continue

        ch = part1.upper()
        if ch == "M":
            days.add(0)
        elif ch == "T":
            days.add(1)
        elif ch == "W":
            days.add(2)
        elif ch == "F":
            days.add(4)
        elif ch == "S":
            days.add(5)
        # unknown chars ignored
        i += 1

    if mode == "O":
        return days  # ONLY those days

    if mode == "X":
        base = set(range(7))  # Mon..Sun (timetable section base)
        return base - days if days else base

    return days


# ----------------------------
# RailReferences
# ----------------------------
@st.cache_data
def load_railrefs_from_repo(path: str) -> pd.DataFrame:
    # RailReferences.csv: TIPLOC, CRS, Description (no header)
    return pd.read_csv(path, header=None, names=["tiploc", "crs", "description"])


@st.cache_data
def load_railrefs_from_upload(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded, header=None, names=["tiploc", "crs", "description"])


def build_desc_to_crs(rail_refs: pd.DataFrame) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for _, r in rail_refs.iterrows():
        desc = norm_name(r["description"])
        crs = str(r["crs"]).strip() if not pd.isna(r["crs"]) else ""
        if desc and crs and desc not in d:
            d[desc] = crs

    # Minimal aliases (extend if needed)
    aliases = {
        "kings cross": "KGX",
        "london kings cross": "KGX",
        "kings x": "KGX",
        "edinburgh": "EDB",
        "edinburgh waverley": "EDB",
    }
    for k, v in aliases.items():
        d.setdefault(norm_name(k), v)
    return d


# ----------------------------
# RTT calls (public/GBTT only)
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
            "operator": s.get("atocName") or "",
            "plannedCancel": bool(s.get("plannedCancel", False)),
            # Public departure at this origin
            "pub_dep_raw": ld.get("gbttBookedDeparture") or ld.get("publicTime") or ld.get("realtimeDeparture"),
        })

    # Always return expected columns, even if zero rows
    return pd.DataFrame(rows, columns=[
        "trainIdentity", "serviceUid", "runDate", "operator", "plannedCancel", "pub_dep_raw"
    ])


def rtt_service_detail(service_uid: str, run_date_iso: str, auth: HTTPBasicAuth) -> dict:
    url = f"{RTT_BASE}/json/service/{service_uid}/{run_date_iso}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_dest_public(payload: dict) -> dict:
    """
    Extract destination name and public arrival.
    Prefer GBTT/public fields; ignore WTT/working.
    """
    locs = payload.get("locations", []) or []
    if not locs:
        return {"dest_name": None, "arr_pub_raw": None}

    dest = locs[-1]

    def pick_arr(loc) -> Optional[str]:
        ld = loc.get("locationDetail") or {}
        return ld.get("gbttBookedArrival") or ld.get("publicTime") or ld.get("realtimeArrival")

    return {
        "dest_name": dest.get("description"),
        "arr_pub_raw": pick_arr(dest),
    }


@st.cache_data(ttl=300)
def cached_location_search(crs: str, run_date: dt.date, user: str, pw: str) -> dict:
    return rtt_location_services(crs, run_date, HTTPBasicAuth(user, pw))


# ----------------------------
# Streamlit app
# ----------------------------
st.set_page_config(page_title="RTT No-Brand Checker", layout="wide")
st.title("RTT No-Brand Checker")
st.caption(
    "Checks PASS trips where Brand is blank. Expected run dates derived from selected date range + JourneyDays Run. "
    "Times are public/GBTT only (exact HH:MM match)."
)

# Secrets
rtt_user = st.secrets.get("RTT_USER", "")
rtt_pass = st.secrets.get("RTT_PASS", "")
auth = HTTPBasicAuth(rtt_user, rtt_pass) if rtt_user and rtt_pass else None

with st.sidebar:
    st.header("1) Date range")
    today = dt.date.today()
    date_from = st.date_input("From", value=today)
    date_to = st.date_input("To", value=today + dt.timedelta(days=7))
    if date_to < date_from:
        date_from, date_to = date_to, date_from

    st.header("2) RailReferences")
    st.caption("Loaded from repo by default. Tick only if you want to update it.")
    update_refs = st.checkbox("Update RailReferences now", value=False)

    st.header("3) Upload pass-trips.csv")
    pass_file = st.file_uploader("pass-trips.csv", type=["csv"])

DEFAULT_RAILREFS_PATH = "RailReferences.csv"

# Load RailReferences (cached)
if update_refs:
    uploaded_refs = st.file_uploader("Upload RailReferences.csv (override)", type=["csv"], key="railrefs_upload")
    if not uploaded_refs:
        st.info("Upload RailReferences.csv to continue (update option is ticked).")
        st.stop()
    rail_refs = load_railrefs_from_upload(uploaded_refs)
    st.success("Using uploaded RailReferences.csv for this session.")
else:
    try:
        rail_refs = load_railrefs_from_repo(DEFAULT_RAILREFS_PATH)
    except Exception as e:
        st.error(
            f"Couldn't load '{DEFAULT_RAILREFS_PATH}' from the repo. "
            "Tick 'Update RailReferences now' and upload it. "
            f"Details: {e}"
        )
        st.stop()

desc_to_crs = build_desc_to_crs(rail_refs)

if not pass_file:
    st.info("Upload pass-trips.csv to begin.")
    st.stop()

# PASS file: metadata first 3 rows, headers in row 4
df = pd.read_csv(pass_file, skiprows=3)

with st.expander("Debug: detected columns in pass-trips.csv"):
    st.write(list(df.columns))

# Required columns
col_brand = find_col(df, "Brand")
col_headcode = find_col(df, "Headcode")
col_origin = find_col(df, "JourneyOrigin", "Journey Origin", "Origin")
col_dest = find_col(df, "JourneyDestination", "Journey Destination", "Destination")
col_dep = find_col(df, "JourneyDeparture", "Journey Departure", "Departure")
col_arr = find_col(df, "JourneyArrival", "Journey Arrival", "Arrival")
col_jdays = find_col(df, "JourneyDays Run", "Journey Days Run", "JourneyDaysRun", "JourneyDays")

# Diagram info columns for reporting
col_resource = find_col(df, "Resource")
col_plan_type = find_col(df, "DiagramPlan Type", "Diagram Plan Type", "Plan Type")
col_depot = find_col(df, "DiagramDepot", "Diagram Depot", "Depot")
col_did = find_col(df, "DiagramID", "Diagram Id", "Diagram ID")
col_ddays = find_col(df, "DiagramDays Run", "Diagram Days Run", "DiagramDaysRun")

required = {
    "Brand": col_brand,
    "Headcode": col_headcode,
    "JourneyOrigin": col_origin,
    "JourneyDestination": col_dest,
    "JourneyDeparture": col_dep,
    "JourneyArrival": col_arr,
    "JourneyDays Run": col_jdays,
}
missing = [k for k, v in required.items() if v is None]

with st.expander("Debug: column mapping used by the app"):
    st.write({
        **required,
        "Resource": col_resource,
        "DiagramPlan Type": col_plan_type,
        "DiagramDepot": col_depot,
        "DiagramID": col_did,
        "DiagramDays Run": col_ddays,
    })

if missing:
    st.error("Missing required columns: " + ", ".join(missing) + ". Check the Debug columns list.")
    st.stop()

# Brand blank only
df_nb = df[df[col_brand].isna() | (df[col_brand].astype(str).str.strip() == "")].copy()

# Parse fields needed for checking
df_nb["dep_hhmm"] = df_nb[col_dep].apply(parse_pass_hhmm)
df_nb["arr_hhmm"] = df_nb[col_arr].apply(parse_pass_hhmm)
df_nb["jdays_set"] = df_nb[col_jdays].apply(parse_journey_days_run)

df_nb["origin_crs"] = df_nb[col_origin].apply(lambda x: desc_to_crs.get(norm_name(x)))
df_nb["dest_crs"] = df_nb[col_dest].apply(lambda x: desc_to_crs.get(norm_name(x)))

st.subheader("Input summary (Brand blank only)")
st.write(f"Rows: **{len(df_nb)}** | Unique headcodes: **{df_nb[col_headcode].nunique()}**")

with st.expander("Preview (Brand blank only)"):
    base_cols = [col_headcode, col_origin, col_dest, col_dep, col_arr, col_jdays]
    diag_cols = [c for c in [col_resource, col_plan_type, col_depot, col_did, col_ddays] if c]
    show_cols = list(dict.fromkeys(base_cols + diag_cols))
    st.dataframe(
        df_nb[show_cols + ["origin_crs", "dest_crs", "dep_hhmm", "arr_hhmm", "jdays_set"]],
        use_container_width=True
    )

with st.expander("Debug: JourneyDays Run values (top 40)"):
    st.write(df_nb[col_jdays].astype(str).value_counts().head(40))

# Build expected checks per selected date range + JourneyDays Run
expected_rows: List[dict] = []
for i in range((date_to - date_from).days + 1):
    d = date_from + dt.timedelta(days=i)
    wd = d.weekday()  # Mon=0..Sun=6

    subset = df_nb[df_nb["jdays_set"].apply(lambda s: wd in s if isinstance(s, set) else False)]
    if subset.empty:
        continue

    for _, r in subset.iterrows():
        # Diagram string: "Resource" - "DiagramPlan Type" "DiagramDepot"."DiagramID" "DiagramDays Run"
        resource = str(r[col_resource]).strip() if col_resource else ""
        plan = str(r[col_plan_type]).strip() if col_plan_type else ""
        depot = str(r[col_depot]).strip() if col_depot else ""
        did = str(r[col_did]).strip() if col_did else ""
        ddays = str(r[col_ddays]).strip() if col_ddays else ""

        if resource or plan:
            diagram_info = f'{resource} - {plan} {depot}.{did} {ddays}'.strip()
        else:
            diagram_info = f'{depot}.{did} {ddays}'.strip()

        expected_rows.append({
            "date": d.isoformat(),
            "headcode": str(r[col_headcode]).strip(),
            "from": str(r[col_origin]).strip(),
            "to": str(r[col_dest]).strip(),
            "dep": r["dep_hhmm"],
            "arr": r["arr_hhmm"],
            "origin_crs": r["origin_crs"],
            "dest_crs": r["dest_crs"],
            "diagram_info": diagram_info,
        })

expected = pd.DataFrame(expected_rows)

st.subheader("Expected checks")
st.write(f"Date range: **{date_from} → {date_to}** | Expected checks: **{len(expected)}**")

if expected.empty:
    st.warning("No expected services found in that date range based on JourneyDays Run.")
    st.stop()

run = st.button("Run RTT check", type="primary")
if not run:
    st.stop()

if not auth:
    st.error("RTT credentials missing. Add RTT_USER/RTT_PASS in Streamlit Cloud Secrets.")
    st.stop()

# Perform checks (grouped by date + origin CRS)
results: List[dict] = []
total = len(expected)
done = 0
progress = st.progress(0)

grouped = expected.groupby(["date", "origin_crs"], dropna=False)

for (date_iso, origin_crs), grp in grouped:
    run_date = dt.date.fromisoformat(date_iso)

    # CRS missing -> not checked
    if not isinstance(origin_crs, str) or not origin_crs.strip():
        for _, row in grp.iterrows():
            results.append({
                **row.to_dict(),
                "status": "NOT CHECKED",
                "error": "Origin CRS not mapped (RailReferences missing mapping / add alias)",
                "operator": "",
                "planned_cancel": "",
            })
            done += 1
            progress.progress(min(1.0, done / total))
        continue

    # RTT location search
    try:
        payload = cached_location_search(origin_crs.strip(), run_date, rtt_user, rtt_pass)
        loc_df = flatten_location_services(payload)
    except Exception as ex:
        for _, row in grp.iterrows():
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": f"RTT location query failed: {ex}",
                "operator": "",
                "planned_cancel": "",
            })
            done += 1
            progress.progress(min(1.0, done / total))
        continue

    # Empty services list -> fail each row in the group (but never crash)
    if loc_df.empty:
        for _, row in grp.iterrows():
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": f"No RTT services returned for origin/date (origin CRS: {origin_crs})",
                "operator": "",
                "planned_cancel": "",
            })
            done += 1
            progress.progress(min(1.0, done / total))
        continue

    for _, row in grp.iterrows():
        hc = str(row["headcode"]).strip()
        exp_dep = row["dep"]
        exp_arr = row["arr"]
        exp_to = row["to"]

        candidates = loc_df[loc_df["trainIdentity"] == hc].copy()
        if candidates.empty:
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": "Headcode not found at origin on date",
                "operator": "",
                "planned_cancel": "",
            })
            done += 1
            progress.progress(min(1.0, done / total))
            continue

        # Departure match (public, exact HH:MM)
        if exp_dep:
            candidates["dep_pub_hhmm"] = candidates["pub_dep_raw"].apply(rtt_public_to_hhmm)
            candidates = candidates[candidates["dep_pub_hhmm"] == exp_dep]
            if candidates.empty:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": "Headcode found but departure time differs (public/GBTT)",
                    "operator": "",
                    "planned_cancel": "",
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

        cand = candidates.iloc[0]
        uid = cand.get("serviceUid")
        rtt_run_date = cand.get("runDate")
        op = cand.get("operator") or ""
        pc = "Y" if bool(cand.get("plannedCancel", False)) else ""

        if not uid or not rtt_run_date:
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": "Insufficient RTT data to verify destination/arrival",
                "operator": op,
                "planned_cancel": pc,
            })
            done += 1
            progress.progress(min(1.0, done / total))
            continue

        # Service detail: destination + public arrival
        try:
            detail = rtt_service_detail(uid, rtt_run_date, auth)
            det = extract_dest_public(detail)
        except Exception:
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": "RTT service detail lookup failed",
                "operator": op,
                "planned_cancel": pc,
            })
            done += 1
            progress.progress(min(1.0, done / total))
            continue

        det_dest = det.get("dest_name") or ""
        if exp_to and det_dest:
            # name-based match (simple and transparent)
            if norm_name(exp_to) not in norm_name(det_dest) and norm_name(det_dest) not in norm_name(exp_to):
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"Destination differs (RTT: {det_dest})",
                    "operator": op,
                    "planned_cancel": pc,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

        rtt_arr_hhmm = rtt_public_to_hhmm(det.get("arr_pub_raw"))
        if exp_arr and rtt_arr_hhmm:
            if exp_arr != rtt_arr_hhmm:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"Arrival time differs (public/GBTT, RTT: {rtt_arr_hhmm})",
                    "operator": op,
                    "planned_cancel": pc,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

        results.append({
            **row.to_dict(),
            "status": "OK",
            "error": "",
            "operator": op,
            "planned_cancel": pc,
        })
        done += 1
        progress.progress(min(1.0, done / total))

progress.empty()

report = pd.DataFrame(results)
if report.empty:
    st.warning("No results produced (unexpected). Check inputs and try again.")
    st.stop()

ok = report[report["status"] == "OK"]
fail = report[report["status"] == "FAIL"]
nc = report[report["status"] == "NOT CHECKED"]

st.subheader("Outcome message")
st.write(f"Checked: **{len(report)}** | OK: **{len(ok)}** | Errors: **{len(fail)}** | Not checked: **{len(nc)}**")

if len(fail) == 0 and len(nc) == 0:
    st.success("All trains are running as booked (public/GBTT) for the selected date range.")
else:
    st.error("Some trains did not match public/GBTT details, or could not be checked. You can create a CSV report.")

with st.expander("Trains running as booked (simplified)"):
    simp = ok.copy()
    simp["line"] = simp.apply(
        lambda r: f"{r['date']} {r['headcode']} {r['dep']} {r['from']} - {r['to']} {r['arr']}",
        axis=1
    )
    lines = simp.sort_values(["date", "dep", "headcode"])["line"].tolist()
    st.write("\n".join(lines) if lines else "None")

with st.expander("Errors / not checked (details)"):
    bad = report[report["status"] != "OK"].copy()
    if bad.empty:
        st.write("None")
    else:
        st.dataframe(
            bad[["date", "headcode", "from", "to", "dep", "arr", "status", "error", "diagram_info", "operator", "planned_cancel"]]
            .sort_values(["date", "status", "dep", "headcode"]),
            use_container_width=True
        )

st.subheader("CSV report")
st.write("Would you like to create a CSV report?")
create_csv = st.checkbox("Yes, create CSV report", value=False)

if create_csv:
    out = report.copy()
    # For OK rows, keep simplified (no error/diagram needed)
    out.loc[out["status"] == "OK", "error"] = ""
    out.loc[out["status"] == "OK", "diagram_info"] = ""

    cols_out = ["date", "headcode", "from", "to", "dep", "arr", "status", "error", "diagram_info", "operator", "planned_cancel"]
    csv_bytes = out[cols_out].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV report",
        data=csv_bytes,
        file_name=f"rtt_no_brand_report_{date_from}_{date_to}.csv",
        mime="text/csv",
    )
