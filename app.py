import re
import datetime as dt
from typing import Optional, Dict, Set, List

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

RTT_BASE = "https://api.rtt.io/api/v1"


# ============================================================
# Utilities
# ============================================================
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


# ============================================================
# Time normalisation
# ============================================================
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
    if len(s) == 4:
        return f"{s[:2]}:{s[2:]}"
    if len(s) == 6:
        return f"{s[:2]}:{s[2:4]}"
    return None


# ============================================================
# Days run parsing (your timetable key)
# ============================================================
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

    # 1) Fallback: railway tokens like MO/TO/WO/ThO/FO/SO/Su
    rail_map = {"MO": 0, "TO": 1, "WO": 2, "THO": 3, "FO": 4, "SO": 5, "SU": 6}
    tokens = re.split(r"[,\s;/]+", raw, flags=re.IGNORECASE)
    rail_days = set()
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        t_up = t.upper().replace("TH0", "THO")
        if t_up == "SU":
            rail_days.add(6)
        elif t_up in rail_map:
            rail_days.add(rail_map[t_up])
    if rail_days:
        return rail_days

    # 2) Timetable key codes like MSX, TWO, ThO
    blob = re.sub(r"[\s,;/]+", "", raw)
    if not blob:
        return set()

    mode = None
    if blob[-1].upper() in ("O", "X"):
        mode = blob[-1].upper()
        blob = blob[:-1]

    i = 0
    days: Set[int] = set()
    while i < len(blob):
        part2 = blob[i:i + 2]
        part1 = blob[i:i + 1]

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
        i += 1

    if mode == "O":
        return days
    if mode == "X":
        base = set(range(7))  # Mon..Sun
        return base - days if days else base
    return days


# ============================================================
# RailReferences
# ============================================================
@st.cache_data
def load_railrefs_from_repo(path: str) -> pd.DataFrame:
    # RailReferences.csv has no header: TIPLOC, CRS, Description
    return pd.read_csv(path, header=None, names=["tiploc", "crs", "description"])


@st.cache_data
def load_railrefs_from_upload(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded, header=None, names=["tiploc", "crs", "description"])


def build_desc_to_crs(rail_refs: pd.DataFrame) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for _, r in rail_refs.iterrows():
        desc = norm_name(r["description"])
        crs = str(r["crs"]).strip().upper() if not pd.isna(r["crs"]) else ""
        if desc and crs and desc not in d:
            d[desc] = crs

    # Minimal aliases (extend anytime)
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


# ============================================================
# RTT calls
# ============================================================
def rtt_location_services(crs_or_tiploc: str, run_date: dt.date, auth: HTTPBasicAuth) -> dict:
    y = run_date.year
    m = f"{run_date.month:02d}"
    d = f"{run_date.day:02d}"
    url = f"{RTT_BASE}/json/search/{crs_or_tiploc}/{y}/{m}/{d}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def flatten_location_services(payload: dict) -> pd.DataFrame:
    """
    Flatten /search response for a given location+date.
    """
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
            "pub_dep_raw": ld.get("gbttBookedDeparture") or ld.get("publicTime") or ld.get("realtimeDeparture"),
        })
    return pd.DataFrame(rows, columns=["trainIdentity", "serviceUid", "runDate", "operator", "plannedCancel", "pub_dep_raw"])


def rtt_service_detail(service_uid: str, run_date_iso: str, auth: HTTPBasicAuth) -> dict:
    """
    IMPORTANT: RTT expects /service/{uid}/{yyyy}/{mm}/{dd}.
    run_date_iso is typically YYYY-MM-DD.
    """
    y, m, d = str(run_date_iso).split("-")
    url = f"{RTT_BASE}/json/service/{service_uid}/{y}/{m}/{d}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_segment_times_by_crs(detail_payload: dict, origin_crs: str, dest_crs: str) -> dict:
    """
    PERMANENT / 'FOR GOOD' FIX:
    Match FROM/TO calling points by CRS ONLY.
    Prevents station names matching junctions/sidings (e.g. Leeds vs Leeds West Junction).
    """
    locs = detail_payload.get("locations", []) or []

    origin_crs = (origin_crs or "").strip().upper()
    dest_crs = (dest_crs or "").strip().upper()

    from_found = False
    to_found = False
    act_from = None
    act_to = None
    dep_raw = None
    arr_raw = None

    for loc in locs:
        ld = loc.get("locationDetail") or {}
        crs = (loc.get("crs") or "").strip().upper()
        desc = loc.get("description") or ""

        if not from_found and crs == origin_crs:
            from_found = True
            act_from = desc
            dep_raw = ld.get("gbttBookedDeparture") or ld.get("publicTime") or ld.get("realtimeDeparture")

        elif from_found and not to_found and crs == dest_crs:
            to_found = True
            act_to = desc
            arr_raw = ld.get("gbttBookedArrival") or ld.get("publicTime") or ld.get("realtimeArrival")

    return {
        "from_found": from_found,
        "to_found": to_found,
        "act_from": act_from,
        "act_to": act_to,
        "act_dep_raw": dep_raw,
        "act_arr_raw": arr_raw,
        "act_dep": rtt_public_to_hhmm(dep_raw),
        "act_arr": rtt_public_to_hhmm(arr_raw),
    }


@st.cache_data(ttl=300)
def cached_location_search(crs: str, run_date: dt.date, user: str, pw: str) -> dict:
    return rtt_location_services(crs, run_date, HTTPBasicAuth(user, pw))


# ============================================================
# Streamlit app
# ============================================================
st.set_page_config(page_title="RTT No-Brand Checker", layout="wide")
st.title("RTT No-Brand Checker")
st.caption(
    "Checks pass-trips.csv rows where Brand is blank. Expected run dates are derived from selected date range + "
    "JourneyDays Run. Matches stations by CRS only inside the RTT service (prevents junction mismatches). "
    "Compares public/GBTT times (exact HH:MM)."
)

# Read secrets for RTT
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

# Load RailReferences
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
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    with st.expander("Debug: detected columns"):
        st.write(list(df.columns))
    st.stop()

# Brand blank only
df_nb = df[df[col_brand].isna() | (df[col_brand].astype(str).str.strip() == "")].copy()

# Parse expected fields
df_nb["exp_dep"] = df_nb[col_dep].apply(parse_pass_hhmm)
df_nb["exp_arr"] = df_nb[col_arr].apply(parse_pass_hhmm)
df_nb["jdays_set"] = df_nb[col_jdays].apply(parse_journey_days_run)

df_nb["origin_crs"] = df_nb[col_origin].apply(lambda x: desc_to_crs.get(norm_name(x)))
df_nb["dest_crs"] = df_nb[col_dest].apply(lambda x: desc_to_crs.get(norm_name(x)))

st.subheader("Input summary (Brand blank only)")
st.write(f"Rows: **{len(df_nb)}** | Unique headcodes: **{df_nb[col_headcode].nunique()}**")

with st.expander("Debug: mapping and raw input preview (first 200 rows)"):
    show_cols = [col_headcode, col_origin, col_dest, col_dep, col_arr, col_jdays]
    diag_cols = [c for c in [col_resource, col_plan_type, col_depot, col_did, col_ddays] if c]
    show_cols = list(dict.fromkeys(show_cols + diag_cols))
    st.dataframe(
        df_nb[show_cols + ["origin_crs", "dest_crs", "exp_dep", "exp_arr", "jdays_set"]].head(200),
        use_container_width=True
    )

# Build expected checks across selected date range
expected_rows: List[dict] = []
for i in range((date_to - date_from).days + 1):
    day = date_from + dt.timedelta(days=i)
    wd = day.weekday()  # Mon=0..Sun=6

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
            "date": day.isoformat(),
            "weekday": day.strftime("%A"),
            "headcode": str(r[col_headcode]).strip(),
            "exp_from": str(r[col_origin]).strip(),
            "exp_to": str(r[col_dest]).strip(),
            "exp_dep": r["exp_dep"],
            "exp_arr": r["exp_arr"],
            "origin_crs": r["origin_crs"],
            "dest_crs": r["dest_crs"],
            "diagram_info": diagram_info,
        })

expected = pd.DataFrame(expected_rows)

st.subheader("Expected checks")
st.write(f"Date range: **{date_from} → {date_to}** | Expected checks: **{len(expected)}**")

with st.expander("Expected checks by weekday"):
    if expected.empty:
        st.write("None")
    else:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        st.dataframe(expected["weekday"].value_counts().reindex(order, fill_value=0))

if expected.empty:
    st.warning("No expected services found in that date range based on JourneyDays Run.")
    st.stop()

run = st.button("Run RTT check", type="primary")

# If we already have a report, allow export without re-running
existing_report = st.session_state.get("report")
if not run and existing_report is None:
    st.stop()

if run:
    if not auth:
        st.error("RTT credentials missing. Add RTT_USER/RTT_PASS in Streamlit Cloud Secrets.")
        st.stop()

    results: List[dict] = []
    total = len(expected)
    done = 0
    progress = st.progress(0)

    grouped = expected.groupby(["date", "origin_crs"], dropna=False)

    for (date_iso, origin_crs), grp in grouped:
        run_date = dt.date.fromisoformat(date_iso)

        # If CRS mapping missing - cannot run
        if not isinstance(origin_crs, str) or not origin_crs.strip():
            for _, row in grp.iterrows():
                results.append({
                    **row.to_dict(),
                    "status": "NOT CHECKED",
                    "error": "Origin CRS not mapped (RailReferences missing mapping / add alias)",
                    "operator": "",
                    "planned_cancel": "",
                    "serviceUid": "",
                    "runDate": "",
                    "act_from": None,
                    "act_to": None,
                    "act_dep": None,
                    "act_arr": None,
                    "act_dep_raw": None,
                    "act_arr_raw": None,
                    "origin_rtt_deps": "",
                })
                done += 1
                progress.progress(min(1.0, done / total))
            continue

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
                    "serviceUid": "",
                    "runDate": "",
                    "act_from": None,
                    "act_to": None,
                    "act_dep": None,
                    "act_arr": None,
                    "act_dep_raw": None,
                    "act_arr_raw": None,
                    "origin_rtt_deps": "",
                })
                done += 1
                progress.progress(min(1.0, done / total))
            continue

        if loc_df.empty:
            for _, row in grp.iterrows():
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"No RTT services returned for origin/date (origin CRS: {origin_crs})",
                    "operator": "",
                    "planned_cancel": "",
                    "serviceUid": "",
                    "runDate": "",
                    "act_from": None,
                    "act_to": None,
                    "act_dep": None,
                    "act_arr": None,
                    "act_dep_raw": None,
                    "act_arr_raw": None,
                    "origin_rtt_deps": "",
                })
                done += 1
                progress.progress(min(1.0, done / total))
            continue

        for _, row in grp.iterrows():
            hc = str(row["headcode"]).strip()
            exp_dep = row["exp_dep"]

            # Destination CRS missing -> not checked (prevents name-based mistakes)
            if not isinstance(row.get("dest_crs"), str) or not str(row.get("dest_crs")).strip():
                results.append({
                    **row.to_dict(),
                    "status": "NOT CHECKED",
                    "error": "Destination CRS not mapped (RailReferences missing mapping / add alias)",
                    "operator": "",
                    "planned_cancel": "",
                    "serviceUid": "",
                    "runDate": "",
                    "act_from": None,
                    "act_to": None,
                    "act_dep": None,
                    "act_arr": None,
                    "act_dep_raw": None,
                    "act_arr_raw": None,
                    "origin_rtt_deps": "",
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            # Find all services at origin for that headcode
            cand_all = loc_df[loc_df["trainIdentity"] == hc].copy()
            if cand_all.empty:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": "Headcode not found at origin on date",
                    "operator": "",
                    "planned_cancel": "",
                    "serviceUid": "",
                    "runDate": "",
                    "act_from": None,
                    "act_to": None,
                    "act_dep": None,
                    "act_arr": None,
                    "act_dep_raw": None,
                    "act_arr_raw": None,
                    "origin_rtt_deps": "",
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            cand_all["dep_pub_hhmm"] = cand_all["pub_dep_raw"].apply(rtt_public_to_hhmm)
            deps = sorted(set(cand_all["dep_pub_hhmm"].dropna().astype(str).tolist()))
            ops = sorted(set(cand_all["operator"].dropna().astype(str).tolist()))
            origin_rtt_deps = ", ".join(deps)

            # If expected departure present, filter to the matching departure at origin
            candidates = cand_all
            if exp_dep:
                candidates = candidates[candidates["dep_pub_hhmm"] == exp_dep]
                if candidates.empty:
                    results.append({
                        **row.to_dict(),
                        "status": "FAIL",
                        "error": f"Origin departure mismatch (expected {exp_dep})",
                        "operator": ", ".join(ops) if ops else "",
                        "planned_cancel": "",
                        "serviceUid": "",
                        "runDate": "",
                        "act_from": None,
                        "act_to": None,
                        "act_dep": deps[0] if deps else None,
                        "act_arr": None,
                        "act_dep_raw": None,
                        "act_arr_raw": None,
                        "origin_rtt_deps": origin_rtt_deps,
                    })
                    done += 1
                    progress.progress(min(1.0, done / total))
                    continue

            cand = candidates.iloc[0]
            uid = cand.get("serviceUid") or ""
            rtt_run_date = cand.get("runDate") or ""
            op = cand.get("operator") or ""
            pc = "Y" if bool(cand.get("plannedCancel", False)) else ""

            if not uid or not rtt_run_date:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": "Insufficient RTT data to verify calling points",
                    "operator": op,
                    "planned_cancel": pc,
                    "serviceUid": uid,
                    "runDate": rtt_run_date,
                    "act_from": None,
                    "act_to": None,
                    "act_dep": None,
                    "act_arr": None,
                    "act_dep_raw": None,
                    "act_arr_raw": None,
                    "origin_rtt_deps": origin_rtt_deps,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            # Service detail: match FROM/TO by CRS ONLY (permanent fix)
            try:
                detail = rtt_service_detail(uid, rtt_run_date, auth)
                seg = extract_segment_times_by_crs(detail, row["origin_crs"], row["dest_crs"])
            except Exception as ex:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"RTT service detail lookup failed: {ex}",
                    "operator": op,
                    "planned_cancel": pc,
                    "serviceUid": uid,
                    "runDate": rtt_run_date,
                    "act_from": None,
                    "act_to": None,
                    "act_dep": None,
                    "act_arr": None,
                    "act_dep_raw": None,
                    "act_arr_raw": None,
                    "origin_rtt_deps": origin_rtt_deps,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            if not seg["from_found"]:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"FROM CRS not found in RTT service calling points (expected {row['origin_crs']})",
                    "operator": op,
                    "planned_cancel": pc,
                    "serviceUid": uid,
                    "runDate": rtt_run_date,
                    "act_from": seg["act_from"],
                    "act_to": seg["act_to"],
                    "act_dep": seg["act_dep"],
                    "act_arr": seg["act_arr"],
                    "act_dep_raw": seg["act_dep_raw"],
                    "act_arr_raw": seg["act_arr_raw"],
                    "origin_rtt_deps": origin_rtt_deps,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            if not seg["to_found"]:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"TO CRS not found in RTT service calling points (expected {row['dest_crs']})",
                    "operator": op,
                    "planned_cancel": pc,
                    "serviceUid": uid,
                    "runDate": rtt_run_date,
                    "act_from": seg["act_from"],
                    "act_to": seg["act_to"],
                    "act_dep": seg["act_dep"],
                    "act_arr": seg["act_arr"],
                    "act_dep_raw": seg["act_dep_raw"],
                    "act_arr_raw": seg["act_arr_raw"],
                    "origin_rtt_deps": origin_rtt_deps,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            # Compare departure at FROM CRS
            if row["exp_dep"] and seg["act_dep"] and row["exp_dep"] != seg["act_dep"]:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"Departure mismatch at FROM CRS (RTT: {seg['act_dep']})",
                    "operator": op,
                    "planned_cancel": pc,
                    "serviceUid": uid,
                    "runDate": rtt_run_date,
                    "act_from": seg["act_from"],
                    "act_to": seg["act_to"],
                    "act_dep": seg["act_dep"],
                    "act_arr": seg["act_arr"],
                    "act_dep_raw": seg["act_dep_raw"],
                    "act_arr_raw": seg["act_arr_raw"],
                    "origin_rtt_deps": origin_rtt_deps,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            # Compare arrival at TO CRS
            if row["exp_arr"] and seg["act_arr"] and row["exp_arr"] != seg["act_arr"]:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"Arrival mismatch at TO CRS (RTT: {seg['act_arr']})",
                    "operator": op,
                    "planned_cancel": pc,
                    "serviceUid": uid,
                    "runDate": rtt_run_date,
                    "act_from": seg["act_from"],
                    "act_to": seg["act_to"],
                    "act_dep": seg["act_dep"],
                    "act_arr": seg["act_arr"],
                    "act_dep_raw": seg["act_dep_raw"],
                    "act_arr_raw": seg["act_arr_raw"],
                    "origin_rtt_deps": origin_rtt_deps,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            # OK
            results.append({
                **row.to_dict(),
                "status": "OK",
                "error": "",
                "operator": op,
                "planned_cancel": pc,
                "serviceUid": uid,
                "runDate": rtt_run_date,
                "act_from": seg["act_from"],
                "act_to": seg["act_to"],
                "act_dep": seg["act_dep"],
                "act_arr": seg["act_arr"],
                "act_dep_raw": seg["act_dep_raw"],
                "act_arr_raw": seg["act_arr_raw"],
                "origin_rtt_deps": origin_rtt_deps,
            })
            done += 1
            progress.progress(min(1.0, done / total))

    progress.empty()
    report = pd.DataFrame(results)
    st.session_state["report"] = report
else:
    report = existing_report

if report is None or report.empty:
    st.warning("No report available. Click 'Run RTT check' to generate one.")
    st.stop()

# ============================================================
# Output
# ============================================================
ok = report[report["status"] == "OK"]
fail = report[report["status"] == "FAIL"]
nc = report[report["status"] == "NOT CHECKED"]

st.subheader("Outcome")
st.write(f"Checked: **{len(report)}** | OK: **{len(ok)}** | FAIL: **{len(fail)}** | NOT CHECKED: **{len(nc)}**")

st.subheader("What the app found (expected vs RTT actual)")
view_cols = [
    "date", "weekday",
    "headcode",
    "origin_crs", "dest_crs",
    "exp_dep", "act_dep",
    "exp_from", "act_from",
    "exp_to", "act_to",
    "exp_arr", "act_arr",
    "operator",
    "status", "error",
    "diagram_info",
    "serviceUid", "runDate",
    "origin_rtt_deps",
    "act_dep_raw", "act_arr_raw",
]
view_cols = [c for c in view_cols if c in report.columns]
st.dataframe(
    report[view_cols].sort_values(["date", "headcode", "exp_dep"], na_position="last"),
    use_container_width=True
)

with st.expander("Only FAIL / NOT CHECKED"):
    bad = report[report["status"] != "OK"].copy()
    if bad.empty:
        st.write("None")
    else:
        st.dataframe(
            bad[view_cols].sort_values(["date", "status", "headcode"], na_position="last"),
            use_container_width=True
        )

st.subheader("CSV report")
st.write("Creates CSV from the latest run (no re-run needed).")
create_csv = st.checkbox("Create CSV report", value=False)

if create_csv:
    out = report.copy()
    out.loc[out["status"] == "OK", "error"] = ""
    out.loc[out["status"] == "OK", "diagram_info"] = ""

    csv_bytes = out[view_cols].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV report",
        data=csv_bytes,
        file_name=f"rtt_no_brand_report_{date_from}_{date_to}.csv",
        mime="text/csv",
    )
