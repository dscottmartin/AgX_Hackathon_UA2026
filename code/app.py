"""Spray or Wait: orchard dashboard for pesticide drift risk from two-height weather data.

Run from the project root:
    streamlit run code/app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from rules import CAUTION, GO, NO_DATA, STOP, TZ, Config, evaluate_series, load_processed

ROOT = Path(__file__).resolve().parents[1]
NEON_TWO = ROOT / "data/processed/soap_2025-06_two_level.csv"
NEON_PROFILE = ROOT / "data/processed/soap_2025-06_temp_profile.csv"
NEON_POS = ROOT / "data/raw/neon/SOAP/2025-06/temperature/sensor_positions_00002.csv"

DEMO_DATE = pd.Timestamp("2025-06-15").date()
NEON_FIRST, NEON_LAST = pd.Timestamp("2025-06-01").date(), pd.Timestamp("2025-06-29").date()
STEPS = 48         # 30-min readings from noon to noon
DEFAULT_IDX = 15   # 19:30

COLORS = {GO: "#1F7A3A", CAUTION: "#E9A23B", STOP: "#B42318", NO_DATA: "#5B6470"}
TEXT_ON = {GO: "#FFFFFF", CAUTION: "#2A1E05", STOP: "#FFFFFF", NO_DATA: "#FFFFFF"}
ACTION = {
    GO: "Conditions support spraying. Follow the label.",
    CAUTION: "Spray only with extra care, and recheck soon.",
    STOP: "Do not spray now.",
    NO_DATA: "Readings are missing. Check conditions on site.",
}
NEON_LABELS = dict(low_label="5 m (17 ft)", high_label="12 m (39 ft)")
SIM_LABELS = dict(low_label="1.5 m (5 ft)", high_label="9 m (30 ft)")

SCENARIOS = {
    "Evening inversion, breezy ground": dict(inv=0.9, day_wind=5.5, night_wind=4.0),
    "Wind shift above the trees": dict(inv=0.2, day_wind=4.5, night_wind=3.6, shear=True),
    "Overcast and breezy": dict(inv=0.05, day_wind=5.0, night_wind=3.8),
    "Typical clear night": dict(inv=0.8, day_wind=5.0, night_wind=1.2),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  font-family: 'Atkinson Hyperlegible', system-ui, -apple-system, sans-serif;
}
.sw-title { font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0; }
.sw-sub { opacity: .75; margin: .25rem 0 1.25rem; max-width: 64ch; }
.sw-status { border-radius: 8px; padding: 18px 22px 16px; margin-bottom: 12px; }
.sw-word { font-size: 3.4rem; font-weight: 700; line-height: 1; }
.sw-action { font-size: 1.2rem; font-weight: 700; margin-top: 8px; }
.sw-when { font-size: .95rem; opacity: .9; margin-top: 8px; }
.sw-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
.sw-mini { border-radius: 8px; padding: 10px 14px; }
.sw-mini .lbl { font-size: .9rem; opacity: .9; }
.sw-mini .w { font-size: 1.5rem; font-weight: 700; }
.sw-note { border-left: 5px solid #B42318; padding: 8px 12px; margin: 2px 0 12px;
           background: rgba(180, 35, 24, .10); }
@media (max-width: 640px) {
  .sw-compare { grid-template-columns: 1fr; }
  .sw-word { font-size: 2.6rem; }
}
</style>
"""


# ---------- Data ----------

@st.cache_data
def neon_log(volatility: str) -> pd.DataFrame:
    return evaluate_series(load_processed(NEON_TWO), Config.for_volatility(volatility, **NEON_LABELS))


@st.cache_data
def neon_profile() -> pd.DataFrame:
    df = load_processed(NEON_PROFILE)
    return df.rename(columns={c: float(c[2:-1]) for c in df.columns})  # "T_0.33m" -> 0.33


@st.cache_data
def neon_location() -> tuple[float, float]:
    try:
        pos = pd.read_csv(NEON_POS)
        return float(pos.locationReferenceLatitude.iloc[0]), float(pos.locationReferenceLongitude.iloc[0])
    except Exception:
        return 37.033, -119.262


@st.cache_data
def simulate(scenario: str, seed: int, date) -> pd.DataFrame:
    """Simulated two-height sensors from noon to noon, with a day/night cycle."""
    p = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(date).tz_localize(TZ) + pd.Timedelta(hours=12)
    idx = pd.date_range(start, periods=STEPS, freq="30min", name="timestamp_local")
    t = np.arange(STEPS) / 2                                   # hours since noon
    night = np.clip(np.sin(np.pi * (t - 7) / 14), 0, None)     # 0 at 19:00 and 09:00, peaks ~02:00
    T_low = 20 + 8 * np.cos(2 * np.pi * (t - 3) / 24) + rng.normal(0, 0.05, STEPS)
    dT = -0.4 * (1 - night) + p["inv"] * night + rng.normal(0, 0.03, STEPS)
    ws_low = np.clip(p["day_wind"] * (1 - night) + p["night_wind"] * night + rng.normal(0, 0.3, STEPS), 0.2, None)
    ws_high = np.clip(ws_low * 1.35 + rng.normal(0, 0.3, STEPS), 0.2, None)
    wd_low = 290 + rng.normal(0, 6, STEPS)
    wd_high = wd_low + rng.normal(0, 6, STEPS)
    if p.get("shear"):
        gust = (t >= 4) & (t <= 7.5)                           # 16:00 to 19:30
        ws_high[gust] = 11 + rng.normal(0, 0.8, gust.sum())
        wd_high[gust] = 200 + rng.normal(0, 8, gust.sum())
    return pd.DataFrame({
        "T_low_C": T_low, "T_high_C": T_low + dT,
        "ws_low_mph": ws_low, "ws_high_mph": ws_high,
        "wd_low_deg": wd_low % 360, "wd_high_deg": wd_high % 360,
    }, index=idx).round(3)


@st.cache_data
def sim_log(scenario: str, seed: int, date, volatility: str) -> pd.DataFrame:
    return evaluate_series(simulate(scenario, seed, date), Config.for_volatility(volatility, **SIM_LABELS))


def noon_window(df: pd.DataFrame, date) -> pd.DataFrame:
    start = pd.Timestamp(date).tz_localize(TZ) + pd.Timedelta(hours=12)
    return df.loc[start:start + pd.Timedelta(minutes=30 * (STEPS - 1))]


def orchard_view(o: dict, volatility: str) -> dict:
    """Everything the page needs for one orchard: its log window, profile, and labels."""
    if o["source"] == "neon":
        date = st.session_state.neon_date
        full = neon_log(volatility)
        view = dict(win=noon_window(full, date), profile=noon_window(neon_profile(), date), full=full,
                    labels=NEON_LABELS, source="Real NEON tower data (replay)", key=f"t_{o['id']}_{date}")
    else:
        date = pd.Timestamp.now(tz=TZ).date()
        win = sim_log(o["scenario"], o["seed"], date, volatility)
        view = dict(win=win, profile=None, full=win, labels=SIM_LABELS,
                    source=f"Simulated sensors: {o['scenario'].lower()}", key=f"t_{o['id']}_{date}")
    n = len(view["win"])
    if n:
        st.session_state.pos.setdefault(view["key"], min(DEFAULT_IDX, n - 1))
    return view


def init_state():
    if "orchards" in st.session_state:
        return
    lat, lon = neon_location()
    st.session_state.orchards = [
        dict(id="neon-soap", name="NEON SOAP tower", lat=lat, lon=lon, source="neon"),
        dict(id="sim-parlier", name="Parlier almond block", lat=36.611, lon=-119.527, source="sim",
             scenario="Evening inversion, breezy ground", seed=1),
        dict(id="sim-kerman", name="Kerman pistachio block", lat=36.723, lon=-120.060, source="sim",
             scenario="Wind shift above the trees", seed=2),
    ]
    st.session_state.selected_id = "neon-soap"
    st.session_state.neon_date = DEMO_DATE
    st.session_state.playing = False
    st.session_state.pos = {}  # selected time index per orchard window (survives widget cleanup)
    st.session_state.new_lat, st.session_state.new_lon = 36.75, -119.80


# ---------- Page ----------

st.set_page_config(page_title="Spray or Wait", page_icon="🌳", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
init_state()
if "pending_select" in st.session_state:
    st.session_state.selected_id = st.session_state.pop("pending_select")

with st.sidebar:
    volatility = st.radio(
        "Product volatility", ["low", "medium", "high"], index=1, format_func=str.capitalize,
        help="Volatile products can keep drifting as vapor, so any real inversion becomes a STOP.")

orchards = st.session_state.orchards
names = {o["id"]: o["name"] for o in orchards}
views = {o["id"]: orchard_view(o, volatility) for o in orchards}
sel_id = st.session_state.selected_id
sel, v = next(o for o in orchards if o["id"] == sel_id), views[sel_id]

# Advance playback before the time slider is drawn
if st.session_state.playing and len(v["win"]):
    cur = st.session_state.pos[v["key"]]
    if cur < len(v["win"]) - 1:
        st.session_state.pos[v["key"]] = cur + 1
    else:
        st.session_state.playing = False


def status_now(view: dict) -> str:
    if not len(view["win"]):
        return NO_DATA
    return view["win"].status.iloc[st.session_state.pos[view["key"]]]


def sync(src: str, dst: str | None = None, pos_key: str | None = None):
    """Copy a widget value into state that persists when the widget is not drawn."""
    if pos_key is not None:
        st.session_state.pos[pos_key] = st.session_state[src]
    else:
        st.session_state[dst] = st.session_state[src]


st.markdown(
    '<p class="sw-title">Spray or Wait</p>'
    '<p class="sw-sub">A two-height weather check for orchard spraying. It catches inversions and '
    'wind changes above the trees that a single ground station cannot see.</p>',
    unsafe_allow_html=True)

left, right = st.columns([5, 7], gap="large")

with left:
    fmap = folium.Map(location=[36.8, -119.7], zoom_start=8, tiles="OpenStreetMap")
    for o in orchards:
        is_sel = o["id"] == sel_id
        folium.CircleMarker(
            [o["lat"], o["lon"]], radius=12 if is_sel else 8,
            color="#1D2621" if is_sel else "#FFFFFF", weight=3 if is_sel else 1.5,
            fill=True, fill_color=COLORS[status_now(views[o["id"]])], fill_opacity=0.95,
            tooltip=o["name"],
        ).add_to(fmap)
    out = st_folium(fmap, height=440, use_container_width=True, key="map",
                    returned_objects=["last_object_clicked_tooltip", "last_clicked"])

    if out:
        tip = out.get("last_object_clicked_tooltip")
        if tip and tip != st.session_state.get("_last_tip"):
            st.session_state._last_tip = tip
            match = next((o["id"] for o in orchards if o["name"] == tip), None)
            if match and match != sel_id:
                st.session_state.pending_select = match
                st.rerun()
        clicked = out.get("last_clicked")
        if clicked and clicked != st.session_state.get("_last_click"):
            st.session_state._last_click = clicked
            st.session_state.new_lat = round(clicked["lat"], 5)
            st.session_state.new_lon = round(clicked["lng"], 5)

    st.selectbox("Orchard", list(names), format_func=names.get, key="selected_id")
    st.caption("Pin color shows each orchard's status at its selected time. Click a pin to open it, "
               "or click an empty spot to copy its coordinates into Add an orchard.")

with right:
    if sel["source"] == "neon":
        st.session_state["_neon_date"] = st.session_state.neon_date
        st.date_input("Night of", key="_neon_date", min_value=NEON_FIRST, max_value=NEON_LAST,
                      on_change=sync, args=("_neon_date", "neon_date"),
                      help="Each window runs from noon to noon the next day.")
    win = v["win"]
    if win.empty:
        st.warning("No readings for this date. Pick another night.")
        st.stop()

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Play the night", width="stretch"):
        st.session_state.playing = True
        st.session_state.pos[v["key"]] = 0
    if b2.button("Pause", width="stretch"):
        st.session_state.playing = False

    labels = [ts.strftime("%a %H:%M") for ts in win.index]
    wkey = "w_" + v["key"]
    st.session_state[wkey] = st.session_state.pos[v["key"]]
    idx = st.select_slider("Time", options=list(range(len(win))), format_func=lambda i: labels[i],
                           key=wkey, on_change=sync, args=(wkey,), kwargs={"pos_key": v["key"]})
    row, ts = win.iloc[idx], win.index[idx]
    status, single = row.status, row.single_station_status

    st.markdown(
        f'<div class="sw-status" style="background:{COLORS[status]};color:{TEXT_ON[status]}">'
        f'<div class="sw-word">{status}</div>'
        f'<div class="sw-action">{ACTION[status]}</div>'
        f'<div class="sw-when">{ts:%A %b %d, %H:%M}<br>{v["source"]}</div></div>',
        unsafe_allow_html=True)

    def mini(label, s):
        return (f'<div class="sw-mini" style="background:{COLORS[s]};color:{TEXT_ON[s]}">'
                f'<div class="lbl">{label}</div><div class="w">{s}</div></div>')

    st.markdown(
        '<div class="sw-compare">'
        + mini(f"Single ground station ({v['labels']['low_label']} wind only)", single)
        + mini("Two heights (this tool)", status) + "</div>",
        unsafe_allow_html=True)
    if single == GO and status != GO:
        st.markdown('<div class="sw-note">A single ground station would say GO right now. '
                    'The second height shows the risk.</div>', unsafe_allow_html=True)

    st.markdown("**Why**")
    st.markdown("\n".join(f"- {r}" for r in row.reasons.split(" | ") if r))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Upper minus lower temp", "n/a" if pd.isna(row.dT_C) else f"{row.dT_C:+.2f} °C",
              help="Positive means warmer air above: an inversion.")
    m2.metric(f"Wind at {v['labels']['low_label']}",
              "n/a" if pd.isna(row.ws_low_mph) else f"{row.ws_low_mph:.1f} mph")
    m3.metric(f"Wind at {v['labels']['high_label']}",
              "n/a" if pd.isna(row.ws_high_mph) else f"{row.ws_high_mph:.1f} mph")
    if pd.isna(row.wd_low_deg) or pd.isna(row.wd_high_deg):
        m4.metric("Direction difference", "n/a", help="Direction is not reliable in light wind.")
    else:
        m4.metric("Direction difference", f"{abs((row.wd_high_deg - row.wd_low_deg + 180) % 360 - 180):.0f}°")

    if status != GO:
        later = win.iloc[idx + 1:]
        nxt = later.index[later.status == GO]
        if len(nxt):
            mins = int((nxt[0] - ts).total_seconds() // 60)
            st.info(f"Next GO reading in this data: {nxt[0]:%a %H:%M}, "
                    f"{mins // 60} h {mins % 60} min after the selected time.")
        else:
            st.info("No GO reading in the rest of this window.")

# ---------- Charts ----------

plot = win.copy()
plot.index = plot.index.tz_localize(None)
plot = plot.reset_index().rename(columns={"timestamp_local": "time"})
plot["end"] = plot.time + pd.Timedelta(minutes=30)
sel_time = pd.DataFrame({"time": [ts.tz_localize(None)]})

x = alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M", labelAngle=0))
rule = alt.Chart(sel_time).mark_rule(color="#1D2621", strokeWidth=2).encode(x="time:T")
low_lbl, high_lbl = v["labels"]["low_label"], v["labels"]["high_label"]
height_scale = alt.Scale(domain=[low_lbl, high_lbl], range=["#8A9A8F", "#1D2621"])

temp_long = plot.melt("time", ["T_low_C", "T_high_C"], var_name="height", value_name="value")
temp_long["height"] = temp_long.height.map({"T_low_C": low_lbl, "T_high_C": high_lbl})
temp = alt.Chart(temp_long).mark_line(strokeWidth=2.5).encode(
    x=x, y=alt.Y("value:Q", title="Air temp (°C)", scale=alt.Scale(zero=False)),
    color=alt.Color("height:N", title=None, scale=height_scale, legend=alt.Legend(orient="top")))

wind_long = plot.melt("time", ["ws_low_mph", "ws_high_mph"], var_name="height", value_name="value")
wind_long["height"] = wind_long.height.map({"ws_low_mph": low_lbl, "ws_high_mph": high_lbl})
wind = alt.Chart(wind_long).mark_line(strokeWidth=2.5).encode(
    x=x, y=alt.Y("value:Q", title="Wind (mph)"),
    color=alt.Color("height:N", title=None, scale=height_scale, legend=None))
limits = alt.Chart(pd.DataFrame({"y": [3, 10]})).mark_rule(strokeDash=[4, 4], color="#8A9A8F").encode(y="y:Q")

strip_long = pd.concat([
    plot[["time", "end", "status", "triggers"]].assign(view="Two heights"),
    plot[["time", "end", "single_station_status", "triggers"]]
    .rename(columns={"single_station_status": "status"}).assign(view="Ground station only"),
])
strip = alt.Chart(strip_long).mark_rect().encode(
    x=x, x2="end:T", y=alt.Y("view:N", title=None, sort=["Two heights", "Ground station only"]),
    color=alt.Color("status:N", title=None, scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                    legend=alt.Legend(orient="bottom")),
    tooltip=[alt.Tooltip("time:T", format="%a %H:%M"), "view:N", "status:N", "triggers:N"])

timeline = alt.vconcat(
    (temp + rule).properties(height=170),
    (wind + limits + rule).properties(height=140),
    (strip + rule).properties(height=70),
).resolve_scale(color="independent")

st.markdown("#### Conditions from noon to noon")
c_chart, c_prof = st.columns([8, 4], gap="large")
c_chart.altair_chart(timeline, width="stretch")

with c_prof:
    prof = v["profile"]
    if prof is not None and ts in prof.index and prof.loc[ts].notna().sum() >= 2:
        p = prof.loc[ts].dropna()
        pdf = pd.DataFrame({"height_m": p.index.astype(float), "temp": p.values})
        st.altair_chart(
            alt.Chart(pdf).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3,
                                     color=COLORS[status]).encode(
                x=alt.X("temp:Q", title="Air temp (°C)", scale=alt.Scale(zero=False)),
                y=alt.Y("height_m:Q", title="Height above ground (m)"),
                order="height_m:Q").properties(height=330, title=f"Tower profile at {ts:%H:%M}"),
            width="stretch")
        st.caption("A line that leans right as it rises means warmer air above: an inversion.")
    else:
        st.caption("A full tower profile is available for NEON data only.")

# ---------- Log ----------

st.markdown("#### Log")
log_cols = ["status", "single_station_status", "triggers", "dT_C", "ws_low_mph", "ws_high_mph", "reasons"]
log_view = win[log_cols].copy()
log_view.index = log_view.index.strftime("%a %b %d %H:%M")
st.dataframe(log_view.round(2), width="stretch", height=300)
st.download_button("Download full log (CSV)", data=v["full"].round(3).to_csv().encode(),
                   file_name=f"{sel['id']}_{volatility}_drift_log.csv", mime="text/csv")

st.caption("This tool supports the applicator's decision. It does not replace the pesticide label, "
           "on-site checks, or regulations, and no system can guarantee zero drift.")


# ---------- Sidebar: add an orchard ----------

with st.sidebar:
    st.markdown("### Add an orchard")
    with st.form("add_orchard"):
        name = st.text_input("Name", placeholder="North almond block")
        lat = st.number_input("Latitude", key="new_lat", min_value=-90.0, max_value=90.0, format="%.5f")
        lon = st.number_input("Longitude", key="new_lon", min_value=-180.0, max_value=180.0, format="%.5f")
        scenario = st.selectbox("Simulated conditions", list(SCENARIOS))
        submitted = st.form_submit_button("Add orchard", width="stretch")
    if submitted:
        name = name.strip()
        if not name:
            st.error("Give the orchard a name.")
        elif name in names.values():
            st.error("An orchard with that name already exists. Pick another name.")
        else:
            new_id = f"user-{len(orchards)}"
            orchards.append(dict(id=new_id, name=name, lat=lat, lon=lon, source="sim",
                                 scenario=scenario, seed=len(orchards) + 10))
            st.session_state.pending_select = new_id
            st.rerun()
    st.caption("New orchards use simulated two-height sensors until real sensors are connected.")

if st.session_state.playing:
    time.sleep(0.5)
    st.rerun()
