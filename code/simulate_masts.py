"""Simulated readings from a network of our two-height masts (5 ft and 30 ft) for Spray or Wait.

Each site gets its own CSV with the same columns as the NEON replay, so rules.py and app.py
use them unchanged:
    timestamp_local, T_low_C, T_high_C, ws_low_mph, ws_high_mph, wd_low_deg, wd_high_deg

How the network is simulated:
  * Each night (noon to noon) has one regional weather pattern shared by every site,
    because nearby orchards see the same large-scale weather.
  * Each site changes that pattern by its terrain (valley floor, foothill edge, west side)
    and adds its own random variation, so sites disagree the way real sites do.
  * A few sensor faults are scheduled so the demo shows NO DATA handling.

Swap a site's CSV for a real mast export later; nothing else has to change.

Run from the project root:
    python code/simulate_masts.py
    python code/simulate_masts.py --start 2026-06-01 --nights 14 --sites 10 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/Los_Angeles"
STEP_MIN = 30

# ---------- Sites ----------
# Orchards across the San Joaquin Valley. Terrain decides how each site bends the regional weather.
SITES = [
    dict(site_id="reedley", name="Reedley peach block", crop="Peach", lat=36.596, lon=-119.450, terrain="floor"),
    dict(site_id="orange-cove", name="Orange Cove citrus", crop="Citrus", lat=36.624, lon=-119.313, terrain="foothill"),
    dict(site_id="kingsburg", name="Kingsburg nectarine block", crop="Nectarine", lat=36.514, lon=-119.554, terrain="floor"),
    dict(site_id="madera", name="Madera almond block", crop="Almond", lat=36.961, lon=-120.060, terrain="floor"),
    dict(site_id="chowchilla", name="Chowchilla almond block", crop="Almond", lat=37.123, lon=-120.260, terrain="floor"),
    dict(site_id="visalia", name="Visalia walnut block", crop="Walnut", lat=36.330, lon=-119.292, terrain="floor"),
    dict(site_id="exeter", name="Exeter citrus", crop="Citrus", lat=36.296, lon=-119.142, terrain="foothill"),
    dict(site_id="hanford", name="Hanford pistachio block", crop="Pistachio", lat=36.327, lon=-119.645, terrain="floor"),
    dict(site_id="mendota", name="Mendota pistachio block", crop="Pistachio", lat=36.754, lon=-120.381, terrain="west"),
    dict(site_id="lindsay", name="Lindsay olive block", crop="Olive", lat=36.203, lon=-119.088, terrain="foothill"),
]

# inv / wind: multipliers on the regional pattern
# drain: on nights with an inversion, the 5 ft wind turns toward this direction (cold air
#        sliding down from the Sierra foothills), while the 30 ft wind keeps the valley direction
TERRAIN = {
    "floor": dict(inv=1.15, wind=0.9),                  # cold air pools on the valley floor
    "foothill": dict(inv=0.9, wind=1.0, drain=70),      # downslope drainage at night
    "west": dict(inv=0.7, wind=1.3),                    # windier west side, weaker inversions
}

# ---------- Regional weather patterns ----------
#   inv        peak inversion before dawn: 30 ft minus 5 ft temperature (°C)
#   day_wind   mid-afternoon wind at 5 ft (mph)
#   night_wind night wind at 5 ft (mph)
#   jet        extra speed-up of the 30 ft wind at night (low-level jet above the canopy)
#   drain      direction (deg) the 5 ft wind turns toward at night, at every site
#   cloud      overcast: smaller day/night temperature swing
PATTERNS = {
    "Clear and calm": dict(inv=1.8, day_wind=6.0, night_wind=1.2),
    "Inversion with a breeze at 5 ft": dict(inv=0.8, day_wind=6.0, night_wind=4.5),
    "Windy front": dict(inv=0.0, day_wind=12.0, night_wind=8.0, cloud=True),
    "Overcast and breezy": dict(inv=0.05, day_wind=5.5, night_wind=4.0, cloud=True),
    "Drainage flow under the canopy": dict(inv=0.25, day_wind=5.5, night_wind=3.4, drain=150),
    "Weak inversion": dict(inv=0.2, day_wind=5.0, night_wind=3.8),
    "Low-level jet above the trees": dict(inv=0.9, day_wind=7.0, night_wind=3.6, jet=1.7),
}

# Night 2 is the main demo: a single ground station says GO while the masts say STOP.
SCHEDULE = [
    "Clear and calm",
    "Inversion with a breeze at 5 ft",
    "Overcast and breezy",
    "Drainage flow under the canopy",
    "Windy front",
    "Weak inversion",
    "Low-level jet above the trees",
    "Clear and calm",
    "Overcast and breezy",
    "Inversion with a breeze at 5 ft",
    "Weak inversion",
    "Clear and calm",
    "Drainage flow under the canopy",
    "Overcast and breezy",
]

# Sensor faults: (site_id, night index, which sensor, start clock hour, hours)
FAULTS = [
    ("reedley", 7, "high", 23.0, 1.5),     # 30 ft sensor drops out
    ("hanford", 1, "low", 2.0, 1.0),       # 5 ft sensor drops out
    ("mendota", 9, "high", 20.5, 3.0),
]
COLS = {"low": ["T_low_C", "ws_low_mph", "wd_low_deg"], "high": ["T_high_C", "ws_high_mph", "wd_high_deg"]}


def _bump(hours: np.ndarray, start: float, end: float) -> np.ndarray:
    """0 outside [start, end], rising smoothly to 1 in the middle (hours since noon)."""
    x = np.clip((hours - start) / (end - start), 0, 1)
    return np.sin(np.pi * x)


def _blend_dir(a_deg, b_deg, w):
    """Turn direction a toward b by weight w (0..1), going the short way around."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    x = (1 - w) * np.cos(a) + w * np.cos(b)
    y = (1 - w) * np.sin(a) + w * np.sin(b)
    return np.degrees(np.arctan2(y, x)) % 360


def regional_night(pattern: str, rng: np.random.Generator) -> dict:
    """This night's regional weather: the pattern plus night-to-night variation."""
    p = dict(PATTERNS[pattern])
    p["inv"] *= 1 + rng.normal(0, 0.15)
    p["day_wind"] *= 1 + rng.normal(0, 0.1)
    p["night_wind"] *= 1 + rng.normal(0, 0.1)
    p["t_mean"] = 24 + rng.normal(0, 1.5)
    p["base_dir"] = 315 + rng.normal(0, 10)          # northwesterly valley wind
    return p


def simulate_site_night(region: dict, site: dict, date, rng: np.random.Generator) -> pd.DataFrame:
    """48 half-hour readings at one site, from noon on `date` to 11:30 the next morning."""
    ter = TERRAIN[site["terrain"]]
    inv = region["inv"] * ter["inv"] * (1 + rng.normal(0, 0.2))
    wind_k = ter["wind"] * (1 + rng.normal(0, 0.12))
    day_wind, night_wind = region["day_wind"] * wind_k, region["night_wind"] * wind_k
    cloud = region.get("cloud", False)

    n = 24 * 60 // STEP_MIN
    start = pd.Timestamp(date).tz_localize(TZ) + pd.Timedelta(hours=12)
    idx = pd.date_range(start, periods=n, freq=f"{STEP_MIN}min", name="timestamp_local")
    h = np.arange(n) * STEP_MIN / 60                               # hours since noon

    night = _bump(h, 7.0, 21.0)             # 19:00 to 09:00, deepest around 02:00
    inv_shape = _bump(h, 7.5, 20.5) ** 0.7  # inversion forms after sunset, breaks after sunrise

    # Temperature at 5 ft: warmest mid-afternoon, coolest at dawn
    t_mean = region["t_mean"] + rng.normal(0, 0.7)
    amp = 4.0 if cloud else 8.0
    T_low = t_mean + amp * np.cos(2 * np.pi * (h - 3.5) / 24) + rng.normal(0, 0.05, n)

    # 30 ft minus 5 ft: slightly cooler aloft in the day, warmer aloft at night if inverted
    day_lapse = -0.1 if cloud else -0.3
    T_high = T_low + day_lapse * (1 - night) + inv * inv_shape + rng.normal(0, 0.04, n)

    # Wind at 5 ft (partly inside the canopy): breezy afternoon, calmer night
    afternoon = 0.75 + 0.25 * np.cos(2 * np.pi * (h - 4) / 24)
    ws_low = day_wind * afternoon * (1 - night) + night_wind * night
    ws_low = np.clip(ws_low + rng.normal(0, 0.35, n), 0.1, None)

    # Wind at 30 ft (above the canopy): faster, and decoupled from the ground at night
    jet = region.get("jet", 0.6 * min(max(inv, 0), 1.0))
    ws_high = np.clip(ws_low * (1.5 + jet * night) + rng.normal(0, 0.4, n), 0.1, None)

    # Direction: noisier when the wind is light
    base = region["base_dir"] + rng.normal(0, 5)
    wd_high = base + rng.normal(0, 1, n) * np.minimum(20 / np.maximum(ws_high, 1), 60)
    wd_low = base + 10 + rng.normal(0, 1, n) * np.minimum(25 / np.maximum(ws_low, 1), 60)
    drain = region.get("drain")
    if drain is None and "drain" in ter and inv >= 0.5:
        drain = ter["drain"]
    if drain is not None:
        wd_low = _blend_dir(wd_low, drain, np.clip(night * 1.4, 0, 1))

    return pd.DataFrame({
        "T_low_C": T_low.round(2), "T_high_C": T_high.round(2),
        "ws_low_mph": ws_low.round(1), "ws_high_mph": ws_high.round(1),
        "wd_low_deg": (wd_low % 360).round(0), "wd_high_deg": (wd_high % 360).round(0),
    }, index=idx)


def apply_faults(df: pd.DataFrame, site_id: str, dates) -> pd.DataFrame:
    for fid, night_i, sensor, clock_start, hours in FAULTS:
        if fid != site_id or night_i >= len(dates):
            continue
        start = pd.Timestamp(dates[night_i]).tz_localize(TZ) + pd.Timedelta(hours=clock_start)
        if clock_start < 12:                        # after midnight belongs to the next calendar day
            start += pd.Timedelta(days=1)
        gap = (df.index >= start) & (df.index < start + pd.Timedelta(hours=hours))
        df.loc[gap, COLS[sensor]] = np.nan
    return df


def simulate_network(start: str, nights: int, n_sites: int, seed: int):
    rng = np.random.default_rng(seed)
    sites = SITES[:n_sites]
    dates = list(pd.date_range(start, periods=nights, freq="D"))
    plan = []
    parts = {s["site_id"]: [] for s in sites}
    for i, d in enumerate(dates):
        pattern = SCHEDULE[i % len(SCHEDULE)]
        region = regional_night(pattern, rng)
        plan.append({"night_of": d.date(), "pattern": pattern})
        for s in sites:
            parts[s["site_id"]].append(simulate_site_night(region, s, d, rng))
    data = {sid: apply_faults(pd.concat(p), sid, dates) for sid, p in parts.items()}
    return pd.DataFrame(sites), data, pd.DataFrame(plan)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Simulate a network of 5 ft / 30 ft masts.")
    ap.add_argument("--start", default="2026-06-01", help="First night (noon of this date)")
    ap.add_argument("--nights", type=int, default=14)
    ap.add_argument("--sites", type=int, default=len(SITES), help=f"How many sites (max {len(SITES)})")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=root / "data/processed/masts")
    args = ap.parse_args()

    sites, data, plan = simulate_network(args.start, args.nights, args.sites, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.csv"):        # clear files from an earlier run
        old.unlink()
    for sid, df in data.items():
        df.to_csv(args.out / f"{sid}.csv")
    sites.to_csv(args.out / "sites.csv", index=False)
    plan.to_csv(args.out / "nights.csv", index=False)

    print(f"Wrote {len(data)} sites x {len(next(iter(data.values())))} readings to {args.out}")
    print(sites[["site_id", "name", "terrain"]].to_string(index=False))
    print()
    print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
