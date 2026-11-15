"""Build clean analysis tables from raw NEON SOAP data.

Outputs (data/processed/):
  <site>_<month>_two_level.csv    lower/upper temperature and wind, one row per 30 min
  <site>_<month>_temp_profile.csv temperature at all tower levels (for the profile chart)

Flagged readings are set to NaN (not dropped), and the time axis is filled to a
full 30-min grid, so missing data stays visible and the rules engine can report it.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE, MONTH = "SOAP", "2025-06"
RAW = ROOT / "data/raw/neon" / SITE / MONTH
OUT = ROOT / "data/processed"
TZ = "America/Los_Angeles"

LOW, HIGH = 20, 30  # NEON verticalPosition codes: 5.22 m and 11.93 m at SOAP
MS_TO_MPH = 2.236936


def read_levels(path, value_col, qf_col):
    """Return a wide table (time x level). Values with a nonzero quality flag become NaN."""
    df = pd.read_csv(path, parse_dates=["startDateTime"])
    df[value_col] = df[value_col].where(df[qf_col] == 0)
    wide = df.pivot(index="startDateTime", columns="verticalPosition", values=value_col)
    wide.index = wide.index.tz_convert(TZ)
    return wide


def level_heights(path):
    """Map verticalPosition code (10, 20, ...) to sensor height in meters."""
    pos = pd.read_csv(path)
    codes = (pos["HOR.VER"] * 1000).round().astype(int)  # 0.01 -> 10, 0.05 -> 50
    return dict(zip(codes, pos["zOffset"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    heights = level_heights(RAW / "temperature/sensor_positions_00002.csv")

    T = read_levels(RAW / "temperature/SAAT_30min.csv", "tempSingleMean", "finalQF")
    wind_path = RAW / "wind/2DWSD_30min.csv"
    WS = read_levels(wind_path, "windSpeedMean", "windSpeedFinalQF") * MS_TO_MPH
    WD = read_levels(wind_path, "windDirMean", "windDirFinalQF")

    # Full 30-min grid so gaps stay visible as NaN rows
    start = min(T.index.min(), WS.index.min())
    end = max(T.index.max(), WS.index.max())
    idx = pd.date_range(start, end, freq="30min")

    two = pd.DataFrame({
        "T_low_C": T[LOW],
        "T_high_C": T[HIGH],
        "ws_low_mph": WS[LOW],
        "ws_high_mph": WS[HIGH],
        "wd_low_deg": WD[LOW],
        "wd_high_deg": WD[HIGH],
    }).reindex(idx)
    two.index.name = "timestamp_local"

    profile = T.reindex(idx)
    profile.columns = [f"T_{heights[c]:.2f}m" for c in profile.columns]
    profile.index.name = "timestamp_local"

    prefix = f"{SITE.lower()}_{MONTH}"
    two_path = OUT / f"{prefix}_two_level.csv"
    profile_path = OUT / f"{prefix}_temp_profile.csv"
    two.round(3).to_csv(two_path)
    profile.round(3).to_csv(profile_path)

    print(f"Lower level: {heights[LOW]} m | Upper level: {heights[HIGH]} m")
    print(f"Time range: {idx[0]} to {idx[-1]} ({len(idx)} rows)")
    print("\nMissing values per column:")
    print(two.isna().sum())
    print(f"\nSaved: {two_path.relative_to(ROOT)}")
    print(f"Saved: {profile_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()