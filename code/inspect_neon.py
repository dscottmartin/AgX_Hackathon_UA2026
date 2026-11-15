"""Inspect NEON SOAP data: sensor heights and inversion frequency."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/neon/SOAP/2025-06"

# Sensor height for each tower level
pos = pd.read_csv(RAW / "temperature/sensor_positions_00002.csv")
print("Sensor heights (zOffset, m):")
print(pos[["HOR.VER", "zOffset"]])

# Temperature: drop flagged rows, one column per level, convert to local time
t = pd.read_csv(RAW / "temperature/SAAT_30min.csv", parse_dates=["startDateTime"])
t = t[t.finalQF == 0]
T = t.pivot(index="startDateTime", columns="verticalPosition", values="tempSingleMean")
T.index = T.index.tz_convert("America/Los_Angeles")

# Temperature difference: top level minus bottom level (positive = inversion)
dT = T[T.columns.max()] - T[T.columns.min()]
print(f"\nInversion frequency: {(dT > 0).mean():.0%}")
print("\nMean dT by local hour (degC):")
print(dT.groupby(dT.index.hour).mean().round(2))

# Compare dT for candidate level pairs (upper - lower), by local hour
pairs = [(10, 30), (20, 30), (20, 50), (10, 50)]
dT_pairs = pd.DataFrame({f"{hi}-{lo}": T[hi] - T[lo] for lo, hi in pairs})
print("\nMean dT by local hour for each level pair (degC):")
print(dT_pairs.groupby(dT_pairs.index.hour).mean().round(2))

# Wind speed by level, converted to mph
w = pd.read_csv(RAW / "wind/2DWSD_30min.csv", parse_dates=["startDateTime"])
w = w[w.windSpeedFinalQF == 0]
WS = w.pivot(index="startDateTime", columns="verticalPosition", values="windSpeedMean") * 2.237
WS.index = WS.index.tz_convert("America/Los_Angeles")
print("\nMean wind speed by local hour and level (mph):")
print(WS.groupby(WS.index.hour).mean().round(1))
print("\nData availability per level (fraction of rows kept):")
print((WS.notna().mean()).round(2))

# Check when wind direction is flagged as missing
two = pd.read_csv(ROOT / "data/processed/soap_2025-06_two_level.csv",
                  parse_dates=["timestamp_local"], index_col="timestamp_local")
two.index = pd.to_datetime(two.index, utc=True).tz_convert("America/Los_Angeles")
missing = two.wd_low_deg.isna()
print("\nFraction of wind direction missing, by local hour:")
print(missing.groupby(two.index.hour).mean().round(2))
print("\nLower wind speed (mph) when direction is missing vs present:")
print(two.ws_low_mph.groupby(missing).describe().round(2))