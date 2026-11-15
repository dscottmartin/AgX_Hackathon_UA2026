"""Find timestamps where a single ground station says GO but multi-height data shows risk."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/neon/SOAP/2025-06"
LOW, HIGH = 20, 30  # 5.22 m and 11.93 m

def load(path, value_cols, qf_col):
    df = pd.read_csv(path, parse_dates=["startDateTime"])
    df = df[df[qf_col] == 0]
    out = {c: df.pivot(index="startDateTime", columns="verticalPosition", values=c) for c in value_cols}
    for v in out.values():
        v.index = v.index.tz_convert("America/Los_Angeles")
    return out

T = load(RAW / "temperature/SAAT_30min.csv", ["tempSingleMean"], "finalQF")["tempSingleMean"]
W = load(RAW / "wind/2DWSD_30min.csv", ["windSpeedMean", "windDirMean"], "windSpeedFinalQF")

df = pd.DataFrame({
    "T_low": T[LOW], "T_high": T[HIGH],
    "ws_low": W["windSpeedMean"][LOW] * 2.237, "ws_high": W["windSpeedMean"][HIGH] * 2.237,
    "wd_low": W["windDirMean"][LOW], "wd_high": W["windDirMean"][HIGH],
}).dropna()

df["dT"] = df.T_high - df.T_low
df["dWS"] = (df.ws_high - df.ws_low).abs()
df["dWD"] = np.abs((df.wd_high - df.wd_low + 180) % 360 - 180)  # handles 350 vs 10 wraparound

# Single ground station view: only checks lower-level wind speed
ground_says_go = df.ws_low.between(3, 10)
# Multi-height view: inversion or shear
multi_flags = (df.dT > 0) | (df.dWD > 45) | (df.dWS > 5)

cases = df[ground_says_go & multi_flags].copy()
print(f"Candidate timestamps: {len(cases)} of {len(df)}")
print("\nCandidates by local hour:")
print(cases.groupby(cases.index.hour).size())
print("\nTop 15 by inversion strength:")
print(cases.sort_values("dT", ascending=False).head(15).round(2))

# Full evening-to-morning window for the demo night
window = df.loc["2025-06-15 16:00":"2025-06-16 09:00",
                ["T_low", "T_high", "dT", "ws_low", "ws_high", "dWD"]]
print("\nDemo night: 2025-06-15 16:00 to 2025-06-16 09:00")
print(window.round(2).to_string())