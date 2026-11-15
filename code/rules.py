"""Drift-risk rules engine: turns two-height weather readings into GO / CAUTION / STOP.

Input: a DataFrame indexed by local timestamp with these columns
    T_low_C, T_high_C, ws_low_mph, ws_high_mph, wd_low_deg, wd_high_deg
Any data source (NEON replay, simulated sensors, weather forecast) only needs to
produce these columns to work with this engine and the app.

This tool supports the applicator's decision. It does not replace the pesticide
label, on-site checks, or regulations, and no system can guarantee zero drift.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

GO, CAUTION, NO_DATA, STOP, INFO = "GO", "CAUTION", "NO DATA", "STOP", "INFO"
RANK = {GO: 0, CAUTION: 1, NO_DATA: 2, STOP: 3}
REQUIRED = ["T_low_C", "T_high_C", "ws_low_mph", "ws_high_mph", "wd_low_deg", "wd_high_deg"]
TZ = "America/Los_Angeles"


@dataclass(frozen=True)
class Config:
    """Thresholds. Wind limits are common guidance; the pesticide label always wins."""
    low_label: str = "5 m (17 ft)"
    high_label: str = "12 m (39 ft)"
    inv_caution_C: float = 0.1       # smaller differences are treated as sensor noise
    inv_stop_C: float = 0.3
    wind_min_mph: float = 3.0
    wind_max_mph: float = 10.0
    speed_shear_mph: float = 5.0
    dir_shear_caution_deg: float = 45.0
    dir_shear_stop_deg: float = 90.0
    dir_min_speed_mph: float = 2.0   # wind direction is unreliable in lighter wind
    clear_after: int = 2             # steady calmer readings needed before downgrading

    @classmethod
    def for_volatility(cls, volatility: str = "medium", **kwargs) -> "Config":
        """High-volatility products can drift as vapor, so any real inversion means STOP."""
        cfg = cls(**kwargs)
        if volatility == "high":
            cfg = replace(cfg, inv_stop_C=cfg.inv_caution_C)
        return cfg


def evaluate_reading(r, cfg: Config = Config()):
    """Evaluate one reading. Returns (status, alerts); each alert is (level, code, message)."""
    alerts = []

    def add(level, code, msg):
        alerts.append((level, code, msg))

    # --- Temperature inversion ---
    if pd.isna(r["T_low_C"]) or pd.isna(r["T_high_C"]):
        add(NO_DATA, "NO_TEMP",
            "Temperature is missing at one height, so an inversion cannot be ruled out. "
            "Check conditions on site.")
    else:
        dT = r["T_high_C"] - r["T_low_C"]
        if dT >= cfg.inv_stop_C:
            add(STOP, "INVERSION",
                f"Temperature inversion: air at {cfg.high_label} is {dT:.1f}°C ({dT * 1.8:.1f}°F) "
                f"warmer than at {cfg.low_label}. Spray will stay trapped near the ground "
                "and can drift a long way sideways.")
        elif dT >= cfg.inv_caution_C:
            add(CAUTION, "INVERSION_WEAK",
                f"Weak inversion forming: air at {cfg.high_label} is {dT:.1f}°C ({dT * 1.8:.1f}°F) "
                f"warmer than at {cfg.low_label}. Watch closely.")

    # --- Wind speed and shear ---
    lo, hi = r["ws_low_mph"], r["ws_high_mph"]
    if pd.isna(lo) or pd.isna(hi):
        add(NO_DATA, "NO_WIND", "Wind speed is missing at one height. Check conditions on site.")
    else:
        if max(lo, hi) > cfg.wind_max_mph:
            where = cfg.low_label if lo >= hi else cfg.high_label
            add(STOP, "WIND_HIGH",
                f"Wind too strong: {max(lo, hi):.1f} mph at {where} "
                f"(above {cfg.wind_max_mph:.0f} mph). Spray will be carried off target.")
        if lo < cfg.wind_min_mph:
            add(CAUTION, "WIND_LOW",
                f"Wind too calm: {lo:.1f} mph at {cfg.low_label} (below {cfg.wind_min_mph:.0f} mph). "
                "Calm air often comes with an inversion and makes drift direction unpredictable.")
        if abs(hi - lo) > cfg.speed_shear_mph:
            add(CAUTION, "SPEED_SHEAR",
                f"Wind speed changes sharply with height: {lo:.1f} mph at {cfg.low_label} vs "
                f"{hi:.1f} mph at {cfg.high_label}. Droplets lifted above the trees may travel farther.")

        # --- Wind direction shear (only when direction is reliable) ---
        dir_missing = pd.isna(r["wd_low_deg"]) or pd.isna(r["wd_high_deg"])
        if dir_missing or min(lo, hi) < cfg.dir_min_speed_mph:
            add(INFO, "DIR_SKIPPED",
                "Wind direction check skipped: wind too light for a reliable direction.")
        else:
            d = abs((r["wd_high_deg"] - r["wd_low_deg"] + 180) % 360 - 180)  # handles 350 vs 10
            if d >= cfg.dir_shear_stop_deg:
                add(STOP, "DIR_SHEAR",
                    f"Wind direction differs by {d:.0f}° between heights. Drift could go a "
                    "different way than the ground wind suggests.")
            elif d >= cfg.dir_shear_caution_deg:
                add(CAUTION, "DIR_SHEAR",
                    f"Wind direction differs by {d:.0f}° between heights. Check where drift would go.")

    levels = [a[0] for a in alerts if a[0] != INFO]
    status = max(levels, key=RANK.get, default=GO)
    if status == GO:
        alerts.insert(0, (GO, "ALL_CLEAR",
                          f"No inversion, wind {lo:.1f} mph within {cfg.wind_min_mph:.0f}–"
                          f"{cfg.wind_max_mph:.0f} mph, and winds consistent between heights."))
    return status, alerts


def evaluate_single_station(ws_low, cfg: Config = Config()):
    """What a single ground-level station sees: only wind speed at one height."""
    if pd.isna(ws_low):
        return NO_DATA
    if ws_low > cfg.wind_max_mph:
        return STOP
    if ws_low < cfg.wind_min_mph:
        return CAUTION
    return GO


def evaluate_series(df: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Evaluate a time series and build the time-stamped log.

    Escalation is immediate (safety first). Downgrading waits until conditions have
    been calmer for `clear_after` readings in a row, so the status does not flicker.
    Missing data is shown right away, and the status from before a data gap carries
    over it (a gap does not erase a STOP).
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    records, held, recent = [], GO, []
    for ts, r in df.iterrows():
        raw, alerts = evaluate_reading(r, cfg)

        if raw == NO_DATA:
            shown = NO_DATA  # show the gap, keep the held state for when data returns
        else:
            if RANK[raw] >= RANK[held]:
                held, recent = raw, []
            else:
                recent.append(raw)
                if len(recent) >= cfg.clear_after:
                    held = max(recent[-cfg.clear_after:], key=RANK.get)
                    recent = []
            shown = held

        reasons = [a[2] for a in alerts]
        if raw != NO_DATA and RANK[held] > RANK[raw]:
            reasons.append(f"Holding {held}: conditions improved but must stay steady for "
                           f"{cfg.clear_after} readings before the status is lowered.")

        records.append({
            "timestamp_local": ts,
            **{c: r[c] for c in REQUIRED},
            "dT_C": r["T_high_C"] - r["T_low_C"],
            "dWS_mph": abs(r["ws_high_mph"] - r["ws_low_mph"]),
            "single_station_status": evaluate_single_station(r["ws_low_mph"], cfg),
            "raw_status": raw,
            "status": shown,
            "triggers": ";".join(a[1] for a in alerts if a[0] not in (GO, INFO)),
            "reasons": " | ".join(reasons),
        })
    return pd.DataFrame(records).set_index("timestamp_local")


def load_processed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="timestamp_local")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
    return df


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Run drift-risk rules on processed two-height data.")
    p.add_argument("--input", default=root / "data/processed/soap_2025-06_two_level.csv", type=Path)
    p.add_argument("--log", default=root / "data/logs/soap_2025-06_drift_log.csv", type=Path)
    p.add_argument("--volatility", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--night", default="2025-06-15", help="Date whose evening-to-morning window to print")
    args = p.parse_args()

    cfg = Config.for_volatility(args.volatility)
    log = evaluate_series(load_processed(args.input), cfg)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log.round(3).to_csv(args.log)

    print(f"Evaluated {len(log)} readings (volatility: {args.volatility})")
    print("\nStatus counts:")
    print(log.status.value_counts().to_string())
    missed = (log.single_station_status == GO) & (log.status != GO)
    single_go = (log.single_station_status == GO).sum()
    print(f"\nReadings where a single ground station says GO but the tool does not: "
          f"{missed.sum()} of {single_go} ({missed.sum() / max(single_go, 1):.0%})")

    start = pd.Timestamp(args.night, tz=TZ) + pd.Timedelta(hours=16)
    window = log.loc[start:start + pd.Timedelta(hours=17)]
    print(f"\nDemo window: {start:%Y-%m-%d %H:%M} to +17 h")
    print(window[["dT_C", "ws_low_mph", "single_station_status", "status", "triggers"]]
          .round(2).to_string())
    print(f"\nSaved log: {args.log}")


if __name__ == "__main__":
    main()