"""Leave-one-out check: estimate each mast from the other masts and compare statuses.

Run from the project root:
    python code/validate_estimates.py
Only spray hours that matter are scored (19:00 to 09:00). With simulated data the numbers
are optimistic, because the simulator itself builds sites from regional weather plus terrain.
"""
from pathlib import Path

import pandas as pd

from estimate import apply_confidence, interpolate, pick_masts
from rules import GO, STOP, Config, evaluate_series, load_processed

MAST_DIR = Path(__file__).resolve().parents[1] / "data/processed/masts"


def main():
    sites = pd.read_csv(MAST_DIR / "sites.csv")
    readings = {s: load_processed(MAST_DIR / f"{s}.csv") for s in sites.site_id}
    cfg = Config.for_volatility("medium")
    logs = {s: evaluate_series(df, cfg) for s, df in readings.items()}

    rows = []
    for _, t in sites.iterrows():
        others = sites[sites.site_id != t.site_id]
        used, same = pick_masts(t.lat, t.lon, t.terrain, others)
        est = evaluate_series(interpolate(readings, used), cfg)
        safe = apply_confidence(est, logs, used, same)
        truth = logs[t.site_id].status
        hours = truth.index.hour
        night = (hours >= 19) | (hours < 9)
        for name, e in [("estimate only", est), ("with conservative rule", safe)]:
            s, tr = e.status[night], truth[night]
            rows.append(dict(method=name, site=t.site_id,
                             agree=(s == tr).mean(),
                             stop_caught=((s == STOP) & (tr == STOP)).sum() / max((tr == STOP).sum(), 1),
                             go_when_stop=((s == GO) & (tr == STOP)).sum(), n=len(s)))
    res = pd.DataFrame(rows)
    summary = res.groupby("method", sort=False).agg(
        status_agreement=("agree", "mean"), stop_caught=("stop_caught", "mean"),
        go_when_actually_stop=("go_when_stop", "sum"), readings=("n", "sum"))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
