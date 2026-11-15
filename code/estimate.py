"""Estimate two-height readings at an orchard that has no mast, from nearby masts.

Method:
  1. Pick masts with the same terrain within SAME_TERRAIN_KM (cold air pooling and drainage
     depend on terrain more than on distance). If fewer than 2 match, use the nearest masts.
  2. Inverse-distance weighting of the INPUTS (temperatures, wind speeds, and wind direction
     as u/v components), then run the same rules.py on the estimate. Statuses are never averaged.
  3. Be conservative: the estimate is never GO when a nearby mast is not GO, or when the
     masts used are far away or on different terrain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rules import CAUTION, GO, RANK, REQUIRED

K = 3                   # masts used per estimate
SAME_TERRAIN_KM = 30.0  # search radius for masts with the same terrain
FAR_KM = 15.0           # nearest mast farther than this means low confidence


def distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, np.asarray(lat2, float), np.asarray(lon2, float)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def pick_masts(lat: float, lon: float, terrain: str, sites: pd.DataFrame, k: int = K):
    """Returns (masts used with a km column, whether they share the orchard's terrain)."""
    s = sites.assign(km=distance_km(lat, lon, sites.lat, sites.lon)).sort_values("km")
    same = s[(s.terrain == terrain) & (s.km <= SAME_TERRAIN_KM)]
    if len(same) >= 2:
        return same.head(k), True
    return s.head(k), False


def interpolate(readings: dict[str, pd.DataFrame], used: pd.DataFrame) -> pd.DataFrame:
    """Inverse-distance-weighted readings. A mast with a missing value is left out of that row."""
    frames = [readings[sid] for sid in used.site_id]
    idx = frames[0].index
    w = 1.0 / np.maximum(used.km.to_numpy(), 1.0) ** 2

    def stack(col):
        return np.column_stack([f[col].reindex(idx).to_numpy(float) for f in frames])

    def wmean(X):
        W = np.where(np.isnan(X), 0.0, w)
        tot = W.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, (np.nan_to_num(X) * W).sum(axis=1) / tot, np.nan)

    out = {c: wmean(stack(c)) for c in ["T_low_C", "T_high_C", "ws_low_mph", "ws_high_mph"]}
    for lvl in ("low", "high"):
        ws, wd = stack(f"ws_{lvl}_mph"), np.radians(stack(f"wd_{lvl}_deg"))
        u, v = wmean(ws * np.sin(wd)), wmean(ws * np.cos(wd))    # average as vectors: 350° and 10° -> 0°
        out[f"wd_{lvl}_deg"] = np.degrees(np.arctan2(u, v)) % 360
    return pd.DataFrame(out, index=idx)[REQUIRED].round(3)


def apply_confidence(log: pd.DataFrame, mast_logs: dict[str, pd.DataFrame], used: pd.DataFrame,
                     same_terrain: bool) -> pd.DataFrame:
    """Lower GO to CAUTION where nearby masts disagree or the estimate is uncertain."""
    log = log.copy()
    names = dict(zip(used.site_id, used.name))
    statuses = pd.DataFrame({sid: mast_logs[sid].status.reindex(log.index) for sid in used.site_id})
    not_go = statuses.apply(lambda c: c.map(RANK).fillna(0) > 0)
    is_go = log.status == GO

    disagree = is_go & not_go.any(axis=1)
    if disagree.any():
        who = not_go[disagree].apply(lambda r: ", ".join(names[s] for s in r.index[r]), axis=1)
        log.loc[disagree, "reasons"] += " | This orchard has no mast. Nearby masts do not all say GO (" \
            + who + "), so the estimate stays at CAUTION."
        log.loc[disagree, "triggers"] = "MASTS_DISAGREE"

    nearest = used.km.min()
    if not same_terrain or nearest > FAR_KM:
        why = (f"the nearest mast is {nearest:.0f} km away" if same_terrain
               else "no nearby mast has the same terrain")
        low = is_go & ~disagree
        log.loc[low, "reasons"] += f" | Estimate is uncertain because {why}. Check conditions on site."
        log.loc[low, "triggers"] = "LOW_CONFIDENCE"
        disagree = disagree | low

    # Drop the "all clear" sentence from rows that are no longer GO
    log.loc[disagree, "reasons"] = log.loc[disagree, "reasons"].str.split(" | ", n=1, regex=False).str[1]
    log.loc[disagree, "status"] = CAUTION
    return log
