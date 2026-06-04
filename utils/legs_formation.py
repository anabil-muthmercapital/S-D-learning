# =============================================================================
# utils/legs_formation.py — Leg measurement & formation classification (Phase 5)
# =============================================================================
#
# Responsibilities
# ----------------
# 1. FORMATION_MAP        — canonical (leg_in_dir, leg_out_dir) → (formation, zone_type)
# 2. classify_move()      — turn a net price displacement into "up" / "down" / "flat"
# 3. measure_legs()       — compute leg-in and leg-out net for one cluster
# 4. detect_formations()  — full pipeline over all passed clusters for one DataFrame
#
# Prerequisites
# -------------
# The input DataFrame must have been produced by:
#   df = add_atr(CandlePrimitives.enrich_dataframe(raw_df))
# Columns required: open, close, high, low, atr, is_base.
# The `passed` list must come from base_detector.detect_bases().
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import LEG_CANDLES, LEG_ATR_MIN

# ---------------------------------------------------------------------------
# Formation map
# ---------------------------------------------------------------------------

FORMATION_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("up", "up"): ("RBR", "demand"),  # Rally-Base-Rally
    ("down", "down"): ("DBD", "supply"),  # Drop-Base-Drop
    ("down", "up"): ("DBR", "demand"),  # Drop-Base-Rally
    ("up", "down"): ("RBD", "supply"),  # Rally-Base-Drop
}


# ---------------------------------------------------------------------------
# Step 1 — direction classifier
# ---------------------------------------------------------------------------


def classify_move(net: float, avg_atr: float) -> str:
    """Return 'up', 'down', or 'flat' based on net displacement vs local ATR.

    A move must clear `LEG_ATR_MIN × avg_atr` to be considered directional.
    Anything smaller is 'flat' (drifting, not impulsive).
    """
    threshold = LEG_ATR_MIN * avg_atr
    if net >= threshold:
        return "up"
    if net <= -threshold:
        return "down"
    return "flat"


# ---------------------------------------------------------------------------
# Step 2 — leg measurement for one cluster
# ---------------------------------------------------------------------------


def measure_legs(
    df: pd.DataFrame,
    bs: int,
    be: int,
) -> dict | None:
    """Measure the leg-in and leg-out for the cluster at iloc [bs, be].

    Windows
    -------
    Leg-in  : bars [bs - LEG_CANDLES, bs - 1]   net = close[bs-1] - open[bs-L]
    Leg-out : bars [be + 1,  be + LEG_CANDLES]   net = close[be+L] - close[be]

    Guard rails
    -----------
    Returns None when there are not enough bars before or after the cluster —
    this prevents the silent look-ahead / index-wrap bugs that arise from
    negative iloc indexing in numpy.

    Returns
    -------
    dict with keys:
        leg_in_dir, leg_out_dir : "up" | "down" | "flat"
        leg_in_net, leg_out_net : raw price displacement (signed)
        avg_atr                 : ATR averaged over the base candles
    """
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    atr = df["atr"].to_numpy()

    avg_atr = atr[bs : be + 1].mean()

    if bs < LEG_CANDLES:
        return None  # not enough history before the base
    if be + LEG_CANDLES >= len(c):
        return None  # not enough future after the base

    leg_in_net = c[bs - 1] - o[bs - LEG_CANDLES]
    leg_out_net = c[be + LEG_CANDLES] - c[be]

    return {
        "leg_in_dir": classify_move(leg_in_net, avg_atr),
        "leg_out_dir": classify_move(leg_out_net, avg_atr),
        "leg_in_net": round(leg_in_net, 5),
        "leg_out_net": round(leg_out_net, 5),
        "avg_atr": round(avg_atr, 5),
    }


# ---------------------------------------------------------------------------
# Step 3 — full pipeline
# ---------------------------------------------------------------------------


def detect_formations(
    df: pd.DataFrame,
    passed_clusters: list[dict],
) -> list[dict]:
    """Run leg measurement + formation classification over all passed clusters.

    Parameters
    ----------
    df              : enriched DataFrame (must have open, close, atr columns)
    passed_clusters : output of base_detector.detect_bases()[0]  (the `passed` list)

    Returns
    -------
    List of dicts, one per confirmed formation.  Each dict contains all fields
    from the original cluster dict plus:
        leg_in_dir, leg_out_dir, leg_in_net, leg_out_net, avg_atr,
        formation ("RBR" | "DBD" | "DBR" | "RBD"),
        zone_type ("demand" | "supply")
    """
    formations: list[dict] = []

    for cluster in passed_clusters:
        bs, be = cluster["start"], cluster["end"]
        legs = measure_legs(df, bs, be)

        if legs is None:
            continue  # not enough bars on either side

        pair = FORMATION_MAP.get((legs["leg_in_dir"], legs["leg_out_dir"]))
        if pair is None:
            continue  # flat leg — no real formation

        formation, zone_type = pair
        formations.append(
            {
                **cluster,
                **legs,
                "formation": formation,
                "zone_type": zone_type,
            }
        )

    return formations
