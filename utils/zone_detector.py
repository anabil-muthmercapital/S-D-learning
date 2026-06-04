# =============================================================================
# utils/zone_detector.py — Proximal/distal levels & departure validation (Phase 6)
# =============================================================================
#
# Responsibilities
# ----------------
# 1. proximal_distal()  — derive the two zone boundary levels from a cluster
# 2. check_departure()  — validate that price left the base with sufficient force
# 3. detect_zones()     — full pipeline over all classified formations for one DataFrame
#
# Geometry reminder
# -----------------
# Demand (price approaches from above on re-entry):
#   proximal = base_high  (hit first as price drops back into zone)
#   distal   = base_low   (back of zone — stop-loss reference)
#
# Supply (price approaches from below on re-entry):
#   proximal = base_low   (hit first as price rallies back into zone)
#   distal   = base_high  (back of zone)
#
# Departure gates (both must pass)
# ---------------------------------
#   dep_atr   = departure / avg_ATR   >= DEPARTURE_ATR_MIN   (volatility-adjusted)
#   dep_ratio = departure / zone_width >= DEPARTURE_RATIO_MIN (zone-relative strength)
#
# Prerequisites
# -------------
# Input DataFrames must come from:
#   df = add_atr(CandlePrimitives.enrich_dataframe(raw_df))
# The `formations` list must come from legs_formation.detect_formations().
# =============================================================================

from __future__ import annotations

import pandas as pd

from utils.config import (
    DEPARTURE_CANDLES,
    DEPARTURE_ATR_MIN,
    DEPARTURE_RATIO_MIN,
)

# ---------------------------------------------------------------------------
# Step 1 — zone boundary levels
# ---------------------------------------------------------------------------


def proximal_distal(
    df: pd.DataFrame,
    bs: int,
    be: int,
    zone_type: str,
) -> tuple[float, float]:
    """Return (proximal, distal) price levels for the cluster at iloc [bs, be].

    Parameters
    ----------
    df        : enriched DataFrame with high/low columns
    bs        : cluster start (iloc)
    be        : cluster end   (iloc)
    zone_type : "demand" or "supply"

    Returns
    -------
    (proximal, distal) — both as floats
    """
    base_high = df["high"].iloc[bs : be + 1].max()
    base_low = df["low"].iloc[bs : be + 1].min()

    if zone_type == "demand":
        return base_high, base_low  # proximal = top, distal = bottom
    else:
        return base_low, base_high  # proximal = bottom, distal = top


# ---------------------------------------------------------------------------
# Step 2 — departure strength check
# ---------------------------------------------------------------------------


def check_departure(
    df: pd.DataFrame,
    proximal: float,
    be: int,
    zone_type: str,
    zone_width: float,
    avg_atr: float,
) -> dict:
    """Measure how far price moved beyond the proximal after the base ends.

    Scans the DEPARTURE_CANDLES bars immediately after the cluster end.
    Uses peak excursion (high/low), not closing price.

    Parameters
    ----------
    df          : enriched DataFrame with high/low columns
    proximal    : proximal price level for this zone
    be          : cluster end (iloc)
    zone_type   : "demand" or "supply"
    zone_width  : abs(proximal - distal)
    avg_atr     : ATR averaged over the base candles

    Returns
    -------
    dict with keys:
        departure  : raw excursion beyond proximal (always positive)
        dep_ratio  : departure / zone_width
        dep_atr    : departure / avg_atr
        passed     : bool — True when both gates clear
    """
    end_idx = min(be + DEPARTURE_CANDLES, len(df) - 1)
    window_h = df["high"].iloc[be + 1 : end_idx + 1]
    window_l = df["low"].iloc[be + 1 : end_idx + 1]

    if window_h.empty:
        return {"departure": 0.0, "dep_ratio": 0.0, "dep_atr": 0.0, "passed": False}

    if zone_type == "demand":
        departure = window_h.max() - proximal  # how far ABOVE the top of the base
    else:
        departure = proximal - window_l.min()  # how far BELOW the bottom of the base

    dep_ratio = departure / zone_width if zone_width > 0 else 0.0
    dep_atr = departure / avg_atr if avg_atr > 0 else 0.0
    passed = (dep_ratio >= DEPARTURE_RATIO_MIN) and (dep_atr >= DEPARTURE_ATR_MIN)

    return {
        "departure": round(departure, 5),
        "dep_ratio": round(dep_ratio, 3),
        "dep_atr": round(dep_atr, 3),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Step 3 — full pipeline
# ---------------------------------------------------------------------------


def detect_zones(
    df: pd.DataFrame,
    formations: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Run proximal/distal + departure checks over all classified formations.

    Parameters
    ----------
    df         : enriched DataFrame (must have high, low, atr columns)
    formations : output of legs_formation.detect_formations()

    Returns
    -------
    (passed_zones, rejected_zones) — each a list of dicts.
    Every dict contains all fields from the formation plus:
        proximal, distal, zone_width,
        departure, dep_ratio, dep_atr, passed
    """
    passed: list[dict] = []
    rejected: list[dict] = []

    for f in formations:
        bs, be = f["start"], f["end"]
        avg_atr = f["avg_atr"]
        zone_type = f["zone_type"]

        prox, dist = proximal_distal(df, bs, be, zone_type)
        zone_width = abs(prox - dist)

        dep_info = check_departure(df, prox, be, zone_type, zone_width, avg_atr)

        zone = {
            **f,
            "proximal": prox,
            "distal": dist,
            "zone_width": round(zone_width, 5),
            **dep_info,
        }

        if dep_info["passed"]:
            passed.append(zone)
        else:
            rejected.append(zone)

    return passed, rejected
