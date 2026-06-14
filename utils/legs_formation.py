# =============================================================================
# utils/legs_formation.py — Leg measurement & formation classification (Phase 5)
# =============================================================================
#
# Responsibilities
# ----------------
# 1. FORMATION_MAP        — canonical (leg_in_dir, leg_out_dir) → (formation, zone_type)
# 2. measure_legs()       — compute leg-in and leg-out for one cluster
# 3. detect_formations()  — full pipeline over all passed clusters for one DataFrame
#
# Prerequisites
# -------------
# The input DataFrame must have been produced by:
#   df = add_atr(CandlePrimitives.enrich_dataframe(raw_df))
# Columns required: open, close, high, low, atr, is_base.
# The `passed` list must come from base_detector.detect_bases().
# =============================================================================

from __future__ import annotations

import pandas as pd
import numpy as np

from utils.config import LEG_CANDLES, LEG_STRONG_BODY_RATIO

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
# Step 1 — direction classifier (dominant-excursion, no ATR threshold)
# ---------------------------------------------------------------------------


# Methodology-literal: leg direction is decided purely by which side of the
# window dominates. The OTA spec does NOT require an ATR-multiple threshold —
# real leg strength is enforced downstream by leg_strength (body-ratio gate).
def _peak_excursion_dir(up_move: float, down_move: float) -> str:
    """Return 'up' if up_move > down_move, 'down' if down_move > up_move,
    'flat' only when both are exactly equal (or both zero).
    """
    if up_move > down_move:
        return "up"
    if down_move > up_move:
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
    Leg-in  : bars [bs - LEG_CANDLES, bs - 1]   reference = open[bs]
    Leg-out : bars [be + 1,  be + LEG_CANDLES]  reference = close[be]

    Both legs use PEAK excursion (high/low) rather than close-to-close so a
    leg that spikes out and retraces still registers (round-trip trap).

    Guard rails
    -----------
    Returns None when there are not enough bars before or after the cluster —
    this prevents the silent look-ahead / index-wrap bugs that arise from
    negative iloc indexing in numpy.

    Returns
    -------
    dict with keys:
        leg_in_dir, leg_out_dir   : "up" | "down" | "flat"
        leg_in_up, leg_in_down    : peak excursions (always positive)
        leg_out_up, leg_out_down  : peak excursions (always positive)
        avg_atr                   : ATR averaged over the base candles
        leg_strength              : strongest qualifying body/range ratio in leg-out
        leg_strength_ok           : True when leg_strength >= LEG_STRONG_BODY_RATIO
        clean_departure           : False when the strongest leg-out candle's wick
                                    pierces the distal edge of the base
                                    (demand: low < base_low; supply: high > base_high)
    """
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    atr = df["atr"].to_numpy()
    btr = df["body_to_range_ratio"].to_numpy()

    avg_atr = atr[bs : be + 1].mean()

    if bs < LEG_CANDLES:
        return None  # not enough history before the base
    if be + LEG_CANDLES >= len(c):
        return None  # not enough future after the base

    # Base geometry — used for the dirty-departure check below.
    base_high = float(h[bs : be + 1].max())
    base_low = float(low_arr[bs : be + 1].min())

    # Leg-in: window [bs-L, bs-1], referenced to where the base BEGINS (open[bs]).
    # Round-trip trap fix: scan peak high/low instead of close-to-close net.
    in_hi = h[bs - LEG_CANDLES : bs].max()
    in_lo = low_arr[bs - LEG_CANDLES : bs].min()
    leg_in_up = o[bs] - in_lo  # rally INTO base from below
    leg_in_down = in_hi - o[bs]  # drop INTO base from above
    leg_in_dir = _peak_excursion_dir(leg_in_up, leg_in_down)

    # Leg-out: window [be+1, be+L], referenced to where the base ENDS (close[be]).
    out_hi = h[be + 1 : be + LEG_CANDLES + 1].max()
    out_lo = low_arr[be + 1 : be + LEG_CANDLES + 1].min()
    leg_out_up = out_hi - c[be]
    leg_out_down = c[be] - out_lo
    leg_out_dir = _peak_excursion_dir(leg_out_up, leg_out_down)

    # Leg strength: strongest directionally-aligned body in the leg-out window.
    # Lives here (not in zone_detector) because it is a LEG property, not zone geometry.
    out_open = o[be + 1 : be + LEG_CANDLES + 1]
    out_close = c[be + 1 : be + LEG_CANDLES + 1]
    out_high = h[be + 1 : be + LEG_CANDLES + 1]
    out_low = low_arr[be + 1 : be + LEG_CANDLES + 1]
    out_btr = btr[be + 1 : be + LEG_CANDLES + 1]
    if leg_out_dir == "up":
        mask = out_close > out_open
    elif leg_out_dir == "down":
        mask = out_close < out_open
    else:
        mask = np.zeros(len(out_btr), dtype=bool)

    if mask.any():
        # Strongest qualifying candle = the one whose body/range ratio is max
        # among the directionally-aligned candles. That's the candle that SET
        # leg_strength, and the one whose wick we must inspect.
        masked_btr = np.where(mask, out_btr, -np.inf)
        strong_idx = int(np.argmax(masked_btr))
        leg_strength = float(out_btr[strong_idx])
        # Dirty-departure trap: consistent with the zone-death definition (a zone
        # dies when price closes beyond the distal). If the strongest leg-out
        # candle's wick already pierces the distal at birth, the departure was
        # not a decisive break — price stabbed through the whole zone.
        if leg_out_dir == "up":
            # demand zone — distal is base_low; lower wick must stay >= base_low.
            clean_departure = bool(out_low[strong_idx] >= base_low)
        else:
            # supply zone — distal is base_high; upper wick must stay <= base_high.
            clean_departure = bool(out_high[strong_idx] <= base_high)
    else:
        leg_strength = 0.0
        clean_departure = False  # no qualifying breakout candle exists

    leg_strength_ok = leg_strength >= LEG_STRONG_BODY_RATIO

    return {
        "leg_in_dir": leg_in_dir,
        "leg_out_dir": leg_out_dir,
        "leg_in_up": round(leg_in_up, 5),
        "leg_in_down": round(leg_in_down, 5),
        "leg_out_up": round(leg_out_up, 5),
        "leg_out_down": round(leg_out_down, 5),
        "avg_atr": round(avg_atr, 5),
        "leg_strength": round(leg_strength, 3),
        "leg_strength_ok": leg_strength_ok,
        "clean_departure": clean_departure,
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
        leg_in_dir, leg_out_dir, leg_in_up, leg_in_down,
        leg_out_up, leg_out_down, avg_atr,
        leg_strength, leg_strength_ok, clean_departure,
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

        # Weak-leg trap: direction alone is not enough — a slow drift could win the
        # excursion contest. The methodology requires at least one full-body candle.
        if not legs["leg_strength_ok"]:
            continue

        # Dirty-departure trap: the strongest leg-out candle's wick pierced the
        # distal edge of the base — equivalent to the zone dying at birth.
        if not legs["clean_departure"]:
            continue

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
