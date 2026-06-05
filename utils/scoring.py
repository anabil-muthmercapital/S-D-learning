# =============================================================================
# utils/scoring.py — Zone quality scoring (Phase 9+)
# =============================================================================
#
# Responsibilities
# ----------------
# add_time_score()   — annotate zones with `base_count` and `time_score`
# add_curve_score()  — annotate zones with `curve_pos`, `curve_third`,
#                      and `curve_score` (sd-concepts.md §7)
#
# Design note
# -----------
# All zone objects in this project are plain dicts produced by
# zone_detector.detect_zones().  Scoring functions follow the same
# annotate-in-place pattern established in freshness.py — they accept a list
# of zone dicts, mutate each one, and return the same list.
#
# Time-score rationale (sd-concepts.md §8)
# -----------------------------------------
# Fewer base candles = the market made a faster decision at that level.
# A single explosive candle that reverses immediately shows the highest
# institutional conviction — no prolonged consolidation needed.
#
# Score table (see TIME_SCORE_TABLE in config.py):
#   1–2 candles → 2  (explosive, single-decision base)
#   3   candles → 1  (compact — still acceptable)
#   4+  candles → 0  (indecisive — weakened conviction)
#
# Curve-score rationale (sd-concepts.md §7)
# ------------------------------------------
# Where a zone sits inside the HTF tradeable range determines how much room
# to run price has.  Range is divided into three equal thirds:
#
#   position = (zone_proximal - htf_low) / (htf_high - htf_low)
#
#   position < 0.333  → Low third  → demand_score=2, supply_score=0
#   0.333–0.667       → Mid third  → both=1
#   position > 0.667  → High third → demand_score=0, supply_score=2
#
# All individual scores (freshness, time, curve, trend, departure)
# feed independently into the final S.E.T.S total (sd-concepts.md §10).
# There is no intermediate composite multiplication between them.
#
# Prerequisites
# -------------
# zones must come from zone_detector.detect_zones(); they must have:
#   start, end        (integer iloc positions in the source DataFrame)
#   proximal, zone_type (for curve scoring)
# =============================================================================

from __future__ import annotations

import pandas as pd

from utils.config import TIME_SCORE_TABLE, HTF_RANGE_LOOKBACK


def add_time_score(zones: list[dict]) -> list[dict]:
    """Add `base_count` and `time_score` to every zone dict in *zones*.

    Parameters
    ----------
    zones : list of zone dicts (from detect_zones — passed or rejected).
            Each dict must have `start` and `end` integer iloc keys.

    Returns
    -------
    The same list with each dict updated in-place.
    """
    for z in zones:
        n = z["end"] - z["start"] + 1
        z["base_count"] = n
        z["time_score"] = TIME_SCORE_TABLE.get(n, 0)
    return zones


_CURVE_DEMAND = {"low": 2, "mid": 1, "high": 0}
_CURVE_SUPPLY = {"low": 0, "mid": 1, "high": 2}


def htf_range_asof(
    htf_df: pd.DataFrame,
    ts,
    lookback: int = HTF_RANGE_LOOKBACK,
) -> "tuple[float, float] | None":
    """Return ``(high, low)`` for the HTF window ending at *ts* (inclusive).

    Only HTF bars with timestamp <= *ts* are used — strictly no lookahead.
    Returns ``None`` if fewer than 5 bars are available in the window.

    Parameters
    ----------
    htf_df   : HTF DataFrame with ``high`` / ``low`` columns and DatetimeIndex.
    ts       : upper timestamp boundary (inclusive); usually ``ltf_index[zone["end"]]``.
    lookback : rolling window length in HTF bars.
    """
    window = htf_df.loc[:ts].tail(lookback)
    if len(window) < 5:
        return None
    return float(window["high"].max()), float(window["low"].min())


def add_curve_score(
    zones: list[dict],
    htf_df: pd.DataFrame,
    ltf_index: pd.DatetimeIndex,
    lookback: int = HTF_RANGE_LOOKBACK,
) -> list[dict]:
    """Add ``curve_pos``, ``curve_third``, and ``curve_score`` to every zone.

    Each zone is scored against its own **point-in-time** HTF range — only
    HTF bars with timestamp <= the zone's formation bar are used (no lookahead).
    The HTF range is divided into three equal thirds (Low / Mid / High).

    Parameters
    ----------
    zones     : list of zone dicts; each must have ``proximal`` and
                ``zone_type`` (``"demand"`` or ``"supply"``).
    htf_df    : HTF DataFrame (e.g. ``data["1d"]``) with ``high``/``low``
                columns and a DatetimeIndex.
    ltf_index : DatetimeIndex of the LTF DataFrame, used to convert
                ``zone["end"]`` (iloc position) to a timestamp.
    lookback  : number of HTF bars in the rolling range window.

    Returns
    -------
    The same list with each dict updated in-place.
    """
    for z in zones:
        ts = ltf_index[z["end"]]
        r  = htf_range_asof(htf_df, ts, lookback)

        if r is None:
            z["curve_pos"]   = None
            z["curve_third"] = "n/a"
            z["curve_score"] = 0
            continue

        hi, lo = r
        htf_range = hi - lo
        if htf_range <= 0:
            z["curve_pos"]   = None
            z["curve_third"] = "n/a"
            z["curve_score"] = 0
            continue

        pos   = (z["proximal"] - lo) / htf_range
        pos   = max(0.0, min(1.0, pos))   # clamp to [0, 1]
        if pos < 0.333:
            third = "low"
        elif pos > 0.667:
            third = "high"
        else:
            third = "mid"

        z["curve_pos"]   = round(pos, 3)
        z["curve_third"] = third
        if z["zone_type"] == "demand":
            z["curve_score"] = _CURVE_DEMAND[third]
        else:
            z["curve_score"] = _CURVE_SUPPLY[third]
    return zones
