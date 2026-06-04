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

from utils.config import TIME_SCORE_TABLE


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


def add_curve_score(
    zones: list[dict],
    htf_high: float,
    htf_low: float,
) -> list[dict]:
    """Add ``curve_pos``, ``curve_third``, and ``curve_score`` to every zone.

    The HTF tradeable range is divided into three equal thirds.  A zone is
    placed by its proximal level (the first edge price will touch).

    Parameters
    ----------
    zones    : list of zone dicts; each must have ``proximal`` and
               ``zone_type`` (``"demand"`` or ``"supply"``).
    htf_high : upper bound of the HTF range (e.g. ``df["high"].max()``).
    htf_low  : lower bound of the HTF range (e.g. ``df["low"].min()``).

    Returns
    -------
    The same list with each dict updated in-place.
    """
    htf_range = htf_high - htf_low
    for z in zones:
        if htf_range <= 0:
            pos, third = 0.5, "mid"
        else:
            pos = (z["proximal"] - htf_low) / htf_range
            if pos < 0.333:
                third = "low"
            elif pos > 0.667:
                third = "high"
            else:
                third = "mid"
        z["curve_pos"] = round(pos, 3)
        z["curve_third"] = third
        if z["zone_type"] == "demand":
            z["curve_score"] = _CURVE_DEMAND[third]
        else:
            z["curve_score"] = _CURVE_SUPPLY[third]
    return zones
