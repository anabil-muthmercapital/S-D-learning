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
from utils.htf_range import (
    htf_range_asof,
    add_curve_score,
)  # noqa: F401  (re-exported for callers)


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
