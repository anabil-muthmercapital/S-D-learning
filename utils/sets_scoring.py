# =============================================================================
# utils/sets_scoring.py — S.E.T.S composite scoring (Phase 12)
# =============================================================================
#
# Responsibilities
# ----------------
# add_sets_score()  — annotate zones with `strength_score`, `sets_total`,
#                     and `sets_rating` (★ / ★★ / ★★★).
#
# S.E.T.S components (all already annotated by earlier phases)
# ------------------------------------------------------------
#   S — Strength      : derived here from departure_ratio  (zone["dep_ratio"])
#   T — Time          : zone["time_score"]          (add_time_score)
#   F — Freshness     : zone["freshness_score"]     (add_freshness)
#   A — Alignment     : zone["trend_score"]         (add_trend_score)
#   C — Curve         : zone["curve_score"]         (add_curve_score)
#
# Strength mapping (sd-concepts.md §14, methodology-literal)
# -----------------------------------------------------------
#   departure_ratio >= SETS_STRENGTH_RATIO_HIGH  →  2  (explosive)
#   departure_ratio >= SETS_STRENGTH_RATIO_LOW   →  1  (adequate)
#   otherwise                                    →  0  (weak)
#
# Rating thresholds
# -----------------
#   total >= SETS_RATING_A  →  ★★★  A-setup (take it)
#   total >= SETS_RATING_B  →  ★★   B-setup (caution)
#   otherwise               →  ★    Skip
#
# Prerequisites
# -------------
# zones must have already been annotated by (in order):
#   add_freshness(), add_time_score(), add_curve_score(), add_trend_score()
# Each zone must also have `dep_ratio` from zone_detector.detect_zones().
# =============================================================================

from __future__ import annotations

from utils.config import (
    SETS_STRENGTH_RATIO_HIGH,
    SETS_STRENGTH_RATIO_LOW,
    SETS_RATING_A,
    SETS_RATING_B,
)


def _strength_score(departure_ratio: float) -> int:
    """Map departure_ratio (departure / zone_width) to a 0-2 strength score."""
    if departure_ratio >= SETS_STRENGTH_RATIO_HIGH:
        return 2
    if departure_ratio >= SETS_STRENGTH_RATIO_LOW:
        return 1
    return 0


def _sets_rating(total: int) -> str:
    """Convert SETS total (0-10) to a star rating string."""
    if total >= SETS_RATING_A:
        return "★★★"
    if total >= SETS_RATING_B:
        return "★★"
    return "★"


def add_sets_score(zones: list[dict]) -> list[dict]:
    """Add ``strength_score``, ``sets_total``, and ``sets_rating`` to every zone.

    All other sub-scores (time, freshness, curve, trend) must already be
    present on each zone dict before calling this function.

    Parameters
    ----------
    zones : list of zone dicts, already annotated by the four earlier scoring
            functions.  Each zone must have:
              - dep_ratio        (from detect_zones)
              - time_score       (from add_time_score)
              - freshness_score  (from add_freshness)
              - curve_score      (from add_curve_score)
              - trend_score      (from add_trend_score)

    Returns
    -------
    The same list with each dict updated in-place.
    """
    for z in zones:
        s = _strength_score(z["dep_ratio"])
        total = (
            s
            + z["time_score"]
            + z["freshness_score"]
            + z["curve_score"]
            + z["trend_score"]
        )
        z["strength_score"] = s
        z["sets_total"] = total
        z["sets_rating"] = _sets_rating(total)
    return zones
