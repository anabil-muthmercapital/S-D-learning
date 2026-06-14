# =============================================================================
# utils/freshness.py — Zone freshness scoring (Phase 8)
# =============================================================================
#
# Responsibilities
# ----------------
# 1. count_touches()  — count how many times price re-entered a zone after
#                       the departure window, stopping if the zone dies.
# 2. add_freshness()  — annotate a list of zone dicts in-place with
#                       `touches` and `freshness_score` fields.
#
# Plotting lives in the notebook (08_freshness.ipynb) so it can be tuned
# interactively — see plot_freshness() defined there.
#
# Freshness definition
# --------------------
# A zone is *touched* every time a candle's wick re-enters the zone box:
#   demand : low[i]  <= proximal  (wick reached the top of the base)
#   supply : high[i] >= proximal  (wick reached the bottom of the base)
#
# A zone *dies* when a candle closes beyond the distal line:
#   demand : close[i] < distal   (structure broken from below)
#   supply : close[i] > distal   (structure broken from above)
# Scanning stops the moment a zone dies.
#
# Score table (see FRESHNESS_SCORE_TABLE in config.py):
#   0 touches → 2  (fresh — orders completely intact)
#   1 touch   → 1  (tested once — partially consumed)
#   2+ touches→ 0  (stale — too much liquidity used up)
#
# Prerequisites
# -------------
# zones must come from zone_detector.detect_zones(); they must have:
#   start, end, zone_type, proximal, distal
# The input DataFrame must have columns: high, low, close.
# =============================================================================

from __future__ import annotations

import pandas as pd

from utils.config import (
    DEPARTURE_CANDLES,
    FRESHNESS_SCORE_TABLE,
)

# ---------------------------------------------------------------------------
# Step 0 — single source of truth for zone-death detection
# ---------------------------------------------------------------------------


def find_death_bar(df: pd.DataFrame, zone: dict) -> int | None:
    """Return the iloc of the bar where *zone* dies, or ``None`` if still alive.

    A zone dies on the first bar (after its departure window) whose close
    is beyond the distal line:

      * demand : close[i] < distal
      * supply : close[i] > distal

    Scanning starts at ``zone["end"] + DEPARTURE_CANDLES + 1`` — the same
    point used by ``count_touches`` — so both functions agree on lifespan.

    Parameters
    ----------
    df   : enriched DataFrame (must have ``close`` column).
    zone : dict with ``end``, ``zone_type``, ``distal``.

    Returns
    -------
    int  — bar index where the zone broke, OR
    None — zone is still alive at the right edge of the data.
    """
    distal = zone["distal"]
    zone_type = zone["zone_type"]
    scan_start = zone["end"] + DEPARTURE_CANDLES + 1

    close = df["close"].to_numpy()
    n = len(df)

    for i in range(scan_start, n):
        if zone_type == "demand" and close[i] < distal:
            return i
        if zone_type == "supply" and close[i] > distal:
            return i
    return None


# ---------------------------------------------------------------------------
# Step 1 — touch counter for one zone
# ---------------------------------------------------------------------------


def count_touches(df: pd.DataFrame, zone: dict) -> int:
    """Count wick re-entries into *zone* after its departure window.

    Parameters
    ----------
    df   : enriched DataFrame (must have high, low, close columns)
    zone : zone dict produced by detect_zones(); needs
           start, end, zone_type, proximal, distal

    Returns
    -------
    int — number of candles whose wick re-entered the zone before it died.
          Scanning stops at the zone's death bar (shared with find_death_bar).
    """
    proximal = zone["proximal"]
    zone_type = zone["zone_type"]
    scan_start = zone["end"] + DEPARTURE_CANDLES + 1

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    death = find_death_bar(df, zone)
    stop = death if death is not None else len(df)

    touches = 0
    for i in range(scan_start, stop):
        if zone_type == "demand" and low[i] <= proximal:
            touches += 1
        elif zone_type == "supply" and high[i] >= proximal:
            touches += 1

    return touches


# ---------------------------------------------------------------------------
# Step 2 — annotate a list of zones in-place
# ---------------------------------------------------------------------------


def add_freshness(df: pd.DataFrame, zones: list[dict]) -> list[dict]:
    """Add `touches` and `freshness_score` to every zone dict in *zones*.

    Parameters
    ----------
    df    : enriched DataFrame
    zones : list of zone dicts (from detect_zones — passed or rejected)

    Returns
    -------
    The same list with each dict updated in-place.
    """
    for z in zones:
        t = count_touches(df, z)
        z["touches"] = t
        z["freshness_score"] = FRESHNESS_SCORE_TABLE.get(t, 0)
    return zones
