# =============================================================================
# utils/trend_alignment.py — Trend alignment scoring (Phase 11)
# =============================================================================
#
# Responsibilities
# ----------------
# find_swings()      — locate swing-high and swing-low bar indices in a DataFrame.
# trend_at()         — classify trend (uptrend / downtrend / sideways) at a given
#                      bar position using the two most recent swing points.
# add_trend_score()  — annotate a list of zone dicts in-place with
#                      `trend` and `trend_score` fields.
#
# Trend definition (sd-concepts.md §13)
# --------------------------------------
# Swing high  : high[i] == max(high[i-w .. i+w])
# Swing low   : low[i]  == min(low[i-w .. i+w])
# where w = SWING_WINDOW (default 3)
#
# At the bar where a zone's base starts (zone["start"]), look back at the
# two most recent swing highs and two most recent swing lows:
#
#   uptrend   : latest SH > prev SH  AND  latest SL > prev SL   (HH + HL)
#   downtrend : latest SH < prev SH  AND  latest SL < prev SL   (LH + LL)
#   sideways  : anything else
#
# Scoring table:
#   demand + uptrend   → 2
#   supply + downtrend → 2
#   any    + sideways  → 1
#   demand + downtrend → 0
#   supply + uptrend   → 0
#
# Prerequisites
# -------------
# zones must come from zone_detector.detect_zones(); they must have:
#   start, zone_type
# The input DataFrame must have columns: high, low.
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import SWING_WINDOW

# ---------------------------------------------------------------------------
# Score lookup table
# ---------------------------------------------------------------------------
_TREND_SCORE: dict[tuple[str, str], int] = {
    ("demand", "uptrend"): 2,
    ("demand", "sideways"): 1,
    ("demand", "downtrend"): 0,
    ("supply", "downtrend"): 2,
    ("supply", "sideways"): 1,
    ("supply", "uptrend"): 0,
}


# ---------------------------------------------------------------------------
# Step 1 — locate swing points
# ---------------------------------------------------------------------------


def find_swings(
    df: pd.DataFrame,
    window: int = SWING_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of bar indices that are swing highs and swing lows.

    A bar ``i`` is a swing high if ``high[i]`` is the maximum over the
    window ``[i-w .. i+w]``; likewise for swing lows.
    Bars within ``window`` of either end of the DataFrame are excluded.

    Parameters
    ----------
    df     : DataFrame with ``high`` and ``low`` columns.
    window : half-window size in bars (default: SWING_WINDOW).

    Returns
    -------
    sh_idx : 1-D int array of swing-high bar indices.
    sl_idx : 1-D int array of swing-low bar indices.
    """
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    n = len(h)

    sh_idx, sl_idx = [], []
    for i in range(window, n - window):
        if h[i] == h[i - window : i + window + 1].max():
            sh_idx.append(i)
        if l[i] == l[i - window : i + window + 1].min():
            sl_idx.append(i)

    return np.array(sh_idx, dtype=int), np.array(sl_idx, dtype=int)


# ---------------------------------------------------------------------------
# Step 2 — classify trend at a single position
# ---------------------------------------------------------------------------


def trend_at(
    pos: int,
    sh_idx: np.ndarray,
    sl_idx: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> str:
    """Return ``"uptrend"``, ``"downtrend"``, or ``"sideways"`` at bar *pos*.

    Uses only swing points that formed strictly before *pos* (no lookahead).
    Returns ``"sideways"`` if fewer than 2 swing highs or 2 swing lows are
    available before *pos*.

    Parameters
    ----------
    pos    : bar index (iloc) to classify — usually ``zone["start"]``.
    sh_idx : swing-high indices from find_swings().
    sl_idx : swing-low  indices from find_swings().
    high   : numpy array of high prices.
    low    : numpy array of low  prices.
    """
    past_sh = sh_idx[sh_idx < pos]
    past_sl = sl_idx[sl_idx < pos]

    if len(past_sh) < 2 or len(past_sl) < 2:
        return "sideways"

    hh = high[past_sh[-1]] > high[past_sh[-2]]  # higher high
    hl = low[past_sl[-1]] > low[past_sl[-2]]  # higher low
    lh = high[past_sh[-1]] < high[past_sh[-2]]  # lower  high
    ll = low[past_sl[-1]] < low[past_sl[-2]]  # lower  low

    if hh and hl:
        return "uptrend"
    if lh and ll:
        return "downtrend"
    return "sideways"


# ---------------------------------------------------------------------------
# Step 3 — annotate a list of zones in-place
# ---------------------------------------------------------------------------


def add_trend_score(
    zones: list[dict],
    df: pd.DataFrame,
    window: int = SWING_WINDOW,
) -> list[dict]:
    """Add ``trend`` and ``trend_score`` to every zone dict in *zones*.

    Trend is evaluated **point-in-time** at each zone's base-start bar,
    using only swing points that formed before that bar (no lookahead).

    Parameters
    ----------
    zones  : list of zone dicts; each must have ``start`` and ``zone_type``.
    df     : enriched DataFrame (must have ``high`` and ``low`` columns).
    window : half-window for swing detection (default: SWING_WINDOW).

    Returns
    -------
    The same list with each dict updated in-place.
    """
    sh_idx, sl_idx = find_swings(df, window)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    for z in zones:
        t = trend_at(z["start"], sh_idx, sl_idx, high, low)
        z["trend"] = t
        z["trend_score"] = _TREND_SCORE.get((z["zone_type"], t), 1)

    return zones
