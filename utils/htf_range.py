# =============================================================================
# utils/htf_range.py — HTF range helpers
# =============================================================================
#
# Provides point-in-time HTF range lookup used by curve scoring and charts.
# Kept separate so notebooks and other modules can import it without pulling
# in the full scoring module.
# =============================================================================

from __future__ import annotations

import pandas as pd

from utils.config import HTF_RANGE_LOOKBACK

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
        r = htf_range_asof(htf_df, ts, lookback)

        if r is None:
            z["curve_pos"] = None
            z["curve_third"] = "n/a"
            z["curve_score"] = 0
            continue

        hi, lo = r
        rng = hi - lo
        if rng <= 0:
            z["curve_pos"] = None
            z["curve_third"] = "n/a"
            z["curve_score"] = 0
            continue

        pos = (z["proximal"] - lo) / rng
        pos = max(0.0, min(1.0, pos))  # clamp to [0, 1]
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
