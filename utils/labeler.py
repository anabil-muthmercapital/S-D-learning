# =============================================================================
# utils/labeler.py — Triple Barrier labelling for ML training
# =============================================================================
#
# Responsibilities
# ----------------
# 1. STEP 1 — direction filter:   curve_third + point-in-time HTF/ITF trend
#                                 decide whether a zone is tradeable.
# 2. STEP 2 — trade levels:       entry = proximal, SL beyond distal (with
#                                 ATR_STOP_BUFFER), TP at RR_RATIO × risk.
# 3. STEP 3 — triple-barrier sim: walk forward bar-by-bar to find which
#                                 of {SL, TP, vertical barrier} is hit first.
# 4. Output a `label` ∈ {0, 1, None} suitable for supervised training plus
#    a rich set of audit fields (entry_bar, exit_bar, exit_reason, bars_held,
#    pnl_r, timeout_pnl_r, ignore_reason, …).
#
# Lookahead safety (the whole point of this module)
# -------------------------------------------------
# The label necessarily looks forward — that is fine, it is the *target*.
# What must NEVER look forward is any feature or decision that goes into
# the trade:
#   * direction filter uses HTF/ITF swings strictly BEFORE the zone's
#     formation timestamp (reuses trend_alignment.find_swings + trend_at,
#     which already enforce `idx < pos`).
#   * the HTF/ITF bar mapped to the zone is the last bar whose timestamp
#     is <= the zone's formation time (`searchsorted(side="right") - 1`),
#     i.e. it has already CLOSED at that moment in time.
#   * the simulation never inspects a bar at iloc < entry_bar.
#   * if a single bar's range hits both SL and TP, the SL is assumed to
#     have been hit first (worst-case, anti-optimism rule).
#
# Death detection reuses freshness.find_death_bar so the lifespan is
# identical to the rest of the system (single source of truth).
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import (
    ATR_STOP_BUFFER,
    DEPARTURE_CANDLES,
    RR_RATIO,
)
from utils.freshness import find_death_bar
from utils.trend_alignment import find_swings, trend_at

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# max_hold_bars default = 60.
# Rationale: with a 3R take-profit, the trade needs time for a full move to
# play out. 60 bars is roughly:
#   * 1h  → 2.5 trading weeks
#   * 4h  → ~2 trading months
#   * 1d  → ~3 trading months
#   * 1wk → over a year
# Caller can override per-timeframe; this is a permissive default that
# avoids forcing too many "timeout" labels on faster frames without letting
# slow frames sit open for years.
DEFAULT_MAX_HOLD_BARS: int = 60

# Default LTF → (ITF, HTF) mapping consistent with HTF_REF in config.py.
# For 1d there is no frame higher than 1wk, so 1wk doubles as both ITF and HTF.
DEFAULT_HTF_LTF_MAP: dict[str, tuple[str, str | None]] = {
    "1h": ("4h", "1d"),
    "4h": ("1d", "1wk"),
    "1d": ("1wk", "1wk"),
}


# ---------------------------------------------------------------------------
# Helpers — point-in-time HTF/ITF mapping
# ---------------------------------------------------------------------------


def _bar_at_or_before(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """Return the iloc of the latest bar whose timestamp is <= *ts*.

    Lookahead-safe: searchsorted(side="right") gives the insertion point to
    the RIGHT of any bar equal to *ts*; subtracting 1 lands on the last bar
    that has already closed at time *ts*. Returns -1 if no such bar exists.
    """
    pos = int(index.searchsorted(ts, side="right")) - 1
    return pos


def _trend_at_timestamp(
    df: pd.DataFrame,
    sh_idx: np.ndarray,
    sl_idx: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ts: pd.Timestamp,
) -> str:
    """Wrap trend_at with a timestamp lookup on a DIFFERENT timeframe.

    The HTF/ITF trend is evaluated at the bar position corresponding to the
    zone's LTF formation timestamp. Mapping uses the last-bar-<=-ts rule so
    no future HTF/ITF bar can leak into the decision.
    """
    pos = _bar_at_or_before(df.index, ts)
    if pos < 0:
        return "sideways"  # the LTF zone formed before this HTF/ITF series begins
    # trend_at uses sh_idx[sh_idx < pos] internally (strictly before),
    # so a swing detected exactly at `pos` is correctly excluded.
    return trend_at(pos, sh_idx, sl_idx, high, low)


# ---------------------------------------------------------------------------
# STEP 1 — direction filter
# ---------------------------------------------------------------------------


def _decide_direction(
    z: dict,
    htf_trend: str | None,
    itf_trend: str,
) -> tuple[bool, str | None, str]:
    """Return (tradeable, direction, ignore_reason).

    Curve-position rules (from the spec):
      * high  : need HTF downtrend + supply  → short
      * low   : need HTF uptrend   + demand  → long
      * mid   : need ITF aligned   (uptrend+demand=long, downtrend+supply=short)
      * n/a   : ignore (we cannot place the zone on the curve)
    """
    third = z.get("curve_third", "n/a")
    ztype = z["zone_type"]

    if third == "n/a":
        return False, None, "curve_third=n/a"

    if third == "high":
        if htf_trend is None:
            return False, None, "no HTF available for curve_third=high"
        if htf_trend == "downtrend" and ztype == "supply":
            return True, "short", ""
        return False, None, f"curve_third=high but HTF={htf_trend}/zone={ztype}"

    if third == "low":
        if htf_trend is None:
            return False, None, "no HTF available for curve_third=low"
        if htf_trend == "uptrend" and ztype == "demand":
            return True, "long", ""
        return False, None, f"curve_third=low but HTF={htf_trend}/zone={ztype}"

    if third == "mid":
        if itf_trend == "uptrend" and ztype == "demand":
            return True, "long", ""
        if itf_trend == "downtrend" and ztype == "supply":
            return True, "short", ""
        return False, None, f"curve_third=mid but ITF={itf_trend}/zone={ztype}"

    return False, None, f"unknown curve_third={third!r}"


# ---------------------------------------------------------------------------
# STEP 2 — trade levels
# ---------------------------------------------------------------------------


def _compute_levels(z: dict) -> dict:
    """Return entry/stop/tp/risk for a tradeable zone.

    For long  (demand) : SL just BELOW the distal (base_low),
                          TP = entry + RR_RATIO × risk.
    For short (supply) : SL just ABOVE the distal (base_high),
                          TP = entry − RR_RATIO × risk.

    The ATR buffer prevents the SL from sitting exactly on the wick that
    formed the base — a single-tick spike beyond the distal would otherwise
    blow through every trade.
    """
    direction = z["direction"]
    proximal = float(z["proximal"])
    distal = float(z["distal"])
    avg_atr = float(z["avg_atr"])
    buf = ATR_STOP_BUFFER * avg_atr

    if direction == "long":
        entry = proximal
        stop = distal - buf
        risk = entry - stop
        tp = entry + RR_RATIO * risk
    else:  # short
        entry = proximal
        stop = distal + buf
        risk = stop - entry
        tp = entry - RR_RATIO * risk

    return {
        "entry": round(entry, 5),
        "stop": round(stop, 5),
        "tp": round(tp, 5),
        "risk": round(risk, 5),
    }


# ---------------------------------------------------------------------------
# STEP 3 — triple-barrier simulation
# ---------------------------------------------------------------------------


def simulate_triple_barrier(
    ltf_df: pd.DataFrame,
    zone: dict,
    max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
) -> dict:
    """Walk forward and return the outcome of one trade.

    Procedure
    ---------
    1. Search for the entry bar — the first bar after the departure window
       whose wick reaches the proximal (price returns to the zone). Search
       stops at the zone's death bar; if none is found before the zone
       breaks, the trade is "no_trade".
    2. From entry+1 onward, check each bar (up to max_hold_bars) for:
         * SL hit  : low <= stop (long) / high >= stop (short)
         * TP hit  : high >= tp  (long) / low  <= tp   (short)
       The entry bar itself is NOT used to resolve outcome — execution is
       assumed to fill at the proximal on bar `entry_bar` and outcome can
       only be observed from the NEXT bar onward (no same-bar cheating).
    3. Same-bar ambiguity: if a single bar shows both SL and TP in its
       range, assume the SL fired first (worst case → anti-lookahead).
    4. If neither barrier hits within max_hold_bars, the vertical barrier
       triggers ("timeout"); record the unrealised R-multiple at that bar.

    Lookahead safety
    ----------------
    - We never inspect a bar with iloc < entry_bar.
    - The same-bar tie rule is the standard López de Prado precaution
      against optimistic outcomes when intra-bar order is unknowable.
    """
    direction = zone["direction"]
    entry = float(zone["entry"])
    stop = float(zone["stop"])
    tp = float(zone["tp"])
    risk = float(zone["risk"])

    high = ltf_df["high"].to_numpy()
    low = ltf_df["low"].to_numpy()
    n = len(ltf_df)

    # ---- entry search window ------------------------------------------------
    # Earliest moment price could realistically retrace to the zone is one
    # bar after the departure window (matches freshness.count_touches).
    entry_search_start = zone["end"] + DEPARTURE_CANDLES + 1
    death = find_death_bar(ltf_df, zone)
    entry_search_stop = death if death is not None else n  # exclusive

    proximal = float(zone["proximal"])
    entry_bar: int | None = None
    for i in range(entry_search_start, entry_search_stop):
        if direction == "long" and low[i] <= proximal:
            entry_bar = i
            break
        if direction == "short" and high[i] >= proximal:
            entry_bar = i
            break

    if entry_bar is None:
        return {
            "entry_bar": None,
            "exit_bar": None,
            "exit_reason": "no_trade",
            "bars_held": 0,
            "label": None,
            "pnl_r": 0.0,
            "timeout_pnl_r": None,
        }

    # ---- triple-barrier walk ------------------------------------------------
    # Outcome is observed from entry_bar + 1 onward — the entry bar itself
    # is the moment of execution, not of resolution.
    walk_stop = min(entry_bar + 1 + max_hold_bars, n)  # exclusive
    last_close: float | None = None

    for j in range(entry_bar + 1, walk_stop):
        bar_high = high[j]
        bar_low = low[j]

        if direction == "long":
            sl_hit = bar_low <= stop
            tp_hit = bar_high >= tp
        else:
            sl_hit = bar_high >= stop
            tp_hit = bar_low <= tp

        # Same-bar ambiguity → SL wins (anti-cheating rule).
        if sl_hit and tp_hit:
            return _outcome(entry_bar, j, "sl", direction, entry, risk, exit_price=stop)
        if sl_hit:
            return _outcome(entry_bar, j, "sl", direction, entry, risk, exit_price=stop)
        if tp_hit:
            return _outcome(entry_bar, j, "tp", direction, entry, risk, exit_price=tp)

        last_close = float(ltf_df["close"].iat[j])

    # ---- vertical barrier ---------------------------------------------------
    # Timeout: the trade is closed at the last observed close inside the hold
    # window. timeout_pnl_r records the unrealised R-multiple at that point so
    # downstream experiments can re-label timeouts (e.g. as wins if pnl_r > 0).
    if last_close is None:
        # walk_stop == entry_bar + 1, i.e. no future bars exist at all.
        return {
            "entry_bar": entry_bar,
            "exit_bar": entry_bar,
            "exit_reason": "no_trade",
            "bars_held": 0,
            "label": None,
            "pnl_r": 0.0,
            "timeout_pnl_r": None,
        }

    exit_bar = walk_stop - 1
    if direction == "long":
        unrealised = (last_close - entry) / risk
    else:
        unrealised = (entry - last_close) / risk

    return {
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "label": 0,  # default: timeouts are losses; flip via timeout_pnl_r if desired
        "pnl_r": round(float(unrealised), 4),
        "timeout_pnl_r": round(float(unrealised), 4),
    }


def _outcome(
    entry_bar: int,
    exit_bar: int,
    reason: str,
    direction: str,
    entry: float,
    risk: float,
    exit_price: float,
) -> dict:
    """Build the outcome dict for a barrier hit (SL or TP)."""
    if direction == "long":
        pnl = (exit_price - entry) / risk
    else:
        pnl = (entry - exit_price) / risk
    return {
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "exit_reason": reason,
        "bars_held": exit_bar - entry_bar,
        "label": 1 if reason == "tp" else 0,
        "pnl_r": round(float(pnl), 4),
        "timeout_pnl_r": None,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def label_zones(
    zones: list[dict],
    ltf_df: pd.DataFrame,
    itf_df: pd.DataFrame | None,
    htf_df: pd.DataFrame | None,
    max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
    htf_ltf_map: dict[str, tuple[str, str | None]] | None = None,  # noqa: ARG001
) -> list[dict]:
    """Annotate every zone with direction-filter, trade-level and label fields.

    Parameters
    ----------
    zones         : scored zones for ONE LTF (output of the full pipeline,
                    must already carry curve_third, zone_type, proximal,
                    distal, avg_atr, start, end).
    ltf_df        : enriched LTF DataFrame for the same timeframe.
    itf_df        : enriched ITF DataFrame, or None (mid-curve filter falls
                    back to "sideways" → most mid zones get rejected).
    htf_df        : enriched HTF DataFrame, or None (high/low-curve zones
                    are rejected with reason "no HTF available").
    max_hold_bars : vertical barrier in LTF bars.
    htf_ltf_map   : retained for API compatibility; the caller already
                    selects which frames to pass in, so this parameter is
                    unused inside the function.

    Returns
    -------
    The same list with each zone updated in-place.
    """
    # ---- precompute ITF/HTF swings ONCE (vectorised, expensive otherwise) ----
    if itf_df is not None and len(itf_df) > 0:
        itf_sh, itf_sl = find_swings(itf_df)
        itf_high = itf_df["high"].to_numpy()
        itf_low = itf_df["low"].to_numpy()
    else:
        itf_sh = itf_sl = np.array([], dtype=int)
        itf_high = itf_low = np.array([])

    if htf_df is not None and len(htf_df) > 0:
        htf_sh, htf_sl = find_swings(htf_df)
        htf_high = htf_df["high"].to_numpy()
        htf_low = htf_df["low"].to_numpy()
        htf_index = htf_df.index
    else:
        htf_sh = htf_sl = np.array([], dtype=int)
        htf_high = htf_low = np.array([])
        htf_index = None

    itf_index = itf_df.index if itf_df is not None and len(itf_df) > 0 else None
    ltf_index = ltf_df.index

    for z in zones:
        # ---- formation timestamp on the LTF ---------------------------------
        # The "moment" the zone exists is when its base finishes forming —
        # iloc `z["end"]` in LTF terms. We map THIS timestamp onto ITF/HTF.
        formation_ts = ltf_index[z["end"]]

        # ---- ITF trend -------------------------------------------------------
        if itf_index is not None:
            # Lookahead-safe: _trend_at_timestamp only uses swings strictly
            # before the bar at-or-before formation_ts.
            itf_trend = _trend_at_timestamp(
                itf_df, itf_sh, itf_sl, itf_high, itf_low, formation_ts
            )
        else:
            itf_trend = "sideways"

        # ---- HTF trend -------------------------------------------------------
        if htf_index is not None:
            htf_trend: str | None = _trend_at_timestamp(
                htf_df, htf_sh, htf_sl, htf_high, htf_low, formation_ts
            )
        else:
            htf_trend = None

        z["itf_trend_at_formation"] = itf_trend
        z["htf_trend_at_formation"] = htf_trend if htf_trend is not None else "n/a"

        # ---- STEP 1: direction filter ---------------------------------------
        tradeable, direction, reason = _decide_direction(z, htf_trend, itf_trend)
        z["tradeable"] = tradeable
        z["direction"] = direction
        z["ignore_reason"] = reason

        if not tradeable:
            z.update(
                {
                    "entry": None,
                    "stop": None,
                    "tp": None,
                    "risk": None,
                    "entry_bar": None,
                    "exit_bar": None,
                    "exit_reason": "filtered",
                    "bars_held": 0,
                    "label": None,
                    "pnl_r": None,
                    "timeout_pnl_r": None,
                }
            )
            continue

        # ---- STEP 2: trade levels -------------------------------------------
        levels = _compute_levels(z)
        z.update(levels)

        # ---- STEP 3: triple-barrier sim -------------------------------------
        outcome = simulate_triple_barrier(ltf_df, z, max_hold_bars=max_hold_bars)
        z.update(outcome)

    return zones
