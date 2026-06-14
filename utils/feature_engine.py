# =============================================================================
# utils/feature_engine.py — ML feature matrix builder for labelled zones
# =============================================================================
#
# Golden rule
# -----------
# EVERY feature must be knowable AT the entry bar (the bar where price first
# touches the proximal). Anything computed using data from after entry is
# poisoned and is explicitly excluded — see the "EXPLICITLY EXCLUDED" block
# below for the full list and the reason each one is dropped.
#
# Pipeline expectation
# --------------------
# The input ``zones`` list must already have been processed by:
#   detect_bases  →  detect_formations  →  detect_zones
#   →  add_freshness  →  add_time_score  →  add_curve_score
#   →  add_trend_score  →  add_sets_score  →  labeler.label_zones
# So each zone carries detection scores, trade levels, and a label/entry_bar.
#
# Output
# ------
# build_features() returns a DataFrame with one row per zone whose
# ``label is not None`` (i.e. zones whose entry was actually triggered).
# Columns:
#   - the safe feature columns listed in FEATURE_COLS (model inputs)
#   - ``label`` (the supervised target)
#   - metadata columns used ONLY for splitting / audit (formation_time,
#     entry_time, zone_type, direction) — never fed to the model
# =============================================================================

from __future__ import annotations

import pandas as pd

from utils.config import DEPARTURE_CANDLES, FRESHNESS_SCORE_TABLE

# ---------------------------------------------------------------------------
# Encoders for categorical features
# ---------------------------------------------------------------------------

# Curve-third encoded as ordinal: low → 0, mid → 1, high → 2. "n/a" → -1 so
# the model can distinguish "not on the curve" from any real position.
_CURVE_THIRD_CODE: dict[str, int] = {"low": 0, "mid": 1, "high": 2, "n/a": -1}

# Trend encoded as signed ternary so up/down are reflections of each other
# around 0 (a sensible inductive bias for a linear model; trees ignore it).
_TREND_CODE: dict[str, int] = {
    "uptrend": 1,
    "sideways": 0,
    "downtrend": -1,
    "n/a": 0,
}


# ---------------------------------------------------------------------------
# Recomputed-safe freshness (this is the lookahead-critical piece)
# ---------------------------------------------------------------------------


def _touches_before_entry(
    ltf_df: pd.DataFrame,
    zone: dict,
    entry_bar: int,
) -> int:
    """Count wick re-entries into *zone* from the departure window up to
    (and INCLUDING) ``entry_bar``.

    Why this exists
    ---------------
    The stored ``touches`` / ``freshness_score`` from ``freshness.count_touches``
    scan the entire post-departure life of the zone — i.e. they include
    touches that happen AFTER entry. Using them as a model feature would
    leak future information. This helper recomputes the same definition but
    bounded by entry_bar (exclusive of the search end via range, so
    ``entry_bar`` itself IS counted) so the feature only reflects what was
    knowable at the moment the trade opened.

    Touch definition (same as freshness.count_touches):
      * demand : low[i]  <= proximal
      * supply : high[i] >= proximal
    """
    proximal = zone["proximal"]
    zone_type = zone["zone_type"]
    scan_start = zone["end"] + DEPARTURE_CANDLES + 1
    # +1 so the entry bar itself is counted as a touch (it IS the touch that
    # triggers entry — knowable at entry time, no lookahead).
    scan_stop = entry_bar + 1

    if scan_stop <= scan_start:
        return 0

    if zone_type == "demand":
        wick = ltf_df["low"].iloc[scan_start:scan_stop].to_numpy()
        # numpy bool sum = touch count
        return int((wick <= proximal).sum())

    wick = ltf_df["high"].iloc[scan_start:scan_stop].to_numpy()
    return int((wick >= proximal).sum())


def _freshness_at_entry(touches: int) -> int:
    """Same mapping as freshness.add_freshness — 0→2, 1→1, 2+→0."""
    return FRESHNESS_SCORE_TABLE.get(touches, 0)


# ---------------------------------------------------------------------------
# Per-zone feature row
# ---------------------------------------------------------------------------


def _build_row(z: dict, ltf_df: pd.DataFrame) -> dict:
    """Build one feature row for a labelled zone.

    Every value here must be computable from information available at or
    before z["entry_bar"]. Comments on each line that touches dataframe
    rows justify why the slice is lookahead-safe.
    """
    entry_bar = z["entry_bar"]
    avg_atr = float(z["avg_atr"])

    # --- recomputed freshness (bounded by entry_bar — NOT by death) -------
    # _touches_before_entry slices ltf_df at [end+L+1 : entry_bar+1], all of
    # which are bars that have already closed by the moment of entry. Safe.
    touches_be = _touches_before_entry(ltf_df, z, entry_bar)
    freshness_be = _freshness_at_entry(touches_be)

    # --- relative / timing features (all derivable at entry) ---------------
    # bars_to_entry: number of bars price spent away from the zone before
    # returning. end+DEPARTURE_CANDLES+1 is the earliest possible touch bar
    # (matches scan_start in freshness). 0 means price returned immediately.
    earliest_touch = z["end"] + DEPARTURE_CANDLES + 1
    bars_to_entry = max(0, entry_bar - earliest_touch)

    risk = float(z["risk"])
    tp = float(z["tp"])
    entry = float(z["entry"])
    # Risk/TP expressed in ATR units → cross-symbol / cross-regime comparable.
    risk_atr = risk / avg_atr if avg_atr > 0 else 0.0
    tp_distance_atr = abs(tp - entry) / avg_atr if avg_atr > 0 else 0.0

    # --- categorical encodings ---------------------------------------------
    curve_third = z.get("curve_third", "n/a")
    curve_pos = z.get("curve_pos")
    # curve_pos can be None when the zone fell outside the HTF range; use
    # -1.0 sentinel so trees can still split on "missing" cleanly.
    curve_pos_num = float(curve_pos) if curve_pos is not None else -1.0

    return {
        # ---- geometry / departure (known at formation, before entry) -----
        "dep_ratio": float(z["dep_ratio"]),
        "dep_atr": float(z["dep_atr"]),
        "departure": float(z["departure"]),
        "zone_width": float(z["zone_width"]),
        "compactness_ratio": float(z["compactness_ratio"]),
        "leg_strength": float(z["leg_strength"]),
        "base_count": int(z["base_count"]),
        "avg_atr": avg_atr,
        # ---- scores (all point-in-time at formation) ---------------------
        "strength_score": int(z["strength_score"]),
        "time_score": int(z["time_score"]),
        "curve_score": int(z["curve_score"]),
        "trend_score": int(z["trend_score"]),
        "curve_pos": curve_pos_num,
        "curve_third_code": _CURVE_THIRD_CODE.get(curve_third, -1),
        # ---- trend context (computed point-in-time by labeler) -----------
        "trend_aligned": int(bool(z.get("trend_aligned", False))),
        "itf_trend_code": _TREND_CODE.get(z.get("itf_trend_at_formation", "n/a"), 0),
        "htf_trend_code": _TREND_CODE.get(z.get("htf_trend_at_formation", "n/a"), 0),
        # ---- zone type ---------------------------------------------------
        "is_demand": 1 if z["zone_type"] == "demand" else 0,
        # ---- recomputed freshness (the key lookahead-safe rebuild) -------
        "touches_before_entry": touches_be,
        "freshness_score_at_entry": freshness_be,
        # ---- relative timing / risk (all knowable at entry) --------------
        "bars_to_entry": int(bars_to_entry),
        "risk_atr": float(risk_atr),
        "tp_distance_atr": float(tp_distance_atr),
        # ---- TARGET ------------------------------------------------------
        "label": int(z["label"]),
        # ---- metadata (NEVER feed to the model) --------------------------
        # ltf_df.index lookups below pull timestamps for bars that have
        # ALREADY closed — no lookahead.
        "formation_time": ltf_df.index[z["end"]],
        "entry_time": ltf_df.index[entry_bar],
        "zone_type": z["zone_type"],
        "direction": z["direction"],
    }


# ---------------------------------------------------------------------------
# Public API + canonical feature list
# ---------------------------------------------------------------------------

# Real feature columns — the training code must feed exactly these to the
# model. `label` is the target; the four metadata columns are for time-
# ordered splitting and audit only.
FEATURE_COLS: list[str] = [
    # geometry / departure
    "dep_ratio",
    "dep_atr",
    "departure",
    "zone_width",
    "compactness_ratio",
    "leg_strength",
    "base_count",
    "avg_atr",
    # scores
    "strength_score",
    "time_score",
    "curve_score",
    "trend_score",
    "curve_pos",
    "curve_third_code",
    # trend context
    "trend_aligned",
    "itf_trend_code",
    "htf_trend_code",
    # zone type
    "is_demand",
    # recomputed-safe freshness
    "touches_before_entry",
    "freshness_score_at_entry",
    # relative timing / risk
    "bars_to_entry",
    "risk_atr",
    "tp_distance_atr",
]

META_COLS: list[str] = ["formation_time", "entry_time", "zone_type", "direction"]
TARGET_COL: str = "label"

# EXPLICITLY EXCLUDED — these are either lookahead or non-features.
# Listed here so the exclusion is grep-able and reviewable:
#   touches, freshness_score        — scan to death (post-entry) → lookahead
#   sets_total, sets_rating          — combine the post-entry freshness above
#   death_bar, is_alive, death_time  — outcome state (post-entry)
#   exit_bar, exit_reason, bars_held — trade OUTCOMES, not features
#   pnl_r, timeout_pnl_r             — outcome PnL
#   entry, stop, tp,                 — raw price levels; not generalisable
#   proximal, distal                   across symbols/regimes — use *_atr versions


def build_features(zones: list[dict], ltf_df: pd.DataFrame) -> pd.DataFrame:
    """Build the ML feature DataFrame for all labelled zones.

    Parameters
    ----------
    zones  : list of zone dicts post-``labeler.label_zones``.
    ltf_df : the SAME enriched LTF DataFrame used to label those zones
             (timestamps + low/high are read from it).

    Returns
    -------
    DataFrame with columns: FEATURE_COLS + [TARGET_COL] + META_COLS.
    Only rows whose label is not None are included (zones that actually
    entered a trade — no_trade entries are dropped because they have no
    target).

    No scaling / normalisation is applied: features are kept raw so the
    caller can fit any scaler INSIDE the cross-validation fold on training
    data only (otherwise scaling on the full dataset would leak the test
    distribution into training).
    """
    rows = [_build_row(z, ltf_df) for z in zones if z.get("label") is not None]
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL] + META_COLS)

    df = pd.DataFrame(rows)
    # Ensure deterministic column order — feature engineering downstream
    # asserts on this ordering.
    return df[FEATURE_COLS + [TARGET_COL] + META_COLS]
