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

from utils.config import DEPARTURE_CANDLES
from utils import costs
from utils import regime as regime_mod

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

# Asset class encoded as a stable integer. "unknown" → -1 sentinel.
# Order is arbitrary but FIXED — do not reorder once the first dataset is
# built, as that would invalidate stored models.
_ASSET_CLASS_CODE: dict[str, int] = {
    "crypto": 0,
    "fx": 1,
    "us_stocks": 2,
    "etfs": 3,
    "indices": 4,
    "commodities": 5,
    "macro": 6,
    "unknown": -1,
}

# Timeframe encoded in ascending duration order (natural ordinal).
_TIMEFRAME_CODE: dict[str, int] = {
    "1h": 0,
    "4h": 1,
    "1d": 2,
    "1wk": 3,
}


# ---------------------------------------------------------------------------
# NOTE — why touches_before_entry / freshness_score_at_entry are gone
# ---------------------------------------------------------------------------
# These features were removed because they have ZERO variance: every single
# labelled row has touches_before_entry = 1 (always) and therefore
# freshness_score_at_entry = 1 (always). The reason is structural:
# simulate_triple_barrier (labeler.py) only opens a trade on the FIRST bar
# that wicks into the proximal after the departure window. That first wicking
# bar is, by definition, touch #1. A second proximal touch would only be
# recorded if the first one was somehow skipped — but the labeler would have
# already opened the trade at bar #1, so a second touch is only reachable
# on a different trade. Zero-variance columns carry no information and should
# never be fed to a model.
# ---------------------------------------------------------------------------
# Per-zone feature row
# ---------------------------------------------------------------------------


def _build_row(
    z: dict,
    ltf_df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    timeframe: str,
    regime_series: pd.Series | None = None,
) -> dict:
    """Build one feature row for a labelled zone.

    Every value here must be computable from information available at or
    before z["entry_bar"]. Comments on each line that touches dataframe
    rows justify why the slice is lookahead-safe.
    """
    entry_bar = z["entry_bar"]
    avg_atr = float(z["avg_atr"])

    # --- relative / timing features (all derivable at entry) ---------------
    # bars_to_entry: number of bars price spent away from the zone before
    # returning. end+DEPARTURE_CANDLES+1 is the earliest possible touch bar
    # (matches scan_start in freshness). 0 means price returned immediately.
    earliest_touch = z["end"] + DEPARTURE_CANDLES + 1
    bars_to_entry = max(0, entry_bar - earliest_touch)

    risk = float(z["risk"])
    tp = float(z["tp"])
    entry = float(z["entry"])
    stop = float(z["stop"])
    # Risk/TP expressed in ATR units → cross-symbol / cross-regime comparable.
    risk_atr = risk / avg_atr if avg_atr > 0 else 0.0
    tp_distance_atr = abs(tp - entry) / avg_atr if avg_atr > 0 else 0.0

    # ---- audit-only post-entry fields (NOT features) --------------------
    # exit_bar may be None on no_trade rows; those are filtered out by the
    # caller (label is None), but be defensive anyway.
    exit_bar = z.get("exit_bar")
    exit_time = ltf_df.index[exit_bar] if exit_bar is not None else pd.NaT
    pnl_r = z.get("pnl_r")
    timeout_pnl_r = z.get("timeout_pnl_r")

    # --- categorical encodings ---------------------------------------------
    curve_third = z.get("curve_third", "n/a")
    curve_pos = z.get("curve_pos")
    # curve_pos can be None when the zone fell outside the HTF range; use
    # -1.0 sentinel so trees can still split on "missing" cleanly.
    curve_pos_num = float(curve_pos) if curve_pos is not None else -1.0

    # ---- regime feature (point-in-time lookup on the daily frame) -------
    # regime_series is a daily Series of {-1, 0, 1, 2} produced upstream by
    # utils.regime.compute_regime_series (expanding-window GMM, refit on
    # past data only). regime_at maps formation_time to the last daily bar
    # with timestamp <= formation_time — same rule htf_trend/curve_score use.
    # When no series is provided (e.g. unit tests), default to -1 = unknown.
    formation_ts = ltf_df.index[z["end"]]
    if regime_series is not None and len(regime_series) > 0:
        regime_code = regime_mod.regime_at(regime_series, formation_ts)
    else:
        regime_code = -1

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
        # ---- cost-awareness features (knowable at entry, no lookahead) ---
        #
        # Design note: `label` is the GROSS win/loss target (did price reach
        # TP before SL?). It deliberately ignores costs so the model learns
        # pure edge. These three features then let the model DOWN-WEIGHT
        # expensive trades on its own, without hard-coding a cost filter.
        #
        # expected_cost_r: round-trip transaction cost in R-multiples at
        #   cost_multiplier=1.0 (base case). Computed from entry price +
        #   risk_price + asset class — all known at entry time, zero
        #   lookahead. A trade needs gross_r > expected_cost_r to be
        #   net-profitable; the model can learn this threshold.
        "expected_cost_r": costs.expected_cost_r(
            asset_class, symbol, entry, risk, cost_multiplier=1.0
        ),
        # asset_class_code / timeframe_code: stable integer encodings so the
        # model can learn asset-class and timeframe interaction effects
        # (e.g. crypto+1h has structurally higher cost_r than etfs+1d).
        "asset_class_code": _ASSET_CLASS_CODE.get(asset_class, -1),
        "timeframe_code": _TIMEFRAME_CODE.get(timeframe, -1),
        # ---- market regime (GMM, expanding-window, lookahead-safe) -------
        # 0=calm  1=medium  2=turbulent  -1=warmup/unknown.
        "regime_code": int(regime_code),
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
        # ---- AUDIT (post-entry; for backtest / analysis only) ------------
        # These columns are deliberately NOT in FEATURE_COLS — the training
        # code selects via df[FEATURE_COLS] so they cannot leak into models.
        # They exist so backtest.py can convert price-based costs to R,
        # estimate trade occupancy windows, and respect timeout_pnl_r.
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "risk": risk,
        "exit_time": exit_time,
        "bars_held": int(z.get("bars_held", 0) or 0),
        "exit_reason": z.get("exit_reason", ""),
        "pnl_r": float(pnl_r) if pnl_r is not None else float("nan"),
        "timeout_pnl_r": (
            float(timeout_pnl_r) if timeout_pnl_r is not None else float("nan")
        ),
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
    # cost-awareness (knowable at entry; label stays GROSS — see note above)
    "expected_cost_r",
    "asset_class_code",
    "timeframe_code",
    # market regime (lookahead-safe GMM on daily frame)
    "regime_code",
    # relative timing / risk
    "bars_to_entry",
    "risk_atr",
    "tp_distance_atr",
]

META_COLS: list[str] = ["formation_time", "entry_time", "zone_type", "direction"]
TARGET_COL: str = "label"

# Audit columns: post-entry observations carried in the dataset for the
# backtester and trade-level analysis. NOT model inputs — kept out of
# FEATURE_COLS on purpose. Training code that does df[FEATURE_COLS] is
# unaffected. Order matters only for the on-disk column layout.
AUDIT_COLS: list[str] = [
    "entry",
    "stop",
    "tp",
    "risk",
    "exit_time",
    "bars_held",
    "exit_reason",
    "pnl_r",
    "timeout_pnl_r",
]

# EXPLICITLY EXCLUDED from FEATURE_COLS — either lookahead, zero-variance, or non-features.
# Listed here so the exclusion is grep-able and reviewable:
#   touches_before_entry,            — zero variance: always == 1 by construction
#   freshness_score_at_entry           (entry bar IS the first proximal touch;
#                                       see the NOTE block above for full explanation)
#   touches, freshness_score         — scan to death (post-entry) → lookahead
#   sets_total, sets_rating          — combine the post-entry freshness above
#   death_bar, is_alive, death_time  — outcome state (post-entry)
#   exit_bar, exit_reason, bars_held — trade OUTCOMES, not features (now in AUDIT_COLS)
#   pnl_r, timeout_pnl_r             — outcome PnL (now in AUDIT_COLS)
#   entry, stop, tp, risk            — raw price levels; not generalisable
#   proximal, distal                   across symbols/regimes (now in AUDIT_COLS)


def build_features(
    zones: list[dict],
    ltf_df: pd.DataFrame,
    symbol: str = "",
    asset_class: str = "unknown",
    timeframe: str = "",
    regime_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the ML feature DataFrame for all labelled zones.

    Parameters
    ----------
    zones       : list of zone dicts post-``labeler.label_zones``.
    ltf_df      : the SAME enriched LTF DataFrame used to label those zones
                  (timestamps + low/high are read from it).
    symbol      : ticker string (e.g. "EURUSD=X"). Used to compute
                  expected_cost_r (JPY-pip rescaling) and asset_class_code.
    asset_class : asset class string from config.WATCHLIST (e.g. "fx").
                  Used to compute expected_cost_r and asset_class_code.
    timeframe   : timeframe string ("1h", "4h", "1d"). Used for
                  timeframe_code. Defaults to empty string → code = -1.

    Returns
    -------
    DataFrame with columns: FEATURE_COLS + [TARGET_COL] + META_COLS + AUDIT_COLS.
    Only rows whose label is not None are included (zones that actually
    entered a trade — no_trade entries are dropped because they have no
    target).

    No scaling / normalisation is applied: features are kept raw so the
    caller can fit any scaler INSIDE the cross-validation fold on training
    data only (otherwise scaling on the full dataset would leak the test
    distribution into training).
    """
    rows = [
        _build_row(z, ltf_df, symbol, asset_class, timeframe, regime_series)
        for z in zones
        if z.get("label") is not None
    ]
    if not rows:
        return pd.DataFrame(
            columns=FEATURE_COLS + [TARGET_COL] + META_COLS + AUDIT_COLS
        )

    df = pd.DataFrame(rows)
    # Ensure deterministic column order — feature engineering downstream
    # asserts on this ordering.
    return df[FEATURE_COLS + [TARGET_COL] + META_COLS + AUDIT_COLS]
