# =============================================================================
# utils/regime.py — Lookahead-safe market regime detection via GMM
# =============================================================================
#
# Purpose
# -------
# Adds a single ordinal feature `regime_code` ∈ {-1, 0, 1, 2} representing
# the macro volatility regime at the time a zone formed:
#
#   0 = calm / low-vol      (sorted by mean realised vol of the GMM
#   1 = medium               components — stable, interpretable codes)
#   2 = turbulent / high-vol
#  -1 = unknown (warmup, or could not be computed)
#
# Why a separate module
# ---------------------
# Regimes are a HIGHER-TIMEFRAME concept: the model wants to know "what kind
# of market are we in right now", which is a macro question best answered on
# the 1d frame. This module computes regimes ONCE per symbol from its daily
# series, then exposes a Series that the feature engine can sample at any
# lower-timeframe zone's formation_time using the project-standard
# `_bar_at_or_before` rule.
#
# Lookahead-safety contract (the whole point of this module)
# ----------------------------------------------------------
# A naive implementation would fit ONE GaussianMixture on the entire history
# and label every bar with the resulting cluster id. That LEAKS THE FUTURE
# into the past — the GMM's component means/covariances were calibrated on
# data that did not exist at the historical bar being classified.
#
# This module REFUSES to do that. Instead:
#
#   1. Per-bar regime features (returns, realised vol) are computed from
#      backward-looking windows only.  ret_t depends on close[t-1..t];
#      vol_t depends on returns[t-19..t].  No window reaches into bar t+1.
#
#   2. GMM is refit on an EXPANDING WINDOW of past bars only.  To label
#      bar `i`, we use the most recent fit whose training data ended at
#      some bar `k` with k < i.  Refitting happens every REFIT_EVERY bars
#      (default 20) to keep cost manageable while still adapting to drift.
#
#   3. Before REGIME_WARMUP bars (default 60) have closed, regime = -1.
#
#   4. When a zone (on any LTF) requests its regime, the caller maps the
#      zone's formation_time to the daily bar position using
#      `_bar_at_or_before` (searchsorted side="right" - 1) — the same rule
#      htf_trend and curve_score already use. This guarantees no daily bar
#      timestamped AFTER the zone formed can be read.
#
# Self-check
# ----------
# `verify_no_lookahead(...)` re-runs the fit/predict for every bar with an
# explicit assertion that the GMM's training set ends strictly before that
# bar's timestamp. Call it once with `--verify` from build_dataset.py during
# the first build; not needed thereafter (the invariant is structural).
# =============================================================================

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Tunables (kept module-level so they can be overridden in tests / experiments)
# ---------------------------------------------------------------------------

REGIME_WARMUP: int = 60  # bars of history required before the first fit
REFIT_EVERY: int = 20  # refit cadence in bars (carry last fit forward)
N_COMPONENTS: int = 3  # 3-regime model: calm / medium / turbulent
RET_VOL_WINDOW: int = 20  # rolling window for realised-vol feature
RET_LONG_WINDOW: int = 5  # multi-bar log return feature
RANDOM_STATE: int = 42  # fixed for reproducibility

# Regime features fed to the GMM. ALL are backward-looking windows ending
# at the bar being characterised — no future bar contributes.
_REGIME_FEATURE_COLS: list[str] = ["ret_1", "ret_5", "vol_20", "ret_atr"]


# ---------------------------------------------------------------------------
# Step 1 — per-bar regime features (backward-looking windows only)
# ---------------------------------------------------------------------------


def _compute_regime_features(htf_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the regime-input feature matrix on the HTF (1d) frame.

    Every column is a function of bars at-or-before the row's timestamp:
      * ret_1   : log(close_t / close_{t-1})           — uses bar t-1, t
      * ret_5   : log(close_t / close_{t-5})           — uses bars t-5..t
      * vol_20  : rolling std of ret_1 over 20 bars    — uses bars t-19..t
      * ret_atr : ret_1 normalised by ATR (cross-asset comparable)
    """
    if "close" not in htf_df.columns:
        raise ValueError("regime: htf_df must have a 'close' column")

    close = htf_df["close"].astype(float)
    ret_1 = np.log(close / close.shift(1))
    ret_5 = np.log(close / close.shift(RET_LONG_WINDOW))
    vol_20 = ret_1.rolling(RET_VOL_WINDOW, min_periods=RET_VOL_WINDOW).std()

    # ATR may already be in the enriched HTF frame (data_loader adds it).
    # Fallback to a 14-bar true-range proxy if absent. Either way it uses
    # only past bars — high/low/close are observed for completed bars.
    if "atr" in htf_df.columns:
        atr = htf_df["atr"].astype(float)
    else:
        h, l = htf_df["high"].astype(float), htf_df["low"].astype(float)
        tr = (
            (h - l)
            .combine((h - close.shift(1)).abs(), max)
            .combine((l - close.shift(1)).abs(), max)
        )
        atr = tr.rolling(14, min_periods=14).mean()

    # Normalise close-to-close return by ATR to give a unit-free move size.
    # Replace inf/zero ATR with NaN so the GMM doesn't see degenerate rows.
    ret_atr = ret_1 / atr.replace(0.0, np.nan)

    out = pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_5": ret_5,
            "vol_20": vol_20,
            "ret_atr": ret_atr,
        },
        index=htf_df.index,
    )
    # Replace any residual infs with NaN; we'll drop NaN rows before fitting.
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Step 2 — map raw GMM cluster ids to stable ordinal codes (sorted by vol)
# ---------------------------------------------------------------------------


def _sort_components_by_vol(gmm: GaussianMixture, vol_col_idx: int) -> np.ndarray:
    """Return an array `mapping[k] = ordinal_code` so that the GMM
    component with the LOWEST mean-realised-vol gets code 0, the next 1,
    the highest 2. Without this the raw GMM cluster ids are arbitrary and
    would flip between refits."""
    means = gmm.means_[:, vol_col_idx]  # shape (n_components,)
    order = np.argsort(means)  # ascending
    mapping = np.empty(len(means), dtype=np.int64)
    for ordinal, raw_id in enumerate(order):
        mapping[raw_id] = ordinal
    return mapping


# ---------------------------------------------------------------------------
# Step 3 — expanding-window fit + predict, lookahead-safe
# ---------------------------------------------------------------------------


def compute_regime_series(
    htf_df: pd.DataFrame,
    warmup: int = REGIME_WARMUP,
    refit_every: int = REFIT_EVERY,
    n_components: int = N_COMPONENTS,
    random_state: int = RANDOM_STATE,
    verify: bool = False,
) -> pd.Series:
    """Return a Series of ordinal regime codes aligned to ``htf_df.index``.

    Algorithm (the lookahead-safety story)
    --------------------------------------
    Let f[0..N-1] be the per-bar regime-feature rows. To assign a regime to
    bar i:
        * If i < warmup, regime = -1.
        * Otherwise let k = warmup + refit_every * ((i - warmup) // refit_every).
          k is the LATEST refit point that is STRICTLY ≤ i. The model used
          to predict bar i was fit on f[0..k-1] — i.e. only bars BEFORE k,
          which (since k ≤ i) is a strict subset of bars BEFORE i. No
          information about bars [i..N-1] reached the fit. QED lookahead-safe.

    Parameters
    ----------
    htf_df       : enriched daily DataFrame with close (and ideally atr).
    warmup       : bars of history required before regimes are produced.
    refit_every  : refit cadence in bars. Smaller = more adaptive, slower.
    n_components : GMM components (kept at 3 for calm / medium / turbulent).
    verify       : when True, assert the lookahead invariant for every
                   refit point (slow; intended for one-off audits).

    Returns
    -------
    pd.Series[int64] aligned to htf_df.index. Values in {-1, 0, 1, 2}.
    Warmup bars and any bar whose regime features contain NaN get -1.
    """
    feats = _compute_regime_features(htf_df)
    n_bars = len(feats)
    codes = np.full(n_bars, -1, dtype=np.int64)

    if n_bars <= warmup:
        # Not enough history for even one fit.
        return pd.Series(codes, index=htf_df.index, name="regime_code")

    vol_col_idx = _REGIME_FEATURE_COLS.index("vol_20")
    last_gmm: Optional[GaussianMixture] = None
    last_mapping: Optional[np.ndarray] = None
    last_fit_end: int = -1  # exclusive upper bound of last fit's training data

    # Walk bars forward. Refit at every `refit_every`-th bar starting at warmup.
    for i in range(warmup, n_bars):
        # Refit if we've crossed a refit boundary. The fit consumes bars
        # [0..i-1], NEVER including bar i. This is the single line where
        # lookahead could sneak in — keep it strict.
        if (i - warmup) % refit_every == 0:
            train_slice = feats.iloc[:i].dropna()  # bars 0..i-1, NaN dropped

            if len(train_slice) < max(warmup, n_components * 5):
                # Not enough clean rows yet (early bars often have NaN from
                # the 20-bar vol window). Carry forward the last fit if any.
                if last_gmm is None:
                    continue
            else:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="full",
                    random_state=random_state,
                    reg_covar=1e-6,
                    max_iter=200,
                )
                try:
                    gmm.fit(train_slice.values)
                    last_gmm = gmm
                    last_mapping = _sort_components_by_vol(gmm, vol_col_idx)
                    last_fit_end = i  # training data was bars [0..i-1]
                    if verify:
                        # Self-check: the last training timestamp must be
                        # strictly before the bar we're about to predict.
                        last_train_ts = train_slice.index[-1]
                        predict_ts = feats.index[i]
                        assert last_train_ts < predict_ts, (
                            f"LOOKAHEAD: GMM fit at i={i} used data up to "
                            f"{last_train_ts} which is NOT < {predict_ts}"
                        )
                except Exception:
                    # Singular covariance, EM convergence failure, etc.
                    # Skip this refit; keep using last good model if any.
                    pass

        # Predict bar i with the most recent valid model.
        if last_gmm is None or last_mapping is None:
            continue

        row = feats.iloc[i]
        if row.isna().any():
            # Don't classify on incomplete features → leave as -1.
            continue

        raw_cluster = int(last_gmm.predict(row.values.reshape(1, -1))[0])
        codes[i] = int(last_mapping[raw_cluster])

    return pd.Series(codes, index=htf_df.index, name="regime_code")


# ---------------------------------------------------------------------------
# Step 4 — point-in-time lookup at an arbitrary timestamp
# ---------------------------------------------------------------------------


def regime_at(regime_series: pd.Series, ts: pd.Timestamp) -> int:
    """Return the regime code valid AT timestamp `ts`.

    Uses the project-standard "last bar at-or-before" rule
    (`searchsorted(side="right") - 1`) — identical to labeler._bar_at_or_before.
    A zone forming intra-day on a lower TF will read the regime of the
    daily bar timestamped at the start of that day (project convention,
    matches how htf_trend and curve_score are sampled).

    Returns -1 if no daily bar exists at-or-before `ts`.
    """
    if len(regime_series) == 0:
        return -1
    pos = int(regime_series.index.searchsorted(ts, side="right")) - 1
    if pos < 0:
        return -1
    return int(regime_series.iloc[pos])


# ---------------------------------------------------------------------------
# Optional one-off audit hook
# ---------------------------------------------------------------------------


def verify_no_lookahead(htf_df: pd.DataFrame) -> None:
    """Re-run regime computation with verify=True. Raises AssertionError on
    the first violation. Intended for a one-off audit, not the hot path."""
    compute_regime_series(htf_df, verify=True)
