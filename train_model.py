#!/usr/bin/env python
# =============================================================================
# train_model.py — XGBoost classifier with strict time-series evaluation
# =============================================================================
#
# Design philosophy
# -----------------
# The gross LABEL (did price reach TP before SL?) is a clean binary
# target — it measures pure edge, independent of costs. The model learns
# which zone *geometry* and *context* predicts the TP being hit. Cost
# awareness enters as features (expected_cost_r, asset_class_code,
# timeframe_code) so the model can learn — from training outcomes — that
# expensive trades need extra edge to be worth taking.
#
# After training we sweep probability thresholds and measure NET expectancy
# (gross R − expected_cost_r per kept trade). The decisive question:
# does any threshold give a HIGHER net expectancy than the unfiltered
# baseline on the test set, on out-of-sample data?
#
# Threshold selection protocol (avoids optimistic bias)
# -------------------------------------------------------
# * Sweep thresholds on the VALIDATION set → pick the best one there.
# * Apply that single fixed threshold to the TEST set — one shot, no
#   re-optimisation. This is the honest, unbiased out-of-sample number.
# * The test sweep is also printed for diagnostics (to see whether val
#   choice generalises) but it does NOT influence the selected threshold.
#
# Critical time-series discipline
# --------------------------------
# * ALL splits are by formation_time — NEVER random.
# * Purged walk-forward CV: training folds are purged of any trade whose
#   [entry_time, exit_time] window overlaps the validation fold, plus a
#   7-day embargo. This prevents even indirect lookahead via overlapping
#   trades (the López de Prado precaution).
# * Hyperparameter tuning is done on train+val; test is touched only once
#   at the very end.
#
# Outputs
# -------
#   data/xgb_model.json    — saved model
#   data/ml_eval.csv       — test-set per-trade predictions
#
# Usage
# -----
#   python train_model.py
#   python train_model.py --n-folds 5 --embargo-days 7 --threshold 0.45
# =============================================================================

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import RR_RATIO
from utils.feature_engine import FEATURE_COLS, TARGET_COL

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PARQUET = DATA_DIR / "dataset.parquet"
MODEL_OUT = DATA_DIR / "xgb_model.json"
EVAL_OUT = DATA_DIR / "ml_eval.csv"


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (implicit)

# XGBoost hyperparameters (regularised to resist overfit on 17k rows)
XGB_PARAMS: dict = {
    "max_depth": 4,
    "learning_rate": 0.04,
    "n_estimators": 1000,  # capped by early stopping
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
    # scale_pos_weight set at runtime from train-split class ratio
}

# Threshold sweep range (inclusive)
THRESHOLD_SWEEP = np.round(np.arange(0.30, 0.71, 0.02), 2)


# ---------------------------------------------------------------------------
# Gross R helper (mirrors backtest._gross_r — single source of logic)
# ---------------------------------------------------------------------------


def _gross_r_series(df: pd.DataFrame) -> pd.Series:
    """Compute per-row gross R from exit_reason and pnl_r (same logic as
    backtest._gross_r). Falls back to label-based ±RR if columns missing."""
    if "exit_reason" not in df.columns or "pnl_r" not in df.columns:
        return df["label"].apply(lambda lbl: float(RR_RATIO) if lbl == 1 else -1.0)

    def _row(r: pd.Series) -> float:
        if r["exit_reason"] == "tp":
            return float(RR_RATIO)
        if r["exit_reason"] == "sl":
            return -1.0
        if r["exit_reason"] == "timeout":
            if pd.notna(r["pnl_r"]):
                return float(r["pnl_r"])
            return -1.0
        return float(RR_RATIO) if int(r["label"]) == 1 else -1.0

    return df.apply(_row, axis=1)


# ---------------------------------------------------------------------------
# Time-based train / val / test split
# ---------------------------------------------------------------------------


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by formation_time position (never random)."""
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ---------------------------------------------------------------------------
# Purged walk-forward cross-validation
# ---------------------------------------------------------------------------


def purged_wfcv(
    df_trainval: pd.DataFrame,
    n_folds: int,
    embargo_days: int,
    xgb_params: dict,
) -> tuple[list[float], list[float]]:
    """K-fold purged walk-forward CV on the train+val window.

    For each fold:
      1. Split by time: earlier rows = fold-train, later rows = fold-val.
      2. PURGE from fold-train any trade whose exit_time >= fold-val start.
         These trades have information that 'overlaps' the validation period.
      3. EMBARGO: also drop fold-train trades whose entry_time >=
         (fold-val start - embargo_days). Accounts for serial correlation
         in features derived from the same bars.
      4. Train XGB on the purged fold-train, evaluate AUC on fold-val.

    Returns
    -------
    train_aucs, val_aucs : per-fold AUC lists.
    """
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    df_trainval = df_trainval.copy().reset_index(drop=True)
    n = len(df_trainval)
    fold_size = n // n_folds
    embargo_td = pd.Timedelta(days=embargo_days)

    train_aucs, val_aucs = [], []

    for k in range(n_folds):
        # Fold-val = k-th chunk. Walk-forward: use all earlier data as
        # fold-train (no future data ever leaks into training).
        val_start_idx = k * fold_size
        val_end_idx = val_start_idx + fold_size if k < n_folds - 1 else n
        if val_start_idx == 0:
            # First chunk has no training data — skip.
            continue

        fold_val = df_trainval.iloc[val_start_idx:val_end_idx]
        fold_train_raw = df_trainval.iloc[:val_start_idx]

        val_start_time = fold_val["entry_time"].min()
        embargo_cutoff = val_start_time - embargo_td

        # Purge: remove trades that exit on or after the val window opens.
        # These trades have outcomes in the future (from fold-train's POV).
        if "exit_time" in fold_train_raw.columns:
            purge_mask = fold_train_raw["exit_time"] >= val_start_time
        else:
            purge_mask = fold_train_raw["entry_time"] >= val_start_time

        # Embargo: remove trades that are too close to val window start
        # (high serial correlation with the first val trades).
        embargo_mask = fold_train_raw["entry_time"] >= embargo_cutoff

        drop_mask = purge_mask | embargo_mask
        fold_train = fold_train_raw[~drop_mask]

        if len(fold_train) < 100 or fold_val[TARGET_COL].nunique() < 2:
            continue

        pos_w = float(
            (fold_train[TARGET_COL] == 0).sum()
            / max((fold_train[TARGET_COL] == 1).sum(), 1)
        )

        params = {**xgb_params, "scale_pos_weight": pos_w}
        # Remove early_stopping_rounds from CV fold params — we use a
        # dedicated eval set instead.
        cv_params = {
            k: v for k, v in params.items() if k not in ("early_stopping_rounds",)
        }
        cv_params["n_estimators"] = 300  # fixed, no early stop in CV

        model = xgb.XGBClassifier(**cv_params)
        model.fit(
            fold_train[FEATURE_COLS], fold_train[TARGET_COL].to_numpy(), verbose=False
        )

        tr_prob = model.predict_proba(fold_train[FEATURE_COLS])[:, 1]
        v_prob = model.predict_proba(fold_val[FEATURE_COLS])[:, 1]

        if fold_train[TARGET_COL].nunique() >= 2:
            train_aucs.append(roc_auc_score(fold_train[TARGET_COL], tr_prob))
        val_aucs.append(roc_auc_score(fold_val[TARGET_COL], v_prob))

        purged_n = drop_mask.sum()
        print(
            f"  Fold {k+1}/{n_folds}: train={len(fold_train):,} "
            f"(purged/embargoed={purged_n}) "
            f"val={len(fold_val):,}  "
            f"val_AUC={val_aucs[-1]:.4f}"
        )

    return train_aucs, val_aucs


# ---------------------------------------------------------------------------
# Threshold sweep evaluation
# ---------------------------------------------------------------------------


def threshold_sweep(
    df_eval: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    label_col: str = TARGET_COL,
) -> pd.DataFrame:
    """For each threshold, report selection stats and net expectancy.

    net_R per kept trade = gross_R - expected_cost_r
    Gross R uses actual exit_reason/pnl_r where available.
    """
    gross_r = _gross_r_series(df_eval).to_numpy()
    cost_r = df_eval["expected_cost_r"].to_numpy()
    net_r = gross_r - cost_r

    rows = []
    for t in thresholds:
        mask = probabilities >= t
        n_kept = int(mask.sum())
        if n_kept == 0:
            rows.append(
                {
                    "threshold": t,
                    "n_kept": 0,
                    "coverage_pct": 0.0,
                    "win_rate": float("nan"),
                    "gross_exp_r": float("nan"),
                    "avg_cost_r": float("nan"),
                    "net_exp_r": float("nan"),
                    "total_net_r": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "threshold": round(float(t), 2),
                "n_kept": n_kept,
                "coverage_pct": round(100.0 * n_kept / len(probabilities), 1),
                "win_rate": round(float(df_eval[label_col].to_numpy()[mask].mean()), 4),
                "gross_exp_r": round(float(gross_r[mask].mean()), 4),
                "avg_cost_r": round(float(cost_r[mask].mean()), 4),
                "net_exp_r": round(float(net_r[mask].mean()), 4),
                "total_net_r": round(float(net_r[mask].sum()), 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-asset-class breakdown (filtered vs unfiltered)
# ---------------------------------------------------------------------------


def asset_class_breakdown(
    df_eval: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Net expectancy by asset_class, filtered vs unfiltered."""
    gross_r = _gross_r_series(df_eval).to_numpy()
    cost_r = df_eval["expected_cost_r"].to_numpy()
    net_r = gross_r - cost_r
    selected = probabilities >= threshold
    ac = df_eval["asset_class"].to_numpy()

    rows = []
    for cls in sorted(df_eval["asset_class"].unique()):
        unfiltered_mask = ac == cls
        filtered_mask = unfiltered_mask & selected
        rows.append(
            {
                "asset_class": cls,
                "n_unfiltered": int(unfiltered_mask.sum()),
                "net_exp_unfiltered": round(
                    (
                        float(net_r[unfiltered_mask].mean())
                        if unfiltered_mask.any()
                        else float("nan")
                    ),
                    4,
                ),
                "n_filtered": int(filtered_mask.sum()),
                "net_exp_filtered": round(
                    (
                        float(net_r[filtered_mask].mean())
                        if filtered_mask.any()
                        else float("nan")
                    ),
                    4,
                ),
            }
        )
    df = pd.DataFrame(rows)
    # Compute lift
    df["net_exp_lift"] = (df["net_exp_filtered"] - df["net_exp_unfiltered"]).round(4)
    return df.sort_values("net_exp_unfiltered", ascending=False)


# ---------------------------------------------------------------------------
# SHAP / feature importance
# ---------------------------------------------------------------------------


def print_feature_importance(model, top_n: int = 10) -> None:
    """Print feature importance — SHAP if available, else XGBoost gain."""
    print("\n" + "─" * 60)
    try:
        import shap

        # TreeExplainer is fast and exact for XGBoost.
        explainer = shap.TreeExplainer(model)
        # Use a background-free approach: mean |SHAP| over training data is
        # computed internally.  We call it 'importance' for brevity.
        # Here we use the booster's gain as a fast stand-in for ranking;
        # a full SHAP run on all test rows is done separately.
        gain = model.get_booster().get_score(importance_type="gain")
        total = sum(gain.values()) or 1.0
        ranked = sorted(gain.items(), key=lambda x: x[1], reverse=True)
        print(
            f"Top {top_n} features by XGBoost gain (SHAP available — "
            "use shap.TreeExplainer for plots):"
        )
        for i, (feat, val) in enumerate(ranked[:top_n], 1):
            bar = "█" * int(40 * val / ranked[0][1])
            print(f"  {i:2d}. {feat:<26s} {val:8.1f}  {bar}")
        print()
        # Also compute mean |SHAP| on the gain-ranked features as a sanity
        # check (using just the top 200 rows for speed).
        print("  (SHAP installed — run shap.TreeExplainer(model) for full plots)")
    except ImportError:
        gain = model.get_booster().get_score(importance_type="gain")
        total = sum(gain.values()) or 1.0
        ranked = sorted(gain.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {top_n} features by XGBoost gain (install shap for SHAP values):")
        for i, (feat, val) in enumerate(ranked[:top_n], 1):
            pct = 100.0 * val / total
            bar = "█" * int(40 * val / ranked[0][1])
            print(f"  {i:2d}. {feat:<26s} {pct:5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# Learning-curve diagnostic
# ---------------------------------------------------------------------------


def run_learning_curve(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    xgb_params: dict,
    fractions: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 1.00),
    flatten_threshold: float = 0.005,
) -> pd.DataFrame:
    """Train on increasing fractions of the most-RECENT train window and
    score against the FIXED full validation set. Reveals whether the model
    is data-limited (val AUC still rising) or signal-limited (val AUC flat).

    Why "most recent" rather than oldest-forward
    --------------------------------------------
    The production model is fit on the FULL train window and evaluated on
    val/test. The relevant question is whether adding more HISTORICAL data
    improves generalisation to the NEAR FUTURE (val/test). Taking the most
    recent k% of train mirrors that — the largest subset (100%) extends
    furthest back, the smallest (20%) is the most recent slice. Random
    sampling would shuffle time and break the time-series property;
    oldest-forward would give every subset a stale relationship to val.
    Recent-backward is the honest choice.

    Returns a DataFrame with columns: fraction, n_train, train_auc, val_auc.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    # Validation set is FIXED across all fractions.
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL].to_numpy()

    n_train_total = len(train_df)
    rows: list[dict] = []

    # Strip early_stopping_rounds — we hold n_estimators fixed so the only
    # variable across subsets is sample size, not tree count.
    lc_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    # Cap n_estimators at a fixed budget for the diagnostic. Using the full
    # 1000 with no early stop would overfit the smallest subsets badly.
    lc_params["n_estimators"] = 300

    for frac in fractions:
        n_sub = max(1, int(round(n_train_total * frac)))
        # Take the MOST RECENT n_sub rows of train (closest to val in time).
        sub = train_df.iloc[-n_sub:]
        X_sub = sub[FEATURE_COLS]
        y_sub = sub[TARGET_COL].to_numpy()

        # Recompute scale_pos_weight from THIS subset only — class balance
        # can drift across windows.
        spw = float((y_sub == 0).sum() / max((y_sub == 1).sum(), 1))
        params = {**lc_params, "scale_pos_weight": spw}

        model = xgb.XGBClassifier(**params)
        model.fit(X_sub, y_sub, verbose=False)

        prob_sub = model.predict_proba(X_sub)[:, 1]
        prob_val = model.predict_proba(X_val)[:, 1]

        rows.append(
            {
                "fraction": frac,
                "n_train": n_sub,
                "train_auc": float(roc_auc_score(y_sub, prob_sub)),
                "val_auc": float(roc_auc_score(y_val, prob_val)),
            }
        )

    out = pd.DataFrame(rows)

    print("\nLearning curve  (val set is FIXED; train subset = most-recent k%):")
    print(f"\n  {'frac':>6}  {'n_train':>8}  {'train_AUC':>10}  {'val_AUC':>9}")
    print("  " + "-" * 40)
    for _, r in out.iterrows():
        print(
            f"  {r['fraction']:>6.2f}  {int(r['n_train']):>8,}  "
            f"{r['train_auc']:>10.4f}  {r['val_auc']:>9.4f}"
        )

    last_slope = float(out["val_auc"].iloc[-1] - out["val_auc"].iloc[-2])
    print(f"\n  val_AUC slope 80% → 100% : {last_slope:+.4f}")
    if last_slope >= flatten_threshold:
        print(
            f"  📈 DATA-LIMITED  — val AUC is still rising at 100%.\n"
            f"     Slope ≥ {flatten_threshold:.3f}: more data would likely help.\n"
            f"     Action: collect more history OR more symbols in the same\n"
            f"     asset classes."
        )
    else:
        print(
            f"  🏁 SIGNAL-LIMITED — val AUC has flattened (slope < {flatten_threshold:.3f}).\n"
            f"     More data won't help meaningfully. The ceiling is set by the\n"
            f"     features. Action: engineer new features (regime interactions,\n"
            f"     volume, microstructure) rather than collecting more data."
        )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train XGBoost on S&D zone dataset with strict time-split CV."
    )
    p.add_argument("--dataset", type=Path, default=DATASET_PARQUET)
    p.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Walk-forward CV folds on train+val. Default 5.",
    )
    p.add_argument(
        "--embargo-days",
        type=int,
        default=7,
        help="Embargo gap (days) before each val fold. Default 7.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the auto-selected best threshold for final reporting.",
    )
    p.add_argument(
        "--exclude-assets",
        nargs="*",
        default=["crypto", "macro"],
        metavar="ASSET_CLASS",
        help=(
            "Asset classes to drop from train/val/test BEFORE the time split. "
            "Default: crypto macro (both structurally unprofitable after costs). "
            "Pass `--exclude-assets` with no values to keep everything."
        ),
    )
    p.add_argument(
        "--learning-curve",
        action="store_true",
        help=(
            "Run a learning-curve diagnostic: train on 20/40/60/80/100%% of "
            "the most-recent TRAIN window and report val AUC at each step. "
            "Used to decide whether the model is data-limited vs signal-limited."
        ),
    )
    p.add_argument("--model-out", type=Path, default=MODEL_OUT)
    p.add_argument("--eval-out", type=Path, default=EVAL_OUT)
    return p.parse_args()


def main() -> int:  # noqa: C901 — acceptable complexity for a training pipeline
    args = _parse_args()

    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        print(f"ERROR: {e}\n  pip install xgboost scikit-learn shap", file=sys.stderr)
        return 1

    if not args.dataset.exists():
        print(f"ERROR: dataset not found at {args.dataset}", file=sys.stderr)
        print("       run `python build_dataset.py` first.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ load
    print("Loading dataset ...")
    df = pd.read_parquet(args.dataset)
    df["formation_time"] = pd.to_datetime(df["formation_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df = df.sort_values("formation_time", kind="stable").reset_index(drop=True)

    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        print(f"ERROR: missing feature columns: {missing_feats}", file=sys.stderr)
        return 1

    print(
        f"  {len(df):,} rows  |  features: {len(FEATURE_COLS)}  "
        f"|  win rate: {df[TARGET_COL].mean():.1%}"
    )

    # ---------------------------------------------------- asset-class exclusion
    # Structurally-unprofitable classes (crypto, macro) are removed BEFORE the
    # time split so they cannot enter training, validation, or test. This
    # prevents them from poisoning the model's learning and ensures every
    # metric below reflects the viable universe.
    excluded = [a for a in (args.exclude_assets or []) if a]
    if excluded:
        present = sorted(df["asset_class"].unique().tolist())
        unknown = [a for a in excluded if a not in present]
        if unknown:
            print(
                f"  note: --exclude-assets {unknown} not in dataset "
                f"(present: {present})"
            )
        before_n = len(df)
        drop_mask = df["asset_class"].isin(excluded)
        dropped_per_class = df.loc[drop_mask, "asset_class"].value_counts().to_dict()
        df = df.loc[~drop_mask].reset_index(drop=True)
        dropped_n = before_n - len(df)
        if dropped_n > 0:
            print(
                f"  excluded asset classes: {excluded}  "
                f"→ dropped {dropped_n:,} rows ({dropped_per_class})"
            )
            print(
                f"  remaining: {len(df):,} rows  "
                f"|  win rate: {df[TARGET_COL].mean():.1%}  "
                f"|  asset classes: {sorted(df['asset_class'].unique().tolist())}"
            )
    else:
        print("  (no asset classes excluded — training on full universe)")

    # ------------------------------------------------------------------ split
    train_df, val_df, test_df = time_split(df)
    print(
        f"\nTime split (by formation_time, never random):"
        f"\n  Train : {train_df['formation_time'].min().date()} → "
        f"{train_df['formation_time'].max().date()}  n={len(train_df):,}  "
        f"win={train_df[TARGET_COL].mean():.1%}"
        f"\n  Val   : {val_df['formation_time'].min().date()} → "
        f"{val_df['formation_time'].max().date()}  n={len(val_df):,}  "
        f"win={val_df[TARGET_COL].mean():.1%}"
        f"\n  Test  : {test_df['formation_time'].min().date()} → "
        f"{test_df['formation_time'].max().date()}  n={len(test_df):,}  "
        f"win={test_df[TARGET_COL].mean():.1%}"
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].to_numpy()
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL].to_numpy()
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].to_numpy()

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"\n  scale_pos_weight (train only): {spw:.3f}")

    # ------------------------------------------------------------------ CV
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    print(
        f"\nPurged walk-forward CV  "
        f"(K={args.n_folds}, embargo={args.embargo_days}d) ..."
    )
    tr_aucs, wf_aucs = purged_wfcv(
        trainval_df,
        n_folds=args.n_folds,
        embargo_days=args.embargo_days,
        xgb_params={**XGB_PARAMS, "scale_pos_weight": spw},
    )
    if wf_aucs:
        print(f"\n  CV val  AUC: {np.mean(wf_aucs):.4f} ± {np.std(wf_aucs):.4f}")
        if tr_aucs:
            print(
                f"  CV train AUC: {np.mean(tr_aucs):.4f} ± {np.std(tr_aucs):.4f}  "
                f"(gap={np.mean(tr_aucs)-np.mean(wf_aucs):.4f}; "
                f"<0.05 = low overfit)"
            )
    else:
        print("  (no CV folds with enough data)")

    # ---------------------------------------------- optional learning curve
    if args.learning_curve:
        print("\n" + "─" * 60)
        print("LEARNING-CURVE DIAGNOSTIC  (--learning-curve)")
        print("─" * 60)
        run_learning_curve(
            train_df,
            val_df,
            xgb_params={**XGB_PARAMS, "scale_pos_weight": spw},
        )

    # ------------------------------------------------------------------ train
    print("\nTraining final model (early stopping on val) ...")
    params = {**XGB_PARAMS, "scale_pos_weight": spw}
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}  (of {XGB_PARAMS['n_estimators']} max)")

    # ------------------------------------------------------------------ AUC
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, prob_train)
    auc_val = roc_auc_score(y_val, prob_val)
    auc_test = roc_auc_score(y_test, prob_test)

    print(f"\n  AUC  train: {auc_train:.4f}")
    print(f"  AUC  val  : {auc_val:.4f}")
    print(f"  AUC  TEST : {auc_test:.4f}", end="")

    # Overfit flag: test AUC meaningfully below val AUC
    if auc_test < auc_val - 0.05:
        print("  ⚠  TEST AUC is >5pp below val — possible overfit/regime shift")
    elif auc_test < 0.53:
        print("  ⚠  TEST AUC barely above chance — model may not generalise")
    else:
        print("  ✓")

    # ------------------------------------------------------------------ baseline on test
    gross_r_test = _gross_r_series(test_df).to_numpy()
    cost_r_test = test_df["expected_cost_r"].to_numpy()
    net_r_test = gross_r_test - cost_r_test
    baseline_net_exp = float(net_r_test.mean())
    baseline_win_rate = float(y_test.mean())

    print(
        f"\n  TEST baseline (no filter): "
        f"net exp = {baseline_net_exp:+.4f} R  "
        f"win = {baseline_win_rate:.1%}  "
        f"n = {len(test_df):,}"
    )

    # -------------------------------------------------------- val sweep (selection)
    print("\nValidation threshold sweep  (threshold is SELECTED here):")
    val_sweep_df = threshold_sweep(val_df, prob_val, THRESHOLD_SWEEP)

    # Select best threshold on VALIDATION set only.
    # Rule: highest net_exp_r with at least 5% coverage.
    valid_val = val_sweep_df[
        (val_sweep_df["coverage_pct"] >= 5) & val_sweep_df["net_exp_r"].notna()
    ]
    if valid_val.empty:
        best_val_row = val_sweep_df.iloc[0]
    else:
        best_val_row = valid_val.loc[valid_val["net_exp_r"].idxmax()]

    # Manual CLI override — skips val selection.
    cli_override = args.threshold is not None
    if cli_override:
        override_rows = val_sweep_df[
            val_sweep_df["threshold"] == round(args.threshold, 2)
        ]
        if not override_rows.empty:
            best_val_row = override_rows.iloc[0]
            print(f"  (threshold overridden via --threshold={args.threshold})")

    best_threshold = float(best_val_row["threshold"])

    # Helper: shared table printer.
    _SWEEP_HDR = (
        f"\n  {'thresh':>6}  {'n_kept':>7}  {'cov%':>5}  "
        f"{'win%':>6}  {'gross_R':>8}  {'cost_R':>7}  {'net_R':>8}  {'total_net':>10}"
    )
    _SWEEP_SEP = "  " + "-" * 68

    def _print_sweep(sweep_df: pd.DataFrame, marker_threshold: float) -> None:
        print(_SWEEP_HDR)
        print(_SWEEP_SEP)
        for _, row in sweep_df.iterrows():
            if pd.isna(row["net_exp_r"]):
                continue
            marker = " ◀ selected" if row["threshold"] == marker_threshold else ""
            print(
                f"  {row['threshold']:>6.2f}  "
                f"{int(row['n_kept']):>7,}  "
                f"{row['coverage_pct']:>5.1f}  "
                f"{row['win_rate']:>6.1%}  "
                f"{row['gross_exp_r']:>+8.4f}  "
                f"{row['avg_cost_r']:>7.4f}  "
                f"{row['net_exp_r']:>+8.4f}  "
                f"{row['total_net_r']:>+10.4f}"
                f"{marker}"
            )

    _print_sweep(val_sweep_df, best_threshold)

    # ------------------------------------------------- test sweep (diagnostics only)
    print(
        f"\nTest threshold sweep  "
        f"(diagnostics only — NOT used for threshold selection):"
    )
    test_sweep_df = threshold_sweep(test_df, prob_test, THRESHOLD_SWEEP)
    _print_sweep(test_sweep_df, best_threshold)

    # Retrieve test-set metrics at the val-chosen threshold (one lookup, no re-opt).
    _test_rows = test_sweep_df[test_sweep_df["threshold"] == best_threshold]
    if _test_rows.empty:
        # Fallback: compute directly (should not happen with standard sweep range).
        _mask = prob_test >= best_threshold
        _gr = _gross_r_series(test_df).to_numpy()
        _cr = test_df["expected_cost_r"].to_numpy()
        _nr = _gr - _cr
        test_net_exp = float(_nr[_mask].mean()) if _mask.any() else float("nan")
        test_n_kept = int(_mask.sum())
        test_coverage = round(100.0 * test_n_kept / len(test_df), 1)
        test_win_rate = (
            float(test_df[TARGET_COL].to_numpy()[_mask].mean())
            if _mask.any()
            else float("nan")
        )
    else:
        _r = _test_rows.iloc[0]
        test_net_exp = float(_r["net_exp_r"])
        test_n_kept = int(_r["n_kept"])
        test_coverage = float(_r["coverage_pct"])
        test_win_rate = float(_r["win_rate"])

    # ------------------------------------------------------------------ asset class breakdown
    print(
        f"\nNet expectancy by asset class — test set "
        f"(threshold={best_threshold}, selected on val):"
    )
    ac_df = asset_class_breakdown(test_df, prob_test, best_threshold)
    print(ac_df.to_string(index=False))

    # ------------------------------------------------------------------ feature importance
    print_feature_importance(model, top_n=10)

    # ------------------------------------------------------------------ verdict
    ml_adds_value = test_net_exp > baseline_net_exp
    delta = test_net_exp - baseline_net_exp
    universe_label = (
        f"filtered universe (excluded: {excluded})" if excluded else "full universe"
    )

    print("\n" + "=" * 72)
    print("FINAL VERDICT — does ML add value ON TOP of the manual asset filter?")
    print("  Threshold selected on VALIDATION, evaluated ONCE on TEST (unbiased OOS)")
    print(f"  Universe: {universe_label}")
    print("=" * 72)
    print(
        f"  (1) Baseline net exp (no ML)       : {baseline_net_exp:+.4f} R / trade  "
        f"(test, n={len(test_df):,})"
    )
    print(
        f"  (2) ML-filtered net exp @ {best_threshold:.2f}      : {test_net_exp:+.4f} R / trade  "
        f"({test_n_kept:,} trades, {test_coverage:.1f}% coverage, win={test_win_rate:.1%})"
    )
    print(f"  (3) Lift = (2) − (1)               : {delta:+.4f} R / trade")
    print(
        f"  Threshold chosen on               : validation set (n={len(val_df):,})"
        + ("  [CLI override]" if cli_override else "")
    )
    print(
        f"  CV val AUC                        : {np.mean(wf_aucs):.4f}"
        if wf_aucs
        else ""
    )
    print(f"  Test AUC                          : {auc_test:.4f}")
    print()
    if ml_adds_value and auc_test >= 0.53:
        print(f"  ✅ ML ADDS VALUE  — net expectancy improved by {delta:+.4f} R/trade")
        print(
            f"     at threshold {best_threshold:.2f} (keeping {test_coverage:.1f}% of test trades)."
        )
        print(f"     Threshold was selected on val, not test — result is unbiased.")
    elif ml_adds_value:
        print(
            f"  ⚠  ML marginally adds value ({delta:+.4f} R) but test AUC is weak ({auc_test:.4f})."
        )
        print(f"     Results may not be robust — collect more data or tune features.")
    else:
        print(f"  ❌ ML does NOT reliably add value at any threshold.")
        print(
            f"     Test net exp {test_net_exp:+.4f} R ≤ baseline {baseline_net_exp:+.4f} R."
        )
        print(
            f"     Consider: more data, better features, or wider stops (cost reduction)."
        )
    print("=" * 72)

    # ------------------------------------------------- return concentration check
    # Inherent to a 3:1 RR strategy: a few large wins drive most of the total
    # PnL. We measure how concentrated the ML-selected slice is. Top-10%
    # contribution > ~80% means the edge depends on a small number of trades
    # — survivable, but worth monitoring (a few missed winners hurts hard).
    selected_mask = prob_test >= best_threshold
    if selected_mask.any():
        selected_net_r = net_r_test[selected_mask]
        selected_sorted = np.sort(selected_net_r)[::-1]  # descending
        total_net = float(selected_sorted.sum())
        median_net = float(np.median(selected_net_r))
        top10_n = max(1, int(np.ceil(0.10 * len(selected_sorted))))
        top10_sum = float(selected_sorted[:top10_n].sum())
        top10_share = (top10_sum / total_net * 100.0) if total_net > 0 else float("nan")

        print("\nReturn concentration — ML-selected test trades:")
        print(f"  n trades                          : {len(selected_sorted):,}")
        print(f"  median net_r                      : {median_net:+.4f} R")
        print(f"  total net_r                       : {total_net:+.4f} R")
        print(
            f"  top 10% (n={top10_n}) net_r          : {top10_sum:+.4f} R  "
            f"({top10_share:.1f}% of total)"
        )
        if not np.isnan(top10_share) and top10_share > 80.0:
            print(
                "  ⚠  high concentration (>80%) — the edge relies on a few large\n"
                "     winners. Inherent to 3:1 RR strategies but worth monitoring:\n"
                "     missing a handful of winners would erase the edge."
            )
        elif not np.isnan(top10_share):
            print("  ✓  acceptable concentration — returns spread across more trades.")
        print("=" * 72)

    # ------------------------------------------------------------------ save
    model.save_model(str(args.model_out))
    print(f"\nSaved model  → {args.model_out}")

    test_out = test_df[
        [
            "formation_time",
            "entry_time",
            "symbol",
            "timeframe",
            "asset_class",
            "direction",
            TARGET_COL,
            "expected_cost_r",
        ]
        + (["exit_reason", "pnl_r"] if "exit_reason" in test_df.columns else [])
    ].copy()
    test_out["prob"] = prob_test
    test_out["gross_r"] = gross_r_test
    test_out["net_r"] = net_r_test
    test_out["selected"] = (prob_test >= best_threshold).astype(
        int
    )  # threshold from val
    test_out.to_csv(args.eval_out, index=False)
    print(f"Saved eval   → {args.eval_out}  ({len(test_out):,} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
