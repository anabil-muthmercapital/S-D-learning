#!/usr/bin/env python
# =============================================================================
# backtest.py — Honest portfolio backtest of the RAW (pre-ML) strategy
# =============================================================================
#
# Purpose
# -------
# Answer a single question: does the raw S&D zone strategy make money AFTER
# realistic transaction costs and capital constraints — BEFORE any ML
# filtering? If gross expectancy survives costs we have a tradeable edge to
# improve; if not, ML must produce enough lift on its own.
#
# Why this script is conservative on purpose
# ------------------------------------------
# * Same-bar SL/TP ambiguity → SL wins (already enforced in labeler).
# * Concurrency cap = real capital constraints: while N trades are open, a
#   new trade is dropped instead of opening a phantom (N+1)th position.
# * Position size is fixed 1% of CURRENT equity (compounded), not of
#   initial — losing streaks shrink size, winning streaks grow it.
# * Costs are charged on BOTH legs, scaled per asset class, expressed as a
#   fraction of entry price → converted to R via the trade's risk_per_unit.
# * Timeouts use the actual `pnl_r` recorded by the labeler (which may be
#   positive or negative). If `pnl_r` is missing, timeout defaults to −1R.
# * Strict time-ordering: trades are processed by `entry_time`, never
#   shuffled.
#
# What this script CANNOT see
# ---------------------------
# * Funding rates (perp crypto) and overnight financing (CFDs/futures).
# * Borrow fees on shorts (irrelevant for label=long trades but matters
#   for the supply→short half of the book).
# * Order-book impact above ~0.1% notional — for the modeled sizes (1%
#   risk × ~5x leverage equivalents) this is usually negligible on liquid
#   instruments but NOT on small-cap or low-volume crypto.
# Tune `cost_multiplier` to stress-test these unknowns.
# =============================================================================

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import RR_RATIO
from utils import costs
from utils.costs import (
    expected_cost_r as _cost_r_fn,
)  # noqa: F401 (re-exported for tests)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PARQUET = DATA_DIR / "dataset.parquet"
OUT_LEDGER = DATA_DIR / "backtest_trades.csv"

# Cost model lives in utils/costs.py — single source of truth shared with
# feature_engine.py. Imported here for convenience; do NOT redefine locally.
COST_MODEL = costs.COST_MODEL


# ---------------------------------------------------------------------------
# Open-trade bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class OpenTrade:
    """Minimal record of a currently-open trade — only needs exit_time so
    we can free the concurrency slot at the right moment."""

    exit_time: pd.Timestamp
    idx: int  # row index in the source dataset (for debugging only)


def _release_expired(open_trades: list[OpenTrade], now: pd.Timestamp) -> None:
    """Remove from `open_trades` every trade whose exit_time <= now.

    A trade that exits AT time T is assumed to have freed its slot BEFORE
    the next trade timestamped T+epsilon, so we use <= (inclusive). This is
    a slight optimism vs. strict < but matches how a real fill engine
    handles same-second events. The bias is tiny in practice (the dataset
    is bar-level so simultaneous opens/closes are rare).
    """
    if not open_trades:
        return
    open_trades[:] = [t for t in open_trades if t.exit_time > now]


# ---------------------------------------------------------------------------
# Per-trade gross R computation
# ---------------------------------------------------------------------------


def _gross_r(row: pd.Series) -> float:
    """R-multiple BEFORE costs, honoring labeler outcomes.

    Logic
    -----
    * TP hit              → +RR_RATIO (e.g. +3R).
    * SL hit              → −1R.
    * Timeout with pnl_r  → that pnl_r (can be positive or negative).
    * Anything missing    → fall back to label-based ±1R/+RR.
    """
    exit_reason = row.get("exit_reason", "")
    pnl_r = row.get("pnl_r", np.nan)

    if exit_reason == "tp":
        return float(RR_RATIO)
    if exit_reason == "sl":
        return -1.0
    if exit_reason == "timeout":
        # pnl_r at the close of the timeout bar — already a signed R-multiple.
        if pd.notna(pnl_r):
            return float(pnl_r)
        return -1.0  # conservative default

    # Fallback: trust the binary label (1 = win at +RR, 0 = loss at −1).
    return float(RR_RATIO) if int(row["label"]) == 1 else -1.0


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float,
    risk_fraction: float,
    max_concurrent: int,
    cost_multiplier: float,
) -> tuple[pd.DataFrame, dict]:
    """Event-driven walk over labelled trades.

    Returns
    -------
    ledger  : per-trade DataFrame with gross/cost/net R, equity_after, …
    summary : dict of aggregate metrics for printing.
    """
    # ---- prep -----------------------------------------------------------
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    # Strict time-ordering for deterministic simulation. Ties broken by
    # the original row index — stable sort guarantees reproducibility.
    df = df.sort_values("entry_time", kind="stable").reset_index(drop=True)

    equity = float(initial_capital)
    open_trades: list[OpenTrade] = []
    ledger_rows: list[dict] = []
    skipped_concurrency = 0

    for i, row in df.iterrows():
        now = row["entry_time"]

        # 1) free slots for trades that already exited by `now`.
        _release_expired(open_trades, now)

        # 2) concurrency cap → drop this trade if the book is full.
        if len(open_trades) >= max_concurrent:
            skipped_concurrency += 1
            continue

        # 3) compute gross R (label/pnl_r driven).
        gross_r = _gross_r(row)

        # 4) cost in R. risk_per_unit_price is just `risk` (price units),
        #    cost_per_unit_price = entry_price × round_trip_cost_frac.
        entry_price = float(row["entry"])
        risk_price = float(row["risk"])
        if risk_price <= 0 or entry_price <= 0:
            # Degenerate row — skip to avoid div-by-zero, count as no-trade.
            continue

        per_side = costs.per_side_cost_frac(
            row["asset_class"], row["symbol"], entry_price
        )
        round_trip_cost_frac = 2.0 * per_side * cost_multiplier
        cost_price = entry_price * round_trip_cost_frac
        cost_r = cost_price / risk_price

        net_r = gross_r - cost_r

        # 5) money P&L = risk_amount × net_r. risk_amount is 1% (default)
        #    of CURRENT equity — compounded sizing.
        risk_amount = equity * risk_fraction
        money_pnl = risk_amount * net_r
        equity_after = equity + money_pnl

        # 6) record + open the slot.
        # exit_time can be NaT in the (rare) audit-edge case where bars_held=0.
        # In that case, free the slot immediately by using `now`.
        exit_t = row["exit_time"] if pd.notna(row["exit_time"]) else now
        open_trades.append(OpenTrade(exit_time=exit_t, idx=int(i)))
        equity = equity_after

        ledger_rows.append(
            {
                "entry_time": now,
                "exit_time": exit_t,
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "asset_class": row["asset_class"],
                "direction": row["direction"],
                "exit_reason": row.get("exit_reason", ""),
                "label": int(row["label"]),
                "gross_r": round(gross_r, 4),
                "cost_r": round(cost_r, 4),
                "net_r": round(net_r, 4),
                "risk_amount": round(risk_amount, 2),
                "money_pnl": round(money_pnl, 2),
                "equity_after": round(equity_after, 2),
                "open_slots_in_use": len(open_trades),
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    summary = _summarize(
        ledger=ledger,
        initial_capital=initial_capital,
        total_input_trades=len(df),
        skipped_concurrency=skipped_concurrency,
        cost_multiplier=cost_multiplier,
    )
    return ledger, summary


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """Peak-to-trough drawdown of an equity series, as a positive percent."""
    if equity_curve.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(-drawdowns.min() * 100.0)


def _sharpe_like(returns_r: np.ndarray) -> float:
    """Sharpe-like ratio on per-trade R-returns (not annualised).

    Annualising requires a meaningful "trades per year" denominator which
    depends on regime + concurrency cap. We report the raw mean/std ratio
    so the value is comparable across runs at fixed parameters.
    """
    if returns_r.size < 2 or returns_r.std(ddof=1) == 0:
        return 0.0
    return float(returns_r.mean() / returns_r.std(ddof=1))


def _summarize(
    ledger: pd.DataFrame,
    initial_capital: float,
    total_input_trades: int,
    skipped_concurrency: int,
    cost_multiplier: float,
) -> dict:
    """Aggregate metrics from the per-trade ledger."""
    if ledger.empty:
        return {
            "trades_taken": 0,
            "trades_skipped_concurrency": skipped_concurrency,
            "total_input_trades": total_input_trades,
            "win_rate": 0.0,
            "gross_expectancy_r": 0.0,
            "net_expectancy_r": 0.0,
            "avg_cost_r": 0.0,
            "total_net_r": 0.0,
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown_pct": 0.0,
            "sharpe_like": 0.0,
            "by_asset_class": pd.DataFrame(),
            "by_timeframe": pd.DataFrame(),
            "cost_multiplier": cost_multiplier,
            "verdict_profitable": False,
        }

    gross_r = ledger["gross_r"].to_numpy()
    cost_r = ledger["cost_r"].to_numpy()
    net_r = ledger["net_r"].to_numpy()
    money_pnl = ledger["money_pnl"].to_numpy()

    wins_mask = net_r > 0
    win_rate = float(wins_mask.mean())

    gross_exp = float(gross_r.mean())
    net_exp = float(net_r.mean())
    avg_cost = float(cost_r.mean())
    total_net = float(net_r.sum())

    final_eq = float(ledger["equity_after"].iloc[-1])
    total_return_pct = (final_eq - initial_capital) / initial_capital * 100.0

    # Profit factor in MONEY terms (the metric that matters for capital).
    gross_profit = float(money_pnl[money_pnl > 0].sum())
    gross_loss = float(-money_pnl[money_pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Drawdown computed on the post-trade equity curve (snapshots after
    # every fill). Doesn't capture intra-trade unrealised drawdown.
    equity_curve = np.concatenate(
        [[initial_capital], ledger["equity_after"].to_numpy()]
    )
    mdd = _max_drawdown_pct(equity_curve)

    sharpe = _sharpe_like(net_r)

    # Per-bucket expectancy + cost: shows which slices SURVIVE costs.
    by_asset = (
        ledger.groupby("asset_class")
        .agg(
            n=("net_r", "size"),
            win_rate=("net_r", lambda s: float((s > 0).mean())),
            gross_exp_r=("gross_r", "mean"),
            cost_r=("cost_r", "mean"),
            net_exp_r=("net_r", "mean"),
            total_net_r=("net_r", "sum"),
        )
        .round(4)
        .sort_values("net_exp_r", ascending=False)
    )
    by_tf = (
        ledger.groupby("timeframe")
        .agg(
            n=("net_r", "size"),
            win_rate=("net_r", lambda s: float((s > 0).mean())),
            gross_exp_r=("gross_r", "mean"),
            cost_r=("cost_r", "mean"),
            net_exp_r=("net_r", "mean"),
            total_net_r=("net_r", "sum"),
        )
        .round(4)
        .sort_values("net_exp_r", ascending=False)
    )

    return {
        "trades_taken": int(len(ledger)),
        "trades_skipped_concurrency": int(skipped_concurrency),
        "total_input_trades": int(total_input_trades),
        "win_rate": win_rate,
        "gross_expectancy_r": gross_exp,
        "net_expectancy_r": net_exp,
        "avg_cost_r": avg_cost,
        "total_net_r": total_net,
        "final_equity": final_eq,
        "total_return_pct": total_return_pct,
        "profit_factor": profit_factor,
        "max_drawdown_pct": mdd,
        "sharpe_like": sharpe,
        "by_asset_class": by_asset,
        "by_timeframe": by_tf,
        "cost_multiplier": cost_multiplier,
        "verdict_profitable": net_exp > 0,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _print_report(s: dict, initial_capital: float) -> None:
    print("\n" + "=" * 72)
    print("BACKTEST REPORT — raw strategy, post-cost")
    print("=" * 72)
    print(f"Cost multiplier        : ×{s['cost_multiplier']:.2f}")
    print(f"Input trades (dataset) : {s['total_input_trades']:,}")
    print(f"Trades taken           : {s['trades_taken']:,}")
    print(f"Skipped (concurrency)  : {s['trades_skipped_concurrency']:,}")
    print(f"Win rate (net > 0)     : {s['win_rate']:.2%}")
    print()
    print(f"Gross expectancy / trade : {s['gross_expectancy_r']:+.4f} R")
    print(f"Net expectancy   / trade : {s['net_expectancy_r']:+.4f} R")
    print(
        f"Avg cost         / trade : {s['avg_cost_r']:.4f} R  "
        f"(eats {abs(s['avg_cost_r'] / s['gross_expectancy_r']) * 100 if s['gross_expectancy_r'] else 0:.1f}% of gross)"
    )
    print()
    print(f"Total net R           : {s['total_net_r']:+.2f}")
    print(f"Starting equity       : ${initial_capital:,.2f}")
    print(f"Final equity          : ${s['final_equity']:,.2f}")
    print(f"Total return          : {s['total_return_pct']:+.2f}%")
    print(f"Profit factor (money) : {s['profit_factor']:.3f}")
    print(f"Max drawdown          : {s['max_drawdown_pct']:.2f}%")
    print(f"Sharpe-like (per-trade): {s['sharpe_like']:+.4f}")
    print()
    if not s["by_asset_class"].empty:
        print("─" * 72)
        print("Net expectancy by ASSET CLASS")
        print("─" * 72)
        print(s["by_asset_class"].to_string())
        print()
    if not s["by_timeframe"].empty:
        print("─" * 72)
        print("Net expectancy by TIMEFRAME")
        print("─" * 72)
        print(s["by_timeframe"].to_string())
        print()

    print("=" * 72)
    verdict = "PROFITABLE" if s["verdict_profitable"] else "NOT PROFITABLE"
    arrow = "✅" if s["verdict_profitable"] else "❌"
    print(
        f"{arrow} VERDICT — raw strategy BEFORE any ML filtering: {verdict}\n"
        f"   Net expectancy after costs = {s['net_expectancy_r']:+.4f} R / trade "
        f"({s['win_rate']:.1%} win rate, ×{s['cost_multiplier']:.2f} costs)."
    )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Out-of-sample validation
# ---------------------------------------------------------------------------

# Default split date: approximately the oldest 70% of the 2024-06-13 →
# 2026-06-13 dataset window (24 months × 0.70 ≈ 16.8 months → 2025-10-01).
DEFAULT_SPLIT_DATE: str = "2025-10-01"

# Default cost multiplier used ONLY for the train-period asset-class
# selection step. Set to 1.5 so only classes with a margin-of-safety edge
# are selected — avoids picking marginal classes that collapse at ×1.5.
DEFAULT_SELECTION_COST_MULT: float = 1.5


def _net_exp_by_class(ledger: pd.DataFrame) -> pd.Series:
    """Per-asset-class mean net R from a backtest ledger."""
    if ledger.empty:
        return pd.Series(dtype=float)
    return ledger.groupby("asset_class")["net_r"].mean().sort_values(ascending=False)


def run_oos(
    df: pd.DataFrame,
    split_date: str,
    selection_cost_mult: float,
    capital: float,
    risk_fraction: float,
    max_concurrent: int,
    oos_cost_mults: list[float],
) -> None:
    """Time-split out-of-sample validation.

    Selection protocol (anti-selection-bias)
    ----------------------------------------
    1. Split the dataset by ``formation_time`` at ``split_date``.
    2. Run the backtest cost model on the TRAIN split ONLY at
       ``selection_cost_mult`` (default ×1.5 = stress level).
    3. Identify which asset classes have positive train-period net expectancy.
    4. Evaluate those same asset classes on the TEST split (never seen during
       selection) at each multiplier in ``oos_cost_mults``.
    5. Show selected classes on train, rejected classes on test, and a
       per-timeframe breakdown within selected classes on test.
    """
    # ------------------------------------------------------------------ split
    df = df.copy()
    df["formation_time"] = pd.to_datetime(df["formation_time"], utc=True)
    cutoff = pd.Timestamp(split_date, tz="UTC")

    train_df = df[df["formation_time"] < cutoff].copy()
    test_df = df[df["formation_time"] >= cutoff].copy()

    if train_df.empty or test_df.empty:
        print(
            f"ERROR: split at {split_date} produced an empty train or test set.",
            file=sys.stderr,
        )
        return

    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 72)
    print(f"Split date         : {split_date}")
    print(
        f"Train period       : {train_df['formation_time'].min().date()}  →  "
        f"{train_df['formation_time'].max().date()}  ({len(train_df):,} trades)"
    )
    print(
        f"Test  period       : {test_df['formation_time'].min().date()}  →  "
        f"{test_df['formation_time'].max().date()}  ({len(test_df):,} trades)"
    )

    # ----------------------------------------------- STEP 1: train selection
    # CRITICAL: asset-class selection uses ONLY train-period data.
    # The test period plays NO role in choosing which classes to keep.
    print(
        f"\nSelecting asset classes using TRAIN period only "
        f"(cost_mult=×{selection_cost_mult:.1f}) ..."
    )
    train_ledger, _ = run_backtest(
        train_df,
        initial_capital=capital,
        risk_fraction=risk_fraction,
        max_concurrent=max_concurrent,
        cost_multiplier=selection_cost_mult,
    )
    train_exp = _net_exp_by_class(train_ledger)

    selected: list[str] = sorted(train_exp[train_exp > 0].index.tolist())
    rejected: list[str] = sorted(train_exp[train_exp <= 0].index.tolist())

    print()
    print("SELECTION USED TRAIN PERIOD ONLY:")
    print(
        f"  Train-period net expectancy by asset class (×{selection_cost_mult:.1f} costs):"
    )
    for cls, exp in train_exp.items():
        tag = "  SELECTED ✓" if cls in selected else "  rejected"
        print(f"    {cls:<14s}  {exp:+.4f} R{tag}")
    print()
    print(f"  → Selected  : {selected}")
    print(f"  → Rejected  : {rejected}")

    if not selected:
        print("\nNo asset classes survived train-period selection. Nothing to test.")
        return

    # ----------------------------- STEP 2: evaluate on test at each cost mult
    all_classes = selected + rejected
    test_selected = test_df[test_df["asset_class"].isin(selected)].copy()
    test_rejected = (
        test_df[test_df["asset_class"].isin(rejected)].copy()
        if rejected
        else pd.DataFrame()
    )
    train_selected = train_df[train_df["asset_class"].isin(selected)].copy()

    for cost_mult in oos_cost_mults:
        print("\n" + "─" * 72)
        print(f"COST MULTIPLIER ×{cost_mult:.1f}")
        print("─" * 72)

        # (A) Selected classes — TEST period (THE honest OOS result)
        oos_ledger, oos_sum = run_backtest(
            test_selected,
            initial_capital=capital,
            risk_fraction=risk_fraction,
            max_concurrent=max_concurrent,
            cost_multiplier=cost_mult,
        )
        _print_oos_section(
            oos_sum,
            oos_ledger,
            capital,
            label=f"[A] SELECTED classes on TEST period (OOS) — ×{cost_mult:.1f}",
            is_key_result=True,
        )

        # (B) Selected classes — TRAIN period (reference / sanity check)
        train_sel_ledger, train_sel_sum = run_backtest(
            train_selected,
            initial_capital=capital,
            risk_fraction=risk_fraction,
            max_concurrent=max_concurrent,
            cost_multiplier=cost_mult,
        )
        _print_oos_section(
            train_sel_sum,
            train_sel_ledger,
            capital,
            label=f"[B] SELECTED classes on TRAIN period (reference) — ×{cost_mult:.1f}",
            is_key_result=False,
        )

        # (C) Rejected classes — TEST period (sanity check: should be worse)
        if not test_rejected.empty:
            rej_ledger, rej_sum = run_backtest(
                test_rejected,
                initial_capital=capital,
                risk_fraction=risk_fraction,
                max_concurrent=max_concurrent,
                cost_multiplier=cost_mult,
            )
            _print_oos_section(
                rej_sum,
                rej_ledger,
                capital,
                label=f"[C] REJECTED classes on TEST period (sanity check) — ×{cost_mult:.1f}",
                is_key_result=False,
            )

        # (D) Per-timeframe within selected assets — TEST only
        if not oos_ledger.empty:
            print(
                f"\n  Timeframe breakdown within SELECTED classes, TEST period, ×{cost_mult:.1f}:"
            )
            tf_tbl = (
                oos_ledger.groupby("timeframe")
                .agg(
                    n=("net_r", "size"),
                    win_rate=("net_r", lambda s: float((s > 0).mean())),
                    gross_exp_r=("gross_r", "mean"),
                    cost_r=("cost_r", "mean"),
                    net_exp_r=("net_r", "mean"),
                    total_net_r=("net_r", "sum"),
                )
                .round(4)
                .sort_values("net_exp_r", ascending=False)
            )
            print(tf_tbl.to_string())

    # ---------------------------------------------------------- final verdict
    print()
    print("=" * 72)
    print("FINAL VERDICT — Out-of-Sample")
    print("=" * 72)
    for cost_mult in oos_cost_mults:
        oos_ledger, oos_sum = run_backtest(
            test_selected,
            initial_capital=capital,
            risk_fraction=risk_fraction,
            max_concurrent=max_concurrent,
            cost_multiplier=cost_mult,
        )
        profitable = oos_sum["net_expectancy_r"] > 0
        arrow = "✅" if profitable else "❌"
        print(
            f"  {arrow}  ×{cost_mult:.1f} costs → OOS net expectancy "
            f"{oos_sum['net_expectancy_r']:+.4f} R / trade  "
            f"({'PROFITABLE' if profitable else 'NOT PROFITABLE'})  "
            f"win={oos_sum['win_rate']:.1%}  mdd={oos_sum['max_drawdown_pct']:.1f}%"
        )
    print("=" * 72)


def _print_oos_section(
    s: dict,
    ledger: pd.DataFrame,
    initial_capital: float,
    label: str,
    is_key_result: bool,
) -> None:
    """Compact section printer for one OOS slice."""
    sep = "━" * 72 if is_key_result else "·" * 72
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    if s["trades_taken"] == 0:
        print("  (no trades in this slice)")
        return
    print(
        f"  Trades taken : {s['trades_taken']:,}  "
        f"(skipped {s['trades_skipped_concurrency']:,} by concurrency cap)"
    )
    print(f"  Win rate     : {s['win_rate']:.2%}")
    print(f"  Gross exp    : {s['gross_expectancy_r']:+.4f} R")
    print(f"  Avg cost     : {s['avg_cost_r']:.4f} R")
    print(
        f"  Net exp      : {s['net_expectancy_r']:+.4f} R  "
        f"{'← KEY RESULT' if is_key_result else ''}"
    )
    print(f"  Total net R  : {s['total_net_r']:+.2f}")
    print(
        f"  Return       : {s['total_return_pct']:+.2f}%  "
        f"(${initial_capital:,.0f} → ${s['final_equity']:,.2f})"
    )
    print(f"  Max drawdown : {s['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe-like  : {s['sharpe_like']:+.4f}")
    if not ledger.empty and "asset_class" in ledger.columns:
        tbl = (
            ledger.groupby("asset_class")
            .agg(
                n=("net_r", "size"),
                net_exp_r=("net_r", "mean"),
                total_net_r=("net_r", "sum"),
            )
            .round(4)
            .sort_values("net_exp_r", ascending=False)
        )
        print(tbl.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Honest portfolio backtest of the raw S&D strategy."
    )
    p.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Starting account equity (USD). Default 10000.",
    )
    p.add_argument(
        "--risk",
        type=float,
        default=0.01,
        help="Fraction of CURRENT equity risked per trade. Default 0.01 (1%%).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum simultaneously open trades (capital constraint). Default 5.",
    )
    p.add_argument(
        "--cost-multiplier",
        type=float,
        default=1.0,
        help="Stress-test multiplier on the cost model. Try 1.5 or 2.0.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PARQUET,
        help=f"Path to the labelled dataset parquet. Default {DATASET_PARQUET}.",
    )
    p.add_argument(
        "--ledger-out",
        type=Path,
        default=OUT_LEDGER,
        help=f"Where to write the per-trade ledger CSV. Default {OUT_LEDGER}.",
    )
    # ---- OOS mode flags --------------------------------------------------
    p.add_argument(
        "--oos",
        action="store_true",
        help="Run out-of-sample validation instead of the full-period backtest.",
    )
    p.add_argument(
        "--split-date",
        type=str,
        default=DEFAULT_SPLIT_DATE,
        help=(
            "ISO date (YYYY-MM-DD) to split train / test. "
            f"Rows with formation_time < split_date are train. "
            f"Default {DEFAULT_SPLIT_DATE}."
        ),
    )
    p.add_argument(
        "--selection-cost-mult",
        type=float,
        default=DEFAULT_SELECTION_COST_MULT,
        help=(
            "Cost multiplier used ONLY for the train-period asset-class "
            "selection step. Higher = more conservative selection. "
            f"Default {DEFAULT_SELECTION_COST_MULT}."
        ),
    )
    p.add_argument(
        "--oos-cost-mults",
        type=float,
        nargs="+",
        default=[1.0, 1.5],
        help="Cost multipliers to evaluate on the test period. Default: 1.0 1.5.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found at {args.dataset}", file=sys.stderr)
        print("       run `python build_dataset.py` first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.dataset)

    required = {
        "entry_time",
        "exit_time",
        "formation_time",
        "symbol",
        "timeframe",
        "asset_class",
        "direction",
        "label",
        "entry",
        "risk",
        "exit_reason",
        "pnl_r",
    }
    missing = required - set(df.columns)
    if missing:
        print(
            f"ERROR: dataset is missing required columns: {sorted(missing)}\n"
            "       rebuild with `python build_dataset.py` after pulling the "
            "feature_engine AUDIT_COLS change.",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded {len(df):,} labelled trades from {args.dataset.name}")

    # ---- OOS mode -------------------------------------------------------
    if args.oos:
        run_oos(
            df=df,
            split_date=args.split_date,
            selection_cost_mult=args.selection_cost_mult,
            capital=args.capital,
            risk_fraction=args.risk,
            max_concurrent=args.max_concurrent,
            oos_cost_mults=args.oos_cost_mults,
        )
        return 0

    # ---- Standard full-period backtest ----------------------------------
    print(
        f"Running event-driven backtest: capital=${args.capital:,.0f} "
        f"risk={args.risk:.2%} max_concurrent={args.max_concurrent} "
        f"cost_mult=×{args.cost_multiplier:.2f}"
    )

    ledger, summary = run_backtest(
        df=df,
        initial_capital=args.capital,
        risk_fraction=args.risk,
        max_concurrent=args.max_concurrent,
        cost_multiplier=args.cost_multiplier,
    )

    args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.ledger_out, index=False)
    print(f"\nWrote per-trade ledger ({len(ledger):,} rows) → {args.ledger_out}")

    _print_report(summary, initial_capital=args.capital)
    return 0


if __name__ == "__main__":
    sys.exit(main())
