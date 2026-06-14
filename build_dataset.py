#!/usr/bin/env python
# =============================================================================
# build_dataset.py — Pool every available symbol/timeframe into one labelled
# universal dataset for ML training.
# =============================================================================
#
# What this script does
# ---------------------
# For each symbol that already has raw CSVs on disk (``data/raw/<SYMBOL>/``)
# and for each LTF in {1h, 4h, 1d}, run the FULL S&D pipeline:
#
#   detect_bases → detect_formations → detect_zones
#     → add_freshness → add_time_score → add_curve_score
#     → add_trend_score → add_sets_score
#     → labeler.label_zones           (Triple Barrier: 0 / 1 / None)
#     → feature_engine.build_features (lookahead-safe model inputs)
#
# Then tag every row with (symbol, timeframe, asset_class), pool everything
# into one DataFrame, sort by formation_time, and write the result to
# data/dataset.parquet (+ a CSV copy for inspection).
#
# Lookahead-safety guarantees
# ---------------------------
# This script delegates ALL detection / scoring / labelling to the existing
# utils.* modules. It does NOT recompute or shortcut any of them — every
# point-in-time rule (HTF/ITF trend mapping, freshness recompute bounded
# by entry_bar, find_death_bar, same-bar SL/TP tie → SL, no entry before
# departure window) stays intact. The pooled DataFrame is sorted by
# `formation_time` so the downstream train/test split can be time-ordered.
#
# CLI
# ---
#   python build_dataset.py
#   python build_dataset.py --symbols USDJPY=X AMZN
#   python build_dataset.py --max-hold 90
#
# No re-download: symbols whose required CSVs are missing are SKIPPED and
# reported. This script never calls the data downloader.
# =============================================================================

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.base_detector import detect_bases
from utils.config import HTF_REF, WATCHLIST
from utils.data_loader import load_enriched_timeframes
from utils.feature_engine import (
    FEATURE_COLS,
    META_COLS,
    TARGET_COL,
    build_features,
)
from utils.freshness import add_freshness
from utils.htf_range import add_curve_score
from utils.labeler import DEFAULT_HTF_LTF_MAP, DEFAULT_MAX_HOLD_BARS, label_zones
from utils.legs_formation import detect_formations
from utils.sets_scoring import add_sets_score
from utils.time_scoring import add_time_score
from utils.trend_alignment import add_trend_score
from utils.zone_detector import detect_zones

# ---------------------------------------------------------------------------
# Constants — LTFs we build training rows for, paths, asset-class lookup
# ---------------------------------------------------------------------------

LTFS: list[str] = ["1h", "4h", "1d"]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_PARQUET = DATA_DIR / "dataset.parquet"
OUT_CSV = DATA_DIR / "dataset.csv"

# Per-LTF the set of TF CSVs that must exist on disk for the row to be built.
# Curve reference (HTF_REF) and labeler HTF/ITF requirements together drive this.
REQUIRED_TFS: dict[str, set[str]] = {
    "1h": {"1h", "4h", "1d"},
    "4h": {"4h", "1d", "1wk"},
    "1d": {"1d", "1wk"},
}


# ---------------------------------------------------------------------------
# Asset-class lookup (inverse of config.WATCHLIST)
# ---------------------------------------------------------------------------


def _build_asset_class_map() -> dict[str, str]:
    """symbol → asset_class, derived once from config.WATCHLIST."""
    out: dict[str, str] = {}
    for cls, symbols in WATCHLIST.items():
        for s in symbols:
            out[s] = cls
    return out


ASSET_CLASS: dict[str, str] = _build_asset_class_map()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_symbols() -> list[str]:
    """Every sub-folder of data/raw that contains at least one of our TFs."""
    if not RAW_DIR.exists():
        return []
    out = []
    for sym_dir in sorted(RAW_DIR.iterdir()):
        if not sym_dir.is_dir():
            continue
        # Anything on disk counts; whether it can be processed at a given LTF
        # is decided later by REQUIRED_TFS.
        if any((sym_dir / f"{tf}.csv").exists() for tf in ("1h", "4h", "1d", "1wk")):
            out.append(sym_dir.name)
    return out


def _available_tfs(symbol: str) -> set[str]:
    """TFs whose CSV exists on disk for *symbol* — no download."""
    return {
        tf
        for tf in ("1h", "4h", "1d", "1wk")
        if (RAW_DIR / symbol / f"{tf}.csv").exists()
    }


def _process_symbol(
    symbol: str,
    max_hold_bars: int,
) -> tuple[list[pd.DataFrame], dict[str, int], list[str]]:
    """Run the full pipeline for one symbol across every applicable LTF.

    Returns
    -------
    frames     : list of per-LTF feature DataFrames (each row tagged with
                 symbol, timeframe, asset_class).
    no_trade   : per-LTF count of zones whose label is None (no_trade).
    skipped    : list of "<symbol>/<ltf>: reason" strings for LTFs that
                 couldn't be processed (missing CSVs, no zones, …).
    """
    frames: list[pd.DataFrame] = []
    no_trade: dict[str, int] = {tf: 0 for tf in LTFS}
    skipped: list[str] = []

    have = _available_tfs(symbol)
    needed_union = set().union(*[REQUIRED_TFS[tf] for tf in LTFS])
    tfs_to_load = sorted(have & needed_union)
    if not tfs_to_load:
        skipped.append(f"{symbol}: no usable CSVs on disk")
        return frames, no_trade, skipped

    # load_enriched_timeframes auto-downloads any missing TF — we only pass
    # the TFs that already exist, so no network call is ever triggered.
    data = load_enriched_timeframes(symbol, timeframes=tfs_to_load)

    asset_cls = ASSET_CLASS.get(symbol, "unknown")

    for ltf in LTFS:
        missing = REQUIRED_TFS[ltf] - set(data)
        if missing:
            skipped.append(f"{symbol}/{ltf}: missing {sorted(missing)}")
            continue

        ltf_df = data[ltf]

        # ---- Detection chain -------------------------------------------------
        passed, _ = detect_bases(ltf_df)
        formations = detect_formations(ltf_df, passed)
        zones, _ = detect_zones(ltf_df, formations)
        if not zones:
            skipped.append(f"{symbol}/{ltf}: zero zones after detection")
            continue

        # ---- Scoring chain (dependency-ordered) ------------------------------
        add_freshness(ltf_df, zones)
        add_time_score(zones)

        # Curve reference per config.HTF_REF (1h→1d, 4h→1d, 1d→1wk). If that
        # specific HTF is missing, fall back to 1d (same fallback the
        # dashboard uses). DataFrames can't be truth-tested, hence the
        # explicit `is not None` checks.
        curve_ref_tf = HTF_REF.get(ltf, "1d")
        curve_ref_df = data.get(curve_ref_tf)
        if curve_ref_df is None:
            curve_ref_df = data.get("1d")
        if curve_ref_df is not None:
            add_curve_score(zones, curve_ref_df, ltf_df.index)

        add_trend_score(zones, ltf_df)
        add_sets_score(zones)

        # ---- Labeling (Triple Barrier) --------------------------------------
        itf_name, htf_name = DEFAULT_HTF_LTF_MAP[ltf]
        itf_df = data.get(itf_name)
        htf_df = data.get(htf_name) if htf_name else None
        label_zones(
            zones,
            ltf_df,
            itf_df,
            htf_df,
            max_hold_bars=max_hold_bars,
        )

        no_trade[ltf] += sum(1 for z in zones if z.get("label") is None)

        # ---- Feature extraction ---------------------------------------------
        feats = build_features(zones, ltf_df)
        if feats.empty:
            skipped.append(f"{symbol}/{ltf}: zero labelled zones")
            continue

        feats["symbol"] = symbol
        feats["timeframe"] = ltf
        feats["asset_class"] = asset_cls
        frames.append(feats)

    return frames, no_trade, skipped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_audit(
    df: pd.DataFrame,
    no_trade_total: int,
    skipped: list[str],
    failures: list[tuple[str, str]],
) -> None:
    print("\n" + "=" * 72)
    print("DATASET AUDIT REPORT")
    print("=" * 72)

    n = len(df)
    wins = int((df[TARGET_COL] == 1).sum())
    losses = int((df[TARGET_COL] == 0).sum())
    print(f"Total labelled trades : {n}")
    print(
        f"  wins  (label=1)     : {wins}  ({wins / n:.2%} win rate)"
        if n
        else "  wins  (label=1)     : 0"
    )
    print(f"  losses(label=0)     : {losses}")
    print(
        f"No-trade zones excluded: {no_trade_total} "
        f"(zone formed but price never reached proximal)"
    )

    if n == 0:
        print("\nNo rows produced — nothing else to report.")
    else:
        # Per-symbol distribution + concentration flag
        print("\nRows per symbol:")
        per_sym = df["symbol"].value_counts()
        for sym, cnt in per_sym.items():
            flag = "  <-- >40% concentration!" if cnt / n > 0.40 else ""
            print(f"  {sym:<14s} {cnt:6d}  ({cnt / n:5.1%}){flag}")

        print("\nRows per asset_class:")
        for cls, cnt in df["asset_class"].value_counts().items():
            print(f"  {cls:<14s} {cnt:6d}  ({cnt / n:5.1%})")

        print("\nRows per timeframe:")
        for tf, cnt in df["timeframe"].value_counts().items():
            print(f"  {tf:<4s}  {cnt:6d}  ({cnt / n:5.1%})")

        print(
            f"\nDate span : {df['formation_time'].min()}"
            f"  →  {df['formation_time'].max()}"
        )

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for line in skipped:
            print(f"  - {line}")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for sym, msg in failures:
            print(f"  - {sym}: {msg}")

    print("\n" + "-" * 72)
    print(
        "REMINDER: Train/test split MUST be time-based (sort by " "`formation_time`),"
    )
    print("          NEVER random — see formation_time column.")
    print("-" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pooled universal S&D training dataset across every "
            "symbol that already has raw CSVs on disk."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Limit to these symbols (default: every folder under data/raw/).",
    )
    parser.add_argument(
        "--max-hold",
        type=int,
        default=DEFAULT_MAX_HOLD_BARS,
        help=f"Vertical barrier in LTF bars (default: {DEFAULT_MAX_HOLD_BARS}).",
    )
    args = parser.parse_args()

    symbols = args.symbols or _discover_symbols()
    if not symbols:
        print(f"No symbols found under {RAW_DIR}. Nothing to do.")
        return 1

    print(
        f"Building dataset for {len(symbols)} symbol(s) "
        f"(max_hold={args.max_hold} bars)..."
    )
    all_frames: list[pd.DataFrame] = []
    all_skipped: list[str] = []
    failures: list[tuple[str, str]] = []
    no_trade_total = 0

    for sym in symbols:
        try:
            frames, no_trade, skipped = _process_symbol(sym, args.max_hold)
        except Exception as exc:  # noqa: BLE001 — per-symbol isolation
            failures.append((sym, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
            continue

        all_frames.extend(frames)
        all_skipped.extend(skipped)
        no_trade_total += sum(no_trade.values())

        per_sym_rows = sum(len(f) for f in frames)
        print(
            f"  {sym:<14s}  frames={len(frames)}  rows={per_sym_rows}  "
            f"no_trade={sum(no_trade.values())}"
        )

    if not all_frames:
        print("\nNo feature frames produced — aborting before write.")
        _print_audit(
            pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL] + META_COLS),
            no_trade_total,
            all_skipped,
            failures,
        )
        return 1

    # Sort by formation_time so any time-based split downstream is trivial
    # and correct. Reset index so row positions are contiguous.
    pooled = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values("formation_time", kind="stable")
        .reset_index(drop=True)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(OUT_PARQUET, index=False)
    pooled.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(pooled)} rows to:")
    print(f"  - {OUT_PARQUET.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUT_CSV.relative_to(PROJECT_ROOT)}")

    _print_audit(pooled, no_trade_total, all_skipped, failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
