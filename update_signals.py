#!/usr/bin/env python
# =============================================================================
# update_signals.py — Track pending forward signals to their outcome.
# =============================================================================
#
# Companion to forward_test.py. forward_test.py EMITS pending signals into
# data/forward_signals.csv. This script RESOLVES them: it walks closed bars
# from each signal's formation forward and updates status as the trade
# progresses through the lifecycle:
#
#                 ┌────────────────────────────────────────┐
#                 │                                        ▼
#   pending  ──►  open  ──►  closed (tp | sl | timeout)
#       │
#       └──►  expired_no_entry   (distal broken before proximal touched)
#
# Lifecycle rules — IDENTICAL to backtest / labeler.simulate_triple_barrier
# ------------------------------------------------------------------------
# * pending → open:
#     The pending order triggers on the first closed bar whose wick
#     reaches the proximal:
#       long  (demand) : low  <= entry
#       short (supply) : high >= entry
#     If the zone DIES first — i.e. a bar closes beyond the distal:
#       long  : close < distal
#       short : close > distal
#     — the setup is invalidated before triggering and status becomes
#     "expired_no_entry". This is the same close-beyond-distal rule used
#     by utils.freshness.find_death_bar.
#
# * open → closed:
#     Triple-barrier walk from the bar AFTER entry (no same-bar cheating).
#     Up to MAX_HOLD_BARS bars (default 60, same as labeler default):
#       SL hit  : low <= stop  (long) / high >= stop (short)
#       TP hit  : high >= tp   (long) / low  <= tp   (short)
#     Same-bar tie → SL wins (anti-optimism, per Lopez de Prado).
#     No barrier hit within MAX_HOLD_BARS → "timeout", recorded at the
#     last observed close inside the hold window with timeout_pnl_r.
#
# Live-lookahead safety
# ---------------------
# Only CLOSED bars are inspected. The currently-forming (last) bar is
# dropped before any walk. The same guard applies as in forward_test.py:
# `ltf_df = full_df.iloc[:-1]`.
#
# Idempotence
# -----------
# - Rows already in a TERMINAL state (closed / expired_no_entry) are skipped.
# - The whole CSV is rewritten in place each run, so concurrent runs would
#   race; assume single-process execution.
# - Re-running with no new bars is a no-op for resolved trades and simply
#   re-walks pending/open ones — same result if nothing new closed.
#
# CLI
# ---
#   python update_signals.py
#   python update_signals.py --no-refresh
#   python update_signals.py --max-hold 90
# =============================================================================

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import DEPARTURE_CANDLES
from utils.data_loader import load_enriched_timeframes
from utils.labeler import DEFAULT_MAX_HOLD_BARS

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SIGNALS_CSV = DATA_DIR / "forward_signals.csv"

LTFS: list[str] = ["1h", "4h", "1d"]

TERMINAL_STATES: set[str] = {"closed", "expired_no_entry"}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _load_signals() -> pd.DataFrame:
    if not SIGNALS_CSV.exists():
        raise SystemExit(
            f"[fatal] {SIGNALS_CSV} does not exist — run forward_test.py first."
        )
    df = pd.read_csv(SIGNALS_CSV)
    if df.empty:
        return df
    if "distal" not in df.columns:
        raise SystemExit(
            "[fatal] forward_signals.csv is missing the 'distal' column "
            "(written by a newer forward_test.py). Re-run forward_test.py to "
            "regenerate pending signals with the new schema, then retry."
        )
    # Coerce outcome-tracking columns to object dtype. pandas infers them
    # as float64 (all NaN on a fresh row) which then raises a FutureWarning
    # when we write strings like ISO timestamps or "sl" into them.
    for col in (
        "entry_time",
        "exit_time",
        "exit_reason",
        "pnl_r",
        "timeout_pnl_r",
        "bars_held",
        "last_checked",
    ):
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), "")
    return df


def _save_signals(df: pd.DataFrame) -> None:
    SIGNALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SIGNALS_CSV, index=False)


# ---------------------------------------------------------------------------
# Trade lifecycle resolvers — mirror labeler.simulate_triple_barrier exactly
# ---------------------------------------------------------------------------


def _find_entry_or_death(
    ltf_df: pd.DataFrame,
    direction: str,
    entry: float,
    distal: float,
    scan_start_idx: int,
) -> tuple[str, int | None]:
    """Walk closed bars from *scan_start_idx* looking for entry trigger or death.

    Returns
    -------
    ("entry", bar_idx)   — proximal wicked at bar_idx → trade opens.
    ("dead",  bar_idx)   — distal broken at bar_idx → setup invalidated.
    ("pending", None)    — neither happened yet; signal still pending.

    Notes
    -----
    Same-bar tie (the bar both wicks the proximal AND closes past the
    distal) is resolved as ENTRY first, because intra-bar a wick to the
    proximal precedes the bar close — execution would have filled at the
    proximal before the death rule could fire on that bar's close.
    """
    high = ltf_df["high"].to_numpy()
    low = ltf_df["low"].to_numpy()
    close = ltf_df["close"].to_numpy()
    n = len(ltf_df)
    for i in range(scan_start_idx, n):
        if direction == "long":
            if low[i] <= entry:
                return "entry", i
            if close[i] < distal:
                return "dead", i
        else:  # short
            if high[i] >= entry:
                return "entry", i
            if close[i] > distal:
                return "dead", i
    return "pending", None


def _walk_triple_barrier(
    ltf_df: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    tp: float,
    risk: float,
    entry_bar_idx: int,
    max_hold_bars: int,
) -> dict:
    """Walk closed bars from entry_bar_idx+1 looking for SL / TP / timeout.

    Returns a dict with status in {"open", "closed"} plus outcome fields.
    "open" means none of SL/TP/timeout has been reached yet — the trade
    is still active.

    The logic is a line-for-line port of labeler.simulate_triple_barrier's
    barrier-walk section, including the same-bar tie rule (SL wins).
    """
    high = ltf_df["high"].to_numpy()
    low = ltf_df["low"].to_numpy()
    close = ltf_df["close"].to_numpy()
    n = len(ltf_df)

    walk_stop = min(entry_bar_idx + 1 + max_hold_bars, n)  # exclusive
    last_close: float | None = None
    last_bar_idx: int | None = None

    for j in range(entry_bar_idx + 1, walk_stop):
        bar_high = high[j]
        bar_low = low[j]

        if direction == "long":
            sl_hit = bar_low <= stop
            tp_hit = bar_high >= tp
        else:
            sl_hit = bar_high >= stop
            tp_hit = bar_low <= tp

        # Same-bar ambiguity → SL wins (anti-cheating, matches labeler).
        if sl_hit:
            pnl = (
                (stop - entry) / risk if direction == "long" else (entry - stop) / risk
            )
            return {
                "status": "closed",
                "exit_reason": "sl",
                "exit_bar_idx": j,
                "pnl_r": round(float(pnl), 4),
                "timeout_pnl_r": None,
                "bars_held": j - entry_bar_idx,
            }
        if tp_hit:
            pnl = (tp - entry) / risk if direction == "long" else (entry - tp) / risk
            return {
                "status": "closed",
                "exit_reason": "tp",
                "exit_bar_idx": j,
                "pnl_r": round(float(pnl), 4),
                "timeout_pnl_r": None,
                "bars_held": j - entry_bar_idx,
            }

        last_close = float(close[j])
        last_bar_idx = j

    # ---- walk completed without barrier hit --------------------------------
    # Two cases:
    #   (a) walk_stop reached MAX_HOLD_BARS bars after entry → TIMEOUT
    #       (vertical barrier — matches labeler).
    #   (b) walk_stop ran out of CLOSED bars before MAX_HOLD_BARS → still OPEN
    #       (the trade hasn't had its full hold window yet).
    full_hold_completed = walk_stop == entry_bar_idx + 1 + max_hold_bars
    if full_hold_completed and last_close is not None and last_bar_idx is not None:
        if direction == "long":
            unreal = (last_close - entry) / risk
        else:
            unreal = (entry - last_close) / risk
        return {
            "status": "closed",
            "exit_reason": "timeout",
            "exit_bar_idx": last_bar_idx,
            "pnl_r": round(float(unreal), 4),
            "timeout_pnl_r": round(float(unreal), 4),
            "bars_held": last_bar_idx - entry_bar_idx,
        }
    return {
        "status": "open",
        "exit_reason": "",
        "exit_bar_idx": None,
        "pnl_r": None,
        "timeout_pnl_r": None,
        "bars_held": (last_bar_idx - entry_bar_idx) if last_bar_idx is not None else 0,
    }


# ---------------------------------------------------------------------------
# Bar-index helpers — translate timestamps ↔ closed-bar indices
# ---------------------------------------------------------------------------


def _idx_after(ltf_df: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Return the iloc of the first bar with index strictly AFTER *ts*.

    Used for both:
      * pending-scan start (one bar after formation_time / departure window)
      * open-walk start    (one bar after entry_time)
    A return value >= len(df) means no such bar has closed yet.
    """
    pos = ltf_df.index.searchsorted(ts, side="right")
    return int(pos)


# ---------------------------------------------------------------------------
# Refresh — re-pull active LTFs so we have the latest closed bars
# ---------------------------------------------------------------------------


def _refresh_bars(symbol: str, timeframes: list[str]) -> None:
    from utils.data_downloader import download_symbol  # noqa: PLC0415

    try:
        download_symbol(symbol, timeframes=timeframes, data_dir=RAW_DIR, overwrite=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] refresh failed for {symbol}: {type(exc).__name__}: {exc}")


def _load_closed_ltf(symbol: str, tf: str) -> pd.DataFrame | None:
    """Load *symbol*/*tf* and drop the last (possibly forming) bar.

    Returns None on failure.
    """
    try:
        data = load_enriched_timeframes(symbol, timeframes=[tf])
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] load failed {symbol}/{tf}: {type(exc).__name__}: {exc}")
        return None
    if tf not in data:
        return None
    full = data[tf]
    if len(full) < 2:
        return None
    return full.iloc[:-1].copy()


# ---------------------------------------------------------------------------
# Per-row update
# ---------------------------------------------------------------------------


def _update_row(
    row: pd.Series,
    ltf_df: pd.DataFrame,
    max_hold_bars: int,
    now_iso: str,
) -> dict:
    """Compute the updated fields for one signal row.

    Returns a dict of {column_name: new_value} to overwrite into the row.
    Empty dict means "nothing changed".
    """
    direction = str(row["direction"])
    entry = float(row["entry"])
    stop = float(row["stop"])
    tp = float(row["tp"])
    risk = float(row["risk"])
    distal = float(row["distal"])

    formation_ts = pd.to_datetime(row["formation_time"], utc=True)
    status = str(row["status"])

    updates: dict = {"last_checked": now_iso}

    # ---------- PENDING → look for entry trigger or death --------------------
    if status == "pending":
        # Skip the departure window (DEPARTURE_CANDLES bars after the base).
        # Wicks during the leg-out are NOT retests — they are part of the
        # departure itself. The labeler (utils/labeler.py) and freshness
        # (utils/freshness.py) both scan from base_end + DEPARTURE_CANDLES + 1;
        # we must match that contract or we'd open trades on the leg-out bar
        # and report fictitious entries (see fix to BZ=F 2026-06-16).
        scan_start = _idx_after(ltf_df, formation_ts) + DEPARTURE_CANDLES
        if scan_start >= len(ltf_df):
            return updates  # no new closed bars past the departure window yet
        result, bar_idx = _find_entry_or_death(
            ltf_df, direction, entry, distal, scan_start
        )
        if result == "pending":
            return updates
        if result == "dead":
            updates.update(
                {
                    "status": "expired_no_entry",
                    "exit_time": ltf_df.index[bar_idx].isoformat(),
                    "exit_reason": "death",
                }
            )
            return updates
        # result == "entry" → fall through to open the trade, then immediately
        # walk barriers in the same call so a same-day TP/SL gets resolved now.
        entry_bar_idx = bar_idx
        updates["status"] = "open"
        updates["entry_time"] = ltf_df.index[entry_bar_idx].isoformat()
    elif status == "open":
        entry_ts = pd.to_datetime(row["entry_time"], utc=True)
        pos = ltf_df.index.searchsorted(entry_ts, side="left")
        if pos >= len(ltf_df) or ltf_df.index[pos] != entry_ts:
            # entry_time isn't on the closed-bar grid we just loaded —
            # likely because the loader trimmed warmup or alignment changed.
            # Treat as not-yet-resolvable; try again next run.
            return updates
        entry_bar_idx = int(pos)
    else:
        return {}  # terminal — no work to do

    # ---------- OPEN → walk triple barrier -----------------------------------
    walk = _walk_triple_barrier(
        ltf_df,
        direction=direction,
        entry=entry,
        stop=stop,
        tp=tp,
        risk=risk,
        entry_bar_idx=entry_bar_idx,
        max_hold_bars=max_hold_bars,
    )
    if walk["status"] == "open":
        # Already marked open above (or was open coming in); nothing to add
        # beyond bars_held / last_checked.
        if walk["bars_held"]:
            updates["bars_held"] = int(walk["bars_held"])
        return updates

    # walk["status"] == "closed"
    updates.update(
        {
            "status": "closed",
            "exit_time": ltf_df.index[walk["exit_bar_idx"]].isoformat(),
            "exit_reason": walk["exit_reason"],
            "pnl_r": walk["pnl_r"],
            "timeout_pnl_r": (
                walk["timeout_pnl_r"] if walk["timeout_pnl_r"] is not None else ""
            ),
            "bars_held": int(walk["bars_held"]),
        }
    )
    return updates


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update pending/open forward signals to their outcome."
    )
    p.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip the yfinance re-pull; use only CSVs already on disk.",
    )
    p.add_argument(
        "--max-hold",
        type=int,
        default=DEFAULT_MAX_HOLD_BARS,
        help=f"Triple-barrier vertical limit in bars (default {DEFAULT_MAX_HOLD_BARS}).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    df = _load_signals()
    if df.empty:
        print(f"[info] {SIGNALS_CSV} has no rows — nothing to update.")
        return 0

    # ---- which rows still need work? ---------------------------------------
    active_mask = ~df["status"].astype(str).isin(TERMINAL_STATES)
    active_df = df.loc[active_mask]
    if active_df.empty:
        print("[info] all signals are in a terminal state — nothing to update.")
        _print_summary(df)
        return 0

    # ---- group by (symbol, timeframe) to load each frame once --------------
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    transitions = {"pending→open": 0, "pending→expired": 0, "open→closed": 0}
    errors: list[str] = []

    pairs = sorted({(r["symbol"], r["timeframe"]) for _, r in active_df.iterrows()})
    print(
        f"[info] {len(active_df)} active signal(s) across {len(pairs)} (symbol, tf) pair(s)"
    )

    for symbol, tf in pairs:
        if not args.no_refresh:
            _refresh_bars(symbol, [tf])
        ltf_df = _load_closed_ltf(symbol, tf)
        if ltf_df is None or len(ltf_df) == 0:
            errors.append(f"{symbol}/{tf}: no usable closed bars")
            continue
        # Index slice of rows we need to update for this (symbol, tf).
        sel = (df["symbol"] == symbol) & (df["timeframe"] == tf) & active_mask
        for idx in df.index[sel]:
            prev_status = str(df.at[idx, "status"])
            try:
                updates = _update_row(df.loc[idx], ltf_df, args.max_hold, now_iso)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}/{tf} row {idx}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            for col, val in updates.items():
                df.at[idx, col] = val
            new_status = str(df.at[idx, "status"])
            if prev_status != new_status:
                key = f"{prev_status}→{'expired' if new_status == 'expired_no_entry' else new_status}"
                transitions[key] = transitions.get(key, 0) + 1

    _save_signals(df)

    print("\n" + "=" * 72)
    print(f"UPDATE SUMMARY  ({now_iso})")
    print("=" * 72)
    for k, v in transitions.items():
        if v:
            print(f"  {k:<20} : {v}")
    if errors:
        print(f"\n  errors ({len(errors)}):")
        for e in errors[:20]:
            print(f"    - {e}")
        if len(errors) > 20:
            print(f"    ... ({len(errors) - 20} more)")
    print()
    _print_summary(df)
    print("=" * 72)
    return 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_summary(df: pd.DataFrame) -> None:
    counts = df["status"].astype(str).value_counts().to_dict()
    n_pending = counts.get("pending", 0)
    n_open = counts.get("open", 0)
    n_closed = counts.get("closed", 0)
    n_expired = counts.get("expired_no_entry", 0)

    print("LIVE PERFORMANCE (closed trades only)")
    print("-" * 72)
    if n_closed == 0:
        print("  no closed trades yet")
    else:
        closed = df[df["status"].astype(str) == "closed"].copy()
        # pnl_r may be stored as string when read from CSV; coerce safely.
        closed["pnl_r"] = pd.to_numeric(closed["pnl_r"], errors="coerce")
        valid = closed.dropna(subset=["pnl_r"])
        wins = int((valid["pnl_r"] > 0).sum())
        win_rate = wins / len(valid) * 100 if len(valid) else 0.0
        total_r = float(valid["pnl_r"].sum())
        avg_r = float(valid["pnl_r"].mean()) if len(valid) else 0.0
        # Reason breakdown
        reason_counts = closed["exit_reason"].astype(str).value_counts().to_dict()
        print(f"  closed trades   : {n_closed}")
        print(
            f"  win rate        : {win_rate:.1f}%   ({wins} W / {len(valid) - wins} L)"
        )
        print(f"  total pnl       : {total_r:+.2f} R")
        print(
            f"  avg expectancy  : {avg_r:+.3f} R / trade  (vs +0.42 R backtest baseline)"
        )
        if reason_counts:
            parts = [f"{k}={v}" for k, v in reason_counts.items()]
            print(f"  by exit reason  : {', '.join(parts)}")

    print()
    print("PIPELINE STATE")
    print("-" * 72)
    print(f"  pending          : {n_pending}")
    print(f"  open             : {n_open}")
    print(f"  closed           : {n_closed}")
    print(f"  expired_no_entry : {n_expired}")
    print(f"  total            : {len(df)}")


if __name__ == "__main__":
    raise SystemExit(main())
