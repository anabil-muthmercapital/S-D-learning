#!/usr/bin/env python
# =============================================================================
# forward_test.py — Paper-trading FORWARD test engine (signal emitter).
# =============================================================================
#
# WHAT THIS IS
# ------------
# One run = one check. For every clean-universe symbol/LTF, this script:
#
#   1. Loads fresh OHLCV (auto-downloads any missing TF and re-pulls 1h/4h/1d
#      so the latest bars are on disk before the pipeline runs).
#   2. DROPS the last bar (it may still be open / forming).
#   3. Runs the EXACT same pipeline as build_dataset.py on the resulting
#      closed-bars-only frame (detect_bases → detect_formations → detect_zones
#      → freshness/time/curve/trend/sets → label_zones).
#   4. Filters to *actionable pending signals*: zones whose departure window
#      has fully CLOSED, that price has not yet touched, that are still alive,
#      and that the frozen XGBoost model rates ≥ ML_THRESHOLD.
#   5. Appends new signals to data/forward_signals.csv (deduplicated by
#      symbol + timeframe + formation_time).
#
# LIVE-LOOKAHEAD SAFETY (the whole point of this file)
# ----------------------------------------------------
# In a backtest every bar is closed by construction — there is no "now".
# In live mode the only bars we may legally touch are bars whose period has
# already ENDED. The two places lookahead could sneak in are:
#
#   (a) The currently-forming bar. Its OHLC can still change before close,
#       so any indicator (ATR, EMA, departure, leg_strength) computed on it
#       is unstable and POISONED relative to its eventual closed value.
#       Mitigation: we drop the last bar before any pipeline call.
#
#   (b) An incomplete departure leg. detect_zones still produces a verdict
#       when fewer than DEPARTURE_CANDLES bars exist after the base — the
#       window simply gets clipped (utils/zone_detector.py uses
#       `min(be + DEPARTURE_CANDLES, len(df) - 1)`). A backtest only sees
#       this clip on the very last bar; a live run sees it every time a
#       freshly-formed base is near the right edge. Such a verdict is
#       LOOKAHEAD-INCONSISTENT with the backtest because the backtest would
#       have re-evaluated it with full data.
#       Mitigation: we require zone["end"] + DEPARTURE_CANDLES <= last_idx,
#       i.e. the full departure window has CLOSED, before emitting.
#
# DESIGN CHOICES
# --------------
# - Pure paper / measurement. No broker calls.
# - Idempotent: existing signals in data/forward_signals.csv are loaded and
#   used as a dedupe key (symbol, timeframe, formation_time).
# - Same encoders / FEATURE_COLS as training — imported, never redefined.
# - Default --exclude-assets matches train_model.py defaults (crypto, macro).
#
# CLI
# ---
#   python forward_test.py
#   python forward_test.py --symbols USDJPY=X AAPL
#   python forward_test.py --no-refresh         # skip the yfinance pull
#   python forward_test.py --exclude-assets crypto macro forex
#   python forward_test.py --since 2026-07-01   # extra scan filter (>= pin)
#   python forward_test.py --reset-forward-test --yes-i-really-want-to-reset
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.base_detector import detect_bases
from utils.config import DEPARTURE_CANDLES, HTF_REF, WATCHLIST
from utils.data_loader import load_enriched_timeframes
from utils.feature_engine import FEATURE_COLS, build_features
from utils.freshness import add_freshness, find_death_bar
from utils.htf_range import add_curve_score
from utils.labeler import DEFAULT_HTF_LTF_MAP, _compute_levels, label_zones
from utils.legs_formation import detect_formations
from utils.regime import compute_regime_series
from utils.sets_scoring import add_sets_score
from utils.time_scoring import add_time_score
from utils.trend_alignment import add_trend_score
from utils.zone_detector import detect_zones

# ---------------------------------------------------------------------------
# Constants — must mirror build_dataset.py / train_model.py
# ---------------------------------------------------------------------------

LTFS: list[str] = ["1h", "4h", "1d"]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_PATH = DATA_DIR / "xgb_model.json"
SIGNALS_CSV = DATA_DIR / "forward_signals.csv"

# Persistent record of when the forward test was started. Once written, every
# subsequent run rejects any zone whose formation_time is BEFORE this date —
# those are historical (back-fill), not forward signals. Created on first run.
FORWARD_START_FILE = DATA_DIR / "forward_test_start.json"

# Same threshold the dashboard and backtest use (train_model best-cv-threshold).
ML_THRESHOLD: float = 0.52

# Same per-LTF dependency set used by build_dataset (curve ref + labeler HTF/ITF).
REQUIRED_TFS: dict[str, set[str]] = {
    "1h": {"1h", "4h", "1d"},
    "4h": {"4h", "1d", "1wk"},
    "1d": {"1d", "1wk"},
}

# Default asset classes to exclude — matches train_model.py's
# --exclude-assets default. The frozen model was trained without these
# classes, so forward-testing them would be off-distribution.
DEFAULT_EXCLUDE_ASSETS: list[str] = ["crypto", "macro"]

# Columns of the persistent signal log. The first block is filled in by
# forward_test.py at signal-detection time; the second block stays empty
# (pending) and is populated later by update_signals.py as outcomes resolve.
SIGNAL_COLS: list[str] = [
    # ---- detection-time fields ----
    "timestamp_detected",
    "symbol",
    "timeframe",
    "zone_type",
    "direction",
    "formation",
    "formation_time",
    "entry",
    "stop",
    "tp",
    "risk",
    "distal",  # needed for death detection (close beyond distal)
    "model_prob",
    "status",
    # ---- outcome-tracking fields (filled by update_signals.py) ----
    "entry_time",
    "exit_time",
    "exit_reason",  # tp | sl | timeout | (empty while pending/open)
    "pnl_r",
    "timeout_pnl_r",
    "bars_held",
    "last_checked",
]

# Dedupe key — same zone observed on the same bar on the same TF is the same
# signal even if re-detected on a later run.
DEDUPE_KEYS: list[str] = ["symbol", "timeframe", "formation_time"]


# ---------------------------------------------------------------------------
# Asset-class lookup (inverse of config.WATCHLIST) — copied from build_dataset.
# ---------------------------------------------------------------------------


def _build_asset_class_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for cls, symbols in WATCHLIST.items():
        for s in symbols:
            out[s] = cls
    return out


ASSET_CLASS: dict[str, str] = _build_asset_class_map()


def _available_tfs(symbol: str) -> set[str]:
    return {
        tf
        for tf in ("1h", "4h", "1d", "1wk")
        if (RAW_DIR / symbol / f"{tf}.csv").exists()
    }


# ---------------------------------------------------------------------------
# Universe discovery
# ---------------------------------------------------------------------------


def _clean_universe(exclude_classes: list[str]) -> list[str]:
    """Symbols on disk whose asset_class is NOT in *exclude_classes*."""
    if not RAW_DIR.exists():
        return []
    out: list[str] = []
    excluded = set(exclude_classes)
    for sym_dir in sorted(RAW_DIR.iterdir()):
        if not sym_dir.is_dir():
            continue
        sym = sym_dir.name
        if ASSET_CLASS.get(sym, "unknown") in excluded:
            continue
        if any((sym_dir / f"{tf}.csv").exists() for tf in ("1h", "4h", "1d", "1wk")):
            out.append(sym)
    return out


# ---------------------------------------------------------------------------
# Fresh-data refresh — re-pull 1h/4h/1d (1wk gets long enough horizon already
# from load_enriched_timeframes' on-demand download). Only the active LTFs
# need fresh closes for the right-edge zone scan.
# ---------------------------------------------------------------------------


def _refresh_bars(symbol: str, timeframes: list[str]) -> None:
    """Re-download the requested TFs so the latest CLOSED bars are on disk.

    We use `overwrite=True` so yfinance returns the full history including
    the most recent bars. Failures here are non-fatal — the pipeline will
    still run on whatever is on disk (it just won't be the very latest).
    """
    from utils.data_downloader import download_symbol  # noqa: PLC0415

    try:
        download_symbol(symbol, timeframes=timeframes, data_dir=RAW_DIR, overwrite=True)
    except Exception as exc:  # noqa: BLE001 — best-effort refresh
        print(f"  [warn] refresh failed for {symbol}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def _load_model(path: Path):
    """Load the frozen XGBoost classifier. Cached at module level."""
    import xgboost as xgb  # noqa: PLC0415

    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


# ---------------------------------------------------------------------------
# Live-safety filter — the heart of the forward test
# ---------------------------------------------------------------------------


def _is_actionable(zone: dict, ltf_df: pd.DataFrame, last_closed_idx: int) -> bool:
    """Return True iff *zone* is a PENDING, model-eligible signal.

    A zone is actionable when ALL of these hold on the closed-bars frame:

      1. Departure window is FULLY closed:
             zone["end"] + DEPARTURE_CANDLES <= last_closed_idx.
         Without this guard we would emit signals whose dep_ratio /
         leg_strength were computed on a half-formed leg-out (see lookahead
         note (b) at the top of this file).

      2. The model-input features computed by detect_zones / scoring used
         only data through last_closed_idx (true by construction — we
         passed the closed-only frame in).

      3. Price has NOT yet touched the proximal:
             zone.get("touches", 0) == 0
         If touches > 0, entry already triggered — in a real run we would
         have entered on that bar; treating it as "pending" now would be
         retroactive.

      4. Zone is still alive:
             find_death_bar(ltf_df, zone) is None
         A zone whose distal has been broken is dead; no further entry
         is possible.

      5. label_zones did NOT assign a label (label is None). label gets
         set only if a touch was found within the closed-data window —
         which is just a stronger form of condition (3), kept here as a
         safety net in case freshness and labeler ever disagree.
    """
    # (1) departure window closed — using last_closed_idx, not len(df), so
    # we are explicit about the "closed bar" frame of reference.
    if zone["end"] + DEPARTURE_CANDLES > last_closed_idx:
        return False
    # (3) no proximal touch yet
    if zone.get("touches", 0) != 0:
        return False
    # (4) zone still alive (close hasn't broken distal)
    if find_death_bar(ltf_df, zone) is not None:
        return False
    # (5) labeler did not open a trade
    if zone.get("label") is not None:
        return False
    return True


# ---------------------------------------------------------------------------
# Per-symbol/per-LTF processing
# ---------------------------------------------------------------------------


def _process_symbol_ltf(
    symbol: str,
    ltf: str,
    data: dict[str, pd.DataFrame],
    model,
    threshold: float,
    now_ts: str,
    forward_start: pd.Timestamp,
) -> tuple[list[dict], int]:
    """Return (new signal rows, count of historical zones skipped) for one (symbol, ltf).

    A zone is considered HISTORICAL — and therefore skipped — when its
    ``formation_time`` is strictly before ``forward_start``. Those are
    back-fill artefacts: the system did not exist at that bar, so claiming
    a signal there would contaminate the forward record.
    """
    missing = REQUIRED_TFS[ltf] - set(data)
    if missing:
        return [], 0

    full_df = data[ltf]
    if len(full_df) < DEPARTURE_CANDLES + 5:
        return [], 0

    # ---- LIVE-LOOKAHEAD GUARD (a): drop the last bar (possibly open) -----
    # Everything downstream sees ONLY bars whose period has fully closed.
    # `ltf_df` is what every util/* function operates on from here on.
    ltf_df = full_df.iloc[:-1].copy()
    last_closed_idx = len(ltf_df) - 1
    if last_closed_idx < DEPARTURE_CANDLES + 1:
        return [], 0

    # The HTF/ITF/curve-ref frames must also be trimmed to closed bars only,
    # otherwise their right edge could leak into trend / curve scoring.
    closed_data: dict[str, pd.DataFrame] = {
        tf: df.iloc[:-1].copy() for tf, df in data.items() if len(df) > 1
    }

    # ---- Detection chain (identical to build_dataset) --------------------
    passed, _ = detect_bases(ltf_df)
    formations = detect_formations(ltf_df, passed)
    zones, _ = detect_zones(ltf_df, formations)
    if not zones:
        return [], 0

    # ---- Scoring chain ---------------------------------------------------
    add_freshness(ltf_df, zones)
    add_time_score(zones)
    curve_ref_tf = HTF_REF.get(ltf, "1d")
    curve_ref_df = closed_data.get(curve_ref_tf)
    if curve_ref_df is None:
        curve_ref_df = closed_data.get("1d")
    if curve_ref_df is not None:
        add_curve_score(zones, curve_ref_df, ltf_df.index)
    add_trend_score(zones, ltf_df)
    add_sets_score(zones)

    # ---- Labeling --------------------------------------------------------
    # label_zones is what tags ALREADY-triggered zones with label∈{0,1};
    # pending zones stay label=None. We need that to filter in _is_actionable.
    itf_name, htf_name = DEFAULT_HTF_LTF_MAP[ltf]
    itf_df = closed_data.get(itf_name)
    htf_df = closed_data.get(htf_name) if htf_name else None
    label_zones(zones, ltf_df, itf_df, htf_df)

    # ---- Actionable filter (live-lookahead-safe) -------------------------
    pending = [z for z in zones if _is_actionable(z, ltf_df, last_closed_idx)]
    if not pending:
        return [], 0

    # ---- Trade levels + dummy fields so build_features will pick them up.
    # build_features only emits rows for zones whose `label is not None`,
    # because in TRAINING a zone without a label is a no-trade. We are
    # deliberately reusing the same code path on pending zones — so we
    # spoof label=0 (the value is discarded; we only use FEATURE_COLS for
    # prediction) and entry_bar=last_closed_idx (a stand-in for "now";
    # affects bars_to_entry only). Mutating the dicts in-place is safe
    # because these zone objects are discarded at function exit.
    for z in pending:
        levels = _compute_levels(z)
        z.update(levels)  # direction, entry, stop, tp, risk
        z["entry_bar"] = last_closed_idx
        z["label"] = 0  # spoof — discarded; build_features needs it set

    # ---- Regime (same lookahead-safe daily GMM as build_dataset) ---------
    regime_series = None
    if "1d" in closed_data:
        try:
            regime_series = compute_regime_series(closed_data["1d"])
        except Exception:  # noqa: BLE001
            regime_series = None

    asset_cls = ASSET_CLASS.get(symbol, "unknown")
    feats = build_features(
        pending,
        ltf_df,
        symbol=symbol,
        asset_class=asset_cls,
        timeframe=ltf,
        regime_series=regime_series,
    )
    if feats.empty:
        return [], 0

    # ---- Score with the frozen model -------------------------------------
    probs = model.predict_proba(feats[FEATURE_COLS])[:, 1]

    rows: list[dict] = []
    n_historical = 0
    for z, prob in zip(pending, probs):
        if prob < threshold:
            continue
        formation_ts = ltf_df.index[z["end"]]
        # ---- RECENCY FILTER ---------------------------------------------
        # Only zones formed at-or-after the forward-test start date count
        # as forward signals. Earlier zones are historical / back-fill and
        # would pollute the live record if logged.
        if formation_ts < forward_start:
            n_historical += 1
            continue
        rows.append(
            {
                "timestamp_detected": now_ts,
                "symbol": symbol,
                "timeframe": ltf,
                "zone_type": z["zone_type"],
                "direction": z["direction"],
                "formation": z.get("formation", ""),
                "formation_time": formation_ts.isoformat(),
                "entry": float(z["entry"]),
                "stop": float(z["stop"]),
                "tp": float(z["tp"]),
                "risk": float(z["risk"]),
                "distal": float(z["distal"]),
                "model_prob": round(float(prob), 4),
                "status": "pending",
                "entry_time": "",
                "exit_time": "",
                "exit_reason": "",
                "pnl_r": "",
                "timeout_pnl_r": "",
                "bars_held": "",
                "last_checked": "",
            }
        )
    return rows, n_historical


# ---------------------------------------------------------------------------
# Signal log I/O
# ---------------------------------------------------------------------------


def _load_existing_signals() -> pd.DataFrame:
    if not SIGNALS_CSV.exists():
        return pd.DataFrame(columns=SIGNAL_COLS)
    try:
        return pd.read_csv(SIGNALS_CSV)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to read {SIGNALS_CSV}: {exc} — treating as empty")
        return pd.DataFrame(columns=SIGNAL_COLS)


def _append_signals(new_rows: list[dict]) -> int:
    """Append *new_rows* to SIGNALS_CSV, deduped against existing entries.

    Returns the number of rows actually written. The log file (with just
    the header) is created on first call even when *new_rows* is empty,
    so downstream consumers can always rely on the path existing.
    """
    SIGNALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not SIGNALS_CSV.exists():
        pd.DataFrame(columns=SIGNAL_COLS).to_csv(SIGNALS_CSV, index=False)
    if not new_rows:
        return 0
    new_df = pd.DataFrame(new_rows, columns=SIGNAL_COLS)
    existing = _load_existing_signals()
    if not existing.empty:
        existing_keys = set(zip(*(existing[k].astype(str) for k in DEDUPE_KEYS)))
        new_keys = list(zip(*(new_df[k].astype(str) for k in DEDUPE_KEYS)))
        keep = [k not in existing_keys for k in new_keys]
        new_df = new_df.loc[keep].reset_index(drop=True)
    if new_df.empty:
        return 0
    new_df.to_csv(SIGNALS_CSV, mode="a", header=False, index=False)
    return len(new_df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward paper-trading engine — emits new S&D signals."
    )
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Subset of symbols to test (default: clean-universe scan).",
    )
    p.add_argument(
        "--exclude-assets",
        nargs="*",
        default=DEFAULT_EXCLUDE_ASSETS,
        metavar="ASSET_CLASS",
        help=(
            "Asset classes to exclude from the live universe. "
            f"Default: {' '.join(DEFAULT_EXCLUDE_ASSETS)} "
            "(same as train_model.py — model was trained without these)."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=ML_THRESHOLD,
        help=f"Model probability threshold for TAKE (default {ML_THRESHOLD}).",
    )
    p.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip the yfinance re-pull (use only CSVs already on disk).",
    )
    p.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "OPTIONAL display/scan filter (UTC). When set, only zones with "
            "formation_time >= this date are logged, IN ADDITION to the pinned "
            "forward-test start. NEVER touches the pin file. Values earlier "
            "than the pinned start are REJECTED (would back-fill history)."
        ),
    )
    p.add_argument(
        "--reset-forward-test",
        action="store_true",
        help=(
            "DANGEROUS: erase the immutable forward-test start pin and re-pin "
            "to today. Requires the companion flag "
            "--yes-i-really-want-to-reset to actually run. Does NOT touch "
            "forward_signals.csv."
        ),
    )
    p.add_argument(
        "--yes-i-really-want-to-reset",
        action="store_true",
        help="Mandatory confirmation companion for --reset-forward-test.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Forward-test start date — IMMUTABLE single source of truth for
# "is this signal historical back-fill or a genuine forward signal?".
#
# Design contract (do not weaken):
#   * The pin file is WRITE-ONCE. Two and only two code paths may create it:
#       (a) first-ever run, when the file does not exist
#       (b) explicit  --reset-forward-test --yes-i-really-want-to-reset
#   * Every other write attempt raises [fatal-bug] and aborts. This is
#     deliberate — if some future code path tries to silently overwrite the
#     pin, the forward record would be retroactively corrupted, so we
#     treat it as a bug and refuse rather than help.
#   * --since is a PURE display/scan filter. It NEVER touches the pin file
#     and is rejected outright if it is older than the pinned start.
# ---------------------------------------------------------------------------


def _parse_date(s: str, *, label: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        raise SystemExit(f"[fatal] {label} value '{s}' is not a valid date.")
    return ts.normalize()  # midnight UTC of that day


def _load_pinned_start() -> pd.Timestamp | None:
    """Read the pin file. Return None if it doesn't exist.

    Never writes. Corrupt content aborts with an explicit message rather
    than silently re-pinning (which would hide the corruption).
    """
    if not FORWARD_START_FILE.exists():
        return None
    try:
        payload = json.loads(FORWARD_START_FILE.read_text())
        return _parse_date(payload["forward_test_start"], label="pin file")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[fatal] pin file {FORWARD_START_FILE} is unreadable: {exc}.\n"
            f"  Inspect it manually. If you accept losing the pin, run:\n"
            f"    python forward_test.py --reset-forward-test "
            f"--yes-i-really-want-to-reset"
        ) from exc


def _create_pin_or_die(ts: pd.Timestamp, *, reason: str) -> None:
    """Write the pin file. Refuses to overwrite an existing pin (write-once).

    `reason` is recorded in the file for forensic traceability — if the pin
    is ever corrupted, the JSON itself records who wrote it.
    """
    if FORWARD_START_FILE.exists():
        raise SystemExit(
            f"[fatal-bug] refusing to overwrite pin file "
            f"{FORWARD_START_FILE} (attempted reason: {reason}). The "
            f"forward-test start date is immutable; overwriting it would "
            f"silently corrupt the forward record. If this is intentional, "
            f"use --reset-forward-test --yes-i-really-want-to-reset."
        )
    FORWARD_START_FILE.parent.mkdir(parents=True, exist_ok=True)
    FORWARD_START_FILE.write_text(
        json.dumps(
            {
                "forward_test_start": ts.isoformat(),
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": reason,
            },
            indent=2,
        )
    )


def _handle_reset(yes_flag: bool) -> pd.Timestamp:
    """Erase the existing pin (if any) and re-pin to today. Loud + guarded."""
    if not yes_flag:
        raise SystemExit(
            "[fatal] --reset-forward-test refused: pass "
            "--yes-i-really-want-to-reset to confirm.\n"
            "  This flag pair erases the immutable forward-test start date "
            "and re-pins to today. Any historical zones formed >= today "
            "would then be eligible to enter the log as 'forward' signals."
        )
    existing = _load_pinned_start()  # may raise on corruption — that's fine
    today = pd.Timestamp.now(tz="UTC").normalize()
    bar = "!" * 72
    print("\n" + bar)
    print("!!  DANGER: --reset-forward-test — ERASING THE FORWARD-TEST PIN")
    print(bar)
    if existing is not None:
        print(f"  current pinned start  : {existing.date()}")
    else:
        print("  current pinned start  : (none — pin file did not exist)")
    print(f"  new pinned start      : {today.date()}")
    print("  forward_signals.csv   : NOT touched by this command")
    print(bar + "\n")
    if FORWARD_START_FILE.exists():
        FORWARD_START_FILE.unlink()
    _create_pin_or_die(
        today, reason="--reset-forward-test --yes-i-really-want-to-reset"
    )
    return today


def _resolve_forward_start(*, reset: bool, yes_flag: bool) -> tuple[pd.Timestamp, bool]:
    """Return (pinned_start_ts, was_created_this_run).

    Resolution order:
      1. --reset-forward-test → _handle_reset (validates --yes flag)
      2. pin file exists → load it (NEVER overwritten by a normal run)
      3. pin file missing → create it with today's date (write-once init)
    """
    if reset:
        return _handle_reset(yes_flag), True

    existing = _load_pinned_start()
    if existing is not None:
        return existing, False

    today = pd.Timestamp.now(tz="UTC").normalize()
    _create_pin_or_die(today, reason="first-run init")
    return today, True


def _print_pin_banner(
    forward_start: pd.Timestamp,
    since_filter: pd.Timestamp | None,
    freshly_created: bool,
) -> None:
    line = "=" * 72
    print(line)
    state = "CREATED THIS RUN" if freshly_created else "IMMUTABLE"
    print(f"  forward-test start (pinned) : {forward_start.date()}  --  {state}")
    print(f"  pin file                    : {FORWARD_START_FILE}")
    if since_filter is not None:
        print(
            f"  extra --since scan filter   : {since_filter.date()} "
            f"(additive; does NOT touch the pin)"
        )
    print(line)


def main() -> int:
    args = _parse_args()

    if not MODEL_PATH.exists():
        print(f"[fatal] model not found at {MODEL_PATH}. Run train_model.py first.")
        return 2

    print(f"[info] loading frozen model: {MODEL_PATH}")
    model = _load_model(MODEL_PATH)

    forward_start, freshly_created = _resolve_forward_start(
        reset=args.reset_forward_test,
        yes_flag=args.yes_i_really_want_to_reset,
    )

    # --since is purely an EXTRA scan filter. Reject silently-corrupting use.
    since_filter: pd.Timestamp | None = None
    if args.since:
        since_filter = _parse_date(args.since, label="--since")
        if since_filter < forward_start:
            raise SystemExit(
                f"[fatal] --since {since_filter.date()} is BEFORE the pinned "
                f"forward-test start {forward_start.date()}. This would "
                f"back-fill historical zones into the forward log.\n"
                f"  If you really mean to restart the forward test, run:\n"
                f"    python forward_test.py --reset-forward-test "
                f"--yes-i-really-want-to-reset"
            )

    _print_pin_banner(forward_start, since_filter, freshly_created)

    # The effective recency cutoff for emitted signals is the LATER of
    # the immutable pin and the optional --since filter.
    effective_start = (
        max(forward_start, since_filter) if since_filter is not None else forward_start
    )

    if args.symbols:
        universe = [s for s in args.symbols if (RAW_DIR / s).exists()]
        missing = [s for s in args.symbols if s not in universe]
        if missing:
            print(f"[warn] no CSV folder for: {missing} — skipping")
    else:
        universe = _clean_universe(args.exclude_assets)

    if not universe:
        print("[fatal] empty universe — nothing to scan.")
        return 1

    print(
        f"[info] scanning {len(universe)} symbol(s) "
        f"× {len(LTFS)} TF(s) — threshold = {args.threshold}"
    )

    now_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    all_new: list[dict] = []
    per_symbol_counts: dict[str, int] = {}
    skipped: list[str] = []
    n_historical_total = 0

    for symbol in universe:
        if not args.no_refresh:
            # Refresh the active LTFs so the right edge of each TF is as
            # recent as yfinance allows. 1wk is large-horizon → no refresh.
            _refresh_bars(symbol, [tf for tf in LTFS])

        try:
            have = _available_tfs(symbol)
            needed = set().union(*(REQUIRED_TFS[t] for t in LTFS))
            tfs_to_load = sorted(have & needed)
            if not tfs_to_load:
                skipped.append(f"{symbol}: no usable CSVs")
                continue
            data = load_enriched_timeframes(symbol, timeframes=tfs_to_load)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{symbol}: load failed → {type(exc).__name__}: {exc}")
            continue

        for ltf in LTFS:
            try:
                rows, n_hist = _process_symbol_ltf(
                    symbol,
                    ltf,
                    data,
                    model,
                    args.threshold,
                    now_ts,
                    effective_start,
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    f"{symbol}/{ltf}: pipeline failed → " f"{type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                continue
            n_historical_total += n_hist
            if rows:
                all_new.extend(rows)
                per_symbol_counts[symbol] = per_symbol_counts.get(symbol, 0) + len(rows)

    n_written = _append_signals(all_new)

    print("\n" + "=" * 72)
    print(f"FORWARD-TEST SUMMARY  ({now_ts})")
    print("=" * 72)
    print(f"  forward-test start : {forward_start.date()}")
    print(f"  symbols scanned    : {len(universe)}")
    print(
        f"  NEW forward signals (formed since {forward_start.date()}, "
        f"prob ≥ {args.threshold:.2f}) : {len(all_new)}"
    )
    print(f"  historical zones skipped (formed before start) : {n_historical_total}")
    print(f"  NEW signals written to log                     : {n_written}")
    print(
        f"  duplicates skipped (already logged)            : {len(all_new) - n_written}"
    )
    print(f"  log file           : {SIGNALS_CSV}")
    if per_symbol_counts:
        print("\n  candidates by symbol:")
        for sym, n in sorted(per_symbol_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {sym:<12}  {n}")
    if skipped:
        print(f"\n  skipped ({len(skipped)}):")
        for s in skipped[:20]:
            print(f"    - {s}")
        if len(skipped) > 20:
            print(f"    ... ({len(skipped) - 20} more)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
