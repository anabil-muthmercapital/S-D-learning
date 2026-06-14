# =============================================================================
# utils/data_loader.py — OHLCV loader for the S&D pipeline
# =============================================================================
# Responsibilities:
#   - Read saved CSV files from  data/raw/{symbol}/{timeframe}.csv
#   - Normalise index to UTC, lowercase columns
#   - Align all timeframes to a common start date (latest first bar wins)
#   - Auto-download via data_downloader if CSV files are missing
#   - Optionally enrich (CandlePrimitives + ATR) on full history BEFORE trimming
#     so Wilder's ATR has its full pre-roll warm-up.
#
# Public API:
#   load_timeframe(symbol, tf)                        → pd.DataFrame
#   load_all_timeframes(symbol, align, warmup_bars)   → dict[str, pd.DataFrame]
#   load_enriched_timeframes(symbol, ...)             → dict[str, pd.DataFrame]
#
# Usage (notebook / production):
#   from utils.data_loader import load_enriched_timeframes
#   data = load_enriched_timeframes("AAPL")          # ATR warmed on FULL history
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from utils.config import DEFAULT_TIMEFRAMES
from utils.models import CandlePrimitives, add_atr

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "raw"

TIMEFRAMES: list[str] = DEFAULT_TIMEFRAMES
"""Ordered list of pipeline timeframes (highest → lowest resolution)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_downloaded(
    symbol: str,
    timeframes: list[str],
    data_dir: Path,
) -> None:
    """Download any missing timeframe CSVs for *symbol*.

    Checks each TF file on disk first.  Only calls the downloader for the
    files that are absent, so an already-downloaded symbol is never re-fetched.
    """
    missing = [
        tf for tf in timeframes if not (data_dir / symbol / f"{tf}.csv").exists()
    ]
    if not missing:
        return

    logger.info(
        "%s: %d timeframe(s) missing — downloading: %s",
        symbol,
        len(missing),
        missing,
    )
    print(f"[data_loader] Downloading {symbol} — missing TFs: {missing} ...")

    # Import here to avoid a circular dependency at module level
    from utils.data_downloader import download_symbol  # noqa: PLC0415

    results = download_symbol(
        symbol, timeframes=missing, data_dir=data_dir, overwrite=False
    )
    failed = [tf for tf, ok in results.items() if not ok]
    if failed:
        raise RuntimeError(
            f"Failed to download {symbol} for timeframes: {failed}.\n"
            "Check your internet connection or yfinance rate limits."
        )
    print(f"[data_loader] Download complete for {symbol}.")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase OHLCV columns, UTC-normalise the DatetimeIndex."""
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_timeframe(
    symbol: str,
    tf: str,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load a single timeframe CSV for *symbol*.

    Parameters
    ----------
    symbol:   ticker string, e.g. ``"AAPL"`` or ``"USDJPY=X"``
    tf:       timeframe key, one of ``TIMEFRAMES`` (``"1wk"``, ``"1d"``, ``"4h"``, ``"1h"``)
    data_dir: override the default ``data/raw/`` directory

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns open/high/low/close/volume.

    Raises
    ------
    FileNotFoundError if the CSV does not exist.
    """
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = root / symbol / f"{tf}.csv"
    if not path.exists():
        _ensure_downloaded(symbol, [tf], root)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = _normalise(df)
    logger.debug(
        "Loaded %s %s — %d rows (%s → %s)",
        symbol,
        tf,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def load_all_timeframes(
    symbol: str,
    timeframes: list[str] | None = None,
    align: bool = True,
    data_dir: Path | str | None = None,
    warmup_bars: int = 0,
) -> dict[str, pd.DataFrame]:
    """Load all timeframes for *symbol*, optionally aligned to a common start.

    Parameters
    ----------
    symbol:      ticker string, e.g. ``"AAPL"``
    timeframes:  list of TF keys to load; defaults to ``TIMEFRAMES``
    align:       if ``True`` (default) all DataFrames are trimmed so their
                 first "live" bar is the same date (the latest first-bar across TFs)
    data_dir:    override the default ``data/raw/`` directory
    warmup_bars: when ``align=True``, keep this many extra bars BEFORE the common
                 start on each timeframe so downstream indicators (ATR, EMAs, …)
                 can warm up on real history. Safe: only past bars are kept — no
                 look-ahead is introduced. The true common-start timestamp is
                 stored on ``df.attrs["common_start"]`` so callers can mask the
                 warm-up region after indicator computation, e.g.:
                     df_live = df[df.index >= df.attrs["common_start"]]

    Returns
    -------
    dict mapping TF key → pd.DataFrame (DatetimeIndex UTC, OHLCV columns).
    """
    tfs = timeframes or TIMEFRAMES
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    # Check all TFs at once and download any that are missing in one shot
    _ensure_downloaded(symbol, tfs, root)

    data: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        data[tf] = load_timeframe(symbol, tf, data_dir=root)

    if align and len(data) > 1:
        common_start = max(df.index[0] for df in data.values())
        aligned: dict[str, pd.DataFrame] = {}
        for tf, df in data.items():
            if warmup_bars > 0:
                # Keep up to `warmup_bars` rows strictly BEFORE common_start
                # (purely historical — no future leak).
                pre = df.index < common_start
                pre_idx = df.index[pre][-warmup_bars:]
                cut = pre_idx[0] if len(pre_idx) else common_start
                sub = df[df.index >= cut].copy()
            else:
                sub = df[df.index >= common_start].copy()
            sub.attrs["common_start"] = common_start
            aligned[tf] = sub
        data = aligned
        logger.info(
            "Aligned %s to common start: %s (warmup_bars=%d)",
            symbol,
            common_start.date(),
            warmup_bars,
        )

    return data


def load_enriched_timeframes(
    symbol: str,
    timeframes: list[str] | None = None,
    align: bool = True,
    data_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load + enrich all timeframes with ATR computed on FULL history before trim.

    Pipeline per timeframe (in this exact order):
        1. Load raw OHLCV (unaligned).
        2. ``CandlePrimitives.enrich_dataframe`` on the full series.
        3. ``add_atr`` on the full series — Wilder's smoothing now gets its
           full pre-roll warm-up.
        4. ONLY THEN trim to the common start across timeframes (if align=True).

    Why this matters
    ----------------
    The previous workflow (`load_all_timeframes` + `add_atr` in the notebook)
    trimmed first, so the first ~14 bars after `common_start` had an unstable
    ATR — worst on the weekly TF where years of pre-roll history were thrown
    away. Computing indicators on full history first fixes that.

    No look-ahead: trimming only the START of the series uses purely past data
    for warm-up, so no future information can leak into earlier bars.

    Parameters
    ----------
    symbol, timeframes, align, data_dir : see ``load_all_timeframes``.

    Returns
    -------
    dict[str, pd.DataFrame] — each DataFrame already carries the columns
    added by ``CandlePrimitives.enrich_dataframe`` plus an ``atr`` column.
    """
    tfs = timeframes or TIMEFRAMES
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    # Load every TF on its FULL history (unaligned) so indicators warm up properly.
    raw = load_all_timeframes(symbol, timeframes=tfs, align=False, data_dir=root)

    enriched: dict[str, pd.DataFrame] = {
        tf: add_atr(CandlePrimitives.enrich_dataframe(df)) for tf, df in raw.items()
    }

    if align and len(enriched) > 1:
        common_start = max(df.index[0] for df in enriched.values())
        enriched = {
            tf: df[df.index >= common_start].copy() for tf, df in enriched.items()
        }
        for df in enriched.values():
            df.attrs["common_start"] = common_start
        logger.info(
            "Enriched + aligned %s to common start: %s",
            symbol,
            common_start.date(),
        )

    return enriched
