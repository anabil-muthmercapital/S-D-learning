# =============================================================================
# utils/data_loader.py — OHLCV loader for the S&D pipeline
# =============================================================================
# Responsibilities:
#   - Read saved CSV files from  data/raw/{symbol}/{timeframe}.csv
#   - Normalise index to UTC, lowercase columns
#   - Align all timeframes to a common start date (latest first bar wins)
#   - Auto-download via data_downloader if CSV files are missing
#
# Public API:
#   load_timeframe(symbol, tf)              → pd.DataFrame
#   load_all_timeframes(symbol, align)      → dict[str, pd.DataFrame]
#
# Usage (notebook / production):
#   from utils.data_loader import load_all_timeframes
#   data = load_all_timeframes("AAPL")              # aligned, auto-download if missing
#   data = load_all_timeframes("AAPL", align=False) # raw, unaligned
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from utils.config import DEFAULT_TIMEFRAMES

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
) -> dict[str, pd.DataFrame]:
    """Load all timeframes for *symbol*, optionally aligned to a common start.

    Parameters
    ----------
    symbol:     ticker string, e.g. ``"AAPL"``
    timeframes: list of TF keys to load; defaults to ``TIMEFRAMES``
    align:      if ``True`` (default) all DataFrames are trimmed so their
                first bar is the same date (the latest first-bar across TFs)
    data_dir:   override the default ``data/raw/`` directory

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
        data = {tf: df[df.index >= common_start] for tf, df in data.items()}
        logger.info("Aligned %s to common start: %s", symbol, common_start.date())

    return data
