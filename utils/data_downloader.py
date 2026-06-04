# =============================================================================
# utils/data_downloader.py — OHLCV downloader
# =============================================================================
# Responsibilities:
#   - Download OHLCV data from yfinance and save as CSV to data/raw/
#   - 4h is fetched as 1h from yfinance then resampled
#
# Functions:
#   fetch_symbol(symbol, timeframe, n_candles)  → DataFrame  (in-memory, no disk)
#   load_symbol(symbol, timeframe)              → DataFrame  (reads saved CSV)
#   download_symbol(symbol, ...)                → {tf: bool} (download + save)
#   download_all(symbols, ...)                  → {sym: {tf: bool}}
#
# File layout:   data/raw/{SYMBOL}/{timeframe}.csv
# =============================================================================

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.config import (
    ALL_SYMBOLS,
    DEFAULT_TIMEFRAMES,
    DOWNLOAD_PERIOD,
    YF_INTERVAL,
    ZONE_SCAN_DEPTH,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase OHLCV columns, UTC-normalise index, drop NaN rows."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a 1h DataFrame to 4h OHLCV."""
    return (
        df.resample("4h", closed="left", label="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def _csv_path(symbol: str, timeframe: str, data_dir: Path) -> Path:
    """Return the CSV path for a symbol/timeframe pair."""
    return data_dir / symbol / f"{timeframe}.csv"


def _ticker_history(
    symbol: str,
    period: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV via yf.Ticker.history() — uses a different endpoint than
    yf.download(), which helps avoid rate-limiting in many cases.
    """
    ticker = yf.Ticker(symbol)
    raw = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=True,
        raise_errors=False,
    )
    return raw if (raw is not None and not raw.empty) else None


# ---------------------------------------------------------------------------


def fetch_symbol(
    symbol: str,
    timeframe: str = "1d",
    n_candles: int = ZONE_SCAN_DEPTH + 50,
) -> Optional[pd.DataFrame]:
    """
    Download the most-recent n_candles of OHLCV data.  Does NOT save to disk.
    Used by the zone detector to get fresh data at runtime.
    """
    if timeframe not in YF_INTERVAL:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Choose from: {list(YF_INTERVAL)}"
        )

    try:
        raw = _ticker_history(
            symbol,
            period=DOWNLOAD_PERIOD[timeframe],
            interval=YF_INTERVAL[timeframe],
        )
    except Exception as exc:
        logger.warning("Download failed — %s [%s]: %s", symbol, timeframe, exc)
        return None

    if raw is None or raw.empty:
        logger.warning("No data — %s [%s]", symbol, timeframe)
        return None

    df = _clean(raw)
    if timeframe == "4h":
        df = _resample_4h(df)

    df = df.iloc[-n_candles:]

    if len(df) < 20:
        logger.warning("Too few rows — %s [%s]: %d", symbol, timeframe, len(df))
        return None

    return df


# ---------------------------------------------------------------------------
# load_symbol — read a saved CSV from disk
# ---------------------------------------------------------------------------


def load_symbol(
    symbol: str,
    timeframe: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Optional[pd.DataFrame]:
    """Read a previously saved CSV.  Returns None if file does not exist."""
    path = _csv_path(symbol, timeframe, data_dir)
    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ---------------------------------------------------------------------------
# download_symbol — fetch full history for one symbol → save CSVs
# ---------------------------------------------------------------------------


def download_symbol(
    symbol: str,
    timeframes: list[str] = DEFAULT_TIMEFRAMES,
    data_dir: Path = DEFAULT_DATA_DIR,
    overwrite: bool = True,
) -> dict[str, bool]:
    """
    Download full history for one symbol across all timeframes and save as CSV.
    Periods are defined in config.py (DOWNLOAD_PERIOD).

    Returns: {timeframe: True if saved successfully, False on failure}
    """
    sym_dir = data_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}

    for tf in timeframes:
        path = _csv_path(symbol, tf, data_dir)

        if not overwrite and path.exists():
            results[tf] = True
            continue

        time.sleep(2)  # throttle — avoids Yahoo Finance rate limiting

        try:
            raw = _ticker_history(
                symbol,
                period=DOWNLOAD_PERIOD[tf],
                interval=YF_INTERVAL[tf],
            )
        except Exception as exc:
            logger.warning("Download failed — %s [%s]: %s", symbol, tf, exc)
            results[tf] = False
            continue

        if raw is None or raw.empty:
            logger.warning("No data — %s [%s]", symbol, tf)
            results[tf] = False
            continue

        try:
            df = _clean(raw)
            if tf == "4h":
                df = _resample_4h(df)

            if len(df) < 20:
                logger.warning("Too few rows — %s [%s]: %d", symbol, tf, len(df))
                results[tf] = False
                continue

            df.to_csv(path)
            logger.info("Saved %d rows — %s [%s] → %s", len(df), symbol, tf, path)
            results[tf] = True
        except Exception as exc:
            logger.warning("Processing failed — %s [%s]: %s", symbol, tf, exc)
            results[tf] = False

    return results


# ---------------------------------------------------------------------------
# download_all — full watchlist, parallel
# ---------------------------------------------------------------------------


def download_all(
    symbols: list[str] = ALL_SYMBOLS,
    timeframes: list[str] = DEFAULT_TIMEFRAMES,
    data_dir: Path = DEFAULT_DATA_DIR,
    overwrite: bool = True,
    n_jobs: int = 4,
) -> dict[str, dict[str, bool]]:
    """Download and save CSVs for all symbols in parallel.

    Uses process-based parallelism (loky backend) so each worker has an
    isolated yfinance session — prevents the data-mixing race condition that
    occurs with thread-based parallelism.
    """
    sym_list = list(symbols)  # materialise so zip alignment is guaranteed

    raw = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(download_symbol)(sym, timeframes, data_dir, overwrite)
        for sym in tqdm(sym_list, desc="Downloading watchlist", unit="sym")
    )

    return dict(zip(sym_list, raw))
