# =============================================================================
# utils/config.py — Central configuration for the sd-ai zone detection system
# Methodology: Origin-To-Algo (OTA) Supply & Demand
# All parameters here are referenced in zones-detection.md (Section 11)
# =============================================================================

# -----------------------------------------------------------------------------
# Data download configuration
# -----------------------------------------------------------------------------

DEFAULT_TIMEFRAMES: list[str] = ["1wk", "1d", "4h", "1h"]
"""The 4 timeframes used by the pipeline (highest → lowest)."""

YF_INTERVAL: dict[str, str] = {
    "1wk": "1wk",
    "1d": "1d",
    "4h": "1h",  # yfinance has no native 4h → fetch as 1h then resample
    "1h": "1h",
}
"""Maps each pipeline timeframe to the yfinance interval string."""

DOWNLOAD_PERIOD: dict[str, str] = {
    "1wk": "5y",  # ~260 weekly candles
    "1d": "5y",  # ~1 260 daily candles
    "4h": "2y",  # yfinance max for 1h interval (~2 190 4h bars after resample)
    "1h": "2y",  # yfinance max for 1h interval (~8 760 1h bars)
}
"""yfinance 'period' string used when saving full history to CSV."""

# -----------------------------------------------------------------------------
# Watchlist
# -----------------------------------------------------------------------------

WATCHLIST: dict[str, list[str]] = {
    "crypto": [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "BNB-USD",
        "XRP-USD",
        "ADA-USD",
        "AVAX-USD",
        "DOT-USD",
        "LINK-USD",
        "DOGE-USD",
        "LTC-USD",
        "MATIC-USD",
        "ATOM-USD",
        "NEAR-USD",
        "UNI-USD",
    ],
    "us_stocks": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AMZN",
        "TSLA",
        "AVGO",
        "BRK-B",
        "JPM",
        "V",
        "MA",
        "UNH",
        "JNJ",
        "WMT",
        "HD",
        "PG",
        "XOM",
        "CVX",
        "COST",
    ],
    "etfs": [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "XLE",
        "XLF",
        "XLK",
        "GLD",
        "SLV",
    ],
    "indices": [
        "^GSPC",
        "^IXIC",
        "^DJI",
        "^RUT",
        "^VIX",
    ],
    "fx": [
        "EURUSD=X",
        "USDJPY=X",
        "GBPUSD=X",
        "AUDUSD=X",
        "USDCAD=X",
        "USDCHF=X",
        "NZDUSD=X",
    ],
    "commodities": [
        "GC=F",  # Gold
        "SI=F",  # Silver
        "CL=F",  # WTI Crude Oil
        "BZ=F",  # Brent Crude Oil
        "NG=F",  # Natural Gas
        "HG=F",  # Copper
    ],
    "macro": [
        "^TNX",  # US 10Y Treasury Yield
        "DX-Y.NYB",  # US Dollar Index (DXY)
    ],
}


# Flat list — useful for iterating all symbols at once
ALL_SYMBOLS: list[str] = [s for group in WATCHLIST.values() for s in group]


# -----------------------------------------------------------------------------
# Global scan depth
# -----------------------------------------------------------------------------

ZONE_SCAN_DEPTH: int = 1300
"""
Only the last ZONE_SCAN_DEPTH candles are scanned for new zones per run.
All phases operate within this window.
"""

MAX_ZONES_PER_TF: int = 5
"""Maximum number of zones to keep per symbol per timeframe (highest-scored kept)."""

ATR_STOP_BUFFER: float = 0.1
"""
Extra buffer added beyond the distal line when placing a stop-loss:
    stop_price = distal_line ± (ATR_STOP_BUFFER × ATR)
"""

RR_RATIO: float = 3.0
"""Minimum risk-reward ratio for a zone to be considered tradeable.
    TP = entry + RR_RATIO × risk   (risk = entry − stop)
"""

# -----------------------------------------------------------------------------
# Candle primitives thresholds
# -----------------------------------------------------------------------------

ATR_PERIOD: int = 14
"""Wilder's ATR lookback period. Used in every ATR-normalised threshold."""

BASE_BODY_RATIO_MAX: float = 0.50
"""A candle is 'base-like' (indecisive) when body / range <= this value."""

DOJI_BODY_RATIO_MAX: float = 0.10
"""A candle is a doji (near-zero body) when body / range <= this value."""

# -----------------------------------------------------------------------------
# Base cluster detection thresholds (Phase 4)
# -----------------------------------------------------------------------------

BASE_MIN_CANDLES: int = 1
"""Minimum number of consecutive base candles to form a valid cluster."""

BASE_MAX_CANDLES: int = 5
"""Maximum number of consecutive base candles in one cluster."""

BASE_MAX_ATR_WIDTH: float = 2.5
"""Cluster height (base_high − base_low) / avg_ATR must not exceed this."""

BASE_COMPACTNESS_MAX: float = 0.80
"""(close_max − close_min) / base_width must not exceed this.
Measures how tightly the cluster's closing prices are grouped.
0 = all closes identical (perfectly calm), 1 = closes cover the full range."""

# -----------------------------------------------------------------------------
# Chart theme (dark TradingView-inspired palette)
# -----------------------------------------------------------------------------

COLOR_BULL: str = "#26a69a"
COLOR_BEAR: str = "#ef5350"
COLOR_BASE: str = "#b0bec5"
COLOR_DOJI: str = "#ffd700"
CHART_BG: str = "#131722"
CHART_GRID: str = "#1e222d"
