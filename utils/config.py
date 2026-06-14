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
    "1d": "1h",  # yfinance Forex daily open=close bug → fetch as 1h then resample
    "4h": "1h",  # yfinance has no native 4h → fetch as 1h then resample
    "1h": "1h",
}
"""Maps each pipeline timeframe to the yfinance interval string."""

DOWNLOAD_PERIOD: dict[str, str] = {
    "1wk": "5y",  # ~260 weekly candles
    "1d": "2y",  # fetched as 1h then resampled; yfinance 1h max is 2y (~500 daily bars)
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

# -----------------------------------------------------------------------------
# Leg measurement thresholds (Phase 5)
# -----------------------------------------------------------------------------

LEG_CANDLES: int = 3
"""Maximum window (1–3 bars) to look back (leg-in) and forward (leg-out) from the base cluster.
A single strong candle within this window is enough to qualify as a leg — the window is not a
fixed count requirement."""

LEG_STRONG_BODY_RATIO: float = 0.60
"""Minimum body/range ratio for the strongest candle inside the leg-out window.
Filters the 'weak-leg trap': a leg that drifts via small bodies is not institutional —
it must contain at least one impulsive (full-body) candle in the chosen direction."""

# -----------------------------------------------------------------------------
# Departure thresholds (Phase 6)
# -----------------------------------------------------------------------------

# Window-divergence guard: legs_formation and zone_detector must scan the SAME
# number of bars after the base. If they diverge, leg direction and departure
# distance are measured over different windows → inconsistent zone validation.
DEPARTURE_CANDLES: int = LEG_CANDLES
"""Bars after the base end to scan for peak excursion (kept equal to LEG_CANDLES)."""

DEPARTURE_ATR_MIN: float = 0.5
"""departure / avg_ATR must be >= this to pass the volatility-adjusted gate."""

DEPARTURE_RATIO_MIN: float = 2.0
"""departure / zone_width must be >= this to pass the zone-relative gate."""

# -----------------------------------------------------------------------------
# Freshness thresholds (Phase 8)
# -----------------------------------------------------------------------------

FRESHNESS_MAX_TOUCHES: int = 2
"""A zone with >= this many touches scores 0 (stale — too much liquidity consumed)."""

FRESHNESS_SCORE_TABLE: dict[int, int] = {0: 2, 1: 1}
"""Maps touch count → freshness score. Any touches >= FRESHNESS_MAX_TOUCHES → score 0.
   0 touches (never re-entered) → 2  (fresh)
   1 touch                      → 1  (tested once, still valid)
   2+ touches                   → 0  (stale)
"""

# -----------------------------------------------------------------------------
# Time-score thresholds (Phase 9)
# -----------------------------------------------------------------------------

TIME_SCORE_TABLE: dict[int, int] = {1: 2, 2: 2, 3: 1}
"""Maps base candle count → time score. Any count >= 4 → score 0.
   1–2 candles → 2  (explosive, single-decision base — highest conviction)
   3   candles → 1  (compact base — acceptable)
   4+ candles  → 0  (indecisive, too much back-and-forth)
"""

# -----------------------------------------------------------------------------
# Curve-score HTF reference (Phase 10)
# -----------------------------------------------------------------------------

HTF_REF: dict[str, str] = {
    "1h": "1d",
    "4h": "1d",
    "1d": "1wk",  # used only when 1wk data is present
}
"""Maps each LTF/ITF to its HTF reference timeframe for curve scoring.
If the mapped HTF key is not present in the loaded data, scoring falls back to "1d"."""

HTF_RANGE_LOOKBACK: int = 60
"""Number of HTF bars used to compute the rolling point-in-time range for curve scoring.
Only bars with timestamp <= the zone's formation bar are included (no lookahead)."""

# -----------------------------------------------------------------------------
# Trend-alignment (Phase 11)
# -----------------------------------------------------------------------------

SWING_WINDOW: int = 3
"""Half-window (in bars) used to identify swing highs and swing lows.
A bar is a swing high if its high is the maximum over [i-w .. i+w]; likewise for lows.
"""

# -----------------------------------------------------------------------------
# S.E.T.S composite scoring (Phase 12)
# -----------------------------------------------------------------------------

SETS_STRENGTH_RATIO_HIGH: float = 3.0
"""departure_ratio (dep_ratio = departure / zone_width) >= this  →  strength score 2 (explosive).
Matches the methodology's example: departure_ratio = 3.75 → score 2."""

SETS_STRENGTH_RATIO_LOW: float = 2.0
"""departure_ratio >= this (and < SETS_STRENGTH_RATIO_HIGH)  →  strength score 1.
Aligned with DEPARTURE_RATIO_MIN so every zone that cleared the departure gate
earns at least 1 point on Strength."""

SETS_RATING_A: int = 7
"""SETS total >= this  →  ★★★ A-setup (take it)."""

SETS_RATING_B: int = 5
"""SETS total >= this (and < SETS_RATING_A)  →  ★★ B-setup (trade with caution)."""

# -----------------------------------------------------------------------------
# Nested zones (Phase 13)
# -----------------------------------------------------------------------------

NESTED_OVERLAP_MIN: float = 0.50
"""Two same-type zones are considered nested when their overlap / min_width >= this.
Nested zones are merged: proximal = sharper entry side, distal = farther stop side.
"""

# -----------------------------------------------------------------------------
# Chart theme (dark TradingView-inspired palette)
# -----------------------------------------------------------------------------

COLOR_BULL: str = "#26a69a"
COLOR_BEAR: str = "#ef5350"
COLOR_BASE: str = "#b0bec5"
COLOR_DOJI: str = "#ffd700"
CHART_BG: str = "#131722"
CHART_GRID: str = "#1e222d"
