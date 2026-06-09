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
