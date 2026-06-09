import pandas as pd
from sd_core.config import CFG


def body_ratio(df: pd.DataFrame) -> pd.Series:
    body = abs(df["close"] - df["open"])
    range_ = df["high"] - df["low"]
    body_ratio = body / range_.replace(0, 1)
    return body_ratio


def true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range


def average_true_range(df: pd.DataFrame, period: int = CFG.atr_period) -> pd.Series:
    tr = true_range(df).to_numpy()
    atr = tr.copy()
    for i in range(1, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(atr, index=df.index)
