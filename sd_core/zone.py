import pandas as pd
from sd_core.config import CFG


def find_bases(
    df: pd.DataFrame, body_ratio_threshold: float = CFG.base_body_ratio_max
) -> pd.Series:
    return df["body_ratio"] <= body_ratio_threshold


def compactness(
    df: pd.DataFrame, compactness_threshold: float = CFG.compactness_ratio_max
) -> pd.Series:
    df["is_base"] = find_bases(df)
    df["base_high"] = df["high"].where(df["is_base"], other=0)
    df["base_low"] = df["low"].where(df["is_base"], other=0)
    df["base_width"] = df["base_high"] - df["base_low"]
    df["compactness_ratio"] = df["base_width"] / df["average_true_range"]
    df["compactness_passed"] = df["compactness_ratio"] <= compactness_threshold
    return df["compactness_passed"]
