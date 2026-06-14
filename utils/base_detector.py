# =============================================================================
# utils/base_detector.py — Base cluster detection (Phase 4)
# =============================================================================
#
# Responsibilities
# ----------------
# 1. find_base_clusters()  — scan a DataFrame for contiguous runs of
#                            is_base == True candles, respecting BASE_MAX_CANDLES.
# 2. evaluate_cluster()    — compute tightness metrics and apply all gates for
#                            one cluster, returning a rich result dict.
# 3. detect_bases()        — full pipeline: find → evaluate → split passed/failed.
#
# Prerequisites
# -------------
# The input DataFrame must have been produced by:
#   df = add_atr(CandlePrimitives.enrich_dataframe(raw_df))
# This guarantees the presence of: is_base, high, low, close, atr.
# =============================================================================

from __future__ import annotations

import pandas as pd

from utils.config import (
    BASE_MIN_CANDLES,
    BASE_MAX_CANDLES,
    BASE_MAX_ATR_WIDTH,
)

# ---------------------------------------------------------------------------
# Step 1 — cluster discovery
# ---------------------------------------------------------------------------


def find_base_clusters(df: pd.DataFrame) -> list[dict]:
    """Scan *df* left-to-right and return every contiguous run of base candles.

    The walk:
    - When a base candle is found, open a new cluster.
    - Extend the cluster while the next candle is also base AND the cluster
      hasn't yet reached BASE_MAX_CANDLES.
    - Save and resume from the candle after the cluster ends.

    Each returned dict contains:
        start : int  — iloc position of the first candle
        end   : int  — iloc position of the last  candle
        count : int  — number of candles  (end - start + 1)
    """
    if "is_base" not in df.columns:
        raise ValueError(
            "DataFrame missing 'is_base' column — "
            "run CandlePrimitives.enrich_dataframe() first."
        )

    clusters: list[dict] = []
    is_base = df["is_base"].to_numpy()
    n = len(df)
    i = 0

    while i < n:
        if not is_base[i]:
            i += 1
            continue

        start = end = i
        while end + 1 < n and is_base[end + 1] and (end - start + 1) < BASE_MAX_CANDLES:
            end += 1

        clusters.append({"start": start, "end": end, "count": end - start + 1})
        i = end + 1

    return clusters


# ---------------------------------------------------------------------------
# Step 2 — tightness evaluation
# ---------------------------------------------------------------------------


def evaluate_cluster(df: pd.DataFrame, cluster: dict) -> dict:
    """Compute tightness metrics for *cluster* and apply both gates.

    Gates
    -----
    1. min_count   : cluster["count"] >= BASE_MIN_CANDLES
    2. compactness : (base_high - base_low) / avg_atr <= BASE_MAX_ATR_WIDTH

    Returns the original cluster dict extended with measured values and
    boolean gate results.  The top-level ``passed`` key is True only when
    both gates pass.
    """
    for col in ("high", "low", "close", "atr"):
        if col not in df.columns:
            raise ValueError(
                f"DataFrame missing '{col}' column — "
                "run add_atr(CandlePrimitives.enrich_dataframe(df)) first."
            )

    s, e = cluster["start"], cluster["end"]
    sub = df.iloc[s : e + 1]

    base_high = sub["high"].max()
    base_low = sub["low"].min()
    base_width = base_high - base_low
    avg_atr = sub["atr"].mean()

    # Gate 2: zone height relative to local volatility (OTA compactness)
    compactness_ratio = base_width / avg_atr if avg_atr > 0 else float("inf")

    min_count_passed = cluster["count"] >= BASE_MIN_CANDLES
    compactness_passed = compactness_ratio <= BASE_MAX_ATR_WIDTH

    return {
        **cluster,
        "base_high": round(base_high, 5),
        "base_low": round(base_low, 5),
        "base_width": round(base_width, 5),
        "avg_atr": round(avg_atr, 5),
        "compactness_ratio": round(compactness_ratio, 3),
        "min_count_passed": min_count_passed,
        "compactness_passed": compactness_passed,
        "passed": min_count_passed and compactness_passed,
    }


# ---------------------------------------------------------------------------
# Step 3 — full pipeline
# ---------------------------------------------------------------------------


def detect_bases(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Run the full base-detection pipeline for one timeframe DataFrame.

    Returns
    -------
    passed : list[dict]
        Evaluated clusters that satisfy all three tightness gates.
    failed : list[dict]
        Evaluated clusters that failed at least one gate.
    """
    clusters = find_base_clusters(df)
    evaluated = [evaluate_cluster(df, c) for c in clusters]
    passed = [e for e in evaluated if e["passed"]]
    failed = [e for e in evaluated if not e["passed"]]
    return passed, failed
