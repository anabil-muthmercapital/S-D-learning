# =============================================================================
# utils/models.py — Core data models
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import pandas as pd
from pydantic import BaseModel, computed_field, model_validator

import numpy as np

from utils.config import ATR_PERIOD, BASE_BODY_RATIO_MAX, DOJI_BODY_RATIO_MAX

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ZoneType = Literal["demand", "supply"]
Formation = Literal["DBR", "RBR", "RBD", "DBD"]
Freshness = Literal["fresh", "tested", "spent"]


# ---------------------------------------------------------------------------
# CandlePrimitives — per-candle computed fields (Phase 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandlePrimitives:
    """Decomposed metrics for a single OHLCV candle."""

    open: float
    high: float
    low: float
    close: float
    prev_close: float | None = None  # previous candle's close (needed for TR)

    @property
    def candle_range(self) -> float:
        """Full range: high − low."""
        return self.high - self.low

    @property
    def body_size(self) -> float:
        """Absolute body: |close − open|."""
        return abs(self.close - self.open)

    @property
    def body_high(self) -> float:
        """Upper edge of the candle body."""
        return max(self.open, self.close)

    @property
    def body_low(self) -> float:
        """Lower edge of the candle body."""
        return min(self.open, self.close)

    @property
    def body_to_range_ratio(self) -> float:
        """Body size as a fraction of the full candle range. 0 if doji."""
        return self.body_size / self.candle_range if self.candle_range else 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_base(self) -> bool:
        """True when body_to_range_ratio ≤ BASE_BODY_RATIO_MAX (indecision candle)."""
        return self.body_to_range_ratio <= BASE_BODY_RATIO_MAX

    @property
    def is_doji(self) -> bool:
        """True when body_to_range_ratio ≤ DOJI_BODY_RATIO_MAX (near-zero body)."""
        return self.body_to_range_ratio <= DOJI_BODY_RATIO_MAX

    @property
    def true_range(self) -> float:
        """True Range using previous close. Falls back to candle_range on the first bar."""
        if self.prev_close is None:
            return self.candle_range
        return max(
            self.candle_range,
            abs(self.high - self.prev_close),
            abs(self.low - self.prev_close),
        )

    # -----------------------------------------------------------------------
    # DataFrame enrichment
    # -----------------------------------------------------------------------

    @classmethod
    def enrich_dataframe(cls, df: "pd.DataFrame") -> "pd.DataFrame":
        """Return a copy of *df* with all per-candle derived columns appended."""
        prev_closes = [None] + list(df["close"].iloc[:-1])
        primitives = [
            cls(
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                prev_close=prev_closes[i],
            )
            for i, r in enumerate(df.itertuples())
        ]
        out = df.copy()
        out["candle_range"] = [c.candle_range for c in primitives]
        out["body_size"] = [c.body_size for c in primitives]
        out["body_to_range_ratio"] = [c.body_to_range_ratio for c in primitives]
        out["is_bullish"] = [c.is_bullish for c in primitives]
        out["is_base"] = [c.is_base for c in primitives]
        out["is_doji"] = [c.is_doji for c in primitives]
        out["prev_close"] = [c.prev_close for c in primitives]
        out["true_range"] = [c.true_range for c in primitives]
        return out


# ---------------------------------------------------------------------------
# ATR enrichment
# ---------------------------------------------------------------------------


def add_atr(df: "pd.DataFrame", period: int = ATR_PERIOD) -> "pd.DataFrame":
    """Return a copy of *df* with an 'atr' column computed via Wilder's smoothing.

    Expects a 'true_range' column — add it first with
    ``CandlePrimitives.enrich_dataframe()``.
    """
    df = df.copy()
    tr = df["true_range"].to_numpy()
    atr = tr.copy()  # seed: ATR[0] = TR[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    df["atr"] = atr
    return df


# ---------------------------------------------------------------------------
# Zone — validated S/D zone object (Phase 5 output)
# ---------------------------------------------------------------------------


class Zone(BaseModel):
    """
    A validated Supply or Demand zone produced by the detection pipeline.
    All index fields refer to integer positions in the source DataFrame.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    zone_type: ZoneType
    formation: Formation

    # ── Index bounds ────────────────────────────────────────────────────────
    base_start: int  # first bar of the base cluster
    base_end: int  # last  bar of the base cluster
    leg_in_start: int  # first bar of the leg-in  (move into base)
    leg_out_end: int  # last  bar of the leg-out (move away from base)

    # ── Price levels ────────────────────────────────────────────────────────
    proximal: float  # price level closest  to current price
    distal: float  # price level furthest from current price

    # ── Metrics ─────────────────────────────────────────────────────────────
    zone_width: float  # abs(proximal − distal)
    departure: float  # leg-out peak displacement from the zone edge (in price)
    departure_ratio: float  # departure / zone_width
    departure_atr: float  # departure / avg_atr (ATR-normalized sanity check)
    base_candle_count: int  # number of base candles

    # ── Diagnostics (captured at detection time, with safe defaults) ────────
    avg_atr: float = 0.0  # ATR averaged across the base candles
    compactness_ratio: float = 0.0  # base_width / avg_atr (lower = tighter)
    leg_strength: float = 0.0  # body_ratio of the strongest qualifying leg-out candle
    base_body_ratio_min: float = (
        0.0  # tightest (smallest) body_ratio among base candles
    )

    # ── State ───────────────────────────────────────────────────────────────
    freshness_state: Freshness = "fresh"
    is_cut: bool = False

    # ── Scores ──────────────────────────────────────────────────────────────
    score_ota: float = 0.0  # OTA grading (0–100)
    score_current: float = 0.0  # live score after freshness/touch decay
    nesting_bonus: float = 0.0  # bonus when confluent with a higher-TF zone
