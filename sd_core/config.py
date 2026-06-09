from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # -----------------------------------------------------------------------------
    # Candle primitives thresholds
    # -----------------------------------------------------------------------------
    base_body_ratio_max: float = 0.5
    doji_body_ratio_max: float = 0.10
    atr_period: int = 14
    # -----------------------------------------------------------------------------
    # Zone detection thresholds
    # -----------------------------------------------------------------------------
    compactness_ratio_max: float = 2.5
    departure_ratio_min: float = 2.0
    departure_atr_min: float = 0.5


CFG = Config()
