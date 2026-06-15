# =============================================================================
# utils/costs.py — Shared transaction cost model
# =============================================================================
#
# Single source of truth for cost assumptions used by BOTH:
#   * backtest.py  — converts costs to R for the simulation
#   * feature_engine.py — computes expected_cost_r as a model feature
#
# Design
# ------
# All costs are expressed as a FRACTION OF ENTRY PRICE so they are
# dimensionless and can be converted to R-multiples by dividing by the
# trade's risk_per_unit (= entry − stop in price terms).
#
#   round_trip_cost_R = (2 × per_side_cost_frac × entry_price) / risk_price
#
# Each charge is per-SIDE; round-trip = 2 × (spread + slip + commission).
#
# Tune `cost_multiplier` in backtest.py to stress-test uncertainty.
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
#
# Shape: { asset_class: { "spread_frac", "slip_frac", "commission_frac" } }
# Each fraction is per-SIDE and is a fraction of entry price.
#
#   fx          : ~0.5 pip spread + ~0.5 pip slip on EURUSD-like majors.
#                 Pip-frac assumed at typical EURUSD price (1.10 → 1 pip ≈
#                 0.91 bps). USDJPY-style pairs auto-rescaled in
#                 per_side_cost_frac() because 1 pip = 0.01 there.
#   crypto      : Top exchange taker fee (~10 bps) + 5 bps each side spread/slip.
#   us_stocks   : Penny-spread (~1–2 bps) + 2 bps slippage, ~zero commission.
#   etfs        : Tighter than stocks; 1 bps spread + 1.5 bps slip.
#   indices     : Index CFDs/futures: slightly wider than ETF.
#   commodities : Futures: ~2 bps spread + 3 bps slip.
#   macro       : Yield indices / DXY — treat like indices.
#   unknown     : Conservative midpoint default for unmapped asset classes.
COST_MODEL: dict[str, dict[str, float]] = {
    "fx": {
        "spread_frac": 0.5 * 1e-4 / 1.10,
        "slip_frac": 0.5 * 1e-4 / 1.10,
        "commission_frac": 0.0,
    },
    "crypto": {
        "spread_frac": 5e-4,
        "slip_frac": 5e-4,
        "commission_frac": 1e-3,
    },
    "us_stocks": {
        "spread_frac": 1.5e-4,
        "slip_frac": 2e-4,
        "commission_frac": 0.0,
    },
    "etfs": {
        "spread_frac": 1e-4,
        "slip_frac": 1.5e-4,
        "commission_frac": 0.0,
    },
    "indices": {
        "spread_frac": 1.5e-4,
        "slip_frac": 2e-4,
        "commission_frac": 0.0,
    },
    "commodities": {
        "spread_frac": 2e-4,
        "slip_frac": 3e-4,
        "commission_frac": 0.0,
    },
    "macro": {
        "spread_frac": 1.5e-4,
        "slip_frac": 2e-4,
        "commission_frac": 0.0,
    },
    "unknown": {
        "spread_frac": 3e-4,
        "slip_frac": 3e-4,
        "commission_frac": 5e-4,
    },
}


def per_side_cost_frac(
    asset_class: str,
    symbol: str,
    entry_price: float,
) -> float:
    """Total cost per SIDE as a fraction of entry price.

    For USDJPY-style fx pairs (price ~150) one pip = 0.01, not 0.0001, so
    the per-pip fraction is computed from the actual price rather than using
    the EURUSD-calibrated default. Detected by "JPY" in the symbol name.

    Parameters
    ----------
    asset_class : one of the keys in COST_MODEL.
    symbol      : ticker string (used for JPY-pip detection on fx).
    entry_price : actual entry price of the trade (the proximal level).
    """
    spec = COST_MODEL.get(asset_class, COST_MODEL["unknown"])
    spread = spec["spread_frac"]
    slip = spec["slip_frac"]
    comm = spec["commission_frac"]

    if asset_class == "fx" and "JPY" in symbol and entry_price > 0:
        # 1 pip = 0.01 on JPY pairs. Compute pip-fraction from the actual
        # entry price (e.g. 150.25 → pip_frac ≈ 6.66e-5 ≈ 0.67 bps).
        pip_frac = 1e-2 / entry_price
        spread = 0.5 * pip_frac
        slip = 0.5 * pip_frac

    return spread + slip + comm


def expected_cost_r(
    asset_class: str,
    symbol: str,
    entry_price: float,
    risk_price: float,
    cost_multiplier: float = 1.0,
) -> float:
    """Round-trip transaction cost expressed as R-multiples.

    Formula
    -------
        cost_R = (2 × per_side_cost_frac × cost_multiplier × entry_price)
                 / risk_price

    Parameters
    ----------
    asset_class     : one of the keys in COST_MODEL.
    symbol          : ticker string (for JPY-pip rescaling).
    entry_price     : actual entry price (proximal level in price units).
    risk_price      : |entry − stop| in price units (the 1R distance).
    cost_multiplier : stress-test multiplier; 1.0 = base case.

    Returns
    -------
    A non-negative float. Returns 0.0 if either price is non-positive to
    avoid division by zero on degenerate rows.
    """
    if entry_price <= 0 or risk_price <= 0:
        return 0.0
    ps = per_side_cost_frac(asset_class, symbol, entry_price)
    return (2.0 * ps * cost_multiplier * entry_price) / risk_price
