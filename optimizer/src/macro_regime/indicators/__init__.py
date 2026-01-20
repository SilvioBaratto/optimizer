"""
Macro Regime Indicators Module - Economic indicator analysis.

Provides:
- RecessionRiskCalculator: Calculate recession probability
- SectorTiltCalculator: Get sector allocation recommendations
- FactorTimingCalculator: Get factor exposure recommendations

Usage:
    from src.macro_regime.indicators import (
        RecessionRiskCalculator,
        SectorTiltCalculator,
        FactorTimingCalculator,
    )

    # Recession risk
    recession_calc = RecessionRiskCalculator()
    risk = recession_calc.calculate(economic_indicators, market_indicators)

    # Sector tilts
    sector_calc = SectorTiltCalculator()
    tilts = sector_calc.get_tilts(regime)

    # Factor timing
    factor_calc = FactorTimingCalculator()
    timing = factor_calc.get_timing(regime)
"""

from optimizer.src.macro_regime.indicators.recession_risk import RecessionRiskCalculator
from optimizer.src.macro_regime.indicators.sector_tilts import SectorTiltCalculator
from optimizer.src.macro_regime.indicators.factor_timing import FactorTimingCalculator

__all__ = [
    "RecessionRiskCalculator",
    "SectorTiltCalculator",
    "FactorTimingCalculator",
]
