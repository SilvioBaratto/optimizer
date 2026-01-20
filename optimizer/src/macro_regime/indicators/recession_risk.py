from __future__ import annotations

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from optimizer.src.macro_regime.strategies.protocol import EconomicIndicators, MarketIndicators


class RecessionRiskCalculator:
    """
    Calculate probability of recession in 6 and 12 months.
    """

    def __init__(self):
        pass

    def calculate(
        self,
        economic_indicators: "EconomicIndicators",
        market_indicators: "MarketIndicators",
    ) -> Dict[str, float]:
        """
        Calculate recession probability.
        """
        risk_6m = 0.05  # Base 5%
        risk_12m = 0.10  # Base 10%

        # Extract indicators
        gdp_qq = economic_indicators.gdp_growth_qq
        gdp_fcst = economic_indicators.gdp_forecast_6m
        unemp = economic_indicators.unemployment
        ind_prod = economic_indicators.industrial_production

        ism = market_indicators.ism_pmi
        curve = market_indicators.yield_curve_2s10s
        hy = market_indicators.hy_spread

        country_in_trouble = False
        trouble_severity = 0.0

        # GDP contraction is strongest signal
        if gdp_qq is not None and gdp_qq < 0:
            country_in_trouble = True
            if gdp_qq < -0.5:
                trouble_severity = 0.9
            elif gdp_qq < -0.3:
                trouble_severity = 0.7
            elif gdp_qq < -0.1:
                trouble_severity = 0.5
            else:
                trouble_severity = 0.3

        # Industrial production decline amplifies significantly
        if ind_prod is not None and ind_prod < -2.0:
            country_in_trouble = True
            production_severity = min(0.5, abs(ind_prod) / 4.0)
            trouble_severity += production_severity

        # High unemployment adds risk
        if unemp is not None and unemp > 6.5:
            trouble_severity += 0.2 * min(1.0, (unemp - 6.5) / 2.0)

        # Cap trouble severity
        trouble_severity = min(1.0, trouble_severity)

        if country_in_trouble:
            country_risk = 0.65 * trouble_severity
            risk_6m += country_risk * 0.7
            risk_12m += country_risk
        else:
            if gdp_fcst is not None and gdp_fcst < 1.0:
                risk_6m += 0.05
                risk_12m += 0.08

        global_weight = 0.8 if not country_in_trouble else 0.15

        # ISM PMI (guide.md page 83: <48 signals defensive rotation)
        if ism is not None:
            ism_risk = self._calculate_ism_risk(ism)
            risk_6m += ism_risk * global_weight * 0.8
            risk_12m += ism_risk * global_weight

        # Yield Curve (guide.md page 85: inversion = 6-18 month warning)
        if curve is not None:
            curve_risk = self._calculate_curve_risk(curve)
            risk_6m += curve_risk * global_weight * 0.7
            risk_12m += curve_risk * global_weight

        # Credit Spreads (guide.md page 87: >500 bps triggers risk-off)
        if hy is not None:
            spread_risk = self._calculate_spread_risk(hy)
            risk_6m += spread_risk * global_weight
            risk_12m += spread_risk * global_weight

        # Cap at 95%
        return {
            "6month": min(0.95, risk_6m),
            "12month": min(0.95, risk_12m),
        }

    def _calculate_ism_risk(self, ism: float) -> float:
        """Calculate risk contribution from ISM PMI."""
        if ism < 43:
            return 0.30  # Severe
        elif ism < 46:
            return 0.20  # Contraction
        elif ism < 49:
            return 0.12  # Weak
        elif ism < 51:
            return 0.06  # Near neutral
        return 0.0

    def _calculate_curve_risk(self, curve: float) -> float:
        """Calculate risk contribution from yield curve."""
        if curve < -25:
            return 0.28  # Deeply inverted
        elif curve < 0:
            return 0.20  # Inverted
        elif curve < 25:
            return 0.10  # Very flat
        elif curve < 75:
            return 0.04  # Flat
        return 0.0

    def _calculate_spread_risk(self, hy_spread: float) -> float:
        """Calculate risk contribution from credit spreads."""
        if hy_spread > 700:
            return 0.22  # Severe stress
        elif hy_spread > 500:
            return 0.14  # Elevated stress
        elif hy_spread > 400:
            return 0.06  # Moderate stress
        return 0.0

    def get_risk_level(self, recession_risk_6m: float) -> str:
        """Get qualitative risk level from probability."""
        if recession_risk_6m >= 0.50:
            return "high"
        elif recession_risk_6m >= 0.25:
            return "elevated"
        elif recession_risk_6m >= 0.10:
            return "moderate"
        return "low"

    def get_defensive_allocation(self, recession_risk_6m: float) -> float:
        """Get recommended defensive allocation percentage."""
        if recession_risk_6m >= 0.50:
            return 0.30  # 30% defensive
        elif recession_risk_6m >= 0.25:
            return 0.20  # 20% defensive
        elif recession_risk_6m >= 0.10:
            return 0.10  # 10% defensive
        return 0.05  # 5% baseline defensive
