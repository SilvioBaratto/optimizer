from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

from optimizer.src.macro_regime.strategies.protocol import (
    BaseClassificationStrategy,
    ClassificationResult,
    EconomicIndicators,
    MarketIndicators,
    MacroRegime,
)
from optimizer.src.macro_regime.indicators.recession_risk import RecessionRiskCalculator
from optimizer.src.macro_regime.indicators.sector_tilts import SectorTiltCalculator
from optimizer.src.macro_regime.indicators.factor_timing import FactorTimingCalculator


class RuleBasedClassifier(BaseClassificationStrategy):
    """
    Rule-based business cycle classifier using quantitative indicators.
    """

    def __init__(self):
        super().__init__()
        self._recession_calc = RecessionRiskCalculator()
        self._sector_calc = SectorTiltCalculator()
        self._factor_calc = FactorTimingCalculator()

    def classify(
        self,
        economic_indicators: EconomicIndicators,
        market_indicators: MarketIndicators,
        country: str = "USA",
    ) -> ClassificationResult:
        """
        Classify regime using quantitative rules.
        """

        # Extract key indicators
        gdp_qq = economic_indicators.gdp_growth_qq
        gdp_forecast = economic_indicators.gdp_forecast_6m
        unemployment = economic_indicators.unemployment
        inflation = economic_indicators.inflation
        inflation_forecast = economic_indicators.inflation_forecast
        earnings_forecast = economic_indicators.earnings_growth_forecast
        industrial_prod = economic_indicators.industrial_production

        ism = market_indicators.ism_pmi
        curve = market_indicators.yield_curve_2s10s
        hy_spread = market_indicators.hy_spread

        # Calculate momentum
        gdp_mom = self.calculate_momentum(gdp_forecast, gdp_qq)
        inf_mom = self.calculate_momentum(inflation_forecast, inflation)

        # Build real and forecast dicts for compatibility
        real = {
            "gdp_growth_qq": gdp_qq,
            "unemployment": unemployment,
            "inflation": inflation,
            "industrial_production": industrial_prod,
        }
        forecast = {
            "gdp_growth_6m": gdp_forecast,
            "inflation_6m": inflation_forecast,
            "earnings_12m": earnings_forecast,
        }

        # Core classification
        regime, confidence, rationale, signals = self._classify_regime(
            real=real,
            forecast=forecast,
            gdp_mom=gdp_mom,
            inf_mom=inf_mom,
            ism=ism,
            curve=curve,
            hy=hy_spread,
        )

        # Calculate recession risk
        recession_risk = self._recession_calc.calculate(
            economic_indicators=economic_indicators,
            market_indicators=market_indicators,
        )

        # Get sector tilts
        sector_tilts = self._sector_calc.get_tilts(regime)

        # Get factor timing
        factor_timing = self._factor_calc.get_timing(regime)

        # Calculate transition probability
        transition_prob = self._calculate_transition_probability(
            regime=regime,
            gdp_mom=gdp_mom,
            ism=ism,
            curve=curve,
        )

        return ClassificationResult(
            regime=regime,
            confidence=confidence,
            rationale=rationale,
            signals=signals,
            recession_risk_6m=recession_risk["6month"],
            recession_risk_12m=recession_risk["12month"],
            sector_tilts=sector_tilts,
            factor_timing=factor_timing,
            transition_probability=transition_prob,
        )

    def _classify_regime(
        self,
        real: Dict[str, Any],
        forecast: Dict[str, Any],
        gdp_mom: Optional[float],
        inf_mom: Optional[float],
        ism: Optional[float],
        curve: Optional[float],
        hy: Optional[float],
    ) -> Tuple[MacroRegime, float, str, Dict[str, Any]]:
        # Base signals dictionary
        base_signals = {
            "gdp_momentum": gdp_mom,
            "inflation_momentum": inf_mom,
        }

        # =====================================================================
        # PRIORITY 1: COUNTRY-SPECIFIC RECESSION SIGNALS
        # =====================================================================
        gdp_qq = real.get("gdp_growth_qq")
        gdp_fcst = forecast.get("gdp_growth_6m")

        # Strong recession signal from country data
        if (gdp_qq is not None and gdp_qq < -0.5) or (gdp_fcst is not None and gdp_fcst < 0.5):
            if real.get("unemployment", 0) > 6.0 and forecast.get("earnings_12m", 0) < 0:
                return (
                    MacroRegime.RECESSION,
                    0.95,
                    f"Country-specific: Negative GDP growth ({gdp_qq}%) + high unemployment + weak earnings",
                    {
                        **base_signals,
                        "gdp_qq": gdp_qq,
                        "gdp_forecast": gdp_fcst,
                        "unemployment": real.get("unemployment"),
                        "earnings_growth": forecast.get("earnings_12m"),
                        "source": "country_specific",
                    },
                )

        # Moderate recession risk from country data
        if gdp_fcst is not None and gdp_fcst < 0.5 and real.get("unemployment", 0) > 5.5:
            return (
                MacroRegime.RECESSION,
                0.85,
                f"Country-specific: Weak growth forecast ({gdp_fcst}%) + elevated unemployment",
                {
                    **base_signals,
                    "gdp_forecast": gdp_fcst,
                    "unemployment": real.get("unemployment"),
                    "source": "country_specific",
                },
            )

        # Use FRED for severe recession signals (all 3 indicators extreme)
        if ism is not None and curve is not None and hy is not None:
            if ism < 45 and curve < 0 and hy > 500:
                return (
                    MacroRegime.RECESSION,
                    0.90,
                    "Global indicators: All leading indicators signal severe recession",
                    {
                        **base_signals,
                        "ism_pmi": ism,
                        "yield_curve": curve,
                        "hy_spread": hy,
                        "source": "global_markets",
                        "consensus": "bearish",
                    },
                )

        if gdp_mom is not None and gdp_mom > 1.0:
            if (
                forecast.get("gdp_growth_6m", 0) > 3.0
                and forecast.get("earnings_12m", 0) > 10.0
                and real.get("unemployment", 10) > 5.0
            ):
                return (
                    MacroRegime.EARLY_CYCLE,
                    0.90,
                    "Strong growth acceleration + robust earnings + policy support",
                    {
                        **base_signals,
                        "earnings_growth": f"{forecast.get('earnings_12m')}%",
                        "ism_pmi": ism,
                        "phase": "recovery",
                    },
                )

        # Early cycle from ISM
        if ism is not None and ism > 55 and curve is not None and curve > 100:
            return (
                MacroRegime.EARLY_CYCLE,
                0.85,
                "Strong ISM + steep curve signal early expansion",
                {
                    **base_signals,
                    "ism_pmi": ism,
                    "yield_curve": curve,
                    "cycle_momentum": "accelerating",
                },
            )

        # Country-specific late cycle: Decelerating growth + high inflation
        if gdp_mom is not None and gdp_mom < -0.5:
            if (inf_mom is not None and inf_mom > 0.5) or forecast.get("inflation_6m", 0) > 3.5:
                return (
                    MacroRegime.LATE_CYCLE,
                    0.85,
                    f"Country-specific: Decelerating growth ({gdp_mom:.1f}%) + persistent inflation",
                    {
                        **base_signals,
                        "policy_stance": "hawkish",
                        "phase": "peak_slowdown",
                        "source": "country_specific",
                    },
                )

        # Country-specific late cycle: Negative GDP growth (not severe for recession)
        if gdp_qq is not None and -0.5 < gdp_qq < 0:
            return (
                MacroRegime.LATE_CYCLE,
                0.80,
                f"Country-specific: Marginal GDP contraction ({gdp_qq:.1f}%) signals late cycle",
                {
                    **base_signals,
                    "gdp_qq": gdp_qq,
                    "phase": "slowdown",
                    "source": "country_specific",
                },
            )

        # Use FRED for late cycle if both ISM weak AND curve inverted/flat
        if ism is not None and curve is not None:
            if ism < 48 and curve < 25:
                return (
                    MacroRegime.LATE_CYCLE,
                    0.75,
                    f"Global indicators: Weak ISM ({ism:.1f}) + flat curve ({curve:.0f} bps)",
                    {
                        **base_signals,
                        "ism_pmi": ism,
                        "yield_curve": curve,
                        "warning": "recession_risk_elevated",
                        "source": "global_markets",
                    },
                )

        if (
            forecast.get("gdp_growth_6m", 0) >= 2.0
            and forecast.get("gdp_growth_6m", 0) <= 3.5
            and forecast.get("inflation_6m", 0) >= 1.5
            and forecast.get("inflation_6m", 0) <= 3.0
        ):
            if gdp_mom is not None and abs(gdp_mom) < 0.5:
                return (
                    MacroRegime.MID_CYCLE,
                    0.80,
                    "Goldilocks: stable growth + contained inflation + steady policy",
                    {
                        **base_signals,
                        "gdp_forecast": forecast.get("gdp_growth_6m"),
                        "inflation_forecast": forecast.get("inflation_6m"),
                        "stability": "high",
                        "phase": "sustained_expansion",
                    },
                )

        return (
            MacroRegime.UNCERTAIN,
            0.60,
            "Mixed signals - monitor closely for clarity",
            {
                **base_signals,
                "ism_pmi": ism,
                "recommendation": "Maintain flexibility, bias toward defensives",
            },
        )

    def _calculate_transition_probability(
        self,
        regime: MacroRegime,
        gdp_mom: Optional[float],
        ism: Optional[float],
        curve: Optional[float],
    ) -> float:
        """Estimate probability of regime change in next 6 months."""
        if regime == MacroRegime.UNCERTAIN:
            return 0.70  # High uncertainty = likely to resolve

        base_prob = 0.15  # Base 15% chance

        if regime == MacroRegime.MID_CYCLE:
            if gdp_mom is not None and abs(gdp_mom) > 1.0:
                base_prob += 0.25
            if ism is not None and (ism < 48 or ism > 55):
                base_prob += 0.20

        elif regime == MacroRegime.LATE_CYCLE:
            base_prob = 0.45
            if curve is not None and curve < 0:
                base_prob += 0.20

        elif regime == MacroRegime.EARLY_CYCLE:
            base_prob = 0.10

        return min(0.95, base_prob)
