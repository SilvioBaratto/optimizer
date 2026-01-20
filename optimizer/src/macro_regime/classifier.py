from __future__ import annotations

from typing import Dict, List, Optional, Any, Literal

from optimizer.src.macro_regime.strategies.protocol import (
    ClassificationStrategy,
    ClassificationResult,
    EconomicIndicators,
    MarketIndicators,
    MacroRegime,
)
from optimizer.src.macro_regime.strategies.rule_based import RuleBasedClassifier
from optimizer.src.macro_regime.strategies.llm_based import LLMBasedClassifier
from optimizer.src.macro_regime.indicators.recession_risk import RecessionRiskCalculator
from optimizer.src.macro_regime.indicators.sector_tilts import SectorTiltCalculator
from optimizer.src.macro_regime.indicators.factor_timing import FactorTimingCalculator


StrategyType = Literal["rule_based", "llm", "hybrid"]


class BusinessCycleClassifier:
    """
    Main orchestrator for business cycle classification.
    """

    # Regime definitions (guide.md pages 73-79)
    REGIME_DEFINITIONS = {
        MacroRegime.EARLY_CYCLE: {
            "duration": "~1 year",
            "returns": "20%+ annualized",
            "description": "Recovery from recession",
        },
        MacroRegime.MID_CYCLE: {
            "duration": "~4 years",
            "returns": "14% annualized",
            "description": "Sustained expansion",
        },
        MacroRegime.LATE_CYCLE: {
            "duration": "~1.5 years",
            "returns": "5% annualized",
            "description": "Peak and slowdown",
        },
        MacroRegime.RECESSION: {
            "duration": "<1 year",
            "returns": "Negative",
            "description": "Economic contraction",
        },
    }

    def __init__(
        self,
        strategy: StrategyType = "llm",
        custom_strategy: Optional[ClassificationStrategy] = None,
    ):
        """
        Initialize the classifier.
        """
        self._strategy_type = strategy

        # Initialize strategies
        if custom_strategy is not None:
            self._strategy = custom_strategy
        elif strategy == "rule_based":
            self._strategy = RuleBasedClassifier()
        elif strategy == "llm":
            self._strategy = LLMBasedClassifier()
        else:  # hybrid
            self._rule_strategy = RuleBasedClassifier()
            self._llm_strategy = LLMBasedClassifier()
            self._strategy = self._rule_strategy  # Default for sync

        # Initialize calculators
        self._recession_calc = RecessionRiskCalculator()
        self._sector_calc = SectorTiltCalculator()
        self._factor_calc = FactorTimingCalculator()

    def classify(
        self,
        ilsole_data: Dict[str, Any],
        fred_data: Dict[str, Any],
        country: str = "USA",
    ) -> ClassificationResult:
        """
        Synchronous classification using the configured strategy.
        """
        economic, market = self._prepare_indicators(ilsole_data, fred_data)
        return self._strategy.classify(economic, market, country)

    async def classify_async(
        self,
        ilsole_data: Dict[str, Any],
        fred_data: Dict[str, Any],
        news_data: Optional[List[Dict[str, Any]]] = None,
        country: str = "USA",
        country_trading_economics_data: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        """
        Asynchronous classification with full data.
        """
        # Merge Trading Economics data if provided
        if country_trading_economics_data:
            fred_data = self._merge_trading_economics(fred_data, country_trading_economics_data)
            ilsole_data = self._merge_ilsole_with_te(ilsole_data, country_trading_economics_data)

        economic, market = self._prepare_indicators(ilsole_data, fred_data)

        if self._strategy_type == "hybrid":
            return await self._hybrid_classify(economic, market, news_data, country)
        elif isinstance(self._strategy, LLMBasedClassifier):
            return await self._strategy.classify_async(economic, market, country, news_data)
        else:
            return self._strategy.classify(economic, market, country)

    async def _hybrid_classify(
        self,
        economic: EconomicIndicators,
        market: MarketIndicators,
        news_data: Optional[List[Dict[str, Any]]],
        country: str,
    ) -> ClassificationResult:
        """
        Hybrid classification combining rule-based and LLM approaches.

        Uses rule-based as base, then validates/adjusts with LLM.
        """
        # Get rule-based result
        rule_result = self._rule_strategy.classify(economic, market, country)

        # Get LLM result
        llm_result = await self._llm_strategy.classify_async(economic, market, country, news_data)

        # Combine results
        return self._combine_results(rule_result, llm_result)

    def _combine_results(
        self,
        rule_result: ClassificationResult,
        llm_result: ClassificationResult,
    ) -> ClassificationResult:
        """
        Combine rule-based and LLM classification results.
        """
        if rule_result.regime == llm_result.regime:
            # Agreement - use LLM result with boosted confidence
            return ClassificationResult(
                regime=llm_result.regime,
                confidence=min(0.95, llm_result.confidence * 1.1),
                rationale=f"[Consensus] {llm_result.rationale}",
                signals={**rule_result.signals, **llm_result.signals},
                recession_risk_6m=(rule_result.recession_risk_6m + llm_result.recession_risk_6m)
                / 2,
                recession_risk_12m=(rule_result.recession_risk_12m + llm_result.recession_risk_12m)
                / 2,
                sector_tilts=llm_result.sector_tilts or rule_result.sector_tilts,
                factor_timing=llm_result.factor_timing or rule_result.factor_timing,
                transition_probability=(
                    rule_result.transition_probability + llm_result.transition_probability
                )
                / 2,
                primary_risks=llm_result.primary_risks,
                conflicting_signals=[],
            )
        else:
            # Disagreement - note the conflict
            if llm_result.confidence >= rule_result.confidence:
                winner = llm_result
                loser = rule_result
            else:
                winner = rule_result
                loser = llm_result

            return ClassificationResult(
                regime=winner.regime,
                confidence=winner.confidence * 0.9,  # Reduce confidence due to conflict
                rationale=f"[Conflict] {winner.rationale}",
                signals={**rule_result.signals, **llm_result.signals},
                recession_risk_6m=max(rule_result.recession_risk_6m, llm_result.recession_risk_6m),
                recession_risk_12m=max(
                    rule_result.recession_risk_12m, llm_result.recession_risk_12m
                ),
                sector_tilts=winner.sector_tilts,
                factor_timing=winner.factor_timing,
                transition_probability=0.50,  # Higher transition risk when signals conflict
                primary_risks=winner.primary_risks,
                conflicting_signals=[
                    f"Rule-based: {rule_result.regime.value}",
                    f"LLM: {llm_result.regime.value}",
                ],
            )

    def _prepare_indicators(
        self,
        ilsole_data: Dict[str, Any],
        fred_data: Dict[str, Any],
    ) -> tuple[EconomicIndicators, MarketIndicators]:
        """Convert raw data to indicator objects."""
        real = ilsole_data.get("real", {})
        forecast = ilsole_data.get("forecast", {})

        economic = EconomicIndicators(
            gdp_growth_qq=real.get("gdp_growth_qq"),
            gdp_growth_yy=real.get("gdp_yy"),
            gdp_forecast_6m=forecast.get("gdp_growth_6m"),
            unemployment=real.get("unemployment"),
            industrial_production=real.get("industrial_production"),
            inflation=real.get("inflation") or real.get("consumer_prices"),
            inflation_forecast=forecast.get("inflation_6m"),
            earnings_growth_forecast=forecast.get("earnings_12m"),
        )

        market = MarketIndicators(
            ism_pmi=fred_data.get("ism_pmi"),
            yield_curve_2s10s=fred_data.get("yield_curve_2s10s"),
            hy_spread=fred_data.get("hy_spread"),
            vix=fred_data.get("vix"),
            vix_signal=fred_data.get("vix_signal"),
        )

        return economic, market

    def _merge_trading_economics(
        self,
        fred_data: Dict[str, Any],
        te_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge Trading Economics data into fred_data."""
        if te_data.get("status") != "success":
            return fred_data

        merged = fred_data.copy()
        indicators = te_data.get("indicators", {})
        bond_yields = te_data.get("bond_yields", {})

        # PMI indicators
        if not merged.get("manufacturing_pmi"):
            merged["manufacturing_pmi"] = indicators.get("manufacturing_pmi", {}).get("value")
        if not merged.get("services_pmi"):
            merged["services_pmi"] = indicators.get("services_pmi", {}).get("value")

        # Bond yields
        for tenor in ["2Y", "5Y", "10Y", "30Y"]:
            key = f"bond_yield_{tenor.lower()}"
            if not merged.get(key) and tenor in bond_yields:
                merged[key] = bond_yields[tenor].get("yield")

        return merged

    def _merge_ilsole_with_te(
        self,
        ilsole_data: Dict[str, Any],
        te_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge Trading Economics data into ilsole_data."""
        if te_data.get("status") != "success":
            return ilsole_data

        merged = {
            "real": ilsole_data.get("real", {}).copy(),
            "forecast": ilsole_data.get("forecast", {}).copy(),
        }

        indicators = te_data.get("indicators", {})

        # Fill missing real indicators
        real = merged["real"]
        if not real.get("gdp_growth_qq"):
            real["gdp_growth_qq"] = indicators.get("gdp_growth_rate", {}).get("value")
        if not real.get("unemployment"):
            real["unemployment"] = indicators.get("unemployment_rate", {}).get("value")
        if not real.get("inflation"):
            real["inflation"] = indicators.get("inflation_rate", {}).get("value")
        if not real.get("industrial_production"):
            ind_prod = te_data.get("industrial_production", {})
            if ind_prod:
                real["industrial_production"] = ind_prod.get("value")

        return merged

    def get_sector_tilts(self, regime: MacroRegime) -> Dict[str, float]:
        """Get sector allocation recommendations for a regime."""
        return self._sector_calc.get_tilts(regime)

    def get_factor_timing(self, regime: MacroRegime) -> Dict[str, str]:
        """Get factor timing recommendations for a regime."""
        return self._factor_calc.get_timing(regime)

    def get_regime_definition(self, regime: MacroRegime) -> Dict[str, str]:
        """Get regime definition and characteristics."""
        return self.REGIME_DEFINITIONS.get(regime, {})

    @staticmethod
    def normalize_regime(regime: Optional[str]) -> MacroRegime:
        """Normalize regime string to enum."""
        return MacroRegime.from_string(regime or "")
