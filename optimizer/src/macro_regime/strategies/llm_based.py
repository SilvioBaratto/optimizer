from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from optimizer.src.macro_regime.strategies.protocol import (
    BaseClassificationStrategy,
    ClassificationResult,
    EconomicIndicators,
    MarketIndicators,
    MacroRegime,
)

if TYPE_CHECKING:
    from baml_client.types import (
        BusinessCycleClassification,
        NewsArticle as BAMLNewsArticle,
        EconomicIndicators as BAMLEconomicIndicators,
        MarketIndicators as BAMLMarketIndicators,
    )


# Macro-relevant keywords for news filtering
MACRO_KEYWORDS = [
    "economy",
    "gdp",
    "inflation",
    "recession",
    "federal",
    "central bank",
    "unemployment",
    "jobs",
    "growth",
    "economic",
    "fiscal",
    "monetary",
    "earnings",
    "layoff",
    "rate",
    "policy",
    "market",
    "financial",
]


class LLMBasedClassifier(BaseClassificationStrategy):
    """
    LLM-based business cycle classifier using BAML.
    """

    def __init__(self):
        super().__init__()
        self._baml_client = None

    def _get_baml_client(self):
        """Lazy initialization of BAML client."""
        if self._baml_client is None:
            try:
                import sys
                from pathlib import Path

                # Add parent directories to path
                script_dir = Path(__file__).parent
                project_root = script_dir.parent.parent.parent

                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))

                from baml_client import b

                self._baml_client = b
            except ImportError as e:
                raise ImportError(
                    f"BAML client not available: {e}. "
                    "Make sure baml_client/ exists in project root."
                )

        return self._baml_client

    def classify(
        self,
        economic_indicators: EconomicIndicators,
        market_indicators: MarketIndicators,
        country: str = "USA",
    ) -> ClassificationResult:
        """
        Synchronous classification (wrapper for async).
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.classify_async(
                economic_indicators=economic_indicators,
                market_indicators=market_indicators,
                country=country,
            )
        )

    async def classify_async(
        self,
        economic_indicators: EconomicIndicators,
        market_indicators: MarketIndicators,
        country: str = "USA",
        news_articles: Optional[List[Dict[str, Any]]] = None,
    ) -> ClassificationResult:
        """
        Async LLM-based classification.
        """
        b = self._get_baml_client()

        from baml_client.types import (
            NewsArticle,
            EconomicIndicators as BAMLEconomicIndicators,
            MarketIndicators as BAMLMarketIndicators,
        )

        # Filter and prepare news
        filtered_news = self._filter_macro_news(news_articles or [])
        if filtered_news:
            filtered_news = filtered_news[:25]  # Limit to 25 articles

        # Convert to BAML types
        baml_economic = self._to_baml_economic(economic_indicators)
        baml_market = self._to_baml_market(market_indicators)
        baml_news = [
            NewsArticle(
                title=article.get("title", ""),
                publisher=article.get("publisher", ""),
                date=article.get("date", ""),
                link=article.get("link"),
                full_content=article.get("full_content"),
            )
            for article in filtered_news
        ]

        try:
            # Step 1: Summarize news
            news_signals = b.SummarizeNewsArticles(news=baml_news, country=country)

            # Step 2: Classify with LLM
            today_str = datetime.now().strftime("%Y-%m-%d")

            llm_result = b.ClassifyBusinessCycleWithLLM(
                today=today_str,
                economic_indicators=baml_economic,
                market_indicators=baml_market,
                news_signals=news_signals,
                country=country,
            )

            return self._convert_llm_result(llm_result)

        except Exception as e:
            raise RuntimeError(f"LLM classification failed for {country}: {e}")

    def _filter_macro_news(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter news to macro-relevant articles only."""
        if not news_data:
            return []

        filtered = []
        for article in news_data:
            title = article.get("title", "").lower()
            if any(keyword in title for keyword in MACRO_KEYWORDS):
                filtered.append(article)

        return filtered

    def _to_baml_economic(self, indicators: EconomicIndicators) -> "BAMLEconomicIndicators":
        """Convert domain EconomicIndicators to BAML type."""
        from baml_client.types import EconomicIndicators as BAMLEconomicIndicators

        return BAMLEconomicIndicators(
            gdp_growth_qq=indicators.gdp_growth_qq,
            gdp_growth_yy=indicators.gdp_growth_yy,
            gdp_forecast_6m=indicators.gdp_forecast_6m,
            unemployment=indicators.unemployment,
            industrial_production=indicators.industrial_production,
            capacity_utilization=indicators.capacity_utilization,
            inflation=indicators.inflation,
            inflation_mom=indicators.inflation_mom,
            core_inflation=indicators.core_inflation,
            inflation_forecast=indicators.inflation_forecast,
            retail_sales_mom=indicators.retail_sales_mom,
            earnings_growth_forecast=indicators.earnings_growth_forecast,
            business_confidence=indicators.business_confidence,
            consumer_confidence=indicators.consumer_confidence,
            trade_balance=indicators.trade_balance,
            current_account=indicators.current_account,
            current_account_gdp=indicators.current_account_gdp,
            government_debt_gdp=indicators.government_debt_gdp,
            budget_balance_gdp=indicators.budget_balance_gdp,
        )

    def _to_baml_market(self, indicators: MarketIndicators) -> "BAMLMarketIndicators":
        """Convert domain MarketIndicators to BAML type."""
        from baml_client.types import MarketIndicators as BAMLMarketIndicators

        return BAMLMarketIndicators(
            ism_pmi=indicators.ism_pmi,
            manufacturing_pmi=indicators.manufacturing_pmi,
            services_pmi=indicators.services_pmi,
            composite_pmi=indicators.composite_pmi,
            interest_rate=indicators.interest_rate,
            yield_curve_2s10s=indicators.yield_curve_2s10s,
            bond_yield_2y=indicators.bond_yield_2y,
            bond_yield_5y=indicators.bond_yield_5y,
            bond_yield_10y=indicators.bond_yield_10y,
            bond_yield_30y=indicators.bond_yield_30y,
            hy_spread=indicators.hy_spread,
            vix=indicators.vix,
            vix_signal=indicators.vix_signal,
        )

    def _convert_llm_result(
        self, llm_result: "BusinessCycleClassification"
    ) -> ClassificationResult:
        """Convert BAML result to domain ClassificationResult."""
        # Parse regime
        regime = MacroRegime.from_string(llm_result.regime)

        # Build sector tilts from recommendations
        sector_tilts = {}
        if hasattr(llm_result, "recommended_overweights") and llm_result.recommended_overweights:
            for sector in llm_result.recommended_overweights:
                sector_tilts[sector] = 0.03  # Default overweight
        if hasattr(llm_result, "recommended_underweights") and llm_result.recommended_underweights:
            for sector in llm_result.recommended_underweights:
                sector_tilts[sector] = -0.03  # Default underweight

        # Parse factor exposure
        factor_timing = {}
        if hasattr(llm_result, "factor_exposure"):
            factor_map = {
                "growth_momentum": {"Growth": "overweight", "Momentum": "overweight"},
                "quality_defensive": {"Quality": "overweight", "Low Volatility": "overweight"},
                "balanced": {"Quality": "neutral", "Value": "neutral", "Growth": "neutral"},
            }
            factor_timing = factor_map.get(llm_result.factor_exposure, {})

        # Build signals dict
        signals = {}
        if hasattr(llm_result, "ism_signal"):
            signals["ism_signal"] = llm_result.ism_signal
        if hasattr(llm_result, "yield_curve_signal"):
            signals["yield_curve_signal"] = llm_result.yield_curve_signal
        if hasattr(llm_result, "credit_spread_signal"):
            signals["credit_spread_signal"] = llm_result.credit_spread_signal

        return ClassificationResult(
            regime=regime,
            confidence=llm_result.confidence,
            rationale=llm_result.rationale,
            signals=signals,
            recession_risk_6m=getattr(llm_result, "recession_risk_6m", 0.10),
            recession_risk_12m=getattr(llm_result, "recession_risk_12m", 0.15),
            sector_tilts=sector_tilts,
            factor_timing=factor_timing,
            primary_risks=getattr(llm_result, "primary_risks", []),
            conflicting_signals=getattr(llm_result, "conflicting_signals", []),
        )
