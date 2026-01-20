"""
View Generator - Generate Black-Litterman views from stock signals.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.domain.models.view import BlackLittermanViewDTO, MacroRegimeDTO

logger = logging.getLogger(__name__)


class ViewGeneratorImpl:
    """
    Generate Black-Litterman views from stock signals using BAML AI.

    This class orchestrates the view generation process:
    1. Fetch news for each stock
    2. Summarize news using BAML
    3. Generate views using BAML with signal + news context
    4. Validate and filter views

    Implements the ViewGenerator protocol from domain.protocols.optimizer.
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        view_scaling: float = 1.0,
    ):
        """
        Initialize view generator.

        Args:
            min_confidence: Minimum view confidence to include
            view_scaling: Scaling factor for view magnitudes
        """
        self._min_confidence = min_confidence
        self._view_scaling = view_scaling
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._baml_client = None

    def _get_baml_client(self):
        """Lazy load BAML client."""
        if self._baml_client is None:
            try:
                from baml_client import b
                self._baml_client = b
            except ImportError as e:
                raise ImportError(
                    f"BAML client not available: {e}. "
                    "Make sure baml_client/ exists in project root."
                )
        return self._baml_client

    async def generate(
        self,
        signals: List[StockSignalDTO],
        macro_regime: Optional[MacroRegimeDTO] = None,
    ) -> List[BlackLittermanViewDTO]:
        """
        Generate views from stock signals.

        Args:
            signals: Stock signals to generate views for
            macro_regime: Optional macro regime for context

        Returns:
            List of BlackLittermanViewDTO objects
        """
        if not signals:
            self._logger.warning("No signals provided for view generation")
            return []

        self._logger.info(f"Generating views for {len(signals)} signals")

        views: List[BlackLittermanViewDTO] = []

        for signal in signals:
            try:
                view = await self._generate_single_view(signal, macro_regime)
                if view and view.confidence >= self._min_confidence:
                    views.append(view)
                elif view:
                    self._logger.debug(
                        f"Filtered view for {signal.ticker}: "
                        f"confidence {view.confidence:.2f} < {self._min_confidence}"
                    )
            except Exception as e:
                self._logger.error(f"Error generating view for {signal.ticker}: {e}")

        self._logger.info(
            f"Generated {len(views)} views from {len(signals)} signals "
            f"(filtered {len(signals) - len(views)})"
        )

        return views

    async def _generate_single_view(
        self,
        signal: StockSignalDTO,
        macro_regime: Optional[MacroRegimeDTO] = None,
    ) -> Optional[BlackLittermanViewDTO]:
        """
        Generate a single view for one stock.

        Args:
            signal: Stock signal
            macro_regime: Macro regime context

        Returns:
            BlackLittermanViewDTO or None if generation fails
        """
        b = self._get_baml_client()

        # Build input data for BAML
        signal_data = self._signal_to_baml_input(signal)
        macro_context = self._macro_to_baml_input(macro_regime) if macro_regime else None
        sector_context = self._build_sector_context(signal)

        try:
            # Fetch and summarize news (optional)
            news_signals = await self._fetch_and_summarize_news(signal.ticker, signal.yfinance_ticker)

            # Generate view using BAML
            from baml_client.types import StockSignalData, MacroRegimeContext, SectorContext

            baml_result = b.GenerateBlackLittermanView(
                signal=signal_data,
                macro_regime=macro_context,
                sector_context=sector_context,
                news_signals=news_signals,
            )

            # Convert to DTO
            view = BlackLittermanViewDTO(
                ticker=signal.ticker,
                expected_return=baml_result.expected_return * self._view_scaling,
                confidence=baml_result.confidence,
                view_uncertainty=baml_result.view_uncertainty,
                rationale=baml_result.rationale,
                view_type="absolute",
                sector=signal.sector,
                generated_at=datetime.now(),
                macro_regime=macro_regime.regime.value if macro_regime else None,
                news_bias=getattr(news_signals, 'news_bias', None) if news_signals else None,
            )

            self._logger.debug(
                f"View for {signal.ticker}: return={view.expected_return:.2%}, "
                f"confidence={view.confidence:.2f}"
            )

            return view

        except Exception as e:
            self._logger.error(f"BAML view generation failed for {signal.ticker}: {e}")
            return self._generate_fallback_view(signal)

    def _generate_fallback_view(
        self,
        signal: StockSignalDTO,
    ) -> BlackLittermanViewDTO:
        """
        Generate a fallback view when BAML fails.

        Uses simple heuristics based on signal metrics.
        """
        # Base return on signal type
        base_returns = {
            "large_gain": 0.12,
            "small_gain": 0.08,
            "neutral": 0.05,
            "small_decline": 0.02,
            "large_decline": -0.02,
        }
        base_return = base_returns.get(signal.signal_type.value, 0.05)

        # Adjust by Sharpe ratio if available
        if signal.sharpe_ratio is not None:
            sharpe_adjustment = (signal.sharpe_ratio - 0.5) * 0.02
            base_return += sharpe_adjustment

        # Confidence based on data quality
        confidence = 0.5
        if signal.data_quality_score is not None:
            confidence = min(0.7, signal.data_quality_score)

        # Uncertainty inversely related to confidence
        uncertainty = 0.10 * (2 - confidence)

        return BlackLittermanViewDTO(
            ticker=signal.ticker,
            expected_return=base_return * self._view_scaling,
            confidence=confidence,
            view_uncertainty=uncertainty,
            rationale="Fallback view based on signal metrics",
            view_type="absolute",
            sector=signal.sector,
            generated_at=datetime.now(),
        )

    def _signal_to_baml_input(self, signal: StockSignalDTO) -> Dict[str, Any]:
        """Convert signal DTO to BAML input format."""
        return {
            "ticker": signal.ticker,
            "yfinance_ticker": signal.yfinance_ticker or signal.ticker,
            "sector": signal.sector,
            "industry": signal.industry,
            "signal_type": signal.signal_type.value.upper(),
            "confidence_level": signal.confidence_level.value.upper() if signal.confidence_level else "MEDIUM",
            "valuation_score": signal.valuation_score,
            "momentum_score": signal.momentum_score,
            "quality_score": signal.quality_score,
            "growth_score": signal.growth_score,
            "technical_score": signal.technical_score,
            "annualized_return": signal.annualized_return,
            "volatility": signal.volatility,
            "sharpe_ratio": signal.sharpe_ratio,
            "beta": signal.beta,
            "alpha": signal.alpha,
            "upside_potential_pct": signal.upside_potential_pct,
            "downside_risk_pct": signal.downside_risk_pct,
            "close_price": signal.close_price,
            "daily_return": signal.daily_return,
        }

    def _macro_to_baml_input(self, regime: MacroRegimeDTO) -> Dict[str, Any]:
        """Convert macro regime DTO to BAML input format."""
        return {
            "current_regime": regime.regime.value.upper(),
            "regime_confidence": regime.confidence,
            "recession_risk_6m": regime.recession_risk_6m,
            "recession_risk_12m": regime.recession_risk_12m,
            "vix": regime.vix,
            "hy_spread": regime.hy_spread,
            "yield_curve_2s10s": regime.yield_curve_2s10s,
            "recommended_overweights": regime.recommended_overweights,
            "recommended_underweights": regime.recommended_underweights,
        }

    def _build_sector_context(self, signal: StockSignalDTO) -> Dict[str, Any]:
        """Build sector context for BAML."""
        return {
            "sector_name": signal.sector or "Unknown",
            "sector_momentum": "moderate",  # Would be calculated from sector-wide data
            "sector_valuation": "fair",
        }

    async def _fetch_and_summarize_news(
        self,
        ticker: str,
        yfinance_ticker: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Fetch and summarize news for a stock.

        Returns BAML StockNewsSignals or None if unavailable.
        """
        try:
            from src.yfinance.client import YFinanceClient
            from baml_client.types import NewsArticle

            yf_client = YFinanceClient.get_instance()
            news_data = yf_client.fetch_news(yfinance_ticker or ticker, fetch_full_content=True)

            if not news_data:
                return None

            # Limit to recent articles
            news_data = news_data[:15]

            # Convert to BAML format
            news_articles = []
            for article in news_data:
                content = article.get('content', {})
                provider = content.get('provider', {})
                publisher = provider.get('displayName', '') if isinstance(provider, dict) else str(provider)

                canonical_url = content.get('canonicalUrl', '')
                link = canonical_url.get('url', '') if isinstance(canonical_url, dict) else str(canonical_url)

                news_article = NewsArticle(
                    title=content.get('title', ''),
                    publisher=publisher,
                    date=str(content.get('pubDate', '')),
                    link=link,
                    summary=content.get('summary', ''),
                    full_content=article.get('full_content'),
                )
                news_articles.append(news_article)

            # Summarize using BAML
            b = self._get_baml_client()
            news_signals = b.SummarizeStockNews(
                ticker=ticker,
                news=news_articles
            )

            return news_signals

        except Exception as e:
            self._logger.debug(f"News fetch/summarize failed for {ticker}: {e}")
            return None

    def construct_matrices(
        self,
        views: List[BlackLittermanViewDTO],
        universe_tickers: List[str],
    ) -> Tuple[Any, Any, Any]:
        """
        Construct P, Q, Omega matrices from views.

        Delegates to ViewMatrixBuilder.
        """
        from src.black_litterman.views.matrix_builder import ViewMatrixBuilder

        builder = ViewMatrixBuilder()
        return builder.construct(views, universe_tickers)
