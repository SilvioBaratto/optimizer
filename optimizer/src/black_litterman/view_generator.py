#!/usr/bin/env python3
"""
Black-Litterman View Generator
===============================

Converts stock signals from the database into structured Black-Litterman views
using BAML AI-driven analysis. Implements the view generation pipeline from Chapter 4.

Flow:
1. Fetch stock signals from database (stock_signals table)
2. Fetch macro regime context (country_regime_assessments, market_indicators)
3. Generate views using BAML GenerateBlackLittermanView function
4. Construct P, Q, Omega matrices for Black-Litterman optimization

Usage:
    from src.black_litterman.view_generator import ViewGenerator

    vg = ViewGenerator()
    views = vg.generate_views(signal_date='2025-10-28')
    P, Q, Omega = vg.construct_matrices(views, universe_tickers)
"""

import logging
import json
from datetime import date as date_type, datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

from app.database import database_manager
from app.models.stock_signals import StockSignal
from app.models.macro_regime import CountryRegimeAssessment, MarketIndicators
from baml_client import b
from baml_client.types import (
    StockSignalData,
    MacroRegimeContext,
    SectorContext,
    BlackLittermanView
)

logger = logging.getLogger(__name__)


class ViewGenerator:
    """
    Generate Black-Litterman views from stock signals using BAML AI analysis.
    """

    def __init__(self):
        """Initialize view generator."""
        database_manager.initialize()
        logger.info("ViewGenerator initialized")

    def fetch_signals_from_db(
        self,
        signal_date: Optional[date_type] = None,
        signal_types: Optional[List[str]] = None,
        confidence_levels: Optional[List[str]] = None,
        min_confidence: Optional[str] = 'MEDIUM'
    ) -> List[StockSignal]:
        """
        Fetch stock signals from database for view generation.

        Args:
            signal_date: Date to fetch signals for (defaults to latest)
            signal_types: Filter by signal types (defaults to ['LARGE_GAIN', 'SMALL_GAIN'])
            confidence_levels: Filter by confidence (defaults to ['HIGH', 'MEDIUM'])
            min_confidence: Minimum confidence level to include

        Returns:
            List of StockSignal objects
        """
        if signal_types is None:
            signal_types = ['LARGE_GAIN', 'SMALL_GAIN']

        if confidence_levels is None:
            confidence_levels = ['HIGH', 'MEDIUM']

        with database_manager.get_session() as session:
            query = session.query(StockSignal)

            # Filter by date
            if signal_date:
                query = query.filter(StockSignal.signal_date == signal_date)
            else:
                # Get latest signal date
                latest_date = session.query(StockSignal.signal_date).order_by(
                    StockSignal.signal_date.desc()
                ).first()
                if latest_date:
                    query = query.filter(StockSignal.signal_date == latest_date[0])

            # Filter by signal type
            query = query.filter(StockSignal.signal_type.in_(signal_types))

            # Filter by confidence
            query = query.filter(StockSignal.confidence_level.in_(confidence_levels))

            signals = query.all()

            logger.info(
                f"Fetched {len(signals)} signals from database "
                f"(date={signal_date or 'latest'}, types={signal_types}, confidence={confidence_levels})"
            )

            return signals

    def fetch_macro_regime(self, country: str = 'USA') -> Optional[MacroRegimeContext]:
        """
        Fetch latest macro regime data from database.

        Args:
            country: Country to fetch regime for

        Returns:
            MacroRegimeContext object or None
        """
        with database_manager.get_session() as session:
            # Get latest regime assessment
            regime = session.query(CountryRegimeAssessment).filter(
                CountryRegimeAssessment.country == country
            ).order_by(
                CountryRegimeAssessment.assessment_timestamp.desc()
            ).first()

            if not regime:
                logger.warning(f"No regime assessment found for {country}")
                return None

            # Get latest market indicators
            indicators = session.query(MarketIndicators).order_by(
                MarketIndicators.data_timestamp.desc()
            ).first()

            # Parse recommended overweights/underweights
            overweights: list[str] = []
            underweights: list[str] = []
            if regime.recommended_overweights:
                # Parse the list stored as JSON string or use directly if already a list
                try:
                    if isinstance(regime.recommended_overweights, str):
                        overweights = json.loads(regime.recommended_overweights)
                    elif isinstance(regime.recommended_overweights, list):
                        overweights = regime.recommended_overweights
                except:
                    pass

            if regime.recommended_underweights:
                try:
                    if isinstance(regime.recommended_underweights, str):
                        underweights = json.loads(regime.recommended_underweights)
                    elif isinstance(regime.recommended_underweights, list):
                        underweights = regime.recommended_underweights
                except:
                    pass

            macro_context = MacroRegimeContext(
                current_regime=getattr(regime.regime, 'value', str(regime.regime)),
                regime_confidence=float(regime.confidence) if regime.confidence else 0.5,
                recession_risk_6m=float(regime.recession_risk_6m) if regime.recession_risk_6m else None,
                recession_risk_12m=float(regime.recession_risk_12m) if regime.recession_risk_12m else None,
                vix=float(indicators.vix) if indicators and indicators.vix else None,
                hy_spread=float(indicators.hy_spread) if indicators and indicators.hy_spread else None,
                yield_curve_2s10s=float(indicators.yield_curve_2s10s) if indicators and indicators.yield_curve_2s10s else None,
                recommended_overweights=overweights,
                recommended_underweights=underweights
            )

            logger.info(
                f"Fetched macro regime for {country}: {macro_context.current_regime} "
                f"(confidence={macro_context.regime_confidence:.2f})"
            )

            return macro_context

    def build_sector_context(
        self,
        sector: str,
        signals: List[StockSignal]
    ) -> SectorContext:
        """
        Build sector context from available signals.

        Args:
            sector: Sector name
            signals: All signals for this batch (to calculate sector stats)

        Returns:
            SectorContext object
        """
        # Count signals in this sector
        sector_signals = [s for s in signals if s.sector == sector]

        # Calculate average expected return for sector
        avg_return = None
        if sector_signals:
            returns = [s.upside_potential_pct for s in sector_signals if s.upside_potential_pct]
            if returns:
                avg_return = float(np.mean(returns)) / 100  # Convert to decimal

        # Determine sector momentum (based on avg momentum scores)
        momentum_scores = [s.momentum_score for s in sector_signals if s.momentum_score is not None]
        sector_momentum = "moderate"
        if momentum_scores:
            avg_mom = np.mean(momentum_scores)
            if avg_mom > 0.5:
                sector_momentum = "strong"
            elif avg_mom > 0.2:
                sector_momentum = "moderate"
            elif avg_mom > -0.2:
                sector_momentum = "weak"
            else:
                sector_momentum = "negative"

        # Determine sector valuation (based on avg valuation scores)
        val_scores = [s.valuation_score for s in sector_signals if s.valuation_score is not None]
        sector_valuation = "fair"
        if val_scores:
            avg_val = np.mean(val_scores)
            if avg_val > 0.3:
                sector_valuation = "cheap"
            elif avg_val < -0.3:
                sector_valuation = "expensive"

        return SectorContext(
            sector_name=sector,
            avg_sector_return=avg_return,
            sector_momentum=sector_momentum,
            sector_valuation=sector_valuation,
            num_signals_in_sector=len(sector_signals)
        )

    async def _generate_single_view(
        self,
        signal: StockSignal,
        instrument,
        macro_regime: MacroRegimeContext,
        sector_context: SectorContext
    ) -> Optional[BlackLittermanView]:
        """
        Generate a single Black-Litterman view for one stock.

        Enhanced with news summarization similar to macro regime classification.

        Process:
        1. Fetch recent news for the stock
        2. Summarize news using BAML SummarizeStockNews
        3. Generate Black-Litterman view with news context

        Args:
            signal: StockSignal object
            instrument: Instrument object
            macro_regime: Macro regime context
            sector_context: Sector context

        Returns:
            BlackLittermanView or None if generation fails
        """
        try:
            # Convert signal to BAML input type
            # Ensure ticker is not None
            ticker = signal.ticker or (instrument.ticker if instrument else "UNKNOWN")
            yfinance_ticker = signal.yfinance_ticker or (instrument.yfinance_ticker if instrument else ticker)

            signal_data = StockSignalData(
                ticker=ticker,
                yfinance_ticker=yfinance_ticker,
                sector=signal.sector,
                industry=signal.industry,
                signal_type=signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                confidence_level=signal.confidence_level.value if signal.confidence_level and hasattr(signal.confidence_level, 'value') else "MEDIUM",
                valuation_score=float(signal.valuation_score) if signal.valuation_score else None,
                momentum_score=float(signal.momentum_score) if signal.momentum_score else None,
                quality_score=float(signal.quality_score) if signal.quality_score else None,
                growth_score=float(signal.growth_score) if signal.growth_score else None,
                technical_score=float(signal.technical_score) if signal.technical_score else None,
                annualized_return=float(signal.annualized_return) if signal.annualized_return else None,
                volatility=float(signal.volatility) if signal.volatility else None,
                sharpe_ratio=float(signal.sharpe_ratio) if signal.sharpe_ratio else None,
                beta=float(signal.beta) if signal.beta else None,
                alpha=float(signal.alpha) if signal.alpha else None,
                volatility_level=signal.volatility_level.value if signal.volatility_level else None,
                beta_risk=signal.beta_risk.value if signal.beta_risk else None,
                debt_risk=signal.debt_risk.value if signal.debt_risk else None,
                liquidity_risk=signal.liquidity_risk.value if signal.liquidity_risk else None,
                upside_potential_pct=float(signal.upside_potential_pct) if signal.upside_potential_pct else None,
                downside_risk_pct=float(signal.downside_risk_pct) if signal.downside_risk_pct else None,
                close_price=float(signal.close_price) if signal.close_price else None,
                daily_return=float(signal.daily_return) if signal.daily_return else None
            )

            # STEP 1: Fetch recent news for the stock
            news_signals = None
            try:
                from src.yfinance.client import YFinanceClient
                from baml_client.types import NewsArticle

                yf_client = YFinanceClient.get_instance()
                news_data = yf_client.fetch_news(yfinance_ticker, fetch_full_content=True)

                if news_data and len(news_data) > 0:
                    # Limit to most recent 15 articles to keep context manageable
                    news_data = news_data[:15]

                    logger.debug(f"Fetched {len(news_data)} news articles for {ticker}")

                    # Convert to BAML NewsArticle objects
                    # Note: news_data structure is nested:
                    # - article['content'] contains metadata (title, publisher, etc.)
                    # - article['full_content'] contains full article text
                    news_articles = []
                    for article in news_data:
                        content = article.get('content', {})

                        # Extract publisher (provider is a dict with displayName)
                        provider = content.get('provider', {})
                        if isinstance(provider, dict):
                            publisher = provider.get('displayName', '')
                        else:
                            publisher = str(provider) if provider else ''

                        # Extract link (canonicalUrl might be a dict with url)
                        canonical_url = content.get('canonicalUrl', '')
                        if isinstance(canonical_url, dict):
                            link = canonical_url.get('url', '')
                        else:
                            link = str(canonical_url) if canonical_url else ''

                        # Extract date (convert to string if needed)
                        pub_date = content.get('pubDate', '')
                        if not isinstance(pub_date, str):
                            pub_date = str(pub_date) if pub_date else ''

                        # Create NewsArticle object
                        news_article = NewsArticle(
                            title=content.get('title', ''),
                            publisher=publisher,
                            date=pub_date,
                            link=link,
                            summary=content.get('summary', ''),
                            full_content=article.get('full_content')  # Full article text from web scraping
                        )
                        news_articles.append(news_article)

                    logger.debug(f"Converted {len(news_articles)} articles to BAML format for {ticker}")

                    # STEP 2: Summarize news using BAML
                    logger.debug(f"Summarizing {len(news_articles)} articles for {ticker}...")
                    news_signals = b.SummarizeStockNews(
                        ticker=ticker,
                        news=news_articles
                    )
                    logger.debug(f"News summary complete for {ticker}: {news_signals.news_bias} bias")

                else:
                    logger.debug(f"No news articles found for {ticker}, proceeding without news context")

            except Exception as news_error:
                logger.warning(f"Failed to fetch/summarize news for {ticker}: {news_error}")
                # Continue without news context

            # STEP 3: Generate view using BAML with news context
            view = b.GenerateBlackLittermanView(
                signal=signal_data,
                macro_regime=macro_regime,
                sector_context=sector_context,
                news_signals=news_signals  # Optional, can be None
            )

            logger.debug(
                f"Generated view for {signal.ticker}: "
                f"return={view.expected_return:.2%}, confidence={view.confidence:.2f}"
            )

            return view

        except Exception as e:
            logger.error(f"Error generating view for {signal.ticker}: {e}")
            return None

    async def generate_views(
        self,
        signal_date: Optional[date_type] = None,
        country: str = 'USA'
    ) -> List[Tuple[StockSignal, BlackLittermanView]]:
        """
        Generate Black-Litterman views for all qualifying signals.

        Args:
            signal_date: Date to generate views for (defaults to latest)
            country: Country for macro regime context

        Returns:
            List of tuples (StockSignal, BlackLittermanView)
        """
        # Fetch signals
        signals = self.fetch_signals_from_db(signal_date=signal_date)

        if not signals:
            logger.warning("No signals found, cannot generate views")
            return []

        # Fetch macro regime
        macro_regime = self.fetch_macro_regime(country=country)

        if not macro_regime:
            logger.warning("No macro regime found, using defaults")
            macro_regime = MacroRegimeContext(
                current_regime='UNCERTAIN',
                regime_confidence=0.5
            )

        # Generate views using BAML
        views = []
        for signal in signals:
            # Build sector context
            sector_context = self.build_sector_context(
                signal.sector or "Unknown",
                signals
            )

            # Generate single view
            view = await self._generate_single_view(
                signal,
                None,  # We don't have instrument in this context
                macro_regime,
                sector_context
            )

            if view:
                views.append((signal, view))

        logger.info(f"Generated {len(views)} views from {len(signals)} signals")
        return views

    def construct_matrices(
        self,
        views: List[Tuple[StockSignal, BlackLittermanView]],
        universe_tickers: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct P, Q, Omega matrices from views for Black-Litterman.

        Args:
            views: List of (StockSignal, BlackLittermanView) tuples
            universe_tickers: Complete list of tickers in portfolio universe

        Returns:
            Tuple of (P, Q, Omega) matrices:
                - P: KxN picking matrix (which assets each view concerns)
                - Q: Kx1 expected returns vector
                - Omega: KxK view uncertainty matrix (diagonal)
        """
        N = len(universe_tickers)
        K = len(views)

        if K == 0:
            logger.warning("No views to construct matrices from")
            return np.zeros((0, N)), np.zeros((0, 1)), np.zeros((0, 0))

        # Create ticker index mapping
        ticker_index = {ticker: i for i, ticker in enumerate(universe_tickers)}

        # Initialize matrices
        P = np.zeros((K, N))
        Q = np.zeros((K, 1))
        Omega = np.zeros((K, K))

        # Fill matrices
        for k, (signal, view) in enumerate(views):
            # Get ticker index
            ticker = signal.yfinance_ticker
            if ticker not in ticker_index:
                logger.warning(f"Ticker {ticker} not in universe, skipping view")
                continue

            idx = ticker_index[ticker]

            # P matrix: absolute view (1 for this asset)
            P[k, idx] = 1.0

            # Q matrix: expected return
            Q[k, 0] = view.expected_return

            # Omega matrix: view uncertainty variance (diagonal element)
            Omega[k, k] = view.view_uncertainty ** 2

        logger.info(
            f"Constructed BL matrices: P({K}x{N}), Q({K}x1), Omega({K}x{K})"
        )

        return P, Q, Omega

    def construct_basket_views(
        self,
        views: List[Tuple[StockSignal, BlackLittermanView]],
        universe_tickers: List[str],
        groupby: str = 'sector'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct basket views grouped by sector or industry.

        Instead of individual stock views, creates views on equal-weighted
        baskets (e.g., "Technology sector will return 12%").

        Args:
            views: List of (StockSignal, BlackLittermanView) tuples
            universe_tickers: Complete list of tickers in portfolio universe
            groupby: 'sector' or 'industry' for grouping

        Returns:
            Tuple of (P, Q, Omega) matrices for basket views
        """
        # Group views by sector/industry
        groups = defaultdict(list)
        for signal, view in views:
            if groupby == 'sector':
                key = signal.sector or "Unknown"
            else:
                key = signal.industry or "Unknown"
            groups[key].append((signal, view))

        N = len(universe_tickers)
        K = len(groups)

        logger.info(f"Creating {K} basket views grouped by {groupby}")

        # Create ticker index mapping
        ticker_index = {ticker: i for i, ticker in enumerate(universe_tickers)}

        # Initialize matrices
        P = np.zeros((K, N))
        Q = np.zeros((K, 1))
        Omega = np.zeros((K, K))

        # Fill matrices
        for k, (group_name, group_views) in enumerate(groups.items()):
            # Get tickers in this group
            group_tickers = [
                signal.yfinance_ticker
                for signal, _ in group_views
                if signal.yfinance_ticker in ticker_index
            ]

            if not group_tickers:
                continue

            n_stocks = len(group_tickers)

            # P matrix: equal-weighted basket
            for ticker in group_tickers:
                idx = ticker_index[ticker]
                P[k, idx] = 1.0 / n_stocks

            # Q matrix: average expected return
            avg_return = np.mean([view.expected_return for _, view in group_views])
            Q[k, 0] = avg_return

            # Omega matrix: average uncertainty (scaled by sqrt(n) for basket)
            avg_uncertainty = np.mean([view.view_uncertainty for _, view in group_views])
            basket_uncertainty = avg_uncertainty / np.sqrt(n_stocks)
            Omega[k, k] = basket_uncertainty ** 2

            logger.debug(
                f"Basket view for {group_name}: {n_stocks} stocks, "
                f"return={avg_return:.2%}, uncertainty={basket_uncertainty:.2%}"
            )

        return P, Q, Omega

    def summary_stats(self, views: List[Tuple[StockSignal, BlackLittermanView]]) -> Dict:
        """
        Calculate summary statistics for generated views.

        Args:
            views: List of (StockSignal, BlackLittermanView) tuples

        Returns:
            Dictionary of summary statistics
        """
        if not views:
            return {}

        returns = [view.expected_return for _, view in views]
        confidences = [view.confidence for _, view in views]
        uncertainties = [view.view_uncertainty for _, view in views]

        # Group by sector
        sectors: dict[str, list[float]] = {}
        for signal, view in views:
            sector = signal.sector or "Unknown"
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(view.expected_return)

        stats = {
            'num_views': len(views),
            'avg_expected_return': np.mean(returns),
            'median_expected_return': np.median(returns),
            'min_expected_return': np.min(returns),
            'max_expected_return': np.max(returns),
            'avg_confidence': np.mean(confidences),
            'avg_uncertainty': np.mean(uncertainties),
            'sectors': {
                sector: {
                    'count': len(rets),
                    'avg_return': np.mean(rets)
                }
                for sector, rets in sectors.items()
            }
        }

        return stats


if __name__ == "__main__":
    # Example usage
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        vg = ViewGenerator()

        # Generate views
        views = await vg.generate_views()

        if views:
            # Print summary
            stats = vg.summary_stats(views)
            print(f"\nGenerated {stats['num_views']} views:")
            print(f"  Avg expected return: {stats['avg_expected_return']:.2%}")
            print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
            print(f"  Sectors: {len(stats['sectors'])}")

            # Show a few examples
            print("\nExample views:")
            for i, (signal, view) in enumerate(views[:3]):
                print(f"\n{i+1}. {signal.ticker} ({signal.sector}):")
                print(f"   Expected return: {view.expected_return:.2%}")
                print(f"   Confidence: {view.confidence:.2f}")
                print(f"   Rationale: {view.rationale[:100]}...")

    asyncio.run(main())
