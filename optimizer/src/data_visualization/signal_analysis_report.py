#!/usr/bin/env python3
"""
Signal Analysis Report - Text-Based Summary
============================================
Generates a comprehensive text report analyzing stock signals:
1. Proportion of LARGE_GAIN signals in overall dataset
2. Industry/sector diversification
3. Company names and business descriptions
4. Detailed breakdown by signal type

Usage:
    python src/data_visualization/signal_analysis_report.py

Output:
    Creates a text file: signal_analysis_report_YYYY-MM-DD.txt
"""

import sys
import logging
from datetime import date as date_type, datetime
from typing import Dict, List, Tuple
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from dotenv import load_dotenv

load_dotenv()

from optimizer.src.yfinance import YFinanceClient

# Import database and models
from optimizer.database.database import database_manager, init_db
from optimizer.database.models.stock_signals import StockSignal, SignalEnum
from optimizer.database.models.universe import Instrument
from optimizer.database.models.macro_regime import (
    CountryRegimeAssessment,
    MacroAnalysisRun,
    MarketIndicators,
    EconomicIndicators,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SignalAnalysisReporter:
    """
    Generates text-based analysis reports for stock signals.
    """

    def __init__(self, signal_date: date_type | None = None):
        """
        Initialize reporter.

        Args:
            signal_date: Date to analyze (defaults to today)
        """
        self.signal_date = signal_date or date_type.today()
        self.all_signals = []
        self.large_gain_stocks = []
        self.signal_distribution = {}
        self.sector_distribution = {}
        self.report_lines = []
        # Track unique instruments to avoid duplicates
        self.seen_instruments = {}  # instrument_id -> signal_data
        self.distinct_large_gain_stocks = []
        # Additional distributions for enhanced reporting
        self.country_distribution = {}
        self.industry_distribution = {}
        self.confidence_distribution = {}
        self.exchange_distribution = {}
        # Macro regime data by country
        self.macro_data_by_country = {}

    def fetch_large_gain_signals(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Fetch ONLY LARGE_GAIN signals for the target date with DISTINCT instruments.

        Much more efficient than fetching all signals and filtering in Python.

        Returns:
            List of tuples (StockSignal, Instrument) with distinct instruments
        """
        logger.info(f"Fetching LARGE_GAIN signals for {self.signal_date}")

        with database_manager.get_session() as session:
            # Fetch ONLY LARGE_GAIN signals (filter at database level)
            query = (
                select(StockSignal)
                .where(
                    StockSignal.signal_date == self.signal_date,
                    StockSignal.signal_type == SignalEnum.LARGE_GAIN,
                )
                .options(
                    joinedload(StockSignal.instrument).joinedload(Instrument.exchange)
                )
                .order_by(StockSignal.close_price.desc())
            )

            signals = session.execute(query).scalars().all()

            if not signals:
                logger.warning(f"No LARGE_GAIN signals found for {self.signal_date}")
                return []

            results = [(signal, signal.instrument) for signal in signals]
            logger.info(f"Found {len(results)} LARGE_GAIN signals")

            return results

    def get_signal_statistics(self) -> dict:
        """
        Get overall signal statistics for the date (count by signal type).

        Returns:
            Dictionary with signal type counts
        """
        logger.info(f"Fetching signal statistics for {self.signal_date}")

        with database_manager.get_session() as session:
            # Count signals by type
            from sqlalchemy import func

            query = (
                select(
                    StockSignal.signal_type, func.count(StockSignal.id).label("count")
                )
                .where(StockSignal.signal_date == self.signal_date)
                .group_by(StockSignal.signal_type)
            )

            results = session.execute(query).all()

            stats = {signal_type.value: count for signal_type, count in results}

            logger.info(f"Signal statistics: {stats}")

            return stats

    def fetch_company_info(self, yf_ticker: str) -> Dict[str, str]:
        """
        Fetch company information from yfinance.

        Args:
            yf_ticker: Yahoo Finance ticker symbol

        Returns:
            Dictionary with company name, sector, industry, and description
        """
        try:
            client = YFinanceClient.get_instance()
            info = client.fetch_info(yf_ticker)

            if info is None:
                return {
                    "name": "Unknown",
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "description": "No description available",
                    "country": "Unknown",
                }

            return {
                "name": info.get("longName", info.get("shortName", "Unknown")),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "description": info.get(
                    "longBusinessSummary", "No description available"
                ),
                "country": info.get("country", "Unknown"),
            }
        except Exception as e:
            logger.warning(f"Could not fetch company info for {yf_ticker}: {e}")
            return {
                "name": "Unknown",
                "sector": "Unknown",
                "industry": "Unknown",
                "description": "No description available",
                "country": "Unknown",
            }

    def fetch_macro_regime_data(self, countries: List[str]) -> Dict[str, Dict]:
        """
        Fetch latest macro regime assessment for countries in the portfolio.

        Args:
            countries: List of country codes

        Returns:
            Dictionary mapping country -> macro data
        """
        logger.info(f"Fetching macro regime data for countries: {countries}")

        macro_data = {}

        with database_manager.get_session() as session:
            for country in countries:
                # Get latest assessment for this country
                query = (
                    select(CountryRegimeAssessment)
                    .where(CountryRegimeAssessment.country == country)
                    .options(
                        joinedload(CountryRegimeAssessment.analysis_run).joinedload(
                            MacroAnalysisRun.market_indicators
                        ),
                        joinedload(CountryRegimeAssessment.economic_indicators),
                    )
                    .order_by(CountryRegimeAssessment.assessment_timestamp.desc())
                    .limit(1)
                )

                result = session.execute(query)
                assessment = result.scalar_one_or_none()

                if assessment:
                    # Get market and economic indicators
                    market_indicators = (
                        assessment.analysis_run.market_indicators
                        if assessment.analysis_run
                        else None
                    )
                    econ_indicators = assessment.economic_indicators

                    macro_data[country] = {
                        "regime": (
                            assessment.regime
                            if isinstance(assessment.regime, str)
                            else assessment.regime.value
                        ),
                        "confidence": float(assessment.confidence),
                        "rationale": assessment.rationale,
                        "assessment_date": assessment.assessment_timestamp.date().isoformat(),
                        # Recession risk
                        "recession_risk_6m": float(assessment.recession_risk_6m),
                        "recession_risk_12m": float(assessment.recession_risk_12m),
                        "recession_drivers": assessment.recession_drivers or [],
                        # Economic indicators
                        "gdp_growth_yy": (
                            float(econ_indicators.gdp_growth_yy)
                            if econ_indicators and econ_indicators.gdp_growth_yy
                            else None
                        ),
                        "unemployment": (
                            float(econ_indicators.unemployment)
                            if econ_indicators and econ_indicators.unemployment
                            else None
                        ),
                        "inflation": (
                            float(econ_indicators.inflation)
                            if econ_indicators and econ_indicators.inflation
                            else None
                        ),
                        # Market indicators
                        "ism_pmi": (
                            float(market_indicators.ism_pmi)
                            if market_indicators and market_indicators.ism_pmi
                            else None
                        ),
                        "ism_signal": (
                            assessment.ism_signal
                            if isinstance(assessment.ism_signal, str)
                            else assessment.ism_signal.value
                        ),
                        "yield_curve_2s10s": (
                            float(market_indicators.yield_curve_2s10s)
                            if market_indicators and market_indicators.yield_curve_2s10s
                            else None
                        ),
                        "yield_curve_signal": (
                            assessment.yield_curve_signal
                            if isinstance(assessment.yield_curve_signal, str)
                            else assessment.yield_curve_signal.value
                        ),
                        "credit_spread_signal": (
                            assessment.credit_spread_signal
                            if isinstance(assessment.credit_spread_signal, str)
                            else assessment.credit_spread_signal.value
                        ),
                        # Portfolio positioning
                        "sector_tilts": assessment.sector_tilts or {},
                        "recommended_overweights": assessment.recommended_overweights
                        or [],
                        "recommended_underweights": assessment.recommended_underweights
                        or [],
                        "factor_exposure": (
                            assessment.factor_exposure
                            if isinstance(assessment.factor_exposure, str)
                            else (
                                assessment.factor_exposure.value
                                if assessment.factor_exposure
                                else None
                            )
                        ),
                        # Risks
                        "primary_risks": assessment.primary_risks or [],
                    }

                    logger.info(
                        f"Loaded macro data for {country}: {macro_data[country]['regime']}"
                    )
                else:
                    logger.warning(f"No macro regime data found for {country}")

        return macro_data

    def analyze_signals(self) -> None:
        """
        Fetch and analyze LARGE_GAIN signals only.
        Tracks DISTINCT stocks by instrument_id to avoid duplicates.

        Much more efficient: filters at database level instead of fetching all signals.
        """
        # First, get overall signal statistics (fast aggregation query)
        self.signal_distribution = self.get_signal_statistics()

        # Now fetch ONLY LARGE_GAIN signals with instruments
        signals_and_instruments = self.fetch_large_gain_signals()

        if not signals_and_instruments:
            logger.warning("No LARGE_GAIN signals to analyze")
            return

        # Track unique instruments (in case same instrument appears multiple times)
        large_gain_by_instrument = {}  # instrument_id -> signal_data
        sectors = []
        countries = []
        industries = []
        confidences = []
        exchanges = []

        for signal, instrument in signals_and_instruments:
            if not instrument.yfinance_ticker:
                logger.debug(f"Skipping {instrument.ticker} - no yfinance ticker")
                continue

            instrument_id = str(instrument.id)

            # Check if we've already processed this instrument
            if instrument_id in large_gain_by_instrument:
                # Skip duplicate - same instrument, already processed
                logger.debug(
                    f"Skipping duplicate instrument: {instrument.yfinance_ticker}"
                )
                continue

            # Fetch company info (only once per unique instrument)
            logger.info(f"Fetching company info for {instrument.yfinance_ticker}")
            company_info = self.fetch_company_info(instrument.yfinance_ticker)

            # Store signal data (use denormalized fields from signal, fallback to instrument)
            signal_data = {
                "instrument_id": instrument_id,
                "ticker": signal.ticker
                or instrument.short_name
                or instrument.yfinance_ticker,
                "yf_ticker": signal.yfinance_ticker or instrument.yfinance_ticker,
                "company_name": company_info["name"],
                # Use denormalized sector/industry from signal if available, otherwise from yfinance
                "sector": signal.sector or company_info["sector"],
                "industry": signal.industry or company_info["industry"],
                "description": company_info["description"],
                "country": company_info["country"],
                "exchange": signal.exchange_name
                or (
                    instrument.exchange.exchange_name
                    if instrument.exchange
                    else "Unknown"
                ),
                "signal_type": signal.signal_type.value,
                "confidence": (
                    signal.confidence_level.value if signal.confidence_level else "N/A"
                ),
                "close_price": signal.close_price,
                "upside_pct": signal.upside_potential_pct,
                "data_quality": signal.data_quality_score,
                "analysis_notes": getattr(
                    signal, "analysis_notes", None
                ),  # Optional field
            }

            # Add to unique LARGE_GAIN stocks
            large_gain_by_instrument[instrument_id] = signal_data

            # Track distributions for this unique stock
            if company_info["sector"] != "Unknown":
                sectors.append(company_info["sector"])

            if company_info["country"] != "Unknown":
                countries.append(company_info["country"])

            if company_info["industry"] != "Unknown":
                industries.append(company_info["industry"])

            if signal.confidence_level:
                confidences.append(signal.confidence_level.value)

            if instrument.exchange:
                exchanges.append(instrument.exchange.exchange_name)

        # Store distinct LARGE_GAIN stocks
        self.distinct_large_gain_stocks = list(large_gain_by_instrument.values())

        # Calculate distributions for LARGE_GAIN only
        self.sector_distribution = dict(Counter(sectors))
        self.country_distribution = dict(Counter(countries))
        self.industry_distribution = dict(Counter(industries))
        self.confidence_distribution = dict(Counter(confidences))
        self.exchange_distribution = dict(Counter(exchanges))

        logger.info(
            f"Analyzed {len(self.distinct_large_gain_stocks)} DISTINCT LARGE_GAIN stocks, "
            f"{len(self.sector_distribution)} unique sectors, "
            f"{len(self.country_distribution)} countries"
        )

        # Fetch macro regime data for countries in portfolio
        if countries:
            unique_countries = list(set(countries))
            self.macro_data_by_country = self.fetch_macro_regime_data(unique_countries)
            logger.info(
                f"Fetched macro data for {len(self.macro_data_by_country)} countries"
            )

    def generate_report(self) -> str:
        """
        Generate comprehensive text report.

        Returns:
            Formatted report as string
        """
        lines = []

        # Header
        lines.append("=" * 100)
        lines.append(f"STOCK SIGNAL ANALYSIS REPORT")
        lines.append(f"Date: {self.signal_date}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 100)
        lines.append("")

        # Overall Statistics
        lines.append("=" * 100)
        lines.append("1. OVERALL SIGNAL DISTRIBUTION")
        lines.append("=" * 100)
        lines.append("")

        # Calculate total from signal distribution
        total_signals = sum(self.signal_distribution.values())

        if total_signals == 0:
            lines.append("No signals found for this date.")
            return "\n".join(lines)

        lines.append(f"Total Signals: {total_signals}")
        lines.append("")

        # Signal type breakdown
        lines.append("Signal Type Breakdown:")
        lines.append("-" * 80)

        # Define signal order for display
        signal_order = [
            "large_gain",
            "small_gain",
            "neutral",
            "small_decline",
            "large_decline",
        ]

        for signal_type in signal_order:
            count = self.signal_distribution.get(signal_type, 0)
            percentage = (count / total_signals) * 100 if total_signals > 0 else 0
            bar = "█" * int(percentage / 2)  # Visual bar (50 chars = 100%)
            lines.append(
                f"  {signal_type.upper():15} {count:4d} ({percentage:5.1f}%) {bar}"
            )

        lines.append("")

        # LARGE_GAIN proportion highlight
        large_gain_count = self.signal_distribution.get("large_gain", 0)
        large_gain_pct = (
            (large_gain_count / total_signals) * 100 if total_signals > 0 else 0
        )

        lines.append(
            f"⭐ LARGE_GAIN Proportion: {large_gain_count}/{total_signals} ({large_gain_pct:.1f}%)"
        )
        lines.append("")

        # LARGE_GAIN Sector Distribution (DISTINCT stocks only)
        if self.distinct_large_gain_stocks:
            lines.append("=" * 100)
            lines.append("2. LARGE_GAIN STOCKS - SECTOR DISTRIBUTION (DISTINCT STOCKS)")
            lines.append("=" * 100)
            lines.append("")

            lines.append(
                f"Total DISTINCT LARGE_GAIN Stocks: {len(self.distinct_large_gain_stocks)}"
            )
            lines.append(f"Sectors Represented: {len(self.sector_distribution)}")
            lines.append("")

            sorted_lg_sectors = sorted(
                self.sector_distribution.items(), key=lambda x: x[1], reverse=True
            )

            lines.append("Sector Distribution (DISTINCT LARGE_GAIN stocks only):")
            lines.append("-" * 80)
            for sector, count in sorted_lg_sectors:
                percentage = (count / len(self.distinct_large_gain_stocks)) * 100
                bar = "█" * int(percentage / 2)
                lines.append(f"  {sector:35} {count:4d} ({percentage:5.1f}%) {bar}")
            lines.append("")

        # Country/Exchange Distribution
        if self.distinct_large_gain_stocks:
            lines.append("=" * 100)
            lines.append(
                "3. LARGE_GAIN STOCKS - GEOGRAPHIC DISTRIBUTION (DISTINCT STOCKS)"
            )
            lines.append("=" * 100)
            lines.append("")

            lines.append(f"Countries Represented: {len(self.country_distribution)}")
            lines.append(f"Exchanges Represented: {len(self.exchange_distribution)}")
            lines.append("")

            # Country distribution
            sorted_countries = sorted(
                self.country_distribution.items(), key=lambda x: x[1], reverse=True
            )

            lines.append("Country Distribution:")
            lines.append("-" * 80)
            for country, count in sorted_countries:
                percentage = (count / len(self.distinct_large_gain_stocks)) * 100
                bar = "█" * int(percentage / 2)
                lines.append(f"  {country:35} {count:4d} ({percentage:5.1f}%) {bar}")
            lines.append("")

            # Exchange distribution
            sorted_exchanges = sorted(
                self.exchange_distribution.items(), key=lambda x: x[1], reverse=True
            )

            lines.append("Exchange Distribution:")
            lines.append("-" * 80)
            for exchange, count in sorted_exchanges:
                percentage = (count / len(self.distinct_large_gain_stocks)) * 100
                bar = "█" * int(percentage / 2)
                lines.append(f"  {exchange:35} {count:4d} ({percentage:5.1f}%) {bar}")
            lines.append("")

        # Industry Distribution (Top 15)
        if self.distinct_large_gain_stocks and self.industry_distribution:
            lines.append("=" * 100)
            lines.append(
                "4. LARGE_GAIN STOCKS - INDUSTRY DISTRIBUTION (TOP 15, DISTINCT STOCKS)"
            )
            lines.append("=" * 100)
            lines.append("")

            lines.append(
                f"Total Industries Represented: {len(self.industry_distribution)}"
            )
            lines.append("")

            sorted_industries = sorted(
                self.industry_distribution.items(), key=lambda x: x[1], reverse=True
            )[:15]

            lines.append("Top 15 Industries:")
            lines.append("-" * 80)
            for industry, count in sorted_industries:
                percentage = (count / len(self.distinct_large_gain_stocks)) * 100
                bar = "█" * int(percentage / 2)
                # Truncate long industry names
                industry_display = (
                    industry[:40] if len(industry) <= 40 else industry[:37] + "..."
                )
                lines.append(
                    f"  {industry_display:40} {count:3d} ({percentage:5.1f}%) {bar}"
                )
            lines.append("")

        # Confidence and Data Quality Distribution
        if self.distinct_large_gain_stocks:
            lines.append("=" * 100)
            lines.append("5. SIGNAL QUALITY METRICS (DISTINCT STOCKS)")
            lines.append("=" * 100)
            lines.append("")

            # Confidence level distribution
            if self.confidence_distribution:
                lines.append("Confidence Level Distribution:")
                lines.append("-" * 80)
                confidence_order = ["high", "medium", "low"]
                for conf_level in confidence_order:
                    count = self.confidence_distribution.get(conf_level, 0)
                    percentage = (
                        (count / len(self.distinct_large_gain_stocks)) * 100
                        if len(self.distinct_large_gain_stocks) > 0
                        else 0
                    )
                    bar = "█" * int(percentage / 2)
                    lines.append(
                        f"  {conf_level.upper():15} {count:4d} ({percentage:5.1f}%) {bar}"
                    )
                lines.append("")

            # Data quality statistics
            data_quality_scores = [
                s["data_quality"]
                for s in self.distinct_large_gain_stocks
                if s["data_quality"] is not None
            ]
            if data_quality_scores:
                avg_quality = sum(data_quality_scores) / len(data_quality_scores)
                min_quality = min(data_quality_scores)
                max_quality = max(data_quality_scores)

                lines.append("Data Quality Score Statistics:")
                lines.append("-" * 80)
                lines.append(f"  Average:  {avg_quality:.3f}")
                lines.append(f"  Minimum:  {min_quality:.3f}")
                lines.append(f"  Maximum:  {max_quality:.3f}")
                lines.append(
                    f"  Stocks with data: {len(data_quality_scores)}/{len(self.distinct_large_gain_stocks)}"
                )
                lines.append("")

            # Upside potential statistics
            upside_values = [
                s["upside_pct"]
                for s in self.distinct_large_gain_stocks
                if s["upside_pct"] is not None
            ]
            if upside_values:
                upside_values_sorted = sorted(upside_values)
                avg_upside = sum(upside_values) / len(upside_values)
                median_upside = upside_values_sorted[len(upside_values_sorted) // 2]
                min_upside = min(upside_values)
                max_upside = max(upside_values)

                lines.append("Upside Potential Statistics:")
                lines.append("-" * 80)
                lines.append(f"  Average:  {avg_upside:+.1f}%")
                lines.append(f"  Median:   {median_upside:+.1f}%")
                lines.append(f"  Minimum:  {min_upside:+.1f}%")
                lines.append(f"  Maximum:  {max_upside:+.1f}%")
                lines.append(
                    f"  Stocks with target: {len(upside_values)}/{len(self.distinct_large_gain_stocks)}"
                )
                lines.append("")

            # Price statistics
            prices = [
                s["close_price"]
                for s in self.distinct_large_gain_stocks
                if s["close_price"] is not None
            ]
            if prices:
                prices_sorted = sorted(prices)
                avg_price = sum(prices) / len(prices)
                median_price = prices_sorted[len(prices_sorted) // 2]
                min_price = min(prices)
                max_price = max(prices)

                lines.append("Price Range Statistics:")
                lines.append("-" * 80)
                lines.append(f"  Average:  ${avg_price:.2f}")
                lines.append(f"  Median:   ${median_price:.2f}")
                lines.append(f"  Minimum:  ${min_price:.2f}")
                lines.append(f"  Maximum:  ${max_price:.2f}")
                lines.append("")

        # Macroeconomic Context by Country
        if self.macro_data_by_country:
            lines.append("=" * 100)
            lines.append(
                "6. MACROECONOMIC CONTEXT BY COUNTRY (RISK BUDGETING & DIVERSIFICATION)"
            )
            lines.append("=" * 100)
            lines.append("")

            lines.append(
                f"Countries with Macro Data: {len(self.macro_data_by_country)}"
            )
            lines.append("")

            # Sort countries by number of stocks
            countries_sorted = sorted(
                self.country_distribution.items(), key=lambda x: x[1], reverse=True
            )

            for country, stock_count in countries_sorted:
                macro = self.macro_data_by_country.get(country)

                if not macro:
                    continue

                lines.append("-" * 100)
                lines.append(
                    f"{country.upper()} - {stock_count} Stocks ({(stock_count / len(self.distinct_large_gain_stocks)) * 100:.1f}% of portfolio)"
                )
                lines.append("-" * 100)
                lines.append(f"  Assessment Date:        {macro['assessment_date']}")
                lines.append(
                    f"  Business Cycle Regime:  {macro['regime']} (Confidence: {macro['confidence']:.1%})"
                )
                lines.append("")

                # Economic Indicators
                lines.append("  Economic Indicators:")
                if macro["gdp_growth_yy"] is not None:
                    lines.append(
                        f"    GDP Growth (YoY):     {macro['gdp_growth_yy']:+.1f}%"
                    )
                if macro["unemployment"] is not None:
                    lines.append(
                        f"    Unemployment:         {macro['unemployment']:.1f}%"
                    )
                if macro["inflation"] is not None:
                    lines.append(f"    Inflation:            {macro['inflation']:.1f}%")
                lines.append("")

                # Recession Risk
                lines.append("  Recession Risk:")
                lines.append(
                    f"    6-Month:              {macro['recession_risk_6m']:.1%}"
                )
                lines.append(
                    f"    12-Month:             {macro['recession_risk_12m']:.1%}"
                )

                if macro["recession_drivers"]:
                    lines.append(
                        f"    Key Drivers:          {', '.join(macro['recession_drivers'][:3])}"
                    )
                lines.append("")

                # Market Indicators
                lines.append("  Market Indicators:")
                if macro["ism_pmi"] is not None:
                    lines.append(
                        f"    ISM PMI:              {macro['ism_pmi']:.1f} ({macro['ism_signal']})"
                    )
                if macro["yield_curve_2s10s"] is not None:
                    lines.append(
                        f"    Yield Curve (2s10s):  {macro['yield_curve_2s10s']:+.0f}bps ({macro['yield_curve_signal']})"
                    )
                lines.append(
                    f"    Credit Spreads:       {macro['credit_spread_signal']}"
                )
                lines.append("")

                # Portfolio Positioning
                lines.append("  Recommended Positioning:")
                lines.append(f"    Factor Exposure:      {macro['factor_exposure']}")

                if macro["recommended_overweights"]:
                    lines.append(
                        f"    Overweight Sectors:   {', '.join(macro['recommended_overweights'][:5])}"
                    )

                if macro["recommended_underweights"]:
                    lines.append(
                        f"    Underweight Sectors:  {', '.join(macro['recommended_underweights'][:5])}"
                    )
                lines.append("")

                # Sector Tilts (weight adjustments)
                if macro["sector_tilts"]:
                    lines.append("  Sector Weight Tilts (Recommended Adjustments):")
                    # Sort by absolute value to show most significant tilts first
                    sorted_tilts = sorted(
                        macro["sector_tilts"].items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )
                    for sector, tilt in sorted_tilts[:10]:  # Show top 10 tilts
                        sign = "+" if tilt >= 0 else ""
                        lines.append(f"    {sector:30} {sign}{tilt:+.2%}")
                    lines.append("")

                # Key Risks
                if macro["primary_risks"]:
                    lines.append("  Primary Risks:")
                    for risk in macro["primary_risks"][:3]:
                        lines.append(f"    • {risk}")
                    lines.append("")

                # Rationale
                lines.append("  Regime Rationale:")
                # Wrap rationale at 90 characters
                rationale_words = macro["rationale"].split()
                current_line = "    "
                for word in rationale_words:
                    if len(current_line) + len(word) + 1 > 90:
                        lines.append(current_line)
                        current_line = "    " + word
                    else:
                        current_line += " " + word if current_line != "    " else word
                if current_line.strip():
                    lines.append(current_line)
                lines.append("")

        # Detailed LARGE_GAIN Company Profiles (DISTINCT stocks only)
        if self.distinct_large_gain_stocks:
            lines.append("=" * 100)
            lines.append(
                "7. LARGE_GAIN STOCKS - DETAILED COMPANY PROFILES (DISTINCT STOCKS)"
            )
            lines.append("=" * 100)
            lines.append("")
            lines.append(
                f"Total DISTINCT Companies: {len(self.distinct_large_gain_stocks)}"
            )
            lines.append("")

            for i, stock in enumerate(self.distinct_large_gain_stocks, 1):
                lines.append("-" * 100)
                lines.append(
                    f"[{i}/{len(self.distinct_large_gain_stocks)}] {stock['company_name']} ({stock['ticker']})"
                )
                lines.append("-" * 100)
                lines.append(f"  Instrument ID:  {stock['instrument_id']}")
                lines.append(f"  Ticker:         {stock['yf_ticker']}")
                lines.append(f"  Country:        {stock['country']}")
                lines.append(f"  Exchange:       {stock['exchange']}")
                lines.append(f"  Sector:         {stock['sector']}")
                lines.append(f"  Industry:       {stock['industry']}")
                lines.append(
                    f"  Close Price:    ${stock['close_price']:.2f}"
                    if stock["close_price"]
                    else "  Close Price:    N/A"
                )
                lines.append(f"  Confidence:     {stock['confidence']}")
                lines.append(
                    f"  Data Quality:   {stock['data_quality']:.2f}"
                    if stock["data_quality"]
                    else "  Data Quality:   N/A"
                )

                if stock["upside_pct"]:
                    lines.append(f"  Upside Target:  {stock['upside_pct']:+.1f}%")

                lines.append("")
                lines.append("  Business Description:")
                # Wrap description text at 90 characters
                desc = stock["description"]
                if desc and desc != "No description available":
                    words = desc.split()
                    current_line = "    "
                    for word in words:
                        if len(current_line) + len(word) + 1 > 90:
                            lines.append(current_line)
                            current_line = "    " + word
                        else:
                            current_line += (
                                " " + word if current_line != "    " else word
                            )
                    if current_line.strip():
                        lines.append(current_line)
                else:
                    lines.append("    No description available")

                lines.append("")

                # Only show analysis notes if they exist and are not None
                if stock.get("analysis_notes"):
                    lines.append("  Signal Analysis:")
                    # Wrap analysis notes
                    notes = stock["analysis_notes"]
                    words = notes.split()
                    current_line = "    "
                    for word in words:
                        if len(current_line) + len(word) + 1 > 90:
                            lines.append(current_line)
                            current_line = "    " + word
                        else:
                            current_line += (
                                " " + word if current_line != "    " else word
                            )
                    if current_line.strip():
                        lines.append(current_line)
                    lines.append("")

        # Summary Statistics
        lines.append("=" * 100)
        lines.append("8. SUMMARY STATISTICS")
        lines.append("=" * 100)
        lines.append("")

        lines.append(f"Total Signals Analyzed:              {total_signals}")
        lines.append(
            f"LARGE_GAIN Signals:                  {large_gain_count} ({large_gain_pct:.1f}%)"
        )
        lines.append(
            f"DISTINCT LARGE_GAIN Stocks:          {len(self.distinct_large_gain_stocks)}"
        )
        lines.append("")

        lines.append("Diversification Metrics:")
        lines.append(
            f"  Unique Sectors:                    {len(self.sector_distribution)}"
        )
        lines.append(
            f"  Unique Industries:                 {len(self.industry_distribution)}"
        )
        lines.append(
            f"  Unique Countries:                  {len(self.country_distribution)}"
        )
        lines.append(
            f"  Unique Exchanges:                  {len(self.exchange_distribution)}"
        )
        lines.append("")

        # Average metrics for LARGE_GAIN (using distinct stocks)
        if self.distinct_large_gain_stocks:
            prices = [
                s["close_price"]
                for s in self.distinct_large_gain_stocks
                if s["close_price"]
            ]
            qualities = [
                s["data_quality"]
                for s in self.distinct_large_gain_stocks
                if s["data_quality"]
            ]

            if prices:
                avg_price = sum(prices) / len(prices)
                lines.append("Key Metrics:")
                lines.append(f"  Average Close Price:               ${avg_price:.2f}")

            if qualities:
                avg_quality = sum(qualities) / len(qualities)
                lines.append(f"  Average Data Quality Score:        {avg_quality:.2f}")

            upside_values = [
                s["upside_pct"]
                for s in self.distinct_large_gain_stocks
                if s["upside_pct"]
            ]
            if upside_values:
                avg_upside = sum(upside_values) / len(upside_values)
                lines.append(f"  Average Upside Potential:          {avg_upside:+.1f}%")

        lines.append("")
        lines.append("=" * 100)
        lines.append("END OF REPORT")
        lines.append("=" * 100)

        return "\n".join(lines)

    def save_report(self, report_text: str) -> str:
        """
        Save report to text file.

        Args:
            report_text: Formatted report string

        Returns:
            Path to saved file
        """
        filename = f"signal_analysis_report_{self.signal_date}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Report saved to: {filename}")
        return filename

    def generate_and_save(self) -> str:
        """
        Run complete analysis and save report.

        Returns:
            Path to saved report file
        """
        logger.info(f"Starting signal analysis for {self.signal_date}")

        # Analyze signals
        self.analyze_signals()

        # Generate report
        report_text = self.generate_report()

        # Print to console
        print("\n" + report_text)

        # Save to file
        filename = self.save_report(report_text)

        logger.info("Analysis complete")

        return filename


def get_most_recent_signal_date() -> date_type | None:
    """
    Fetch the most recent signal date from the database.

    Returns:
        Most recent signal date or None if no signals found
    """
    from sqlalchemy import func

    with database_manager.get_session() as session:
        query = select(func.max(StockSignal.signal_date))
        result = session.execute(query).scalar_one_or_none()

        if result:
            logger.info(f"Most recent signal date: {result}")
            return result
        else:
            logger.warning("No signals found in database")
            return None


def main():
    """
    Main function - generate signal analysis report.
    """
    print("=" * 100)
    print("STOCK SIGNAL ANALYSIS REPORT GENERATOR")
    print("=" * 100)

    try:
        # Initialize database
        logger.info("Initializing database connection...")
        init_db()

        if not database_manager.is_initialized:
            logger.error("Database initialization failed")
            sys.exit(1)

        logger.info("Database connection established")

        # Get the most recent signal date from database
        signal_date = get_most_recent_signal_date()

        if signal_date is None:
            logger.error("No signals found in database. Run signal analysis first.")
            sys.exit(1)

        logger.info(f"Using most recent signal date: {signal_date}")

        # Create reporter
        reporter = SignalAnalysisReporter(signal_date=signal_date)

        # Generate and save report
        report_file = reporter.generate_and_save()

        print("\n" + "=" * 100)
        print(f"✓ Report saved to: {report_file}")
        print("=" * 100)

    except KeyboardInterrupt:
        logger.info("\nReport generation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
