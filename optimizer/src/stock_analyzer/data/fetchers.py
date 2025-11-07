"""
Data Fetchers
============

Provides functions to fetch data from various sources:
- Price data from yfinance
- Macro regime data from database
- Economic indicators from database
- PMI and unemployment data from Trading Economics
"""

from typing import Optional, Tuple, Dict
import logging
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.orm import joinedload

from app.database import database_manager
from src.yfinance import YFinanceClient
from app.models.macro_regime import (
    CountryRegimeAssessment,
    MacroAnalysisRun,
    EconomicIndicators,
)
from app.models.trading_economics import TradingEconomicsIndicator, TradingEconomicsSnapshot

logger = logging.getLogger(__name__)


async def fetch_price_data(
    yf_ticker: str,
    period: str = "2y",
    max_retries: int = 3
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Fetch historical price data for stock and benchmark with retry logic.

    Now uses centralized YFinanceClient for caching and rate limiting.
    Runs blocking yfinance calls in thread pool for true async parallelism.

    Args:
        yf_ticker: Yahoo Finance ticker symbol
        period: Historical period to fetch (default: "2y" for 2 years)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Tuple of (stock_data, benchmark_data, stock_info)
        Returns (None, None, None) if fetch fails
    """
    import asyncio
    from functools import partial

    # Get singleton YFinanceClient instance
    client = YFinanceClient.get_instance()

    # Run blocking yfinance call in thread pool to avoid blocking event loop
    # This enables TRUE parallel fetching across 100+ stocks
    loop = asyncio.get_event_loop()
    fetch_func = partial(
        client.fetch_price_and_benchmark,
        symbol=yf_ticker,
        benchmark="SPY",
        period=period,
        max_retries=max_retries
    )

    stock_hist, spy_hist, stock_info = await loop.run_in_executor(None, fetch_func)

    return stock_hist, spy_hist, stock_info


def get_country_from_ticker(yf_ticker: str, info: Optional[Dict]) -> Optional[str]:
    """
    Determine country from ticker and info dict.

    Tries database first (exchange → country mapping), then falls back to
    yfinance info['country'] field.

    Args:
        yf_ticker: Yahoo Finance ticker symbol
        info: Stock info dict from yfinance (optional)

    Returns:
        Country name (USA, Germany, France, UK, Japan) or None if unknown
    """
    # Try database first
    try:
        with database_manager.get_session() as session:
            from app.models.universe import Instrument

            # Handle multiple instruments with same ticker (prefer active)
            query = (
                select(Instrument)
                .where(Instrument.yfinance_ticker == yf_ticker)
                .options(joinedload(Instrument.exchange))
                .order_by(Instrument.is_active.desc())  # Active instruments first
                .limit(1)
            )

            result = session.execute(query)
            instrument = result.scalars().first()

            if instrument and instrument.exchange:
                exchange_to_country = {
                    "NYSE": "USA",
                    "NASDAQ": "USA",
                    "London Stock Exchange": "UK",
                    "Deutsche Börse Xetra": "Germany",
                    "Gettex": "Germany",
                    "Euronext Paris": "France",
                }
                country = exchange_to_country.get(instrument.exchange.exchange_name)
                if country:
                    return country
    except Exception as e:
        logger.debug(f"Error looking up country in database: {e}")

    # Fallback to info dict
    if info and 'country' in info:
        country_map = {
            "United States": "USA",
            "Germany": "Germany",
            "France": "France",
            "United Kingdom": "UK",
        }
        return country_map.get(info['country'], info['country'])

    return None


async def fetch_macro_regime(country: str) -> Optional[Dict]:
    """
    Fetch latest macro regime assessment from database.

    Args:
        country: Country name (USA, Germany, France, UK, Japan)

    Returns:
        Dict with regime, confidence, recession risks, sector tilts, etc.
        None if no assessment available
    """
    try:
        with database_manager.get_session() as session:
            query = (
                select(CountryRegimeAssessment)
                .where(CountryRegimeAssessment.country == country)
                .options(
                    joinedload(CountryRegimeAssessment.analysis_run).joinedload(
                        MacroAnalysisRun.market_indicators
                    )
                )
                .order_by(CountryRegimeAssessment.assessment_timestamp.desc())
                .limit(1)
            )

            result = session.execute(query)
            assessment = result.scalar_one_or_none()

            if not assessment:
                return None

            return {
                'regime': (
                    assessment.regime
                    if isinstance(assessment.regime, str)
                    else assessment.regime.value
                ),
                'confidence': float(assessment.confidence),
                'recession_risk_6m': float(assessment.recession_risk_6m),
                'recession_risk_12m': float(assessment.recession_risk_12m),
                'sector_tilts': assessment.sector_tilts or {},
                'factor_exposure': (
                    assessment.factor_exposure
                    if isinstance(assessment.factor_exposure, str)
                    else (assessment.factor_exposure.value if assessment.factor_exposure else None)
                ),
                'recommended_overweights': assessment.recommended_overweights or [],
                'recommended_underweights': assessment.recommended_underweights or [],
            }
    except Exception as e:
        logger.debug(f"Error fetching macro regime: {e}")
        return None


def fetch_economic_forecasts(
    country: str,
    max_age_days: int = 30
) -> Optional[Dict]:
    """
    Query forward-looking economic forecasts from database.

    Args:
        country: Country name (USA, Germany, France, UK, Japan)
        max_age_days: Maximum age of data to consider (default: 30 days)

    Returns:
        Dict with gdp_forecast_6m, inflation_forecast_6m, earnings_forecast_12m
        or None if not available
    """
    try:
        with database_manager.get_session() as session:
            query = (
                select(
                    EconomicIndicators.gdp_forecast_6m,
                    EconomicIndicators.inflation_forecast_6m,
                    EconomicIndicators.earnings_forecast_12m,
                    EconomicIndicators.data_timestamp,
                )
                .where(EconomicIndicators.country == country)
                .order_by(desc(EconomicIndicators.data_timestamp))
                .limit(1)
            )

            result = session.execute(query).first()

            if result:
                gdp_forecast, inflation_forecast, earnings_forecast, data_timestamp = result

                # Check data age
                from datetime import datetime

                age_days = (datetime.now() - data_timestamp.replace(tzinfo=None)).days

                if age_days > max_age_days:
                    logger.debug(
                        f"Economic forecasts for {country} are {age_days} days old "
                        f"(> {max_age_days} days)"
                    )
                    return None

                logger.debug(
                    f"Using economic forecasts for {country}: "
                    f"GDP={gdp_forecast}%, Inflation={inflation_forecast}%, "
                    f"Earnings={earnings_forecast}% (age: {age_days} days)"
                )

                return {
                    'gdp_forecast_6m': (
                        float(gdp_forecast) if gdp_forecast is not None else None
                    ),
                    'inflation_forecast_6m': (
                        float(inflation_forecast) if inflation_forecast is not None else None
                    ),
                    'earnings_forecast_12m': (
                        float(earnings_forecast) if earnings_forecast is not None else None
                    ),
                }
            else:
                logger.debug(f"No economic forecasts found for {country}")
                return None

    except Exception as e:
        logger.debug(f"Error fetching economic forecasts for {country}: {e}")
        return None


def fetch_pmi_data(
    country: str,
    pmi_type: str = 'composite',
    max_age_days: int = 30
) -> Optional[float]:
    """
    Query actual PMI values from Trading Economics database.

    Args:
        country: Country name (USA, Germany, France, UK, Japan)
        pmi_type: Type of PMI - 'composite', 'manufacturing', or 'services'
        max_age_days: Maximum age of data to consider (default: 30 days)

    Returns:
        PMI value (float) or None if not available

    PMI Interpretation:
        > 52: Strong expansion (EARLY_CYCLE)
        50-52: Mild expansion (MID_CYCLE)
        48-50: Mild contraction (LATE_CYCLE)
        < 48: Strong contraction (RECESSION)
    """
    try:
        with database_manager.get_session() as session:
            # Map pmi_type to indicator_name
            indicator_map = {
                'composite': 'composite_pmi',
                'manufacturing': 'manufacturing_pmi',
                'services': 'services_pmi',
            }
            indicator_name = indicator_map.get(pmi_type, 'composite_pmi')

            # Query most recent PMI value for country
            query = (
                select(
                    TradingEconomicsIndicator.value,
                    TradingEconomicsSnapshot.fetch_timestamp,
                )
                .join(
                    TradingEconomicsSnapshot,
                    TradingEconomicsIndicator.snapshot_id == TradingEconomicsSnapshot.id,
                )
                .where(
                    TradingEconomicsSnapshot.country == country,
                    TradingEconomicsIndicator.indicator_name == indicator_name,
                )
                .order_by(desc(TradingEconomicsSnapshot.fetch_timestamp))
                .limit(1)
            )

            result = session.execute(query).first()

            if result:
                pmi_value, fetch_timestamp = result

                # Check data age
                from datetime import datetime

                age_days = (datetime.now() - fetch_timestamp.replace(tzinfo=None)).days

                if age_days > max_age_days:
                    logger.debug(
                        f"PMI data for {country} is {age_days} days old "
                        f"(> {max_age_days} days), falling back to regime-based estimate"
                    )
                    return None

                logger.debug(
                    f"Using actual {pmi_type} PMI for {country}: {pmi_value:.1f} "
                    f"(age: {age_days} days)"
                )
                return float(pmi_value)
            else:
                logger.debug(
                    f"No {pmi_type} PMI data found for {country}, "
                    f"using regime-based estimate"
                )
                return None

    except Exception as e:
        logger.debug(f"Error fetching {pmi_type} PMI for {country}: {e}")
        return None


def fetch_unemployment_rate(country: str, max_age_days: int = 30) -> Optional[float]:
    """
    Query unemployment rate from Trading Economics database.

    Args:
        country: Country name (USA, Germany, France, UK, Japan)
        max_age_days: Maximum age of data to consider (default: 30 days)

    Returns:
        Unemployment rate (%) or None if not available
    """
    try:
        with database_manager.get_session() as session:
            query = (
                select(
                    TradingEconomicsIndicator.value,
                    TradingEconomicsSnapshot.fetch_timestamp,
                )
                .join(
                    TradingEconomicsSnapshot,
                    TradingEconomicsIndicator.snapshot_id == TradingEconomicsSnapshot.id,
                )
                .where(
                    TradingEconomicsSnapshot.country == country,
                    TradingEconomicsIndicator.indicator_name == 'unemployment_rate',
                )
                .order_by(desc(TradingEconomicsSnapshot.fetch_timestamp))
                .limit(1)
            )

            result = session.execute(query).first()

            if result:
                unemployment_rate, fetch_timestamp = result

                # Check data age
                from datetime import datetime

                age_days = (datetime.now() - fetch_timestamp.replace(tzinfo=None)).days

                if age_days > max_age_days:
                    logger.debug(
                        f"Unemployment data for {country} is {age_days} days old "
                        f"(> {max_age_days} days)"
                    )
                    return None

                logger.debug(
                    f"Using unemployment rate for {country}: {unemployment_rate:.1f}% "
                    f"(age: {age_days} days)"
                )
                return float(unemployment_rate)
            else:
                logger.debug(f"No unemployment data found for {country}")
                return None

    except Exception as e:
        logger.debug(f"Error fetching unemployment rate for {country}: {e}")
        return None
