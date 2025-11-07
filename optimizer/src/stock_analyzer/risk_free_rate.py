"""
Risk-Free Rate Calculator - Institutional Grade
================================================
Provides dynamic risk-free rate calculation using:
1. Database (Trading Economics - 10Y government bonds)
2. TIPS adjustment using real inflation forecasts from EconomicIndicators
3. Data quality validation using macro regime confidence
4. Rate smoothing via averaging over lookback period
5. Fallback to hardcoded values with warnings

CRITICAL: Must use NOMINAL treasury yields, NOT TIPS (inflation-protected).
When TIPS is detected, automatically adjusts using real inflation forecasts.

Data Sources:
- Bond yields: TradingEconomicsBondYield table
- Inflation forecasts: EconomicIndicators.inflation_forecast_6m (Il Sole 24 Ore)
- Macro confidence: CountryRegimeAssessment.confidence (LLM forecasts)
- Recession risk: CountryRegimeAssessment.recession_risk_6m/12m
"""

import logging
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from sqlalchemy import select, desc, func
from sqlalchemy.exc import SQLAlchemyError

from app.database import database_manager
from app.models.trading_economics import (
    TradingEconomicsSnapshot,
    TradingEconomicsBondYield
)
from app.models.macro_regime import EconomicIndicators, CountryRegimeAssessment

logger = logging.getLogger(__name__)

# Fallback values (only used when database is unavailable)
FALLBACK_RISK_FREE_RATES = {
    'USA': 0.045,      # 4.5%
    'Germany': 0.026,  # 2.6%
    'France': 0.034,   # 3.4%
    'UK': 0.045,       # 4.5%
    'Japan': 0.017,    # 1.7%
}

# Reasonable ranges for risk-free rates by country (in %)
# Based on historical ranges - used for data quality validation
REASONABLE_RANGES = {
    'USA': (-2.0, 15.0),
    'UK': (-2.0, 15.0),
    'Germany': (-2.0, 10.0),
    'France': (-2.0, 12.0),
    'Japan': (-2.0, 5.0),
}

# Default parameters
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_LOOKBACK_DAYS = 5
DEFAULT_USE_AVERAGE = True
DEFAULT_FALLBACK_TO_STALE = True


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RiskFreeRateResult:
    """
    Institutional-grade result for risk-free rate queries.

    Provides complete diagnostic information for audit trails and
    portfolio optimization per best practices.
    """
    rate: Optional[float]
    """Risk-free rate as decimal (e.g., 0.045 for 4.5%), or None if unavailable"""

    status: str
    """Status: 'success', 'no_data', 'stale_data', 'invalid_data', 'error'"""

    age_days: Optional[int] = None
    """Age of data in days"""

    data_source: Optional[str] = None
    """Description of data source (e.g., 'US 10Y Treasury TIPS adjusted')"""

    warnings: List[str] = field(default_factory=list)
    """List of warning messages for audit trail"""

    rate_pct: Optional[float] = None
    """Rate as percentage (e.g., 4.5 for 4.5%)"""

    inflation_adjusted: bool = False
    """Whether TIPS adjustment was applied"""

    macro_confidence: Optional[float] = None
    """Macro regime confidence (0.0-1.0) if available"""

    def __post_init__(self):
        """Calculate derived fields"""
        if self.rate is not None and self.rate_pct is None:
            self.rate_pct = self.rate * 100

    def is_successful(self) -> bool:
        """Check if rate was successfully retrieved"""
        return self.status == 'success' and self.rate is not None


# ============================================================================
# HELPER FUNCTIONS - Query real database data
# ============================================================================

def _get_inflation_forecast(country: str, max_age_days: int = 30) -> Optional[float]:
    """
    Get real inflation forecast from EconomicIndicators table (Il Sole 24 Ore data).

    Args:
        country: Country code
        max_age_days: Maximum age of forecast data

    Returns:
        Inflation forecast as percentage, or None if unavailable
    """
    try:
        with database_manager.get_session() as session:
            query = (
                select(EconomicIndicators.inflation_forecast_6m, EconomicIndicators.data_timestamp)
                .where(
                    EconomicIndicators.country == country,
                    EconomicIndicators.inflation_forecast_6m.isnot(None)
                )
                .order_by(desc(EconomicIndicators.data_timestamp))
                .limit(1)
            )

            result = session.execute(query).first()

            if result:
                inflation_forecast, data_timestamp = result

                # Check freshness
                age_days = (datetime.now() - data_timestamp.replace(tzinfo=None)).days
                if age_days > max_age_days:
                    logger.debug(
                        f"{country} inflation forecast is {age_days} days old (max: {max_age_days})"
                    )
                    return None

                logger.debug(
                    f"Using {country} inflation forecast from Il Sole 24 Ore: {inflation_forecast:.2f}% "
                    f"(age: {age_days} days)"
                )
                return inflation_forecast

            return None

    except Exception as e:
        logger.debug(f"Error fetching inflation forecast for {country}: {e}")
        return None


def _get_macro_confidence(country: str, max_age_days: int = 30) -> Optional[float]:
    """
    Get macro regime confidence from CountryRegimeAssessment (LLM forecasts).

    Args:
        country: Country code
        max_age_days: Maximum age of assessment

    Returns:
        Confidence score (0.0-1.0), or None if unavailable
    """
    try:
        with database_manager.get_session() as session:
            query = (
                select(
                    CountryRegimeAssessment.confidence,
                    CountryRegimeAssessment.assessment_timestamp
                )
                .where(CountryRegimeAssessment.country == country)
                .order_by(desc(CountryRegimeAssessment.assessment_timestamp))
                .limit(1)
            )

            result = session.execute(query).first()

            if result:
                confidence, assessment_timestamp = result

                # Check freshness
                age_days = (datetime.now() - assessment_timestamp.replace(tzinfo=None)).days
                if age_days > max_age_days:
                    logger.debug(
                        f"{country} macro confidence is {age_days} days old (max: {max_age_days})"
                    )
                    return None

                logger.debug(
                    f"Using {country} macro regime confidence: {confidence:.2f} (age: {age_days} days)"
                )
                return confidence

            return None

    except Exception as e:
        logger.debug(f"Error fetching macro confidence for {country}: {e}")
        return None


def _validate_rate(country: str, rate: float) -> bool:
    """
    Validate if rate is within reasonable range for the country.

    Args:
        country: Country code
        rate: Rate as percentage

    Returns:
        True if rate is reasonable, False otherwise
    """
    min_rate, max_rate = REASONABLE_RANGES.get(country, (-2.0, 20.0))

    if not (min_rate <= rate <= max_rate):
        logger.warning(
            f"Suspicious {country} risk-free rate: {rate:.2f}% "
            f"(expected range: {min_rate:.1f}% to {max_rate:.1f}%)"
        )
        return False

    return True


def _query_averaged_rate(
    session,
    country: str,
    lookback_days: int
) -> Tuple[Optional[float], Optional[datetime], Optional[str]]:
    """
    Query averaged bond yield over lookback period for smoothing.

    Args:
        session: Database session
        country: Country code
        lookback_days: Number of days to average over

    Returns:
        Tuple of (averaged_rate, latest_timestamp, data_source)
    """
    cutoff_date = datetime.now() - timedelta(days=lookback_days)

    query = (
        select(
            func.avg(TradingEconomicsBondYield.yield_value).label('avg_yield'),
            func.max(TradingEconomicsSnapshot.fetch_timestamp).label('latest_date'),
            TradingEconomicsBondYield.raw_name
        )
        .join(TradingEconomicsBondYield.snapshot)
        .where(
            TradingEconomicsSnapshot.country == country,
            TradingEconomicsBondYield.maturity == '10Y',
            TradingEconomicsSnapshot.fetch_timestamp >= cutoff_date
        )
        .group_by(TradingEconomicsBondYield.raw_name)
        .order_by(desc('latest_date'))
        .limit(1)
    )

    result = session.execute(query).first()

    if result:
        return result.avg_yield, result.latest_date, result.raw_name

    return None, None, None


def _query_latest_rate(
    session,
    country: str
) -> Tuple[Optional[float], Optional[datetime], Optional[str]]:
    """
    Query latest bond yield without averaging.

    Args:
        session: Database session
        country: Country code

    Returns:
        Tuple of (rate, timestamp, data_source)
    """
    query = (
        select(
            TradingEconomicsSnapshot.fetch_timestamp,
            TradingEconomicsBondYield.yield_value,
            TradingEconomicsBondYield.raw_name
        )
        .join(TradingEconomicsBondYield.snapshot)
        .where(
            TradingEconomicsSnapshot.country == country,
            TradingEconomicsBondYield.maturity == '10Y'
        )
        .order_by(desc(TradingEconomicsSnapshot.fetch_timestamp))
        .limit(1)
    )

    result = session.execute(query).first()

    if result:
        return result.yield_value, result.fetch_timestamp, result.raw_name

    return None, None, None


# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def get_risk_free_rate(
    country: str = 'USA',
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    use_average: bool = DEFAULT_USE_AVERAGE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    fallback_to_stale: bool = DEFAULT_FALLBACK_TO_STALE
) -> float:
    """
    Get current risk-free rate from Trading Economics database (10Y government bond).

    Institutional-grade features:
    - Automatic TIPS adjustment using real inflation forecasts from Il Sole 24 Ore
    - Rate smoothing via averaging over lookback period
    - Data quality validation with reasonable range checking
    - Macro regime confidence tracking for audit trails

    Uses country-specific 10Y government bond yields:
    - USA → US 10Y Treasury (TIPS adjusted if needed)
    - UK → UK 10Y Gilt
    - Germany → German 10Y Bund
    - France → French 10Y OAT
    - Japan → Japanese 10Y JGB

    Args:
        country: Country code (USA, UK, Germany, France, Japan)
        max_age_days: Maximum age of database data (default: 30 days)
        use_average: If True, average recent observations to smooth noise (default: True)
        lookback_days: Number of days to average over (default: 5)
        fallback_to_stale: If True, use stale data with warning (default: True)

    Returns:
        Risk-free rate as decimal (e.g., 0.045 for 4.5%)

    Example:
        >>> rf = get_risk_free_rate(country='USA')
        >>> sharpe = (return - rf) / volatility
    """

    # Try database first with institutional-grade logic
    result = _get_risk_free_rate_from_db(
        country,
        max_age_days,
        use_average,
        lookback_days,
        fallback_to_stale
    )

    if result.is_successful():
        # Type guard: is_successful() guarantees rate is not None
        assert result.rate is not None, "is_successful() guarantees rate is not None"

        logger.info(
            f"Using {country} risk-free rate from database: {result.rate:.4f} ({result.rate_pct:.2f}%) "
            f"[source: {result.data_source}, age: {result.age_days}d]"
        )
        if result.warnings:
            for warning in result.warnings:
                logger.info(f"  ⚠ {warning}")
        return result.rate

    # Fallback to hardcoded value
    fallback = FALLBACK_RISK_FREE_RATES.get(country, 0.045)
    logger.warning(
        f"Using fallback {country} risk-free rate: {fallback:.4f} ({fallback*100:.2f}%) "
        f"[reason: {result.status}]"
    )
    return fallback


def _get_risk_free_rate_from_db(
    country: str,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    use_average: bool = DEFAULT_USE_AVERAGE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    fallback_to_stale: bool = DEFAULT_FALLBACK_TO_STALE
) -> RiskFreeRateResult:
    """
    Get risk-free rate from Trading Economics database with institutional-grade features.

    Features:
    - Automatic TIPS adjustment using real inflation forecasts from Il Sole 24 Ore
    - Rate smoothing via averaging over lookback period
    - Data quality validation with reasonable range checking
    - Macro regime confidence for audit trails
    - Robust error handling with detailed status tracking

    Args:
        country: Country code (USA, UK, Germany, France, Japan)
        max_age_days: Maximum age of database data (default: 30 days)
        use_average: If True, average recent observations (default: True)
        lookback_days: Number of days to average over (default: 5)
        fallback_to_stale: If True, use stale data with warning (default: True)

    Returns:
        RiskFreeRateResult with rate, status, and diagnostic information
    """
    try:
        with database_manager.get_session() as session:
            # Query bond yield (with optional averaging for smoothing)
            if use_average:
                rate, latest_date, data_source = _query_averaged_rate(
                    session, country, lookback_days
                )
            else:
                rate, latest_date, data_source = _query_latest_rate(
                    session, country
                )

            if rate is None or latest_date is None:
                return RiskFreeRateResult(
                    rate=None,
                    status='no_data',
                    data_source=data_source
                )

            # Check data freshness
            age_days = (datetime.now() - latest_date.replace(tzinfo=None)).days

            # Validate rate is reasonable
            if not _validate_rate(country, rate):
                return RiskFreeRateResult(
                    rate=None,
                    status='invalid_data',
                    age_days=age_days,
                    data_source=data_source,
                    warnings=[f"Rate {rate:.2f}% outside reasonable range"]
                )

            # Handle stale data
            warnings = []
            if age_days > max_age_days:
                warning_msg = f"Data is {age_days} days old (max: {max_age_days})"
                if fallback_to_stale:
                    warnings.append(warning_msg)
                    logger.debug(f"{country}: {warning_msg}. Using anyway.")
                else:
                    return RiskFreeRateResult(
                        rate=None,
                        status='stale_data',
                        age_days=age_days,
                        data_source=data_source,
                        warnings=[warning_msg]
                    )

            # Handle TIPS adjustment using REAL inflation forecasts
            inflation_adjusted = False
            if data_source and 'TIPS' in data_source.upper():
                # Try to get real inflation forecast from Il Sole 24 Ore data
                inflation_forecast = _get_inflation_forecast(country, max_age_days)

                if inflation_forecast is not None:
                    # Adjust TIPS to nominal equivalent
                    # Nominal ≈ TIPS + Expected Inflation
                    adjusted_rate = rate + inflation_forecast
                    warnings.append(
                        f"TIPS adjusted: {rate:.2f}% + {inflation_forecast:.2f}% inflation forecast = {adjusted_rate:.2f}%"
                    )
                    rate = adjusted_rate
                    inflation_adjusted = True
                    data_source = f"{data_source} (inflation-adjusted)"
                else:
                    # No inflation forecast available, warn user
                    warnings.append(
                        f"Using TIPS {rate:.2f}% without adjustment (no inflation forecast available)"
                    )

            # Get macro confidence for audit trail (optional)
            macro_confidence = _get_macro_confidence(country, max_age_days)
            if macro_confidence is not None and macro_confidence < 0.5:
                warnings.append(
                    f"Low macro regime confidence: {macro_confidence:.2f}"
                )

            return RiskFreeRateResult(
                rate=rate / 100,  # Convert from % to decimal
                status='success',
                age_days=age_days,
                data_source=data_source,
                warnings=warnings,
                inflation_adjusted=inflation_adjusted,
                macro_confidence=macro_confidence
            )

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching {country} risk-free rate: {e}")
        return RiskFreeRateResult(
            rate=None,
            status='error',
            warnings=[f"Database error: {str(e)}"]
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching {country} risk-free rate: {e}")
        return RiskFreeRateResult(
            rate=None,
            status='error',
            warnings=[f"Unexpected error: {str(e)}"]
        )


# get_country_risk_free_rate is now just an alias to get_risk_free_rate
# for backward compatibility
def get_country_risk_free_rate(country: str, max_age_days: int = 30) -> float:
    """
    Get country-specific risk-free rate (10Y government bond).

    This is an alias to get_risk_free_rate() for backward compatibility.

    Args:
        country: Country code (USA, Germany, France, UK, Japan)
        max_age_days: Maximum age of database data (default: 30 days)

    Returns:
        Risk-free rate as decimal
    """
    return get_risk_free_rate(country=country, max_age_days=max_age_days)


def prefetch_all_risk_free_rates(
    countries: Optional[List[str]] = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    use_average: bool = DEFAULT_USE_AVERAGE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> dict:
    """
    Pre-fetch risk-free rates for multiple countries in a single batch.

    This is a performance optimization to avoid redundant database queries
    when processing many stocks from the same countries.

    Args:
        countries: List of country codes (defaults to all major countries)
        max_age_days: Maximum age of database data
        use_average: If True, average recent observations
        lookback_days: Number of days to average over

    Returns:
        Dictionary mapping country -> risk_free_rate (decimal)

    Example:
        >>> rates = prefetch_all_risk_free_rates()
        >>> us_rate = rates.get('USA', 0.045)
        >>> sharpe = (return - us_rate) / volatility
    """
    if countries is None:
        countries = ['USA', 'UK', 'Germany', 'France', 'Japan']

    logger.info(f"Pre-fetching risk-free rates for {len(countries)} countries...")

    rates = {}
    for country in countries:
        result = _get_risk_free_rate_from_db(
            country,
            max_age_days=max_age_days,
            use_average=use_average,
            lookback_days=lookback_days,
            fallback_to_stale=True
        )

        if result.is_successful():
            assert result.rate is not None
            rates[country] = result.rate
            logger.info(
                f"  ✓ {country}: {result.rate:.4f} ({result.rate_pct:.2f}%) "
                f"[source: {result.data_source}, age: {result.age_days}d]"
            )
            if result.warnings:
                for warning in result.warnings:
                    logger.info(f"    ⚠ {warning}")
        else:
            # Use fallback
            fallback = FALLBACK_RISK_FREE_RATES.get(country, 0.045)
            rates[country] = fallback
            logger.warning(
                f"  ✗ {country}: Using fallback {fallback:.4f} ({fallback*100:.2f}%) "
                f"[reason: {result.status}]"
            )

    logger.info(f"✓ Pre-fetched {len(rates)} risk-free rates")
    return rates


if __name__ == "__main__":
    """Test institutional-grade risk-free rate fetching"""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from app.database import init_db

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Initialize database
    init_db()

    print("=" * 80)
    print("INSTITUTIONAL-GRADE RISK-FREE RATE CALCULATOR")
    print("=" * 80)
    print()
    print("Features:")
    print("  ✓ Automatic TIPS adjustment using Il Sole 24 Ore inflation forecasts")
    print("  ✓ Rate smoothing via 5-day averaging")
    print("  ✓ Data quality validation with reasonable range checking")
    print("  ✓ Macro regime confidence tracking")
    print("  ✓ Robust error handling with audit trails")
    print()
    print("=" * 80)
    print()

    # Test country-specific rates with full diagnostic output
    print("Country-specific 10Y bond yields:")
    print()

    for country in ['USA', 'UK', 'Germany', 'France', 'Japan']:
        print(f"\n{country}:")
        print("-" * 40)

        # Get detailed result for demonstration
        result = _get_risk_free_rate_from_db(country)

        if result.is_successful():
            print(f"  Rate:             {result.rate:.4f} ({result.rate_pct:.2f}%)")
            print(f"  Status:           {result.status}")
            print(f"  Data Source:      {result.data_source}")
            print(f"  Data Age:         {result.age_days} days")
            print(f"  TIPS Adjusted:    {result.inflation_adjusted}")
            if result.macro_confidence:
                print(f"  Macro Confidence: {result.macro_confidence:.2f}")
            if result.warnings:
                print(f"  Warnings:")
                for warning in result.warnings:
                    print(f"    - {warning}")
        else:
            print(f"  Status:  {result.status}")
            print(f"  Rate:    Using fallback: {FALLBACK_RISK_FREE_RATES.get(country, 0.045):.4f}")
            if result.warnings:
                print(f"  Reason:  {', '.join(result.warnings)}")

    print()
    print("=" * 80)
    print()
    print("NOTE: All rates are from Trading Economics database with real-time")
    print("      inflation forecasts from Il Sole 24 Ore and macro confidence")
    print("      from LLM regime assessments.")
    print("=" * 80)
