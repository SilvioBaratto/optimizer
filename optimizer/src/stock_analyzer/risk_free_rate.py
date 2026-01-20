from typing import Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from sqlalchemy import select, desc, func
from sqlalchemy.exc import SQLAlchemyError

from optimizer.database.database import database_manager
from optimizer.database.models.trading_economics import (
    TradingEconomicsSnapshot,
    TradingEconomicsBondYield,
)
from optimizer.database.models.macro_regime import EconomicIndicators, CountryRegimeAssessment

# Fallback values (only used when database is unavailable)
FALLBACK_RISK_FREE_RATES = {
    "USA": 0.045,  # 4.5%
    "Germany": 0.026,  # 2.6%
    "France": 0.034,  # 3.4%
    "UK": 0.045,  # 4.5%
    "Japan": 0.017,  # 1.7%
}

# Reasonable ranges for risk-free rates by country (in %)
# Based on historical ranges - used for data quality validation
REASONABLE_RANGES = {
    "USA": (-2.0, 15.0),
    "UK": (-2.0, 15.0),
    "Germany": (-2.0, 10.0),
    "France": (-2.0, 12.0),
    "Japan": (-2.0, 5.0),
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
        return self.status == "success" and self.rate is not None


def _get_inflation_forecast(country: str, max_age_days: int = 30) -> Optional[float]:
    """Get real inflation forecast from EconomicIndicators table (Il Sole 24 Ore data)."""
    try:
        with database_manager.get_session() as session:
            query = (
                select(
                    EconomicIndicators.inflation_forecast_6m,
                    EconomicIndicators.data_timestamp,
                )
                .where(
                    EconomicIndicators.country == country,
                    EconomicIndicators.inflation_forecast_6m.isnot(None),
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
                    return None

                return inflation_forecast

            return None

    except Exception:
        return None


def _get_macro_confidence(country: str, max_age_days: int = 30) -> Optional[float]:
    """Get macro regime confidence from CountryRegimeAssessment (LLM forecasts)."""
    try:
        with database_manager.get_session() as session:
            query = (
                select(
                    CountryRegimeAssessment.confidence,
                    CountryRegimeAssessment.assessment_timestamp,
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
                    return None

                return confidence

            return None

    except Exception:
        return None


def _validate_rate(country: str, rate: float) -> bool:
    """Validate if rate is within reasonable range for the country."""
    min_rate, max_rate = REASONABLE_RANGES.get(country, (-2.0, 20.0))
    return min_rate <= rate <= max_rate


def _query_averaged_rate(
    session, country: str, lookback_days: int
) -> Tuple[Optional[float], Optional[datetime], Optional[str]]:
    """Query averaged bond yield over lookback period for smoothing."""
    cutoff_date = datetime.now() - timedelta(days=lookback_days)

    query = (
        select(
            func.avg(TradingEconomicsBondYield.yield_value).label("avg_yield"),
            func.max(TradingEconomicsSnapshot.fetch_timestamp).label("latest_date"),
            TradingEconomicsBondYield.raw_name,
        )
        .join(TradingEconomicsBondYield.snapshot)
        .where(
            TradingEconomicsSnapshot.country == country,
            TradingEconomicsBondYield.maturity == "10Y",
            TradingEconomicsSnapshot.fetch_timestamp >= cutoff_date,
        )
        .group_by(TradingEconomicsBondYield.raw_name)
        .order_by(desc("latest_date"))
        .limit(1)
    )

    result = session.execute(query).first()

    if result:
        return result.avg_yield, result.latest_date, result.raw_name

    return None, None, None


def _query_latest_rate(
    session, country: str
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
            TradingEconomicsBondYield.raw_name,
        )
        .join(TradingEconomicsBondYield.snapshot)
        .where(
            TradingEconomicsSnapshot.country == country,
            TradingEconomicsBondYield.maturity == "10Y",
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
    country: str = "USA",
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    use_average: bool = DEFAULT_USE_AVERAGE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    fallback_to_stale: bool = DEFAULT_FALLBACK_TO_STALE,
) -> float:
    """Get current risk-free rate from Trading Economics database (10Y government bond)."""

    # Try database first with institutional-grade logic
    result = _get_risk_free_rate_from_db(
        country, max_age_days, use_average, lookback_days, fallback_to_stale
    )

    if result.is_successful():
        # Type guard: is_successful() guarantees rate is not None
        assert result.rate is not None, "is_successful() guarantees rate is not None"
        return result.rate

    # Fallback to hardcoded value
    fallback = FALLBACK_RISK_FREE_RATES.get(country, 0.045)
    return fallback


def _get_risk_free_rate_from_db(
    country: str,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    use_average: bool = DEFAULT_USE_AVERAGE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    fallback_to_stale: bool = DEFAULT_FALLBACK_TO_STALE,
) -> RiskFreeRateResult:
    """Get risk-free rate from Trading Economics database with institutional-grade features."""
    try:
        with database_manager.get_session() as session:
            # Query bond yield (with optional averaging for smoothing)
            if use_average:
                rate, latest_date, data_source = _query_averaged_rate(
                    session, country, lookback_days
                )
            else:
                rate, latest_date, data_source = _query_latest_rate(session, country)

            if rate is None or latest_date is None:
                return RiskFreeRateResult(rate=None, status="no_data", data_source=data_source)

            # Check data freshness
            age_days = (datetime.now() - latest_date.replace(tzinfo=None)).days

            # Validate rate is reasonable
            if not _validate_rate(country, rate):
                return RiskFreeRateResult(
                    rate=None,
                    status="invalid_data",
                    age_days=age_days,
                    data_source=data_source,
                    warnings=[f"Rate {rate:.2f}% outside reasonable range"],
                )

            # Handle stale data
            warnings = []
            if age_days > max_age_days:
                warning_msg = f"Data is {age_days} days old (max: {max_age_days})"
                if fallback_to_stale:
                    warnings.append(warning_msg)
                else:
                    return RiskFreeRateResult(
                        rate=None,
                        status="stale_data",
                        age_days=age_days,
                        data_source=data_source,
                        warnings=[warning_msg],
                    )

            # Handle TIPS adjustment using REAL inflation forecasts
            inflation_adjusted = False
            if data_source and "TIPS" in data_source.upper():
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
                warnings.append(f"Low macro regime confidence: {macro_confidence:.2f}")

            return RiskFreeRateResult(
                rate=rate / 100,  # Convert from % to decimal
                status="success",
                age_days=age_days,
                data_source=data_source,
                warnings=warnings,
                inflation_adjusted=inflation_adjusted,
                macro_confidence=macro_confidence,
            )

    except SQLAlchemyError as e:
        return RiskFreeRateResult(rate=None, status="error", warnings=[f"Database error: {str(e)}"])
    except Exception as e:
        return RiskFreeRateResult(
            rate=None, status="error", warnings=[f"Unexpected error: {str(e)}"]
        )


# get_country_risk_free_rate is now just an alias to get_risk_free_rate
# for backward compatibility
def get_country_risk_free_rate(country: str, max_age_days: int = 30) -> float:
    """Get country-specific risk-free rate (10Y government bond)."""
    return get_risk_free_rate(country=country, max_age_days=max_age_days)


def prefetch_all_risk_free_rates(
    countries: Optional[List[str]] = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    use_average: bool = DEFAULT_USE_AVERAGE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict:
    """Pre-fetch risk-free rates for multiple countries in a single batch."""
    if countries is None:
        countries = ["USA", "UK", "Germany", "France", "Japan"]

    rates = {}
    for country in countries:
        result = _get_risk_free_rate_from_db(
            country,
            max_age_days=max_age_days,
            use_average=use_average,
            lookback_days=lookback_days,
            fallback_to_stale=True,
        )

        if result.is_successful():
            assert result.rate is not None
            rates[country] = result.rate
        else:
            # Use fallback
            fallback = FALLBACK_RISK_FREE_RATES.get(country, 0.045)
            rates[country] = fallback

    return rates
