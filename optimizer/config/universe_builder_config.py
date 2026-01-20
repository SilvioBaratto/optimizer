"""
Universe Builder Configuration - Externalized thresholds for universe building.

This module extracts all constants from build_universe.py into a structured
configuration dataclass following the existing pattern in the codebase.

Design Principles:
- Frozen dataclass for immutability
- Factory methods for different contexts (from_env, for_regime)
- All thresholds documented with their purpose
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass(frozen=True)
class LiquidityTier:
    """Liquidity requirements for a market cap segment."""

    min_adv_dollars: float  # Minimum average daily dollar volume
    min_adv_shares: int  # Minimum average daily share volume

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "min_adv_dollars": self.min_adv_dollars,
            "min_adv_shares": self.min_adv_shares,
        }


@dataclass(frozen=True)
class InstitutionalFieldSpec:
    """Specification for an institutional data category."""

    fields: Tuple[str, ...]
    description: str
    required: bool = True
    at_least_one: bool = False
    all_required: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "fields": list(self.fields),
            "description": self.description,
            "required": self.required,
            "at_least_one": self.at_least_one,
            "all_required": self.all_required,
        }


@dataclass(frozen=True)
class UniverseBuilderConfig:
    """
    Configuration for universe building from Trading212 API.

    All thresholds and parameters for the UniverseBuilder are externalized here.
    This enables easy testing and environment-based configuration.

    Attributes:
        # Market Cap Thresholds
        min_market_cap: Absolute minimum market cap ($100M default)
        small_cap_threshold: Upper bound for small-cap ($2B)
        mid_cap_threshold: Upper bound for mid-cap ($10B)
        large_cap_threshold: Lower bound for large-cap ($10B)

        # Price Filters
        min_price: Minimum share price ($5)
        max_price: Maximum share price ($10,000 - data error check)

        # Liquidity Tiers
        liquidity_tiers: Dict mapping market cap segment to LiquidityTier

        # Historical Data Requirements
        min_trading_days: Minimum days of historical data (~3 years sanity check)

        # Portfolio Countries
        portfolio_countries: Tuple of country names to include
        country_to_exchanges: Mapping of country -> list of exchange names
    """

    # Market Cap Thresholds
    min_market_cap: float = 100_000_000  # $100M absolute minimum
    small_cap_threshold: float = 2_000_000_000  # $100M-2B (small-cap)
    mid_cap_threshold: float = 10_000_000_000  # $2B-10B (mid-cap)
    large_cap_threshold: float = 10_000_000_000  # $10B+ (large-cap)

    # Price Filters
    min_price: float = 5.0  # $5 minimum share price
    max_price: float = 10_000.0  # $10,000 maximum (data error check)

    # Liquidity by market cap segment
    liquidity_tiers: Dict[str, LiquidityTier] = field(
        default_factory=lambda: {
            "large_cap": LiquidityTier(
                min_adv_dollars=10_000_000, min_adv_shares=500_000
            ),
            "mid_cap": LiquidityTier(
                min_adv_dollars=5_000_000, min_adv_shares=250_000
            ),
            "small_cap": LiquidityTier(
                min_adv_dollars=1_000_000, min_adv_shares=100_000
            ),
        }
    )

    # Historical data requirements
    min_trading_days: int = 750  # ~3 years sanity check for period='5y'

    # Portfolio countries
    portfolio_countries: Tuple[str, ...] = (
        "USA",  # NYSE, NASDAQ
        "Germany",  # Deutsche Börse Xetra
        "France",  # Euronext Paris
        "UK",  # London Stock Exchange
    )

    # Mapping of countries to their Trading212 exchanges
    country_to_exchanges: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "USA": ("NYSE", "NASDAQ"),
            "Germany": ("Deutsche Börse Xetra",),  # Gettex removed - not supported by Yahoo Finance
            "France": ("Euronext Paris",),
            "UK": ("London Stock Exchange",),
        }
    )

    # Institutional data coverage requirements
    institutional_fields: Dict[str, InstitutionalFieldSpec] = field(
        default_factory=lambda: {
            "market_cap": InstitutionalFieldSpec(
                fields=("marketCap",),
                description="Market capitalization (MIN $100M)",
                required=True,
            ),
            "price": InstitutionalFieldSpec(
                fields=("currentPrice", "regularMarketPrice"),
                description="Current stock price (MIN $5)",
                required=True,
                at_least_one=True,
            ),
            "volume": InstitutionalFieldSpec(
                fields=("averageVolume", "averageVolume10days"),
                description="Average daily volume (MIN $1M dollar volume)",
                required=True,
                at_least_one=True,
            ),
            "shares_outstanding": InstitutionalFieldSpec(
                fields=("sharesOutstanding",),
                description="Total shares outstanding",
                required=True,
            ),
            "beta": InstitutionalFieldSpec(
                fields=("beta",),
                description="Market beta (risk metric)",
                required=False,
            ),
            "sector_industry": InstitutionalFieldSpec(
                fields=("sector", "industry"),
                description="GICS sector and industry classification",
                required=True,
                all_required=True,
            ),
            "exchange": InstitutionalFieldSpec(
                fields=("exchange",),
                description="Primary exchange listing",
                required=True,
            ),
            "financial_ratios": InstitutionalFieldSpec(
                fields=("trailingPE", "priceToBook"),
                description="Valuation ratios (P/E, P/B)",
                required=True,
                at_least_one=True,
            ),
            "profitability": InstitutionalFieldSpec(
                fields=("returnOnEquity", "returnOnAssets", "profitMargins"),
                description="Profitability metrics (ROE, ROA, margins)",
                required=True,
                at_least_one=True,
            ),
            "debt_metrics": InstitutionalFieldSpec(
                fields=("debtToEquity", "totalDebt", "totalAssets"),
                description="Debt and balance sheet metrics",
                required=True,
                at_least_one=True,
            ),
            "dividend_data": InstitutionalFieldSpec(
                fields=("dividendYield", "dividendRate"),
                description="Dividend information",
                required=False,
                at_least_one=True,
            ),
            "52week_range": InstitutionalFieldSpec(
                fields=("fiftyTwoWeekHigh", "fiftyTwoWeekLow"),
                description="52-week price range",
                required=True,
                all_required=True,
            ),
        }
    )

    # Yahoo Finance suffix mapping by exchange
    yahoo_suffix_map: Dict[str, str] = field(
        default_factory=lambda: {
            # US exchanges (no suffix)
            "NYSE": "",
            "NASDAQ": "",
            # UK
            "London Stock Exchange": ".L",
            # France
            "Euronext Paris": ".PA",
            # Germany (only XETRA - Gettex not supported by Yahoo Finance)
            "Deutsche Börse Xetra": ".DE",
        }
    )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "min_market_cap": self.min_market_cap,
            "small_cap_threshold": self.small_cap_threshold,
            "mid_cap_threshold": self.mid_cap_threshold,
            "large_cap_threshold": self.large_cap_threshold,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "liquidity_tiers": {k: v.to_dict() for k, v in self.liquidity_tiers.items()},
            "min_trading_days": self.min_trading_days,
            "portfolio_countries": list(self.portfolio_countries),
        }

    def get_allowed_exchanges(self) -> set:
        """
        Get set of allowed exchange names based on portfolio_countries.

        Returns:
            Set of exchange names to include in universe
        """
        allowed = set()
        for country in self.portfolio_countries:
            exchanges = self.country_to_exchanges.get(country, ())
            allowed.update(exchanges)
        return allowed

    def is_exchange_allowed(self, exchange_name: str) -> bool:
        """
        Check if an exchange should be included based on portfolio_countries.

        Args:
            exchange_name: Name of the exchange

        Returns:
            True if exchange is in a portfolio country
        """
        return exchange_name in self.get_allowed_exchanges()

    def determine_market_cap_segment(self, market_cap: float) -> str:
        """
        Determine market cap segment for a given market cap value.

        Args:
            market_cap: Market capitalization in dollars

        Returns:
            'large_cap', 'mid_cap', or 'small_cap'
        """
        if market_cap >= self.large_cap_threshold:
            return "large_cap"
        elif market_cap >= self.small_cap_threshold:
            return "mid_cap"
        else:
            return "small_cap"

    def get_liquidity_requirements(self, market_cap: float) -> LiquidityTier:
        """
        Get liquidity requirements for a given market cap.

        Args:
            market_cap: Market capitalization in dollars

        Returns:
            LiquidityTier with ADV requirements
        """
        segment = self.determine_market_cap_segment(market_cap)
        return self.liquidity_tiers[segment]

    def get_yahoo_suffix(self, exchange_name: str) -> Optional[str]:
        """
        Get Yahoo Finance ticker suffix for an exchange.

        Args:
            exchange_name: Trading212 exchange name

        Returns:
            Yahoo Finance suffix (e.g., '.L', '.DE') or None if not found
        """
        return self.yahoo_suffix_map.get(exchange_name)

    @classmethod
    def from_env(cls) -> "UniverseBuilderConfig":
        """
        Create configuration from environment variables.

        Environment variables:
        - UNIVERSE_MIN_MARKET_CAP
        - UNIVERSE_MIN_PRICE
        - UNIVERSE_MAX_PRICE
        - UNIVERSE_MIN_TRADING_DAYS
        - UNIVERSE_PORTFOLIO_COUNTRIES (comma-separated)
        """

        def get_float(name: str, default: float) -> float:
            val = os.getenv(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def get_int(name: str, default: int) -> int:
            val = os.getenv(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def get_countries(default: Tuple[str, ...]) -> Tuple[str, ...]:
            val = os.getenv("UNIVERSE_PORTFOLIO_COUNTRIES")
            if val is None:
                return default
            return tuple(c.strip() for c in val.split(",") if c.strip())

        return cls(
            min_market_cap=get_float("UNIVERSE_MIN_MARKET_CAP", 100_000_000),
            min_price=get_float("UNIVERSE_MIN_PRICE", 5.0),
            max_price=get_float("UNIVERSE_MAX_PRICE", 10_000.0),
            min_trading_days=get_int("UNIVERSE_MIN_TRADING_DAYS", 750),
            portfolio_countries=get_countries(cls.portfolio_countries),
        )

    @classmethod
    def for_regime(cls, regime: str) -> "UniverseBuilderConfig":
        """
        Create regime-appropriate configuration.

        Args:
            regime: Macro regime (early_cycle, mid_cycle, late_cycle, recession)

        Returns:
            Regime-appropriate UniverseBuilderConfig
        """
        if regime == "recession":
            # Very strict in recession - only highest quality
            return cls(
                min_market_cap=500_000_000,  # $500M minimum
                min_price=10.0,
                liquidity_tiers={
                    "large_cap": LiquidityTier(
                        min_adv_dollars=20_000_000, min_adv_shares=1_000_000
                    ),
                    "mid_cap": LiquidityTier(
                        min_adv_dollars=10_000_000, min_adv_shares=500_000
                    ),
                    "small_cap": LiquidityTier(
                        min_adv_dollars=5_000_000, min_adv_shares=250_000
                    ),
                },
            )
        elif regime == "late_cycle":
            # More conservative in late cycle
            return cls(
                min_market_cap=200_000_000,  # $200M minimum
                min_price=7.0,
                liquidity_tiers={
                    "large_cap": LiquidityTier(
                        min_adv_dollars=15_000_000, min_adv_shares=750_000
                    ),
                    "mid_cap": LiquidityTier(
                        min_adv_dollars=7_500_000, min_adv_shares=375_000
                    ),
                    "small_cap": LiquidityTier(
                        min_adv_dollars=2_000_000, min_adv_shares=150_000
                    ),
                },
            )
        elif regime == "early_cycle":
            # More aggressive in early cycle
            return cls(
                min_market_cap=75_000_000,  # $75M minimum
                min_price=3.0,
                liquidity_tiers={
                    "large_cap": LiquidityTier(
                        min_adv_dollars=7_500_000, min_adv_shares=400_000
                    ),
                    "mid_cap": LiquidityTier(
                        min_adv_dollars=3_500_000, min_adv_shares=200_000
                    ),
                    "small_cap": LiquidityTier(
                        min_adv_dollars=750_000, min_adv_shares=75_000
                    ),
                },
            )
        else:
            # Default (mid_cycle or uncertain)
            return cls()
