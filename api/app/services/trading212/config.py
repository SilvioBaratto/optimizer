"""
Universe Builder Configuration - Externalized thresholds for universe building.

Design Principles:
- Frozen dataclass for immutability
- Factory methods for different contexts (for_regime)
- All thresholds documented with their purpose
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LiquidityTier:
    """Liquidity requirements for a market cap segment."""

    min_adv_dollars: float
    min_adv_shares: int

    def to_dict(self) -> dict:
        return {
            "min_adv_dollars": self.min_adv_dollars,
            "min_adv_shares": self.min_adv_shares,
        }


@dataclass(frozen=True)
class InstitutionalFieldSpec:
    """Specification for an institutional data category."""

    fields: tuple[str, ...]
    description: str
    required: bool = True
    at_least_one: bool = False
    all_required: bool = False

    def to_dict(self) -> dict:
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
    """

    # Market Cap Thresholds
    min_market_cap: float = 100_000_000
    small_cap_threshold: float = 2_000_000_000
    mid_cap_threshold: float = 10_000_000_000
    large_cap_threshold: float = 10_000_000_000

    # Price Filters
    min_price: float = 5.0
    max_price: float = 10_000.0

    # Liquidity by market cap segment
    liquidity_tiers: dict[str, LiquidityTier] = field(
        default_factory=lambda: {
            "large_cap": LiquidityTier(
                min_adv_dollars=10_000_000, min_adv_shares=500_000
            ),
            "mid_cap": LiquidityTier(min_adv_dollars=5_000_000, min_adv_shares=250_000),
            "small_cap": LiquidityTier(
                min_adv_dollars=1_000_000, min_adv_shares=100_000
            ),
        }
    )

    # Historical data requirements
    min_trading_days: int = 750

    # Portfolio countries
    portfolio_countries: tuple[str, ...] = (
        "USA",
        "Germany",
        "France",
        "UK",
    )

    # Mapping of countries to their Trading212 exchanges
    country_to_exchanges: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "USA": ("NYSE", "NASDAQ"),
            "Germany": ("Deutsche Börse Xetra",),
            "France": ("Euronext Paris",),
            "UK": ("London Stock Exchange",),
        }
    )

    # Institutional data coverage requirements
    institutional_fields: dict[str, InstitutionalFieldSpec] = field(
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
    yahoo_suffix_map: dict[str, str] = field(
        default_factory=lambda: {
            "NYSE": "",
            "NASDAQ": "",
            "London Stock Exchange": ".L",
            "Euronext Paris": ".PA",
            "Deutsche Börse Xetra": ".DE",
        }
    )

    def to_dict(self) -> dict:
        return {
            "min_market_cap": self.min_market_cap,
            "small_cap_threshold": self.small_cap_threshold,
            "mid_cap_threshold": self.mid_cap_threshold,
            "large_cap_threshold": self.large_cap_threshold,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "liquidity_tiers": {
                k: v.to_dict() for k, v in self.liquidity_tiers.items()
            },
            "min_trading_days": self.min_trading_days,
            "portfolio_countries": list(self.portfolio_countries),
        }

    def get_allowed_exchanges(self) -> set:
        allowed = set()
        for country in self.portfolio_countries:
            exchanges = self.country_to_exchanges.get(country, ())
            allowed.update(exchanges)
        return allowed

    def is_exchange_allowed(self, exchange_name: str) -> bool:
        return exchange_name in self.get_allowed_exchanges()

    def determine_market_cap_segment(self, market_cap: float) -> str:
        if market_cap >= self.large_cap_threshold:
            return "large_cap"
        elif market_cap >= self.small_cap_threshold:
            return "mid_cap"
        else:
            return "small_cap"

    def get_liquidity_requirements(self, market_cap: float) -> LiquidityTier:
        segment = self.determine_market_cap_segment(market_cap)
        return self.liquidity_tiers[segment]

    def get_yahoo_suffix(self, exchange_name: str) -> str | None:
        return self.yahoo_suffix_map.get(exchange_name)

    @classmethod
    def for_regime(cls, regime: str) -> "UniverseBuilderConfig":
        if regime == "recession":
            return cls(
                min_market_cap=500_000_000,
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
            return cls(
                min_market_cap=200_000_000,
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
            return cls(
                min_market_cap=75_000_000,
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
            return cls()
