"""Configuration for investability screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from optimizer.exceptions import ConfigurationError


class ExchangeRegion(str, Enum):
    """Exchange region for region-specific screening thresholds."""

    US = "us"
    EUROPE = "europe"


@dataclass(frozen=True)
class HysteresisConfig:
    """Entry/exit thresholds with hysteresis to reduce turnover.

    Setting exit below entry prevents marginal stocks from
    oscillating in and out of the universe with small fluctuations.

    Parameters
    ----------
    entry : float
        Threshold a stock must exceed to enter the universe.
    exit_ : float
        Threshold below which a current member is removed.
        Must be <= ``entry``.
    """

    entry: float
    exit_: float

    def __post_init__(self) -> None:
        if self.exit_ > self.entry:
            msg = f"exit_ ({self.exit_}) must be <= entry ({self.entry})"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class InvestabilityScreenConfig:
    """Immutable configuration for investability screening.

    Enforces minimum standards of market capitalization, liquidity,
    price level, listing history, and data availability.  All
    hysteresis thresholds use separate entry/exit values to reduce
    turnover at screen boundaries.

    Parameters
    ----------
    market_cap : HysteresisConfig
        Free-float market capitalization thresholds (USD).
    addv_12m : HysteresisConfig
        12-month average daily dollar volume thresholds (USD).
    addv_3m : HysteresisConfig
        3-month average daily dollar volume thresholds (USD).
    trading_frequency : HysteresisConfig
        Fraction of trading days with nonzero volume (0-1).
    price_us : HysteresisConfig
        Minimum price for US-listed equities (USD).
    price_europe : HysteresisConfig
        Minimum price for European-listed equities (local currency).
    min_trading_history : int
        Minimum trading days of price history required.
    min_ipo_seasoning : int
        Minimum trading days since first price observation.
    min_annual_reports : int
        Minimum annual financial statements required.
    min_quarterly_reports : int
        Minimum quarterly financial statements required.
    exchange_region : ExchangeRegion
        Region for price threshold selection.
    mcap_percentile_entry : float
        Minimum exchange-percentile rank (0-1) for entry.  A stock must
        exceed BOTH the absolute ``market_cap.entry`` floor AND this
        percentile within its exchange to enter the universe.  Defaults
        to the 10th percentile (0.10).  Requires an ``exchange`` column
        in the ``fundamentals`` DataFrame passed to
        ``apply_investability_screens``.
    mcap_percentile_exit : float
        Minimum exchange-percentile rank (0-1) for existing members to
        avoid removal.  Must be <= ``mcap_percentile_entry``.  Defaults
        to the 7.5th percentile (0.075).
    """

    market_cap: HysteresisConfig = field(
        default_factory=lambda: HysteresisConfig(entry=200_000_000, exit_=150_000_000)
    )
    addv_12m: HysteresisConfig = field(
        default_factory=lambda: HysteresisConfig(entry=750_000, exit_=500_000)
    )
    addv_3m: HysteresisConfig = field(
        default_factory=lambda: HysteresisConfig(entry=500_000, exit_=350_000)
    )
    trading_frequency: HysteresisConfig = field(
        default_factory=lambda: HysteresisConfig(entry=0.95, exit_=0.90)
    )
    price_us: HysteresisConfig = field(
        default_factory=lambda: HysteresisConfig(entry=3.0, exit_=2.0)
    )
    price_europe: HysteresisConfig = field(
        default_factory=lambda: HysteresisConfig(entry=2.0, exit_=1.5)
    )
    min_trading_history: int = 252
    min_ipo_seasoning: int = 60
    min_annual_reports: int = 3
    min_quarterly_reports: int = 8
    exchange_region: ExchangeRegion = ExchangeRegion.US
    mcap_percentile_entry: float = 0.10
    mcap_percentile_exit: float = 0.075

    def __post_init__(self) -> None:
        if not (0.0 <= self.mcap_percentile_exit <= self.mcap_percentile_entry <= 1.0):
            msg = (
                f"mcap_percentile_exit ({self.mcap_percentile_exit}) must be "
                f"<= mcap_percentile_entry ({self.mcap_percentile_entry}) "
                f"and both must be in [0, 1]"
            )
            raise ConfigurationError(msg)

    @classmethod
    def for_developed_markets(cls) -> InvestabilityScreenConfig:
        """Strict thresholds for developed-market institutional universes."""
        return cls()

    @classmethod
    def for_broad_universe(cls) -> InvestabilityScreenConfig:
        """Relaxed thresholds for broader coverage."""
        return cls(
            market_cap=HysteresisConfig(entry=100_000_000, exit_=75_000_000),
            addv_12m=HysteresisConfig(entry=500_000, exit_=350_000),
            addv_3m=HysteresisConfig(entry=300_000, exit_=200_000),
            trading_frequency=HysteresisConfig(entry=0.90, exit_=0.85),
            price_us=HysteresisConfig(entry=2.0, exit_=1.0),
            price_europe=HysteresisConfig(entry=1.0, exit_=0.5),
            min_trading_history=126,
            min_ipo_seasoning=30,
            min_annual_reports=2,
            min_quarterly_reports=4,
        )

    @classmethod
    def for_small_cap(cls) -> InvestabilityScreenConfig:
        """Thresholds appropriate for small-cap universes."""
        return cls(
            market_cap=HysteresisConfig(entry=50_000_000, exit_=35_000_000),
            addv_12m=HysteresisConfig(entry=250_000, exit_=150_000),
            addv_3m=HysteresisConfig(entry=150_000, exit_=100_000),
            trading_frequency=HysteresisConfig(entry=0.90, exit_=0.85),
            price_us=HysteresisConfig(entry=1.0, exit_=0.5),
            price_europe=HysteresisConfig(entry=0.5, exit_=0.25),
            min_trading_history=126,
            min_ipo_seasoning=30,
            min_annual_reports=2,
            min_quarterly_reports=4,
        )
