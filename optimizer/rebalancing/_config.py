"""Configuration for rebalancing frameworks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RebalancingFrequency(str, Enum):
    """Calendar-based rebalancing frequency.

    Each value corresponds to the approximate number of trading
    days in the rebalancing period.
    """

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"


class ThresholdType(str, Enum):
    """Threshold convention for drift-based rebalancing."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


# Trading days per frequency
TRADING_DAYS: dict[RebalancingFrequency, int] = {
    RebalancingFrequency.MONTHLY: 21,
    RebalancingFrequency.QUARTERLY: 63,
    RebalancingFrequency.SEMIANNUAL: 126,
    RebalancingFrequency.ANNUAL: 252,
}


@dataclass(frozen=True)
class CalendarRebalancingConfig:
    """Immutable configuration for calendar-based rebalancing.

    Triggers portfolio reconstruction at fixed intervals regardless
    of portfolio drift.

    Parameters
    ----------
    frequency : RebalancingFrequency
        Rebalancing frequency.
    """

    frequency: RebalancingFrequency = RebalancingFrequency.QUARTERLY

    @property
    def trading_days(self) -> int:
        """Number of trading days between rebalances."""
        return TRADING_DAYS[self.frequency]

    @classmethod
    def for_monthly(cls) -> CalendarRebalancingConfig:
        """Monthly rebalancing (21 trading days)."""
        return cls(frequency=RebalancingFrequency.MONTHLY)

    @classmethod
    def for_quarterly(cls) -> CalendarRebalancingConfig:
        """Quarterly rebalancing (63 trading days)."""
        return cls(frequency=RebalancingFrequency.QUARTERLY)

    @classmethod
    def for_semiannual(cls) -> CalendarRebalancingConfig:
        """Semiannual rebalancing (126 trading days)."""
        return cls(frequency=RebalancingFrequency.SEMIANNUAL)

    @classmethod
    def for_annual(cls) -> CalendarRebalancingConfig:
        """Annual rebalancing (252 trading days)."""
        return cls(frequency=RebalancingFrequency.ANNUAL)


@dataclass(frozen=True)
class ThresholdRebalancingConfig:
    """Immutable configuration for threshold-based rebalancing.

    Rebalances only when portfolio drift exceeds specified limits,
    avoiding unnecessary turnover during stable periods.

    Parameters
    ----------
    threshold_type : ThresholdType
        Whether to use absolute or relative drift thresholds.
    threshold : float
        Drift threshold.  For absolute: percentage points of weight
        (e.g. 0.05 = 5pp).  For relative: fraction of target weight
        (e.g. 0.25 = 25% deviation).
    """

    threshold_type: ThresholdType = ThresholdType.ABSOLUTE
    threshold: float = 0.05

    @classmethod
    def for_absolute(cls, threshold: float = 0.05) -> ThresholdRebalancingConfig:
        """Absolute drift threshold (default 5pp)."""
        return cls(
            threshold_type=ThresholdType.ABSOLUTE,
            threshold=threshold,
        )

    @classmethod
    def for_relative(cls, threshold: float = 0.25) -> ThresholdRebalancingConfig:
        """Relative drift threshold (default 25%)."""
        return cls(
            threshold_type=ThresholdType.RELATIVE,
            threshold=threshold,
        )


@dataclass(frozen=True)
class HybridRebalancingConfig:
    """Hybrid rebalancing: check threshold only at calendar review dates.

    Combines calendar and threshold strategies: the portfolio is reviewed
    at regular calendar intervals, but trades are executed only when drift
    exceeds the threshold at that review date.  Between review dates,
    ``should_rebalance_hybrid`` always returns ``False`` regardless of drift.

    Parameters
    ----------
    calendar : CalendarRebalancingConfig
        Calendar schedule that defines review dates.
    threshold : ThresholdRebalancingConfig
        Drift threshold evaluated at each review date.
    """

    calendar: CalendarRebalancingConfig = field(
        default_factory=CalendarRebalancingConfig
    )
    threshold: ThresholdRebalancingConfig = field(
        default_factory=ThresholdRebalancingConfig
    )

    @classmethod
    def for_monthly_with_5pct_threshold(cls) -> HybridRebalancingConfig:
        """Monthly review with 5pp absolute drift threshold."""
        return cls(
            calendar=CalendarRebalancingConfig.for_monthly(),
            threshold=ThresholdRebalancingConfig.for_absolute(threshold=0.05),
        )

    @classmethod
    def for_quarterly_with_10pct_threshold(cls) -> HybridRebalancingConfig:
        """Quarterly review with 10pp absolute drift threshold."""
        return cls(
            calendar=CalendarRebalancingConfig.for_quarterly(),
            threshold=ThresholdRebalancingConfig.for_absolute(threshold=0.10),
        )
