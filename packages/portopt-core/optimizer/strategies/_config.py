"""Configuration and enums for the strategies module."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from optimizer.exceptions import ConfigurationError

_MIN_PERIODS_PER_YEAR: Final = 1


class MovingAverageKind(str, Enum):
    """Weighting scheme of a moving average."""

    SMA = "sma"
    EMA = "ema"


@dataclass(frozen=True)
class CostConfig:
    """Immutable configuration of the trading cost model.

    Attributes:
        proportional_bps: Cost in basis points of every unit of traded notional,
            where the notional is expressed as a fraction of equity. The cost is
            one-way: selling 1 and buying 1 pays it twice.
    """

    proportional_bps: float = 10.0

    def __post_init__(self) -> None:
        """Reject a cost that is negative or not a finite number."""
        if not math.isfinite(self.proportional_bps) or self.proportional_bps < 0.0:
            msg = (
                "proportional_bps must be finite and non-negative, "
                f"got {self.proportional_bps}"
            )
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable configuration of a weight backtest.

    Attributes:
        periods_per_year: Periods per year, used to annualise returns, volatility
            and the Sharpe ratio (252 for daily data).
        execution_lag: Periods between the date a weight is decided and the first
            period it earns a return. The weight decided at the close of row ``t``
            earns the return of row ``t + execution_lag``, so the default of 1
            means "decide at the close, hold over the next period". A lag of 0
            credits the weight with the return that ended at the very close it
            was decided on, which is a look-ahead leak kept only so that the leak
            can be measured.
        costs: Trading cost model.
        initial_capital: Equity at the start, in any currency unit; it scales the
            equity curve only.
    """

    periods_per_year: int = 252
    execution_lag: int = 1
    costs: CostConfig = field(default_factory=CostConfig)
    initial_capital: float = 1.0

    def __post_init__(self) -> None:
        """Reject conventions the backtest cannot honour."""
        if self.periods_per_year < _MIN_PERIODS_PER_YEAR:
            msg = (
                f"periods_per_year must be at least {_MIN_PERIODS_PER_YEAR}, "
                f"got {self.periods_per_year}"
            )
            raise ConfigurationError(msg)
        if self.execution_lag < 0:
            msg = f"execution_lag must not be negative, got {self.execution_lag}"
            raise ConfigurationError(msg)
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            msg = (
                "initial_capital must be finite and positive, "
                f"got {self.initial_capital}"
            )
            raise ConfigurationError(msg)
