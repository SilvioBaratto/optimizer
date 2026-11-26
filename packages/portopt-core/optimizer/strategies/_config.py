"""Configuration and enums for the strategies module."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from optimizer.exceptions import ConfigurationError

_MIN_PERIODS_PER_YEAR: Final = 1
_MIN_EXECUTION_LAG: Final = 0


def require_integer(name: str, value: object, minimum: int) -> int:
    """Return a setting as an ``int``, raising unless it is a whole number.

    ``bool`` is rejected although it subclasses ``int``, and so is every float,
    including NaN and integral values such as ``3.0``: a window or a lag written
    as a float is a mistake worth surfacing before it reaches pandas or numpy,
    where a NaN would slip past a ``<`` comparison.

    Args:
        name: Argument name used in the error message.
        value: Candidate setting; Python and numpy integers are accepted.
        minimum: Smallest accepted value.

    Returns:
        The value as a Python ``int``.

    Raises:
        ConfigurationError: If the value is not an integer or is below ``minimum``.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        msg = f"{name} must be an integer of at least {minimum}, got {value!r}"
        raise ConfigurationError(msg)
    return int(value)


def require_real(name: str, value: object) -> float:
    """Return a setting as a ``float``, raising unless it is a real number.

    Args:
        name: Argument name used in the error message.
        value: Candidate setting; ``bool``, text and ``None`` are rejected.

    Returns:
        The value as a Python ``float``; it may still be NaN or infinite, which
        the caller must rule out for its own setting.

    Raises:
        ConfigurationError: If the value is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"{name} must be a real number, got {value!r}"
        raise ConfigurationError(msg)
    return float(value)


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
        bps = require_real("proportional_bps", self.proportional_bps)
        if not math.isfinite(bps) or bps < 0.0:
            msg = f"proportional_bps must be finite and non-negative, got {bps}"
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
        require_integer(
            "periods_per_year", self.periods_per_year, _MIN_PERIODS_PER_YEAR
        )
        require_integer("execution_lag", self.execution_lag, _MIN_EXECUTION_LAG)
        capital = require_real("initial_capital", self.initial_capital)
        if not math.isfinite(capital) or capital <= 0.0:
            msg = f"initial_capital must be finite and positive, got {capital}"
            raise ConfigurationError(msg)
        if not isinstance(self.costs, CostConfig):
            msg = f"costs must be a CostConfig, got {type(self.costs).__name__}"
            raise ConfigurationError(msg)
