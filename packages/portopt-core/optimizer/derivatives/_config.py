"""Option types and the delta-hedging simulation settings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from optimizer.derivatives._validation import (
    FloatArray,
    FloatOrArray,
    require_integer,
    require_real,
)
from optimizer.exceptions import ConfigurationError

_MIN_PATHS = 2


class OptionType(str, Enum):
    """Whether the option pays on a rise (call) or on a fall (put) of the spot."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: OptionType | str) -> OptionType:
        """Return the option type for an enum member or its string value.

        Args:
            value: ``OptionType.CALL``, ``OptionType.PUT``, ``"call"`` or ``"put"``.

        Returns:
            The matching enum member.

        Raises:
            ConfigurationError: If the value names neither a call nor a put.
        """
        try:
            return cls(value)
        except ValueError as exc:
            msg = f"option_type must be 'call' or 'put', got {value!r}"
            raise ConfigurationError(msg) from exc

    @property
    def sign(self) -> float:
        """Return +1.0 for a call and -1.0 for a put."""
        return 1.0 if self is OptionType.CALL else -1.0

    def payoff(self, spot: FloatOrArray, strike: FloatOrArray) -> FloatArray:
        """Return the exercise value, the payoff at expiry.

        Args:
            spot: Underlying price or prices.
            strike: Strike price or prices.

        Returns:
            ``max(S - K, 0)`` for a call and ``max(K - S, 0)`` for a put, with the
            broadcast shape of the inputs.
        """
        exercise = self.sign * (np.asarray(spot) - np.asarray(strike))
        return np.asarray(np.maximum(exercise, 0.0), dtype=np.float64)


@dataclass(frozen=True)
class DeltaHedgeConfig:
    """Immutable settings of a delta-hedging simulation.

    The hedger sells one option at the Black-Scholes price for ``hedge_volatility``
    and hedges with the Black-Scholes delta for that same volatility, while the
    underlying moves with ``true_volatility``. Setting the two apart shows what a
    wrong volatility estimate costs.

    Attributes:
        n_paths: Number of simulated price paths.
        n_rebalances: Number of equally spaced hedge dates, the first at inception.
        drift: Real-world annual drift of the underlying.
        true_volatility: Annual volatility the underlying actually has.
        hedge_volatility: Annual volatility used to price and hedge.
        rate: Continuously compounded risk-free rate that finances the hedge.
        transaction_cost_bps: Proportional cost of every trade in basis points of
            the traded value, the initial hedge included.
        random_state: Seed of the random generator; ``None`` draws fresh entropy.
    """

    n_paths: int = 5000
    n_rebalances: int = 252
    drift: float = 0.05
    true_volatility: float = 0.2
    hedge_volatility: float = 0.2
    rate: float = 0.02
    transaction_cost_bps: float = 0.0
    random_state: int | None = None

    def __post_init__(self) -> None:
        """Reject settings the simulation cannot use."""
        require_integer("n_paths", self.n_paths, _MIN_PATHS)
        require_integer("n_rebalances", self.n_rebalances, 1)
        _require_positive("true_volatility", self.true_volatility)
        _require_positive("hedge_volatility", self.hedge_volatility)
        _require_finite("drift", self.drift)
        _require_finite("rate", self.rate)
        if _require_finite("transaction_cost_bps", self.transaction_cost_bps) < 0.0:
            msg = (
                "transaction_cost_bps must be non-negative, "
                f"got {self.transaction_cost_bps}"
            )
            raise ConfigurationError(msg)
        if self.random_state is not None:
            require_integer("random_state", self.random_state, 0)


def _require_finite(name: str, value: object) -> float:
    number = require_real(name, value)
    if not math.isfinite(number):
        msg = f"{name} must be finite, got {value}"
        raise ConfigurationError(msg)
    return number


def _require_positive(name: str, value: object) -> None:
    number = require_real(name, value)
    if not math.isfinite(number) or number <= 0.0:
        msg = f"{name} must be positive and finite, got {value}"
        raise ConfigurationError(msg)
