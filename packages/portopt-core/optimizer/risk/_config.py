"""Settings for the Value-at-Risk calculator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np

from optimizer.exceptions import ConfigurationError

# A tail quantile of fewer than 100 draws rests on a handful of paths.
_MIN_SIMULATIONS: Final = 100
# The standard deviation and the sample moments need a few points to mean anything.
_MIN_OBSERVATIONS_FLOOR: Final = 10


class VarMethod(str, Enum):
    """Estimator used for the loss quantile."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    CORNISH_FISHER = "cornish_fisher"
    MONTE_CARLO = "monte_carlo"


def require_confidence_level(confidence_level: float) -> None:
    """Reject a confidence level that is not strictly between 0 and 1.

    Args:
        confidence_level: Probability the loss estimate is meant to cover.

    Raises:
        ConfigurationError: If the value is outside the open unit interval or is
            not a real number (a bool, a string or ``None`` is rejected).
    """
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float, np.integer, np.floating))
        or not 0.0 < confidence_level < 1.0
    ):
        msg = (
            "confidence_level must be a number strictly between 0 and 1, "
            f"got {confidence_level!r}"
        )
        raise ConfigurationError(msg)


def require_finite_real(name: str, value: object) -> None:
    """Reject ``value`` unless it is a finite real number.

    Args:
        name: Argument name used in the error message.
        value: Candidate number; a bool, string, ``None``, NaN or infinity is
            rejected, a numpy scalar is accepted.

    Raises:
        ConfigurationError: If ``value`` is not a finite real number.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
    ):
        msg = f"{name} must be a finite real number, got {value!r}"
        raise ConfigurationError(msg)


def require_integer_at_least(name: str, value: object, minimum: int) -> None:
    """Reject ``value`` unless it is a whole number that is at least ``minimum``.

    Args:
        name: Argument name used in the error message.
        value: Candidate setting; a bool, float (even a whole one), NaN, string or
            ``None`` is rejected, a numpy integer is accepted.
        minimum: Smallest accepted value.

    Raises:
        ConfigurationError: If ``value`` is not an integer of at least ``minimum``.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        msg = f"{name} must be an integer of at least {minimum}, got {value!r}"
        raise ConfigurationError(msg)


@dataclass(frozen=True)
class VarConfig:
    """Immutable configuration for the Value-at-Risk calculator.

    Attributes:
        confidence_level: Probability the loss estimate covers, strictly between 0
            and 1 (0.99 means the loss is exceeded on 1% of the periods).
        method: Estimator of the loss quantile and the expected shortfall.
        horizon: Holding period in number of return periods, at least 1.
        n_simulations: Paths drawn by the Monte Carlo method, at least 100.
        random_state: Seed of the Monte Carlo draws; ``None`` takes fresh entropy.
        min_observations: Fewest returns a sample may hold, at least 10.
    """

    confidence_level: float = 0.95
    method: VarMethod = VarMethod.HISTORICAL
    horizon: int = 1
    n_simulations: int = 10_000
    random_state: int | None = None
    min_observations: int = 30

    def __post_init__(self) -> None:
        """Reject settings the estimators cannot use."""
        require_confidence_level(self.confidence_level)
        if not isinstance(self.method, VarMethod):
            msg = f"method must be a VarMethod, got {self.method!r}"
            raise ConfigurationError(msg)
        require_integer_at_least("horizon", self.horizon, 1)
        require_integer_at_least("n_simulations", self.n_simulations, _MIN_SIMULATIONS)
        require_integer_at_least(
            "min_observations", self.min_observations, _MIN_OBSERVATIONS_FLOOR
        )
        if self.random_state is not None:
            require_integer_at_least("random_state", self.random_state, 0)
