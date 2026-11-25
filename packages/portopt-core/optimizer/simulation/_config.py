"""Settings of the simulators and the checks every simulator applies to them."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from optimizer.exceptions import ConfigurationError

DEFAULT_TIME_STEP: Final = 1.0 / 252.0


def require_finite(name: str, value: float) -> None:
    """Raise unless ``value`` is a finite number.

    Args:
        name: Setting name used in the error message.
        value: Value to check.

    Raises:
        ConfigurationError: If ``value`` is NaN or infinite.
    """
    if not math.isfinite(value):
        msg = f"{name} must be finite, got {value}"
        raise ConfigurationError(msg)


def require_positive(name: str, value: float) -> None:
    """Raise unless ``value`` is finite and strictly positive.

    Args:
        name: Setting name used in the error message.
        value: Value to check.

    Raises:
        ConfigurationError: If ``value`` is not finite or not above zero.
    """
    if not (math.isfinite(value) and value > 0.0):
        msg = f"{name} must be finite and positive, got {value}"
        raise ConfigurationError(msg)


def require_at_least_one(name: str, value: int) -> None:
    """Raise unless ``value`` is an integer of at least one.

    Args:
        name: Setting name used in the error message.
        value: Count to check.

    Raises:
        ConfigurationError: If ``value`` is below one.
    """
    if value < 1:
        msg = f"{name} must be at least 1, got {value}"
        raise ConfigurationError(msg)


def require_seed(random_state: int | None) -> None:
    """Raise if ``random_state`` is a seed numpy would refuse.

    Args:
        random_state: Seed, or ``None`` for fresh entropy.

    Raises:
        ConfigurationError: If the seed is negative.
    """
    if random_state is not None and random_state < 0:
        msg = f"random_state must be non-negative or None, got {random_state}"
        raise ConfigurationError(msg)


@dataclass(frozen=True)
class GbmConfig:
    """Immutable settings for a single-asset geometric Brownian motion simulation.

    Attributes:
        initial_price: Price at time zero, strictly positive.
        mu: Drift of the price per unit of time: ``E[S_t] = S_0 exp(mu t)``.
        sigma: Volatility of the log price per square root of a unit of time; zero
            gives the deterministic path.
        n_steps: Number of time steps per path.
        n_paths: Number of paths.
        dt: Length of one step in the same unit as ``mu`` and ``sigma``; the default
            is one trading day of a 252-day year.
        antithetic: Whether the second half of the paths reuses the shocks of the
            first half with the opposite sign. With an odd ``n_paths`` the last
            first-half path has no mirror image.
        random_state: Seed of the random generator; ``None`` draws fresh entropy.
    """

    initial_price: float
    mu: float
    sigma: float
    n_steps: int
    n_paths: int
    dt: float = DEFAULT_TIME_STEP
    antithetic: bool = True
    random_state: int | None = None

    def __post_init__(self) -> None:
        """Reject settings the simulation cannot use."""
        require_positive("initial_price", self.initial_price)
        require_finite("mu", self.mu)
        require_finite("sigma", self.sigma)
        if self.sigma < 0.0:
            msg = f"sigma must be non-negative, got {self.sigma}"
            raise ConfigurationError(msg)
        require_positive("dt", self.dt)
        require_at_least_one("n_steps", self.n_steps)
        require_at_least_one("n_paths", self.n_paths)
        require_seed(self.random_state)


@dataclass(frozen=True)
class BootstrapConfig:
    """Immutable settings for a bootstrap simulation of wealth paths.

    Attributes:
        n_paths: Number of paths.
        n_steps: Number of periods per path.
        block_size: Length of the blocks of consecutive returns that are resampled
            together; 1 is the plain i.i.d. bootstrap.
        random_state: Seed of the random generator; ``None`` draws fresh entropy.
    """

    n_paths: int
    n_steps: int
    block_size: int = 1
    random_state: int | None = None

    def __post_init__(self) -> None:
        """Reject settings the simulation cannot use."""
        require_at_least_one("n_paths", self.n_paths)
        require_at_least_one("n_steps", self.n_steps)
        require_at_least_one("block_size", self.block_size)
        require_seed(self.random_state)
