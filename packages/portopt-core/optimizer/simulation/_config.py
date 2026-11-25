"""Settings of the simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from optimizer.simulation._validation import (
    require_at_least_one,
    require_finite,
    require_non_negative,
    require_positive,
    require_seed,
)

DEFAULT_TIME_STEP: Final = 1.0 / 252.0


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
        require_non_negative("sigma", self.sigma)
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
