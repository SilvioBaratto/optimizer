"""Bootstrap wealth paths and a summary of simulated paths.

The bootstrap resamples the observed returns with replacement and compounds them,
so the simulated paths keep the empirical distribution of the returns (fat tails,
skew) without assuming a model. Resampling single returns is the i.i.d. bootstrap
and destroys any serial dependence. Resampling blocks of consecutive returns, the
moving-block bootstrap of Kunsch (1989) and Liu and Singh (1992), keeps the
dependence inside a block: short-range autocorrelation and volatility clustering
survive up to the block length. Efron and Tibshirani, "An Introduction to the
Bootstrap" (1993), cover the i.i.d. case.

Errors follow the rule stated in ``_gbm``: ``ConfigurationError`` for settings and
for wealth that overflows, ``DataError`` for the returns and the paths summarised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.simulation._config import BootstrapConfig
from optimizer.simulation._validation import (
    as_float_array,
    as_returns,
    as_vector,
    require_no_overflow,
    require_positive,
)

# A bootstrap needs something to resample; one return is enough.
_MIN_RETURNS: Final = 1


@dataclass(frozen=True)
class PathSummary:
    """Distribution of the outcomes of a set of simulated paths.

    Attributes:
        mean_terminal: Mean of the final value divided by the starting value.
        terminal_quantiles: Quantiles of the final value divided by the starting
            value, keyed by quantile level.
        probability_of_loss: Share of paths that end below their starting value.
        mean_max_drawdown: Mean over paths of the largest fall from a running
            maximum, as a fraction of that maximum between 0 and 1.
        worst_max_drawdown: Largest of those drawdowns over all paths.
    """

    mean_terminal: float
    terminal_quantiles: dict[float, float]
    probability_of_loss: float
    mean_max_drawdown: float
    worst_max_drawdown: float


def _resample(
    returns: npt.NDArray[np.float64], config: BootstrapConfig
) -> npt.NDArray[np.float64]:
    """Draw ``(n_paths, n_steps)`` returns in blocks of ``config.block_size``."""
    block = config.block_size
    if block > returns.size:
        msg = (
            f"block_size ({block}) cannot exceed the number of returns ({returns.size})"
        )
        raise DataError(msg)
    rng = np.random.default_rng(config.random_state)
    n_blocks = -(-config.n_steps // block)
    starts = rng.integers(0, returns.size - block + 1, size=(config.n_paths, n_blocks))
    positions = (starts[:, :, np.newaxis] + np.arange(block)).reshape(
        config.n_paths, n_blocks * block
    )
    resampled: npt.NDArray[np.float64] = returns[positions[:, : config.n_steps]]
    return resampled


def simulate_bootstrap(
    returns: npt.ArrayLike, config: BootstrapConfig, initial_value: float = 1.0
) -> npt.NDArray[np.float64]:
    """Simulate wealth paths by compounding resampled simple returns.

    With ``config.block_size`` above one the returns are drawn as moving blocks:
    each block starts at a random position and takes that many consecutive
    observations, without wrapping around the end of the series. Blocks overlap in
    the source, so every start position is equally likely. This keeps short-range
    autocorrelation up to the block length; the last block is cut to fit the horizon.

    Args:
        returns: Simple per-period returns to resample, at least one, each at least
            -100%.
        config: Number of paths and steps, block length and seed.
        initial_value: Wealth at the start of every path, finite and positive.

    Returns:
        An array of shape ``(n_paths, n_steps + 1)``; column zero is
        ``initial_value``.

    Raises:
        ConfigurationError: If ``initial_value`` is not positive and finite or the
            wealth overflows the float range.
        DataError: If ``returns`` is not a numeric, finite, one-dimensional series
            of at least ``config.block_size`` values, each at least -100%.
    """
    require_positive("initial_value", initial_value)
    sample = _resample(as_returns(returns, _MIN_RETURNS), config)
    with np.errstate(over="ignore"):
        wealth = initial_value * np.cumprod(1.0 + sample, axis=1)
    require_no_overflow(wealth, "reduce n_steps or initial_value, or rescale returns")
    start = np.full((config.n_paths, 1), initial_value)
    return np.concatenate([start, wealth], axis=1)


def _max_drawdown(paths: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return each path's largest fall from its running maximum, as a fraction."""
    running_max = np.maximum.accumulate(paths, axis=1)
    return (1.0 - paths / running_max).max(axis=1)


def _validated_paths(paths: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``paths`` as a finite, non-negative matrix with a positive start."""
    matrix = as_float_array("paths", paths, DataError)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        msg = (
            "paths must be a matrix with at least one path and two time points, "
            f"got shape {matrix.shape}"
        )
        raise DataError(msg)
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        msg = "paths must be finite and non-negative"
        raise DataError(msg)
    if (matrix[:, 0] <= 0.0).any():
        msg = "every path must start above zero"
        raise DataError(msg)
    return matrix


def summarize_paths(
    paths: npt.ArrayLike, quantiles: Sequence[float] = (0.05, 0.5, 0.95)
) -> PathSummary:
    """Summarise the final values and drawdowns of simulated paths.

    Args:
        paths: Matrix of shape ``(n_paths, n_steps + 1)`` of prices or wealth, as
            returned by ``simulate_gbm`` or ``simulate_bootstrap``; column zero is
            the starting value.
        quantiles: Levels in ``[0, 1]`` of the terminal distribution to report,
            interpolated linearly between order statistics.

    Returns:
        The summary; terminal values are relative to each path's starting value and
        drawdowns are positive fractions measured from the running maximum.

    Raises:
        ConfigurationError: If ``quantiles`` is empty, not numeric or has a level
            outside ``[0, 1]``.
        DataError: If ``paths`` is not a numeric, finite, non-negative matrix of at
            least two time points that starts above zero.
    """
    levels = as_vector("quantiles", quantiles, error=ConfigurationError)
    if not ((levels >= 0.0) & (levels <= 1.0)).all():
        msg = f"quantiles must be levels in [0, 1], got {quantiles}"
        raise ConfigurationError(msg)
    matrix = _validated_paths(paths)
    terminal = matrix[:, -1] / matrix[:, 0]
    drawdowns = _max_drawdown(matrix)
    return PathSummary(
        mean_terminal=float(terminal.mean()),
        terminal_quantiles={
            float(level): float(value)
            for level, value in zip(levels, np.quantile(terminal, levels), strict=True)
        },
        probability_of_loss=float((terminal < 1.0).mean()),
        mean_max_drawdown=float(drawdowns.mean()),
        worst_max_drawdown=float(drawdowns.max()),
    )
