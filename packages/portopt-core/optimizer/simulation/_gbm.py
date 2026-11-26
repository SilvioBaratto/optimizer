"""Geometric Brownian motion: simulation, correlated simulation and calibration.

A price follows geometric Brownian motion when its log price moves as a Brownian
motion with drift, ``dS = mu S dt + sigma S dW``. Over a step of length ``dt`` the
solution is exact, so no discretisation error enters the paths:

    S_{t+dt} = S_t * exp((mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z),  Z ~ N(0, 1)

Hence ``E[S_T] = S_0 exp(mu T)`` and ``E[ln S_T] = ln S_0 + (mu - sigma^2 / 2) T``.
See Hull, "Options, Futures, and Other Derivatives" (the Black-Scholes-Merton
model), and Glasserman, "Monte Carlo Methods in Financial Engineering" (2004),
chapter 3 for sample paths, including correlated geometric Brownian motion, and
section 4.2 for antithetic variates.

The whole ``optimizer.simulation`` package raises one error type per kind of input:
``ConfigurationError`` for settings (``initial_price``, ``mu``, ``sigma``,
``n_steps``, ``n_paths``, ``dt``, the seed, and the start prices and drifts of a
market) and for a simulated path that overflows the float range, because those
settings decide it; ``DataError`` for data arrays (observed prices, returns, a
covariance matrix, simulated paths passed to a summary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.simulation._config import DEFAULT_TIME_STEP, GbmConfig
from optimizer.simulation._validation import (
    as_float_array,
    as_vector,
    require_at_least_one,
    require_no_overflow,
    require_positive,
    require_seed,
)

# Relative size below which an eigenvalue or an asymmetry is rounding noise rather
# than a property of the matrix.
_MATRIX_TOLERANCE: Final = 1e-10
# The log returns need two values for a variance with one degree of freedom.
_MIN_CALIBRATION_PRICES: Final = 3


@dataclass(frozen=True)
class GbmFit:
    """Parameters of a geometric Brownian motion fitted to a price series.

    Attributes:
        mu: Drift of the price per unit of time.
        sigma: Volatility of the log price per square root of a unit of time.
    """

    mu: float
    sigma: float


def _shocks(
    rng: np.random.Generator, n_paths: int, n_steps: int, *, antithetic: bool
) -> npt.NDArray[np.float64]:
    """Draw the standard normal shocks, mirroring the first half if antithetic."""
    if not antithetic:
        return rng.standard_normal((n_paths, n_steps))
    half = (n_paths + 1) // 2
    base = rng.standard_normal((half, n_steps))
    return np.concatenate([base, -base])[:n_paths]


def _compound(
    initial: npt.NDArray[np.float64] | float,
    log_increments: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return ``initial * exp`` of the running sum of the log increments.

    The increments are ``(paths, steps)`` or ``(paths, steps, assets)``. A leading
    column of zeros makes the first time point equal ``initial`` exactly.

    Raises:
        ConfigurationError: If a price leaves the float range.
    """
    zeros = np.zeros_like(log_increments[:, :1])
    log_growth = np.concatenate([zeros, np.cumsum(log_increments, axis=1)], axis=1)
    with np.errstate(over="ignore"):
        paths = initial * np.exp(log_growth)
    require_no_overflow(paths, "reduce mu, the volatility, dt or n_steps")
    return paths


def simulate_gbm(config: GbmConfig) -> npt.NDArray[np.float64]:
    """Simulate price paths of one asset under geometric Brownian motion.

    With ``config.antithetic`` the second half of the paths reuses the shocks of the
    first half with the opposite sign, which cancels the first-order error of the
    sample mean. For an odd ``n_paths`` one extra mirror image is drawn and
    discarded, so the last path of the first half has no partner.

    Args:
        config: Drift, volatility, horizon and path settings.

    Returns:
        An array of shape ``(n_paths, n_steps + 1)``; column zero is the initial
        price.

    Raises:
        ConfigurationError: If a simulated price overflows the float range.
    """
    rng = np.random.default_rng(config.random_state)
    shocks = _shocks(rng, config.n_paths, config.n_steps, antithetic=config.antithetic)
    drift = (config.mu - config.sigma**2 / 2.0) * config.dt
    log_increments = drift + config.sigma * math.sqrt(config.dt) * shocks
    return _compound(config.initial_price, log_increments)


def _covariance_factor(
    covariance: npt.ArrayLike, n_assets: int
) -> npt.NDArray[np.float64]:
    """Return ``A`` with ``A @ A.T`` equal to a symmetric positive semi-definite matrix.

    A positive definite matrix gets its Cholesky factor. A singular one (perfectly
    correlated or constant assets, a rank-deficient estimate) has no Cholesky
    factor, so the square root from its eigen-decomposition is used instead.

    Raises:
        DataError: If the matrix is not numeric, has the wrong shape, is not finite,
            not symmetric or has a negative eigenvalue beyond rounding noise.
    """
    matrix = as_float_array("covariance", covariance, DataError)
    if matrix.shape != (n_assets, n_assets):
        msg = (
            f"covariance must have shape ({n_assets}, {n_assets}) for {n_assets} "
            f"assets, got {matrix.shape}"
        )
        raise DataError(msg)
    if not np.isfinite(matrix).all():
        msg = "covariance must be finite"
        raise DataError(msg)
    if np.abs(matrix - matrix.T).max() > _MATRIX_TOLERANCE * np.abs(matrix).max():
        msg = "covariance must be symmetric"
        raise DataError(msg)
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = _MATRIX_TOLERANCE * max(float(eigenvalues.max()), 0.0)
    if eigenvalues.min() < -tolerance:
        msg = (
            "covariance must be positive semi-definite, its smallest eigenvalue is "
            f"{eigenvalues.min():.3g}"
        )
        raise DataError(msg)
    if eigenvalues.min() > tolerance:
        return np.linalg.cholesky(symmetric)
    # Directions below the tolerance carry no risk; keeping their rounding noise
    # would break exact linear relations between assets.
    return eigenvectors * np.sqrt(np.where(eigenvalues > tolerance, eigenvalues, 0.0))


def _market_inputs(
    initial_prices: npt.ArrayLike, mu: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the start prices and drifts as matching, validated vectors."""
    prices = as_vector("initial_prices", initial_prices, error=ConfigurationError)
    drifts = as_vector("mu", mu, error=ConfigurationError)
    if drifts.shape != prices.shape:
        msg = (
            f"mu needs one drift per asset: {prices.size} prices, {drifts.size} drifts"
        )
        raise ConfigurationError(msg)
    if not (prices > 0.0).all():
        msg = "initial_prices must be positive"
        raise ConfigurationError(msg)
    return prices, drifts


def simulate_correlated_gbm(
    initial_prices: npt.ArrayLike,
    mu: npt.ArrayLike,
    covariance: npt.ArrayLike,
    n_steps: int,
    n_paths: int,
    dt: float = DEFAULT_TIME_STEP,
    random_state: int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulate price paths of correlated assets under geometric Brownian motion.

    The log increments of asset ``i`` over a step are
    ``(mu_i - covariance_ii / 2) dt + sqrt(dt) * (A z)_i`` where ``A A^T`` is the
    covariance and ``z`` is a vector of independent standard normals.

    Args:
        initial_prices: Start price of each asset, finite and positive.
        mu: Drift of each price per unit of time, one per asset.
        covariance: Covariance matrix of the log returns per unit of time, symmetric
            and positive semi-definite, ordered like ``initial_prices``.
        n_steps: Number of time steps per path.
        n_paths: Number of paths.
        dt: Length of one step in the same unit as ``mu`` and ``covariance``.
        random_state: Seed of the random generator; ``None`` draws fresh entropy.

    Returns:
        An array of shape ``(n_paths, n_steps + 1, n_assets)``; index zero of the
        time axis holds the initial prices.

    Raises:
        ConfigurationError: If ``n_steps`` or ``n_paths`` is not an integer of at
            least 1, ``dt`` is not positive and finite, ``random_state`` is not a
            non-negative integer, ``initial_prices`` or ``mu`` is not a finite
            vector (positive prices, one drift per asset) or a simulated price
            overflows the float range.
        DataError: If the covariance is not numeric, not a square matrix with one
            row per asset, not finite, not symmetric or not positive semi-definite.
    """
    require_at_least_one("n_steps", n_steps)
    require_at_least_one("n_paths", n_paths)
    require_positive("dt", dt)
    require_seed(random_state)
    prices, drifts = _market_inputs(initial_prices, mu)
    factor = _covariance_factor(covariance, prices.size)
    shocks = np.random.default_rng(random_state).standard_normal(
        (n_paths, n_steps, prices.size)
    )
    variances = np.einsum("ij,ij->i", factor, factor)
    log_increments = (drifts - variances / 2.0) * dt + math.sqrt(dt) * shocks @ factor.T
    return _compound(prices, log_increments)


def calibrate_gbm(prices: npt.ArrayLike, dt: float = DEFAULT_TIME_STEP) -> GbmFit:
    """Fit a geometric Brownian motion to a price series from its log returns.

    The log returns are independent normals with mean ``(mu - sigma^2 / 2) dt`` and
    variance ``sigma^2 dt``, so the estimates are ``sigma^2 = var / dt`` and
    ``mu = mean / dt + sigma^2 / 2``, with ``mean`` and ``var`` the sample mean and
    the unbiased sample variance (``ddof=1``) of the log returns. This is a moment
    estimator; the maximum likelihood estimate divides the variance by the number of
    returns instead, which differs by a factor ``(n - 1) / n``.

    Args:
        prices: Prices at equally spaced times, at least three, finite and positive.
        dt: Time between two prices, in the unit ``mu`` and ``sigma`` are quoted in.

    Returns:
        The fitted drift and volatility.

    Raises:
        ConfigurationError: If ``dt`` is not positive and finite.
        DataError: If ``prices`` is not a numeric, one-dimensional, finite, positive
            series of at least three values.
    """
    require_positive("dt", dt)
    series = as_vector(
        "prices", prices, error=DataError, min_size=_MIN_CALIBRATION_PRICES
    )
    if not (series > 0.0).all():
        msg = "prices must be positive"
        raise DataError(msg)
    log_returns = np.diff(np.log(series))
    variance_rate = float(log_returns.var(ddof=1)) / dt
    mean_rate = float(log_returns.mean()) / dt
    return GbmFit(mu=mean_rate + variance_rate / 2.0, sigma=math.sqrt(variance_rate))
