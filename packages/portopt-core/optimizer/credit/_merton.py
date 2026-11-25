"""Merton structural credit model.

Merton (1974), "On the Pricing of Corporate Debt: The Risk Structure of Interest
Rates", treats a firm's equity as a European call option on its assets ``V`` with
the face value of its zero-coupon debt ``D`` as the strike and the debt's maturity
``T`` as the expiry::

    E = V N(d1) - D exp(-r T) N(d2)
    sigma_E E = N(d1) sigma_V V
    d1 = (ln(V / D) + (r + sigma_V^2 / 2) T) / (sigma_V sqrt(T))
    d2 = d1 - sigma_V sqrt(T)

Assets and their volatility cannot be observed, but equity value ``E`` and equity
volatility ``sigma_E`` can, so the two equations are solved for ``(V, sigma_V)``.
Everything else follows: the market value of the debt is ``V - E``, its yield is
``y = -ln((V - E) / D) / T`` and the credit spread is ``y - r``. The textbook
example this module is checked against is the Merton model worked example in
Hull, *Options, Futures, and Other Derivatives*.

The solve is a two-parameter nonlinear least-squares problem in
``(ln V, ln sigma_V)`` using ``scipy.optimize.least_squares`` (trust-region
reflective). Logs keep both unknowns positive without bounds. ``scipy.optimize.root``
with ``method="hybr"`` was tried first and missed a 1e-10 residual on 60 of 1920
balance sheets (equity from 1e-4 to 1e6 times the debt, equity volatility from 1% to
800%, maturities from 3 weeks to 30 years); the trust-region solver reached the
tolerance on every one of them. On a 280-balance-sheet grid a bracketing scan over
the asset volatility found exactly one root per case, and the solver agreed with it
to a relative 1e-10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.special import ndtr

from optimizer.credit._config import MertonConfig, require_positive_finite
from optimizer.exceptions import ConfigurationError, ConvergenceError, DataError

# exp(-r T) must stay representable: exp(709) is the largest double.
_MAX_RATE_TIMES_MATURITY: Final = 700.0
# The solver is stopped this much tighter than the acceptance test, because it
# measures progress in the step size and its last step can leave the residual just
# above a tolerance it was stopped at.
_STOP_TOLERANCE_FRACTION: Final = 0.01
# scipy ignores, and warns about, termination tolerances below machine epsilon.
_MIN_STOP_TOLERANCE: Final = 1e-15
_MIN_PRICES: Final = 3
_MIN_WINDOW: Final = 2


@dataclass(frozen=True)
class MertonResult:
    """Outcome of a Merton structural credit model solve.

    Attributes:
        asset_value: Implied market value of the firm's assets ``V``.
        asset_volatility: Implied annual volatility of the assets ``sigma_V``.
        d1: Risk-neutral ``d1`` at the solution.
        d2: Risk-neutral ``d2`` at the solution; ``N(-d2)`` is the risk-neutral
            default probability.
        distance_to_default: Number of asset-volatility standard deviations between
            the expected log asset value at maturity and the default point, with
            the configured asset drift (the risk-free rate when none is set).
        default_probability: Probability that the assets end below the debt face
            value at maturity, ``N(-distance_to_default)``.
        debt_value: Market value of the debt, ``V - E``.
        credit_spread: Yield on the debt in excess of the risk-free rate, in
            continuously compounded terms, ``-ln(debt_value / D) / T - r``. Never
            negative. When the debt is tiny next to the equity the figure is
            limited by double precision: its absolute error is roughly
            ``1e-16 * E / D``.
        leverage: Present value of the debt over the asset value,
            ``D exp(-r T) / V``.
        residual_norm: Euclidean norm of the relative residuals of the equity
            value and equity volatility equations at the solution, never above the
            configured ``solver_tolerance``.
    """

    asset_value: float
    asset_volatility: float
    d1: float
    d2: float
    distance_to_default: float
    default_probability: float
    debt_value: float
    credit_spread: float
    leverage: float
    residual_norm: float


@dataclass(frozen=True)
class _Problem:
    """The validated market inputs of one solve."""

    equity_value: float
    equity_volatility: float
    debt_face_value: float
    risk_free_rate: float
    maturity: float

    @property
    def discounted_debt(self) -> float:
        """Present value of the debt at the risk-free rate."""
        return self.debt_face_value * math.exp(-self.risk_free_rate * self.maturity)

    def description(self) -> str:
        """Return the inputs as text for an error message."""
        return (
            f"equity_value={self.equity_value:g}, "
            f"equity_volatility={self.equity_volatility:g}, "
            f"debt_face_value={self.debt_face_value:g}, "
            f"risk_free_rate={self.risk_free_rate:g}, maturity={self.maturity:g}"
        )


def _build_problem(
    equity_value: float,
    equity_volatility: float,
    debt_face_value: float,
    risk_free_rate: float,
    maturity: float,
) -> _Problem:
    """Validate the market inputs of a solve and bundle them."""
    require_positive_finite("equity_value", equity_value, ConfigurationError)
    require_positive_finite("equity_volatility", equity_volatility, ConfigurationError)
    require_positive_finite("debt_face_value", debt_face_value, ConfigurationError)
    if not math.isfinite(risk_free_rate):
        msg = f"risk_free_rate must be finite, got {risk_free_rate}"
        raise ConfigurationError(msg)
    if abs(risk_free_rate * maturity) > _MAX_RATE_TIMES_MATURITY:
        msg = (
            "risk_free_rate times maturity must be below "
            f"{_MAX_RATE_TIMES_MATURITY:g} in absolute value so that the discount "
            f"factor is representable, got {risk_free_rate * maturity:g}"
        )
        raise ConfigurationError(msg)
    return _Problem(
        equity_value, equity_volatility, debt_face_value, risk_free_rate, maturity
    )


def _d1(asset_value: float, asset_volatility: float, problem: _Problem) -> float:
    """Return the risk-neutral ``d1`` for an asset value and volatility."""
    drift = problem.risk_free_rate + 0.5 * asset_volatility * asset_volatility
    numerator = np.log(asset_value / problem.debt_face_value) + drift * problem.maturity
    return float(numerator / (asset_volatility * math.sqrt(problem.maturity)))


def _relative_residuals(
    log_unknowns: npt.NDArray[np.float64], problem: _Problem
) -> npt.NDArray[np.float64]:
    """Return both Merton equations as relative errors at ``(ln V, ln sigma_V)``."""
    asset_value, asset_volatility = np.exp(log_unknowns)
    d1 = _d1(asset_value, asset_volatility, problem)
    d2 = d1 - asset_volatility * math.sqrt(problem.maturity)
    equity = (
        asset_value * ndtr(d1) - problem.discounted_debt * ndtr(d2)
    ) / problem.equity_value
    volatility_term = (
        ndtr(d1)
        * asset_volatility
        * asset_value
        / (problem.equity_volatility * problem.equity_value)
    )
    return np.array([equity - 1.0, volatility_term - 1.0], dtype=np.float64)


def _solve_unknowns(
    problem: _Problem, config: MertonConfig
) -> tuple[float, float, float]:
    """Return ``(V, sigma_V, residual_norm)`` that solve the two equations."""
    # Textbook start: assets as the equity plus the discounted debt, and the
    # equity volatility scaled down by the equity's share of those assets.
    start_value = problem.equity_value + problem.discounted_debt
    start_volatility = problem.equity_volatility * problem.equity_value / start_value
    stop = max(config.solver_tolerance * _STOP_TOLERANCE_FRACTION, _MIN_STOP_TOLERANCE)
    try:
        fit = least_squares(
            _relative_residuals,
            np.log([start_value, start_volatility]),
            args=(problem,),
            method="trf",
            xtol=stop,
            ftol=stop,
            gtol=stop,
            max_nfev=config.max_iterations,
        )
    except ValueError as error:
        msg = (
            "the Merton equations are not finite at the starting guess for "
            f"{problem.description()}"
        )
        raise ConvergenceError(msg) from error
    residual_norm = float(np.linalg.norm(fit.fun))
    if not residual_norm <= config.solver_tolerance:
        msg = (
            f"Merton solve did not converge: residual norm {residual_norm:.3e} is "
            f"above solver_tolerance {config.solver_tolerance:.3e} "
            f"({fit.message}) for {problem.description()}"
        )
        raise ConvergenceError(msg)
    asset_value, asset_volatility = np.exp(fit.x)
    return float(asset_value), float(asset_volatility), residual_norm


def _credit_spread(debt_value: float, problem: _Problem) -> float:
    """Return the debt's continuously compounded yield over the risk-free rate."""
    if debt_value <= 0.0:
        msg = (
            "the debt value V - E is not positive, so its yield is undefined: "
            "the debt is too small next to the equity, or the equity volatility too "
            f"large, to resolve in double precision for {problem.description()}"
        )
        raise ConvergenceError(msg)
    debt_yield = -math.log(debt_value / problem.debt_face_value) / problem.maturity
    # Mathematically the spread is non-negative; rounding in V - E can push it a
    # few ulps below zero for a firm with almost no credit risk.
    return max(debt_yield - problem.risk_free_rate, 0.0)


def _build_result(
    problem: _Problem,
    config: MertonConfig,
    unknowns: tuple[float, float, float],
) -> MertonResult:
    """Derive every reported quantity from the solved assets."""
    asset_value, asset_volatility, residual_norm = unknowns
    root_maturity = math.sqrt(problem.maturity)
    d1 = _d1(asset_value, asset_volatility, problem)
    drift = problem.risk_free_rate if config.asset_drift is None else config.asset_drift
    distance = (
        math.log(asset_value / problem.debt_face_value)
        + (drift - 0.5 * asset_volatility * asset_volatility) * problem.maturity
    ) / (asset_volatility * root_maturity)
    if not (math.isfinite(d1) and math.isfinite(distance)):
        msg = (
            "the Merton solution is not finite in double precision, so the inputs "
            f"are too extreme in scale or volatility: {problem.description()}"
        )
        raise ConvergenceError(msg)
    debt_value = asset_value - problem.equity_value
    return MertonResult(
        asset_value=asset_value,
        asset_volatility=asset_volatility,
        d1=d1,
        d2=d1 - asset_volatility * root_maturity,
        distance_to_default=distance,
        default_probability=float(ndtr(-distance)),
        debt_value=debt_value,
        credit_spread=_credit_spread(debt_value, problem),
        leverage=problem.discounted_debt / asset_value,
        residual_norm=residual_norm,
    )


def solve_merton(
    equity_value: float,
    equity_volatility: float,
    debt_face_value: float,
    risk_free_rate: float,
    config: MertonConfig | None = None,
) -> MertonResult:
    """Back out a firm's asset value and volatility from its equity.

    Args:
        equity_value: Market value of the equity (market capitalisation), above
            zero.
        equity_volatility: Annual volatility of the equity, above zero.
        debt_face_value: Face value of the zero-coupon debt due at the configured
            maturity, in the same currency as ``equity_value``, above zero.
        risk_free_rate: Continuously compounded annual risk-free rate; it may be
            zero or negative.
        config: Settings; defaults to ``MertonConfig()``.

    Returns:
        The implied assets, the default probability and distance to default, the
        debt value, the credit spread and the leverage.

    Raises:
        ConfigurationError: If a market input is not finite and positive (the rate
            only has to be finite), or the rate times the maturity is too large for
            the discount factor to be representable.
        ConvergenceError: If the solver cannot bring the residual of both equations
            under ``config.solver_tolerance`` within ``config.max_iterations``, or
            the solution is not finite, or the debt value ``V - E`` is not positive
            in double precision (a debt that is negligible next to the equity, or
            practically worthless because the equity is tiny and extremely
            volatile).
    """
    settings = config or MertonConfig()
    problem = _build_problem(
        equity_value,
        equity_volatility,
        debt_face_value,
        risk_free_rate,
        settings.maturity,
    )
    # A trust-region trial step can overflow exp() or divide by a vanishing
    # volatility; the inf or NaN residual that results is how the solver learns to
    # shrink the step, so it must neither raise nor warn. Non-finite output is
    # caught explicitly in _build_result instead.
    with np.errstate(all="ignore"):
        unknowns = _solve_unknowns(problem, settings)
        return _build_result(problem, settings, unknowns)


def _as_prices(prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``prices`` as a validated one-dimensional float array."""
    try:
        series = np.asarray(prices, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "prices must be numeric"
        raise DataError(msg) from error
    if series.ndim != 1:
        msg = f"prices must be one-dimensional, got shape {series.shape}"
        raise DataError(msg)
    if series.size < _MIN_PRICES:
        msg = f"at least {_MIN_PRICES} prices are needed, got {series.size}"
        raise DataError(msg)
    if not np.isfinite(series).all():
        msg = "prices must be finite: drop or fill missing values first"
        raise DataError(msg)
    if (series <= 0.0).any():
        msg = "prices must be positive to take log returns"
        raise DataError(msg)
    return series


def _latest_returns(
    log_returns: npt.NDArray[np.float64], window: int | None
) -> npt.NDArray[np.float64]:
    """Return the last ``window`` log returns, or all of them without a window."""
    if window is None:
        return log_returns
    if window < _MIN_WINDOW:
        msg = f"window must be at least {_MIN_WINDOW}, got {window}"
        raise ConfigurationError(msg)
    if window > log_returns.size:
        msg = (
            f"window of {window} returns is longer than the {log_returns.size} "
            "returns the prices give"
        )
        raise DataError(msg)
    return log_returns[-window:]


def estimate_equity_volatility(
    prices: npt.ArrayLike,
    periods_per_year: int = 252,
    window: int | None = None,
) -> float:
    """Return the annualised volatility of an equity price series.

    The estimate is the sample standard deviation (``ddof=1``) of the log returns,
    scaled by the square root of ``periods_per_year``.

    Args:
        prices: Equity prices in time order, at a regular frequency.
        periods_per_year: Observations per year of the price series, used to
            annualise: 252 for daily, 52 for weekly, 12 for monthly prices.
        window: Number of the most recent log returns to use; ``None`` uses all.

    Returns:
        The annualised volatility; zero when the price never changes.

    Raises:
        ConfigurationError: If ``periods_per_year`` is below 1 or ``window`` is
            below 2.
        DataError: If ``prices`` is not a numeric, one-dimensional series of at
            least 3 finite, positive values, or ``window`` exceeds the number of
            returns.
    """
    if periods_per_year < 1:
        msg = f"periods_per_year must be at least 1, got {periods_per_year}"
        raise ConfigurationError(msg)
    log_returns = np.diff(np.log(_as_prices(prices)))
    latest = _latest_returns(log_returns, window)
    return float(np.std(latest, ddof=1) * math.sqrt(periods_per_year))


def merton_from_market(
    market_cap: float,
    equity_prices: npt.ArrayLike,
    total_debt: float,
    risk_free_rate: float,
    config: MertonConfig | None = None,
    periods_per_year: int = 252,
) -> MertonResult:
    """Solve the Merton model from a market capitalisation and a price history.

    The equity volatility is estimated from ``equity_prices`` with
    ``estimate_equity_volatility`` and the debt face value is ``total_debt``.

    Args:
        market_cap: Market value of the equity, above zero.
        equity_prices: Equity prices in time order; their log returns give the
            equity volatility.
        total_debt: Face value of the debt, taken as zero-coupon and due at the
            configured maturity, above zero.
        risk_free_rate: Continuously compounded annual risk-free rate.
        config: Settings; defaults to ``MertonConfig()``.
        periods_per_year: Observations per year of ``equity_prices``.

    Returns:
        The solution of the Merton model for these market inputs.

    Raises:
        DataError: If the prices are invalid or constant.
        ConfigurationError: If a market input or setting is invalid.
        ConvergenceError: If the solver does not converge.
    """
    volatility = estimate_equity_volatility(equity_prices, periods_per_year)
    if volatility <= 0.0:
        msg = "equity prices are constant, so the equity volatility is zero"
        raise DataError(msg)
    return solve_merton(market_cap, volatility, total_debt, risk_free_rate, config)
