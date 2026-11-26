"""Kelly criterion: the bet size that maximises long-run growth of wealth.

Staking a fraction ``f`` of wealth on an outcome with return ``r`` multiplies wealth
by ``1 + f r``. Repeated over many independent periods, wealth grows at the rate
``g(f) = E[ln(1 + f r)]``, and the fraction that maximises ``g`` is the Kelly
fraction (Kelly, 1956; Thorp, 1969). It has closed forms in two cases:

* an even-odds bet that wins with probability ``p`` has ``f* = 2p - 1``, and more
  generally ``f* = (b p - q) / b`` for net odds ``b`` and ``q = 1 - p``;
* returns that are approximately normal with mean ``mu`` and volatility ``sigma``
  have ``f* = (mu - r_f) / sigma^2`` for a risk-free rate ``r_f`` (Thorp, 2006,
  "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market").

Betting more than the Kelly fraction lowers the growth rate and raises the risk of
deep drawdowns; for near-normal returns the growth rate at twice the Kelly fraction
has fallen back to about the riskless rate, and above that it is lower still.
``simulate_kelly_growth`` makes that trade-off visible on a grid of fractions.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize

from optimizer.exceptions import ConfigurationError, ConvergenceError
from optimizer.simulation._validation import (
    as_returns,
    as_vector,
    require_at_least_one,
    require_finite,
    require_open_unit_interval,
    require_positive,
    require_seed,
)

# Kelly's objective needs at least two outcomes to describe a bet with a risk.
_MIN_OUTCOMES: Final = 2
_FRACTION_TOLERANCE: Final = 1e-10
# The objective tends to minus infinity where a wealth multiplier reaches zero, so
# the search stops a hair short of that fraction.
_BOUND_MARGIN: Final = 1e-9


class _FractionOutcome(NamedTuple):
    """Summary of one betting fraction; the field names are the frame's columns."""

    growth_rate: float
    median_terminal_wealth: float
    probability_below_threshold: float
    probability_of_loss: float


def kelly_fraction_gaussian(
    mu: float, sigma: float, risk_free_rate: float = 0.0
) -> float:
    """Return the Kelly fraction for normally distributed returns.

    Args:
        mu: Expected return per period.
        sigma: Standard deviation of the return per period, positive.
        risk_free_rate: Return per period of the riskless alternative, in the same
            unit as ``mu``.

    Returns:
        ``(mu - risk_free_rate) / sigma^2``: the fraction of wealth to hold in the
        risky asset. It is negative when the asset earns less than the riskless
        rate and above one when leverage pays.

    Raises:
        ConfigurationError: If ``sigma`` is not positive and finite or ``mu`` or
            ``risk_free_rate`` is not finite.
    """
    require_finite("mu", mu)
    require_finite("risk_free_rate", risk_free_rate)
    require_positive("sigma", sigma)
    return (mu - risk_free_rate) / sigma**2


def _require_period_rate(rate: float) -> None:
    """Raise unless a per-period rate is finite and above -100%."""
    require_finite("risk_free_rate_per_period", rate)
    if rate <= -1.0:
        msg = f"risk_free_rate_per_period must be above -1, got {rate}"
        raise ConfigurationError(msg)


def _loss_free_upper_bound(
    excess: npt.NDArray[np.float64], risk_free_rate: float, max_leverage: float
) -> float:
    """Return the largest fraction below which every wealth multiplier is positive.

    The multiplier ``1 + r_f + f * excess`` of an outcome with a negative excess
    return reaches zero at ``f = (1 + r_f) / -excess``; the worst outcome is the
    first to do so.
    """
    shortfall = -excess[excess < 0.0]
    if shortfall.size == 0:
        return max_leverage
    ruinous = (1.0 + risk_free_rate) / float(shortfall.max())
    return min(max_leverage, ruinous * (1.0 - _BOUND_MARGIN))


def kelly_fraction_empirical(
    returns: npt.ArrayLike,
    risk_free_rate_per_period: float = 0.0,
    max_leverage: float = 5.0,
) -> float:
    """Return the fraction that maximises the mean log growth of observed returns.

    Maximises ``mean(ln(1 + r_f + f (r - r_f)))`` over the observed returns ``r``
    for ``f`` in ``[0, max_leverage]``, with ``f`` the fraction held in the risky
    outcome and the rest earning ``r_f``. The objective is concave, so the bounded
    search finds the global optimum. The upper bound is lowered below the point
    where the worst observed return would wipe out the stake (``1 + r_f + f
    (r - r_f) <= 0``), where the objective is undefined; an outcome of -100% hence
    keeps the fraction below one.

    The wealth model is ``1 + r_f + f (r - r_f)``: the part of wealth that is not
    staked earns ``r_f``. ``simulate_kelly_growth`` uses ``1 + f r`` instead, with
    no riskless return on the rest, so for a non-zero ``risk_free_rate_per_period``
    the optimum found here is not the peak of the curve that function simulates; the
    two coincide at ``r_f = 0``.

    Args:
        returns: Simple per-period returns of the risky outcome, at least two, each
            at least -100%.
        risk_free_rate_per_period: Return per period of the riskless alternative,
            above -100%.
        max_leverage: Largest fraction considered, positive. It is the answer when
            log growth keeps rising up to the cap.

    Returns:
        The Kelly fraction in ``[0, max_leverage]``; exactly zero when the mean
        return does not exceed the riskless rate, since no stake then raises growth.

    Raises:
        ConfigurationError: If ``max_leverage`` is not positive and finite or the
            risk-free rate is not finite and above -100%.
        DataError: If ``returns`` is not a finite one-dimensional series of at
            least two values, each at least -100%.
        ConvergenceError: If the scalar optimiser does not converge.
    """
    require_positive("max_leverage", max_leverage)
    _require_period_rate(risk_free_rate_per_period)
    excess = as_returns(returns, _MIN_OUTCOMES) - risk_free_rate_per_period
    if excess.mean() <= 0.0:
        return 0.0
    upper = _loss_free_upper_bound(excess, risk_free_rate_per_period, max_leverage)
    base = 1.0 + risk_free_rate_per_period

    def negative_growth(fraction: float) -> float:
        """Return minus the mean log wealth multiplier at ``fraction``."""
        return -float(np.log(base + fraction * excess).mean())

    result = optimize.minimize_scalar(
        negative_growth,
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": _FRACTION_TOLERANCE},
    )
    if not result.success:
        msg = f"the Kelly fraction search did not converge: {result.message}"
        raise ConvergenceError(msg)
    return float(result.x)


def _as_fractions(fractions: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``fractions`` as a non-empty vector of finite non-negative numbers."""
    levels = as_vector("fractions", fractions, error=ConfigurationError)
    if (levels < 0.0).any():
        msg = "fractions must be non-negative"
        raise ConfigurationError(msg)
    return levels


def _fraction_outcome(
    draws: npt.NDArray[np.float64], fraction: float, ruin_threshold: float
) -> _FractionOutcome:
    """Return the summary of betting ``fraction`` on ``draws``.

    Wealth cannot fall below zero, so a multiplier that would turn negative is
    floored at zero and the path stays at zero from then on.
    """
    multipliers = np.maximum(1.0 + fraction * draws, 0.0)
    wealth = np.cumprod(multipliers, axis=1)
    terminal = wealth[:, -1]
    with np.errstate(divide="ignore"):
        growth_rate = float(np.log(multipliers).mean())
    return _FractionOutcome(
        growth_rate=growth_rate,
        median_terminal_wealth=float(np.median(terminal)),
        probability_below_threshold=float((wealth.min(axis=1) < ruin_threshold).mean()),
        probability_of_loss=float((terminal < 1.0).mean()),
    )


def simulate_kelly_growth(
    returns: npt.ArrayLike,
    fractions: npt.ArrayLike,
    n_paths: int,
    n_periods: int,
    random_state: int | None = None,
    ruin_threshold: float = 0.5,
) -> pd.DataFrame:
    """Simulate wealth under repeated betting for a grid of Kelly fractions.

    The observed returns are resampled i.i.d. with replacement into ``n_paths``
    sequences of ``n_periods`` outcomes. Every fraction is evaluated on the same
    sequences, so differences between rows come from the fraction and not from
    sampling noise. Wealth starts at one and is multiplied by ``1 + f r`` each
    period; it is floored at zero, so a fraction that can lose more than the whole
    stake ends in ruin rather than in negative wealth.

    The wealth model has no riskless return on the part that is not staked.
    ``kelly_fraction_empirical`` uses ``1 + r_f + f (r - r_f)``, so for a non-zero
    risk-free rate its optimum is not the peak of the curve simulated here; the two
    coincide at ``r_f = 0``.

    Args:
        returns: Simple per-period returns of the outcome to bet on, at least two,
            each at least -100%.
        fractions: Fractions of wealth staked each period, finite and non-negative.
        n_paths: Number of simulated paths.
        n_periods: Number of periods per path.
        random_state: Seed of the random generator; ``None`` draws fresh entropy.
        ruin_threshold: Fraction of the starting wealth, strictly between 0 and 1,
            below which a path counts as ruined.

    Returns:
        A frame indexed by ``fraction`` with the columns ``growth_rate`` (mean of
        the log wealth multiplier per period, minus infinity if some simulated
        multiplier is zero), ``median_terminal_wealth``,
        ``probability_below_threshold`` (share of paths whose wealth is ever below
        ``ruin_threshold``) and ``probability_of_loss`` (share that end below the
        starting wealth).

    Raises:
        ConfigurationError: If ``fractions`` is empty, not numeric, not finite or
            negative, ``n_paths`` or ``n_periods`` is not an integer of at least 1,
            ``ruin_threshold`` is not strictly between 0 and 1 or ``random_state``
            is not a non-negative integer.
        DataError: If ``returns`` is not a numeric, finite, one-dimensional series
            of at least two values, each at least -100%.
    """
    require_at_least_one("n_paths", n_paths)
    require_at_least_one("n_periods", n_periods)
    require_seed(random_state)
    require_open_unit_interval("ruin_threshold", ruin_threshold)
    levels = _as_fractions(fractions)
    outcomes = as_returns(returns, _MIN_OUTCOMES)
    rng = np.random.default_rng(random_state)
    draws = outcomes[rng.integers(0, outcomes.size, size=(n_paths, n_periods))]
    rows = [
        _fraction_outcome(draws, float(fraction), ruin_threshold) for fraction in levels
    ]
    return pd.DataFrame(
        rows,
        index=pd.Index(levels, name="fraction"),
        columns=list(_FractionOutcome._fields),
    )
