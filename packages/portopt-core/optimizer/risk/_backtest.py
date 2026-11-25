"""Backtest of Value-at-Risk forecasts.

A VaR forecast at confidence ``c`` claims that the loss exceeds it on a share
``p = 1 - c`` of the days, independently from one day to the next. The backtest
counts the days on which the realised loss did exceed the forecast (the
*exceptions*) and tests both claims:

* Kupiec (1995), the proportion-of-failures test: is the exception rate ``p``?
* Christoffersen (1998), the independence test: does an exception make another one
  more likely the next day? Its sum with Kupiec's statistic is the conditional
  coverage test.

The Basel Committee (1996) traffic-light zones classify the exception count of a
99% VaR over 250 days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special, stats

from optimizer.exceptions import DataError
from optimizer.risk._config import require_confidence_level
from optimizer.risk._var import as_series

TrafficLight = Literal["green", "yellow", "red"]

_MIN_OBSERVATIONS: Final = 2
_BASEL_CONFIDENCE: Final = 0.99
_BASEL_OBSERVATIONS: Final = 250
_GREEN_MAX_EXCEPTIONS: Final = 4
_YELLOW_MAX_EXCEPTIONS: Final = 9


@dataclass(frozen=True)
class VarBacktestResult:
    """Outcome of a Value-at-Risk backtest.

    Attributes:
        n_observations: Dates on which both a return and a forecast exist.
        n_exceptions: Dates on which the return fell below minus the forecast.
        expected_exceptions: Exceptions a correct forecast produces on average,
            ``n_observations * (1 - confidence_level)``.
        exception_rate: ``n_exceptions / n_observations``.
        kupiec_statistic: Likelihood-ratio statistic of the proportion-of-failures
            test, chi-square with one degree of freedom under a correct forecast.
        kupiec_p_value: Probability of a statistic at least that large.
        independence_statistic: Likelihood-ratio statistic of the Christoffersen
            independence test, chi-square with one degree of freedom.
        independence_p_value: Probability of a statistic at least that large.
        conditional_coverage_statistic: Sum of the two statistics, chi-square with
            two degrees of freedom.
        conditional_coverage_p_value: Probability of a statistic at least that large.
        traffic_light: Basel zone (``"green"``, ``"yellow"`` or ``"red"``) when the
            confidence level is 99% and there are exactly 250 observations,
            otherwise ``None``.
    """

    n_observations: int
    n_exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_statistic: float
    kupiec_p_value: float
    independence_statistic: float
    independence_p_value: float
    conditional_coverage_statistic: float
    conditional_coverage_p_value: float
    traffic_light: TrafficLight | None


def _align(
    returns: npt.ArrayLike | pd.Series, var_forecasts: npt.ArrayLike | pd.Series
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the returns and forecasts on their common non-missing dates."""
    realised = as_series(returns, "returns")
    forecast = as_series(var_forecasts, "var_forecasts")
    if not (realised.index.is_unique and forecast.index.is_unique):
        msg = "returns and var_forecasts must each have unique index labels"
        raise DataError(msg)
    joined = pd.concat([realised, forecast], axis=1, join="inner").dropna()
    if len(joined) < _MIN_OBSERVATIONS:
        msg = (
            f"at least {_MIN_OBSERVATIONS} dates with both a return and a forecast "
            f"are needed, got {len(joined)}"
        )
        raise DataError(msg)
    values = joined.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        msg = "returns and var_forecasts must be finite"
        raise DataError(msg)
    return values[:, 0], values[:, 1]


def _bernoulli_log_likelihood(
    successes: float, trials: float, probability: float
) -> float:
    """Return the log-likelihood of ``successes`` in ``trials`` Bernoulli draws.

    ``xlogy`` and ``xlog1py`` take ``0 * ln(0)`` as 0, which the corner cases of no
    exception and all exceptions need.
    """
    return float(
        special.xlogy(successes, probability)
        + special.xlog1py(trials - successes, -probability)
    )


def _ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or 0 when there is nothing to divide."""
    return numerator / denominator if denominator else 0.0


def _proportion_of_failures(
    n_observations: int, n_exceptions: int, tail_probability: float
) -> float:
    """Return Kupiec's likelihood-ratio statistic."""
    observed_rate = n_exceptions / n_observations
    unrestricted = _bernoulli_log_likelihood(
        n_exceptions, n_observations, observed_rate
    )
    restricted = _bernoulli_log_likelihood(
        n_exceptions, n_observations, tail_probability
    )
    return max(0.0, 2.0 * (unrestricted - restricted))


def _transition_counts(
    exceptions: npt.NDArray[np.bool_],
) -> tuple[int, int, int, int]:
    """Return the counts of day pairs (no, no), (no, yes), (yes, no), (yes, yes)."""
    previous, current = exceptions[:-1], exceptions[1:]
    return (
        int(np.sum(~previous & ~current)),
        int(np.sum(~previous & current)),
        int(np.sum(previous & ~current)),
        int(np.sum(previous & current)),
    )


def _independence(exceptions: npt.NDArray[np.bool_]) -> float:
    """Return Christoffersen's likelihood-ratio statistic for independence."""
    no_no, no_yes, yes_no, yes_yes = _transition_counts(exceptions)
    after_no = no_no + no_yes
    after_yes = yes_no + yes_yes
    total = after_no + after_yes
    unrestricted = _bernoulli_log_likelihood(
        no_yes, after_no, _ratio(no_yes, after_no)
    ) + _bernoulli_log_likelihood(yes_yes, after_yes, _ratio(yes_yes, after_yes))
    pooled_rate = _ratio(no_yes + yes_yes, total)
    restricted = _bernoulli_log_likelihood(no_yes + yes_yes, total, pooled_rate)
    return max(0.0, 2.0 * (unrestricted - restricted))


def _traffic_light(
    confidence_level: float, n_observations: int, n_exceptions: int
) -> TrafficLight | None:
    """Return the Basel zone, or ``None`` outside the setting it is defined for."""
    if (
        not math.isclose(confidence_level, _BASEL_CONFIDENCE)
        or n_observations != _BASEL_OBSERVATIONS
    ):
        return None
    if n_exceptions <= _GREEN_MAX_EXCEPTIONS:
        return "green"
    if n_exceptions <= _YELLOW_MAX_EXCEPTIONS:
        return "yellow"
    return "red"


def backtest_var(
    returns: npt.ArrayLike | pd.Series,
    var_forecasts: npt.ArrayLike | pd.Series,
    confidence_level: float,
) -> VarBacktestResult:
    """Test whether VaR exceptions are as frequent and as independent as claimed.

    An exception is a date on which the return is strictly below minus the
    forecast. Returns and forecasts are matched on their common index and dates
    where either is missing are dropped, so the output of ``rolling_var_forecasts``
    can be passed as it is. The statistics are likelihood ratios in which
    ``0 * ln(0)`` is taken as 0; the p-values come from the chi-square survival
    function.

    Args:
        returns: Realised simple per-period returns.
        var_forecasts: VaR forecasts for the same dates, as positive loss fractions.
        confidence_level: Confidence level the forecasts were made at, strictly
            between 0 and 1.

    Returns:
        The exception count, the Kupiec, independence and conditional coverage
        tests, and the Basel traffic-light zone where it applies.

    Raises:
        ConfigurationError: If ``confidence_level`` is not strictly between 0 and 1.
        DataError: If an input is not a one-dimensional numeric series, fewer than
            two dates are common to both, or a value is infinite.
    """
    require_confidence_level(confidence_level)
    realised, forecasts = _align(returns, var_forecasts)
    exceptions = realised < -forecasts
    n_observations = int(exceptions.size)
    n_exceptions = int(exceptions.sum())
    tail_probability = 1.0 - confidence_level
    kupiec = _proportion_of_failures(n_observations, n_exceptions, tail_probability)
    independence = _independence(exceptions)
    conditional_coverage = kupiec + independence
    return VarBacktestResult(
        n_observations=n_observations,
        n_exceptions=n_exceptions,
        expected_exceptions=n_observations * tail_probability,
        exception_rate=n_exceptions / n_observations,
        kupiec_statistic=kupiec,
        kupiec_p_value=float(stats.chi2.sf(kupiec, 1)),
        independence_statistic=independence,
        independence_p_value=float(stats.chi2.sf(independence, 1)),
        conditional_coverage_statistic=conditional_coverage,
        conditional_coverage_p_value=float(stats.chi2.sf(conditional_coverage, 2)),
        traffic_light=_traffic_light(confidence_level, n_observations, n_exceptions),
    )
