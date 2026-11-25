"""Engle-Granger cointegration test for a pair, and the mean-reversion half-life.

Two price series are cointegrated when a linear combination of them is stationary
although each of them is not. Engle and Granger (1987) test it in two steps:
regress ``y`` on ``x`` by least squares, then run an Augmented Dickey-Fuller (ADF)
test on the residuals ``e``. The ADF regression has no constant or trend because
the residuals have zero mean by construction:

    diff(e)[t] = gamma e[t-1] + sum_i phi_i diff(e)[t-i] + u[t]

and the statistic is the t-statistic of ``gamma``. Because the residuals come from
a fitted regression, the statistic does not follow the Dickey-Fuller
distribution; the critical values are MacKinnon's (2010) for the two-variable case
with a constant.

Research tools only: nothing in ``optimizer.statarb`` models transaction costs,
borrow costs or short-sale limits, and the fund mandate is long-only and
paper-only, so none of it is wired to execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.exceptions import DataError
from optimizer.statarb._config import ENGLE_GRANGER_CRITICAL_VALUES, EngleGrangerConfig
from optimizer.statarb._inputs import EXACT_FIT_RATIO, aligned_pair, as_float_array

# Schwert (1989): the longest lag worth trying grows with the fourth root of T.
_SCHWERT_SCALE: Final = 12.0
_SCHWERT_REFERENCE_SIZE: Final = 100.0
_SCHWERT_EXPONENT: Final = 0.25
# The Akaike search compares fits that each carry one coefficient per lag, so the
# longest of them needs rows to spare.
_SEARCH_MIN_DEGREES_OF_FREEDOM: Final = 2
_MIN_HALF_LIFE_OBSERVATIONS: Final = 3


@dataclass(frozen=True, eq=False)
class EngleGrangerResult:
    """Outcome of an Engle-Granger cointegration test.

    The result holds a pandas Series, so it compares by identity.

    Attributes:
        hedge_ratio: Slope ``b`` of the regression ``y = a + b x``.
        intercept: Intercept ``a`` of that regression.
        adf_statistic: t-statistic of the lagged residual in the ADF regression.
        critical_values: Asymptotic MacKinnon critical values by level (``"1%"``,
            ``"5%"``, ``"10%"``); a statistic below one rejects "no cointegration"
            at that level.
        is_cointegrated: Whether ``adf_statistic`` is below the critical value of
            the configured significance.
        half_life: Half-life of the spread in periods, ``ou_half_life(spread)``;
            infinite when the spread does not mean revert.
        spread: The regression residual ``y - a - b x``, on ``y``'s index.
        adf_lags: Lagged differences used in the ADF regression, given or chosen.
    """

    hedge_ratio: float
    intercept: float
    adf_statistic: float
    critical_values: dict[str, float]
    is_cointegrated: bool
    half_life: float
    spread: pd.Series
    adf_lags: int


class _AdfFit(NamedTuple):
    """The pieces of one ADF regression that the callers need."""

    t_statistic: float
    rss: float
    rows: int
    params: int


def _ols_line(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Return the intercept and slope of the least-squares line of ``y`` on ``x``."""
    if np.ptp(x) == 0.0:
        msg = "x is constant, so the hedge ratio is undefined"
        raise DataError(msg)
    x_centred = x - x.mean()
    slope = float(np.sum(x_centred * (y - y.mean())) / np.sum(x_centred * x_centred))
    return float(y.mean() - slope * x.mean()), slope


def _adf_design(
    residuals: npt.NDArray[np.float64], lags: int, first: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the target and regressors of the ADF regression from row ``first``."""
    changes = np.diff(residuals)
    last = len(changes)
    columns = [residuals[first:last]]
    columns += [changes[first - lag : last - lag] for lag in range(1, lags + 1)]
    return changes[first:], np.column_stack(columns)


def _fit_adf(residuals: npt.NDArray[np.float64], lags: int, first: int) -> _AdfFit:
    """Fit the no-constant ADF regression on the rows from ``first`` onwards."""
    target, design = _adf_design(residuals, lags, first)
    rows, params = design.shape
    if rows - params < 1:
        msg = (
            f"the ADF regression with {lags} lags has no degrees of freedom: "
            f"{rows} rows for {params} coefficients"
        )
        raise DataError(msg)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    errors = target - design @ coefficients
    rss = float(errors @ errors)
    if rss <= EXACT_FIT_RATIO * float(target @ target):
        msg = (
            "the spread is degenerate (constant or an exact recursion), "
            "so there is no ADF statistic"
        )
        raise DataError(msg)
    precision = np.linalg.pinv(design.T @ design)[0, 0]
    standard_error = math.sqrt(rss / (rows - params) * precision)
    return _AdfFit(float(coefficients[0] / standard_error), rss, rows, params)


def _lag_limit(size: int, max_lags: int | None) -> int:
    """Return the largest lag count the Akaike search may try."""
    schwert = math.floor(
        _SCHWERT_SCALE * (size / _SCHWERT_REFERENCE_SIZE) ** _SCHWERT_EXPONENT
    )
    requested = schwert if max_lags is None else max_lags
    # With m lags the longest regression has size - 1 - m rows and m + 1 columns.
    capacity = (size - 2 - _SEARCH_MIN_DEGREES_OF_FREEDOM) // 2
    return min(requested, capacity)


def _select_lags(residuals: npt.NDArray[np.float64], max_lags: int) -> int:
    """Return the lag count in ``0..max_lags`` with the lowest Akaike criterion.

    Every candidate is fitted on the rows the longest one can use, so the criteria
    are computed on the same sample and can be compared.
    """
    scores = []
    for lags in range(max_lags + 1):
        fit = _fit_adf(residuals, lags, first=max_lags)
        scores.append(fit.rows * math.log(fit.rss / fit.rows) + 2 * fit.params)
    return int(np.argmin(scores))


def engle_granger(
    y: pd.Series, x: pd.Series, config: EngleGrangerConfig | None = None
) -> EngleGrangerResult:
    """Test whether ``y`` and ``x`` are cointegrated.

    Regresses ``y`` on ``x`` with a constant, then runs the no-constant ADF test on
    the residuals. The critical values are the *asymptotic* MacKinnon values for
    two variables with a constant, -3.90 (1%), -3.34 (5%) and -3.04 (10%). No
    finite-sample correction is applied, so the decision is approximate for short
    samples. The test is not symmetric: swapping ``y`` and ``x`` changes the
    statistic, so try both orders when neither series is the natural regressand.

    This is a research tool. It ignores costs and borrow limits, and nothing here
    is wired to execution.

    Args:
        y: Dependent series, for example a log price.
        x: Independent series on exactly the same index.
        config: Settings; defaults to ``EngleGrangerConfig()``.

    Returns:
        The hedge ratio, the ADF statistic, the decision and the spread.

    Raises:
        DataError: If the inputs are misaligned, too short, not numeric, not finite
            or ``x`` is constant, or the ADF regression has no degrees of freedom
            or the spread is degenerate.
    """
    settings = config or EngleGrangerConfig()
    y_values, x_values = aligned_pair(y, x, settings.min_observations)
    intercept, slope = _ols_line(y_values, x_values)
    residuals = y_values - intercept - slope * x_values
    lags = settings.adf_lags
    if lags is None:
        lags = _select_lags(residuals, _lag_limit(len(residuals), settings.max_lags))
    statistic = _fit_adf(residuals, lags, first=lags).t_statistic
    spread = pd.Series(residuals, index=y.index, name="spread")
    return EngleGrangerResult(
        hedge_ratio=slope,
        intercept=intercept,
        adf_statistic=statistic,
        critical_values=dict(ENGLE_GRANGER_CRITICAL_VALUES),
        is_cointegrated=bool(
            statistic < ENGLE_GRANGER_CRITICAL_VALUES[settings.significance]
        ),
        half_life=ou_half_life(spread),
        spread=spread,
        adf_lags=lags,
    )


def ou_half_life(series: npt.ArrayLike) -> float:
    """Return the half-life of mean reversion of a series, in periods.

    Regresses ``diff(s)[t]`` on ``s[t-1]`` with a constant; the slope ``lambda``
    is the discrete-time Ornstein-Uhlenbeck speed, and a shock decays by half
    after ``-ln 2 / ln(1 + lambda)`` periods when ``-1 < lambda < 0``.

    The slope of a unit-root series is biased downward, so a random walk often
    gives a long finite half-life rather than infinity: compare the result with
    the length of the sample before reading it as mean reversion.

    Args:
        series: One-dimensional series of levels, such as a spread.

    Returns:
        The half-life in periods, or ``math.inf`` when the slope is zero or
        positive (no reversion) or at most -1 (the series overshoots its mean).

    Raises:
        DataError: If the input is not numeric or not one-dimensional, has fewer
            than three values, has a non-finite value, or its lagged values are
            constant.
    """
    values = as_float_array(series, "series")
    if values.ndim != 1:
        msg = f"series must be one-dimensional, got shape {values.shape}"
        raise DataError(msg)
    if values.size < _MIN_HALF_LIFE_OBSERVATIONS:
        msg = (
            f"at least {_MIN_HALF_LIFE_OBSERVATIONS} observations are needed, "
            f"got {values.size}"
        )
        raise DataError(msg)
    if not np.isfinite(values).all():
        msg = "series must be finite: drop or fill missing values first"
        raise DataError(msg)
    lagged = values[:-1] - values[:-1].mean()
    variance_sum = float(lagged @ lagged)
    if variance_sum == 0.0:
        msg = "the lagged series is constant, so the half-life is undefined"
        raise DataError(msg)
    changes = np.diff(values)
    # Centring both sides keeps an exactly constant change (a linear trend) from
    # producing a rounding-sized negative slope and a finite half-life.
    slope = float(lagged @ (changes - changes.mean())) / variance_sum
    if -1.0 < slope < 0.0:
        return -math.log(2.0) / math.log1p(slope)
    return math.inf
