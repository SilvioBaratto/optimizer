"""Ordinary least squares with classical and Newey-West standard errors.

The coefficients solve ``min ||y - X b||^2`` through ``numpy.linalg.lstsq``. Two
estimates of their covariance are offered:

* the classical ``s^2 (X'X)^-1`` with ``s^2 = SSR / (n - k)``, valid when the errors
  are independent with a common variance;
* the Newey-West (1987) estimator in the sandwich form ``(X'X)^-1 S (X'X)^-1``,
  where ``S`` is the long-run variance of the scores ``x_t e_t`` with Bartlett
  weights ``1 - l / (L + 1)`` on lag ``l``. It stays valid when the errors are
  heteroskedastic or autocorrelated up to ``L`` lags, as daily and monthly return
  residuals often are. It carries no small-sample correction.

Newey and West (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
Autocorrelation Consistent Covariance Matrix", Econometrica 55(3), 703-708.

Nothing is dropped or imputed: a missing or infinite value, a rank-deficient design
or a sample too short to leave a residual degree of freedom raises ``DataError``.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from optimizer.exceptions import DataError
from optimizer.regression._config import CONSTANT_NAME, require_hac_lags
from optimizer.regression._inputs import to_float_array

_Array = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class OlsResult:
    """Outcome of an ordinary least squares fit.

    The comparison operators are left at identity because a field holding a pandas
    object has no unambiguous equality.

    Attributes:
        coefficients: Estimate for each regressor, labelled by regressor name; the
            intercept, when added, comes first and is called ``"const"``.
        std_errors: Standard error of each coefficient, classical or Newey-West.
        t_statistics: Coefficient over standard error; infinite for a perfect fit.
        p_values: Two-sided p-values of the t-statistics under a Student t
            distribution with ``degrees_of_freedom`` degrees of freedom.
        r_squared: Share of the variance of ``y`` the regressors explain. It is
            centred when the regressors span a constant and uncentred otherwise.
        adjusted_r_squared: ``r_squared`` corrected for the number of regressors.
        residuals: ``y`` minus the fitted values, on the index of ``y``.
        n_observations: Number of rows used.
        degrees_of_freedom: Rows minus estimated parameters.
    """

    coefficients: pd.Series
    std_errors: pd.Series
    t_statistics: pd.Series
    p_values: pd.Series
    r_squared: float
    adjusted_r_squared: float
    residuals: pd.Series
    n_observations: int
    degrees_of_freedom: int


@dataclass(frozen=True)
class _Problem:
    """A validated least squares problem, ready to solve.

    Attributes:
        response: Finite response vector.
        design: Design matrix, with the constant first when one was added.
        names: Name of each design column.
        index: Index the residuals are reported on.
        gram_inverse: ``(X'X)^-1``, computed once from the singular values.
        add_constant: Whether the constant was added by the caller's request.
    """

    response: _Array
    design: _Array
    names: list[Hashable]
    index: pd.Index
    gram_inverse: _Array
    add_constant: bool


def _response(y: pd.Series | npt.ArrayLike) -> tuple[_Array, pd.Index | None]:
    """Return the response as a finite vector, with its index if it has one."""
    index = y.index if isinstance(y, pd.Series) else None
    values = to_float_array(y, "y")
    if values.ndim != 1:
        msg = f"y must be one-dimensional, got shape {values.shape}"
        raise DataError(msg)
    return values, index


def _regressor_labels(
    x: pd.Series | pd.DataFrame | npt.ArrayLike, ndim: int, n_columns: int
) -> list[Hashable]:
    """Return one name per regressor column, keeping pandas names."""
    if isinstance(x, pd.DataFrame):
        return list(x.columns)
    if isinstance(x, pd.Series):
        return ["x" if x.name is None else x.name]
    if ndim == 1:
        return ["x"]
    return [f"x{position}" for position in range(1, n_columns + 1)]


def _regressors(
    x: pd.Series | pd.DataFrame | npt.ArrayLike,
) -> tuple[_Array, list[Hashable], pd.Index | None]:
    """Return the regressors as a matrix, with their names and index."""
    index = x.index if isinstance(x, (pd.Series, pd.DataFrame)) else None
    values = to_float_array(x, "x")
    if values.ndim not in (1, 2):
        msg = f"x must be one- or two-dimensional, got shape {values.shape}"
        raise DataError(msg)
    matrix = values[:, None] if values.ndim == 1 else values
    if matrix.shape[1] == 0:
        msg = "x needs at least one column"
        raise DataError(msg)
    labels = _regressor_labels(x, values.ndim, matrix.shape[1])
    return matrix, labels, index


def _result_index(
    y_index: pd.Index | None, x_index: pd.Index | None, n_obs: int
) -> pd.Index:
    """Return the index residuals are reported on, refusing to align silently."""
    if y_index is not None and x_index is not None and not y_index.equals(x_index):
        msg = "y and x must share the same index: align them before the call"
        raise DataError(msg)
    chosen = y_index if y_index is not None else x_index
    return chosen if chosen is not None else pd.RangeIndex(n_obs)


def _design(
    regressors: _Array, labels: list[Hashable], add_constant: bool
) -> tuple[_Array, list[Hashable]]:
    """Return the design matrix and its column names, constant first if added."""
    if len(set(labels)) != len(labels):
        msg = "regressor names must be unique"
        raise DataError(msg)
    if not add_constant:
        return regressors, labels
    if CONSTANT_NAME in labels:
        msg = f"a regressor is called {CONSTANT_NAME!r}, which the intercept owns"
        raise DataError(msg)
    constant = np.ones((regressors.shape[0], 1))
    return np.hstack([constant, regressors]), [CONSTANT_NAME, *labels]


def _gram_inverse(design: _Array) -> _Array:
    """Return ``(X'X)^-1`` from the singular values, raising if ``X`` is rank deficient.

    Working from the SVD of ``X`` avoids squaring its condition number, which
    forming ``X'X`` first would do.
    """
    _, singular, right = np.linalg.svd(design, full_matrices=False)
    tolerance = singular.max() * max(design.shape) * np.finfo(np.float64).eps
    rank = int((singular > tolerance).sum())
    if rank < design.shape[1]:
        msg = (
            f"the regressors are collinear: the design has rank {rank} "
            f"for {design.shape[1]} columns"
        )
        raise DataError(msg)
    return (right.T / singular**2) @ right


def _require_estimable(design: _Array, hac_lags: int | None) -> _Array:
    """Check the sample can support the fit and return ``(X'X)^-1``."""
    n_obs, n_params = design.shape
    if n_obs < n_params + 1:
        msg = (
            f"at least {n_params + 1} observations are needed to estimate "
            f"{n_params} parameters with a residual degree of freedom, got {n_obs}"
        )
        raise DataError(msg)
    if hac_lags is not None and hac_lags >= n_obs:
        msg = f"hac_lags ({hac_lags}) must be smaller than the {n_obs} observations"
        raise DataError(msg)
    return _gram_inverse(design)


def _spans_constant(design: _Array) -> bool:
    """Return whether a constant lies in the column space of the design."""
    with_ones = np.column_stack([design, np.ones(design.shape[0])])
    return bool(np.linalg.matrix_rank(with_ones) == design.shape[1])


def _goodness_of_fit(
    problem: _Problem, residuals: _Array, degrees_of_freedom: int
) -> tuple[float, float]:
    """Return the R-squared and adjusted R-squared of a fit."""
    response = problem.response
    centred = problem.add_constant or _spans_constant(problem.design)
    degenerate = np.ptp(response) == 0.0 if centred else not response.any()
    if degenerate:
        msg = "y is constant, so R-squared is undefined"
        raise DataError(msg)
    reference = response - response.mean() if centred else response
    r_squared = 1.0 - float(residuals @ residuals) / float(reference @ reference)
    total_df = response.size - 1 if centred else response.size
    return r_squared, 1.0 - (1.0 - r_squared) * total_df / degrees_of_freedom


def _newey_west_meat(scores: _Array, lags: int) -> _Array:
    """Return the Bartlett-weighted long-run variance of the scores."""
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        autocovariance = scores[lag:].T @ scores[:-lag]
        meat = meat + weight * (autocovariance + autocovariance.T)
    return meat


def _covariance(
    problem: _Problem,
    residuals: _Array,
    hac_lags: int | None,
    degrees_of_freedom: int,
) -> _Array:
    """Return the covariance of the coefficients, classical or Newey-West."""
    if hac_lags is None:
        return float(residuals @ residuals) / degrees_of_freedom * problem.gram_inverse
    meat = _newey_west_meat(problem.design * residuals[:, None], hac_lags)
    return problem.gram_inverse @ meat @ problem.gram_inverse


def _validated_problem(
    y: pd.Series | npt.ArrayLike,
    x: pd.Series | pd.DataFrame | npt.ArrayLike,
    add_constant: bool,
    hac_lags: int | None,
) -> _Problem:
    """Check every input and assemble the least squares problem."""
    response, y_index = _response(y)
    regressors, labels, x_index = _regressors(x)
    if regressors.shape[0] != response.size:
        msg = f"y has {response.size} rows but x has {regressors.shape[0]}"
        raise DataError(msg)
    index = _result_index(y_index, x_index, response.size)
    design, names = _design(regressors, labels, add_constant)
    return _Problem(
        response=response,
        design=design,
        names=names,
        index=index,
        gram_inverse=_require_estimable(design, hac_lags),
        add_constant=add_constant,
    )


def _fit(problem: _Problem, hac_lags: int | None) -> OlsResult:
    """Solve a validated problem and report its estimates."""
    design, response = problem.design, problem.response
    degrees_of_freedom = design.shape[0] - design.shape[1]
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ coefficients
    r_squared, adjusted = _goodness_of_fit(problem, residuals, degrees_of_freedom)
    covariance = _covariance(problem, residuals, hac_lags, degrees_of_freedom)
    # A negative diagonal can only be rounding on a perfect fit, where it is zero.
    std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = coefficients / std_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), degrees_of_freedom)
    labelled = pd.Index(problem.names)
    return OlsResult(
        coefficients=pd.Series(coefficients, index=labelled),
        std_errors=pd.Series(std_errors, index=labelled),
        t_statistics=pd.Series(t_statistics, index=labelled),
        p_values=pd.Series(p_values, index=labelled),
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        residuals=pd.Series(residuals, index=problem.index),
        n_observations=design.shape[0],
        degrees_of_freedom=degrees_of_freedom,
    )


def ols(
    y: pd.Series | npt.ArrayLike,
    x: pd.Series | pd.DataFrame | npt.ArrayLike,
    add_constant: bool = True,
    hac_lags: int | None = None,
) -> OlsResult:
    """Fit ``y`` on ``x`` by ordinary least squares.

    Args:
        y: Response, one value per observation.
        x: Regressors, one column per regressor. A Series or DataFrame keeps its
            names for the coefficients; a bare array is called ``x`` (one
            dimension) or ``x1``, ``x2`` and so on.
        add_constant: Whether to add an intercept column, named ``"const"``, in
            front of ``x``.
        hac_lags: ``None`` for classical standard errors. An integer ``L >= 0``
            selects Newey-West errors with Bartlett weights up to lag ``L``;
            ``0`` is White's heteroskedasticity-robust estimator.

    Returns:
        The coefficients with their standard errors, t-statistics and p-values,
        the R-squared figures and the residuals.

    Raises:
        ConfigurationError: If ``hac_lags`` is not ``None`` or a non-negative
            integer.
        DataError: If a value is missing or infinite, the shapes or indexes of
            ``y`` and ``x`` differ, a name repeats or clashes with ``"const"``, the
            sample has fewer than ``parameters + 1`` observations, ``hac_lags``
            reaches the sample length, the regressors are collinear, or ``y`` is
            constant.
    """
    require_hac_lags(hac_lags)
    return _fit(_validated_problem(y, x, add_constant, hac_lags), hac_lags)
