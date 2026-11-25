"""Efficient frontier chart.

The frontier is the set of portfolios that offer the highest expected return for
each level of risk (Markowitz, 1952, "Portfolio Selection", Journal of Finance
7(1)). The figure plots risk on the x axis and expected return on the y axis, and
can mark named portfolios such as the maximum-Sharpe and minimum-variance ones.
``frontier_from_weights`` turns a table of portfolio weights into the points to
plot, with risk ``sqrt(w' S w)`` and expected return ``w' mu``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go

from optimizer.exceptions import DataError
from optimizer.plotting._config import ChartTheme, resolve_theme
from optimizer.plotting._validation import (
    as_finite_vector,
    as_labels,
    require_non_negative,
)

_MIN_POINTS: Final = 2
_HIGHLIGHT_SIZE: Final = 14
# A covariance matrix estimated from data can carry negative eigenvalues of the
# order of rounding error, relative to its largest entry. Anything larger is a
# genuine indefiniteness.
_PSD_TOLERANCE: Final = 1e-10
_SYMMETRY_TOLERANCE: Final = 1e-10


def efficient_frontier_figure(
    risks: npt.ArrayLike,
    returns: npt.ArrayLike,
    *,
    labels: Sequence[object] | None = None,
    highlight: Mapping[str, tuple[float, float]] | None = None,
    risk_label: str = "Volatility",
    return_label: str = "Expected return",
    theme: ChartTheme | None = None,
) -> go.Figure:
    """Plot an efficient frontier, and optionally mark named portfolios on it.

    Args:
        risks: Risk of each frontier portfolio, non-negative and finite.
        returns: Expected return of each frontier portfolio, finite, same length as
            ``risks`` and at least two points.
        labels: Optional text shown when hovering over each frontier point.
        highlight: Optional mapping of a portfolio name to its ``(risk, return)``
            pair, drawn as one separate named marker each, for example the maximum
            Sharpe and minimum variance portfolios.
        risk_label: Title of the risk (x) axis.
        return_label: Title of the return (y) axis.
        theme: Chart appearance; the default theme when ``None``.

    Returns:
        A figure with the frontier line sorted by risk as its first trace, followed
        by one marker trace per highlighted portfolio.

    Raises:
        ConfigurationError: If ``theme`` is invalid.
        DataError: If the points, labels or highlights are malformed.
    """
    risk_values, return_values = _frontier_points(risks, returns)
    point_labels = as_labels(labels, risk_values.size)
    markers = _highlight_points(highlight)
    chart_theme = resolve_theme(theme)
    order = np.argsort(risk_values, kind="stable")
    text = None if point_labels is None else [point_labels[i] for i in order]
    figure = go.Figure(
        go.Scatter(
            x=risk_values[order],
            y=return_values[order],
            mode="lines+markers",
            name="Efficient frontier",
            text=text,
        )
    )
    for name, (risk, expected_return) in markers.items():
        figure.add_trace(
            go.Scatter(
                x=[risk],
                y=[expected_return],
                mode="markers",
                name=name,
                marker={"size": _HIGHLIGHT_SIZE, "symbol": "star"},
            )
        )
    figure.update_layout(
        title="Efficient frontier", xaxis_title=risk_label, yaxis_title=return_label
    )
    chart_theme.apply(figure)
    return figure


def frontier_from_weights(
    weights: pd.DataFrame,
    mu: npt.ArrayLike | pd.Series,
    covariance: npt.ArrayLike | pd.DataFrame,
) -> pd.DataFrame:
    """Return the risk and expected return of each portfolio in a weights table.

    Args:
        weights: Portfolio weights, one row per portfolio and one column per asset.
        mu: Expected return of each asset: a Series aligned to the weight columns by
            name, or an array in the column order.
        covariance: Covariance matrix of the asset returns: a DataFrame aligned to
            the weight columns by name, or a square array in the column order. It
            must be symmetric and positive semi-definite.

    Returns:
        A frame indexed like ``weights`` with columns ``risk``
        (``sqrt(w' S w)``) and ``expected_return`` (``w' mu``).

    Raises:
        DataError: If an input is not numeric and finite, the shapes or asset names
            do not match, or the covariance matrix is not symmetric positive
            semi-definite.
    """
    matrix = _weights_matrix(weights)
    expected = _aligned_vector(mu, weights.columns, "mu")
    sigma = _aligned_covariance(covariance, weights.columns)
    variance = np.einsum("ij,jk,ik->i", matrix, sigma, matrix)
    # A positive semi-definite matrix can still produce a variance a hair below
    # zero from rounding, and the square root would turn that into NaN.
    risk = np.sqrt(np.clip(variance, 0.0, None))
    return pd.DataFrame(
        {"risk": risk, "expected_return": matrix @ expected}, index=weights.index
    )


def _frontier_points(
    risks: npt.ArrayLike, returns: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the validated risk and return vectors of the frontier."""
    risk_values = as_finite_vector(risks, "risks", min_size=_MIN_POINTS)
    return_values = as_finite_vector(returns, "returns", min_size=_MIN_POINTS)
    if risk_values.size != return_values.size:
        msg = (
            "risks and returns must have the same length, got "
            f"{risk_values.size} and {return_values.size}"
        )
        raise DataError(msg)
    require_non_negative(risk_values, "risks")
    return risk_values, return_values


def _highlight_points(
    highlight: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    """Return the validated ``name -> (risk, return)`` pairs to mark."""
    if highlight is None:
        return {}
    if not isinstance(highlight, Mapping):
        msg = "highlight must be a mapping of portfolio name to (risk, return)"
        raise DataError(msg)
    points: dict[str, tuple[float, float]] = {}
    for name, pair in highlight.items():
        label = f"highlight[{name!r}]"
        values = as_finite_vector(pair, label, min_size=_MIN_POINTS)
        if values.size != _MIN_POINTS:
            msg = f"{label} must be a (risk, return) pair, got {values.size} values"
            raise DataError(msg)
        require_non_negative(values[:1], f"risk of {label}")
        points[str(name)] = (float(values[0]), float(values[1]))
    return points


def _weights_matrix(weights: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return the weights as a validated float matrix, portfolios by assets."""
    if not isinstance(weights, pd.DataFrame):
        msg = f"weights must be a pandas DataFrame, got {type(weights).__name__}"
        raise DataError(msg)
    if weights.shape[0] < 1 or weights.shape[1] < 1:
        msg = f"weights needs at least one portfolio and one asset, got {weights.shape}"
        raise DataError(msg)
    if not weights.columns.is_unique:
        msg = "weights columns must be unique asset names"
        raise DataError(msg)
    try:
        matrix = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "weights must be numeric"
        raise DataError(msg) from error
    if not np.isfinite(matrix).all():
        msg = "weights must be finite"
        raise DataError(msg)
    return matrix


def _aligned_vector(
    values: npt.ArrayLike | pd.Series, assets: pd.Index, name: str
) -> npt.NDArray[np.float64]:
    """Return ``values`` as a vector in the order of ``assets``."""
    if isinstance(values, pd.Series):
        if not assets.isin(values.index).all() or not values.index.is_unique:
            msg = f"{name} must hold a unique entry for every weight column"
            raise DataError(msg)
        values = values.reindex(assets)
    vector = as_finite_vector(values, name)
    if vector.size != len(assets):
        msg = f"{name} has {vector.size} entries for {len(assets)} assets"
        raise DataError(msg)
    return vector


def _covariance_by_name(covariance: pd.DataFrame, assets: pd.Index) -> pd.DataFrame:
    """Return ``covariance`` with rows and columns in the order of ``assets``."""
    labelled = covariance.index.is_unique and covariance.columns.is_unique
    complete = (
        assets.isin(covariance.index).all() and assets.isin(covariance.columns).all()
    )
    if not (labelled and complete):
        msg = "covariance must hold a unique row and column for every weight column"
        raise DataError(msg)
    return covariance.reindex(index=assets, columns=assets)


def _aligned_covariance(
    covariance: npt.ArrayLike | pd.DataFrame, assets: pd.Index
) -> npt.NDArray[np.float64]:
    """Return the covariance matrix in the order of ``assets``, once validated."""
    if isinstance(covariance, pd.DataFrame):
        covariance = _covariance_by_name(covariance, assets)
    try:
        sigma = np.array(covariance, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "covariance must be numeric"
        raise DataError(msg) from error
    if sigma.shape != (len(assets), len(assets)):
        msg = f"covariance must be {len(assets)} x {len(assets)}, got {sigma.shape}"
        raise DataError(msg)
    _require_positive_semidefinite(sigma)
    return sigma


def _require_positive_semidefinite(sigma: npt.NDArray[np.float64]) -> None:
    """Raise unless ``sigma`` is finite, symmetric and positive semi-definite."""
    if not np.isfinite(sigma).all():
        msg = "covariance must be finite"
        raise DataError(msg)
    scale = float(np.abs(sigma).max())
    if not np.allclose(sigma, sigma.T, rtol=0.0, atol=_SYMMETRY_TOLERANCE * scale):
        msg = "covariance must be symmetric"
        raise DataError(msg)
    if np.linalg.eigvalsh(sigma).min() < -_PSD_TOLERANCE * scale:
        msg = "covariance must be positive semi-definite"
        raise DataError(msg)
