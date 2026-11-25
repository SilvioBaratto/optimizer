"""Correlation heatmap.

A heatmap of the pairwise correlations of a set of assets, with a diverging colour
scale centred on zero so positive and negative dependence read at a glance. Assets
can be reordered by hierarchical clustering so that groups of related assets form
visible blocks along the diagonal; the ordering is the leaf order of an
average-linkage clustering on the distance ``sqrt((1 - rho) / 2)``, the metric of
Lopez de Prado (2016), "Building Diversified Portfolios that Outperform Out of
Sample", Journal of Portfolio Management 42(4).
"""

from __future__ import annotations

from typing import Final, Literal, cast, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.plotting._config import ChartTheme, CorrelationOrder, resolve_theme

_CorrelationMethod = Literal["pearson", "spearman", "kendall"]
_METHODS: Final = frozenset(get_args(_CorrelationMethod))
_MIN_ASSETS: Final = 2
_MIN_OBSERVATIONS: Final = 2
# Absorbs the rounding noise of a correlation matrix computed elsewhere, so one that
# is off by float error is not refused.
_TOLERANCE: Final = 1e-8
_ANNOTATION_DECIMALS: Final = 2
_COLOR_SCALE: Final = "RdBu"


def correlation_heatmap(
    data: pd.DataFrame,
    *,
    is_correlation: bool = False,
    method: str = "pearson",
    order: CorrelationOrder | str = CorrelationOrder.NONE,
    annotate: bool = True,
    theme: ChartTheme | None = None,
) -> go.Figure:
    """Draw the correlation matrix of a set of assets as a heatmap.

    Args:
        data: Asset returns, one column per asset, or with ``is_correlation`` a
            correlation matrix. Missing returns are dropped pairwise.
        is_correlation: Treat ``data`` as an already computed correlation matrix:
            square, symmetric, with a unit diagonal and entries in ``[-1, 1]``.
        method: ``"pearson"``, ``"spearman"`` or ``"kendall"``; ignored when
            ``is_correlation`` is true.
        order: A ``CorrelationOrder`` or its value. ``"none"`` keeps the column order;
            ``"cluster"`` reorders the assets by hierarchical clustering so similar
            assets sit next to each other.
        annotate: Write each value, rounded to two decimals, in its cell.
        theme: Chart appearance; the default theme when ``None``.

    Returns:
        A plotly figure with one heatmap trace, colour range ``-1`` to ``1``.

    Raises:
        ConfigurationError: If ``method``, ``order`` or ``theme`` is invalid.
        DataError: If ``data`` is not a numeric frame of at least two assets, is not a
            valid correlation matrix when ``is_correlation`` is set, or has a
            correlation that cannot be computed (for example a constant column).
    """
    _require_method(method)
    asset_order = _resolve_order(order)
    chart_theme = resolve_theme(theme)
    if is_correlation:
        matrix = _checked_correlation(data)
        title = "Correlation matrix"
    else:
        matrix = _correlation_of_returns(data, method)
        title = f"Correlation heatmap ({method})"
    if asset_order is CorrelationOrder.CLUSTER:
        matrix = _cluster_ordered(matrix)
    figure = go.Figure(_heatmap_trace(matrix, annotate=annotate))
    figure.update_layout(title=title)
    figure.update_xaxes(type="category")
    figure.update_yaxes(type="category", autorange="reversed")
    chart_theme.apply(figure)
    return figure


def _require_method(method: str) -> None:
    """Raise if ``method`` is not a correlation coefficient this chart supports."""
    if method not in _METHODS:
        msg = f"method must be one of {sorted(_METHODS)}, got {method!r}"
        raise ConfigurationError(msg)


def _resolve_order(order: CorrelationOrder | str) -> CorrelationOrder:
    """Return ``order`` as a ``CorrelationOrder``, accepting its plain string value."""
    try:
        return CorrelationOrder(order)
    except ValueError as error:
        allowed = [member.value for member in CorrelationOrder]
        msg = f"order must be one of {allowed}, got {order!r}"
        raise ConfigurationError(msg) from error


def _numeric_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return ``data`` after checking it is a numeric frame of enough assets."""
    if not isinstance(data, pd.DataFrame):
        msg = f"data must be a pandas DataFrame, got {type(data).__name__}"
        raise DataError(msg)
    if data.shape[1] < _MIN_ASSETS:
        msg = f"data needs at least {_MIN_ASSETS} assets (columns), got {data.shape[1]}"
        raise DataError(msg)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        msg = "data must be numeric in every column"
        raise DataError(msg)
    return data


def _correlation_of_returns(data: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return the correlation matrix of the asset returns in ``data``."""
    frame = _numeric_frame(data)
    if frame.shape[0] < _MIN_OBSERVATIONS:
        msg = (
            f"data needs at least {_MIN_OBSERVATIONS} observations (rows), "
            f"got {frame.shape[0]}"
        )
        raise DataError(msg)
    matrix = frame.corr(method=cast("_CorrelationMethod", method))
    if matrix.isna().to_numpy().any():
        msg = (
            "a correlation is undefined: a column is constant or shares fewer than "
            f"{_MIN_OBSERVATIONS} observations with another column"
        )
        raise DataError(msg)
    return matrix


def _checked_correlation(data: pd.DataFrame) -> pd.DataFrame:
    """Return ``data`` after checking it is a valid correlation matrix."""
    frame = _numeric_frame(data)
    values = frame.to_numpy(dtype=np.float64)
    if values.shape[0] != values.shape[1]:
        msg = f"a correlation matrix must be square, got shape {values.shape}"
        raise DataError(msg)
    if not np.isfinite(values).all():
        msg = "a correlation matrix must be finite"
        raise DataError(msg)
    if not np.allclose(values, values.T, rtol=0.0, atol=_TOLERANCE):
        msg = "a correlation matrix must be symmetric"
        raise DataError(msg)
    if not np.allclose(np.diag(values), 1.0, rtol=0.0, atol=_TOLERANCE):
        msg = "a correlation matrix must have a unit diagonal"
        raise DataError(msg)
    if np.abs(values).max() > 1.0 + _TOLERANCE:
        msg = "correlations must lie between -1 and 1"
        raise DataError(msg)
    return frame


def _cluster_ordered(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return ``matrix`` with rows and columns in hierarchical-clustering order."""
    values = matrix.to_numpy(dtype=np.float64)
    distance = np.sqrt(np.clip((1.0 - values) / 2.0, 0.0, None))
    # Rounding can leave the two triangles a hair apart and the diagonal off zero,
    # while squareform and linkage need an exactly symmetric, zero-diagonal matrix.
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    leaves = leaves_list(tree)
    return matrix.iloc[leaves, leaves]


def _annotation_text(values: npt.NDArray[np.float64]) -> list[list[str]]:
    """Return the cell labels: values rounded to two decimals, without ``-0.00``."""
    # Adding 0.0 turns a rounded negative zero into a positive one.
    rounded = np.round(values, _ANNOTATION_DECIMALS) + 0.0
    return [[f"{value:.{_ANNOTATION_DECIMALS}f}" for value in row] for row in rounded]


def _heatmap_trace(matrix: pd.DataFrame, *, annotate: bool) -> go.Heatmap:
    """Return the heatmap trace for ``matrix``."""
    values = matrix.to_numpy(dtype=np.float64)
    labels: dict[str, object] = {}
    if annotate:
        labels = {"text": _annotation_text(values), "texttemplate": "%{text}"}
    return go.Heatmap(
        z=values,
        x=[str(label) for label in matrix.columns],
        y=[str(label) for label in matrix.index],
        colorscale=_COLOR_SCALE,
        reversescale=True,
        zmin=-1.0,
        zmax=1.0,
        zmid=0.0,
        colorbar={"title": {"text": "Correlation"}},
        hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        **labels,
    )
