"""Plotly charts for portfolio analysis.

Figure builders for the standard pictures of a portfolio study: a correlation
heatmap, an efficient frontier, cumulative return with rolling volatility and
drawdown, and the security market line. Each builder returns a
``plotly.graph_objects.Figure`` that the caller can show, customise or save with
``save_figure_html``.

This is the only part of ``optimizer`` that imports plotly directly, and it is not
imported by ``import optimizer``: load it with ``import optimizer.plotting``.
Static image export needs the separate ``kaleido`` package and is not provided.
"""

from optimizer.plotting._config import ChartTheme, CorrelationOrder
from optimizer.plotting._correlation import correlation_heatmap
from optimizer.plotting._export import save_figure_html
from optimizer.plotting._frontier import (
    efficient_frontier_figure,
    frontier_from_weights,
)
from optimizer.plotting._returns import returns_volatility_figure
from optimizer.plotting._sml import security_market_line_figure

__all__ = [
    "ChartTheme",
    "CorrelationOrder",
    "correlation_heatmap",
    "efficient_frontier_figure",
    "frontier_from_weights",
    "returns_volatility_figure",
    "save_figure_html",
    "security_market_line_figure",
]
