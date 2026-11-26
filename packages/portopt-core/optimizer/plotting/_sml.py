"""Security market line chart.

The capital asset pricing model (Sharpe, 1964, "Capital Asset Prices: A Theory of
Market Equilibrium under Conditions of Risk", Journal of Finance 19(3)) prices an
asset's expected return linearly in its market beta:

    E[r] = rf + beta * (E[r_m] - rf)

The security market line is that relation drawn over beta. Assets above it return
more than their beta justifies (positive alpha) and assets below it less.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from optimizer.exceptions import DataError
from optimizer.plotting._config import ChartTheme, resolve_theme
from optimizer.plotting._validation import (
    as_finite_scalar,
    as_finite_vector,
    as_labels,
)


def security_market_line_figure(
    betas: npt.ArrayLike,
    mean_returns: npt.ArrayLike,
    risk_free_rate: float,
    market_premium: float,
    *,
    labels: Sequence[object] | None = None,
    theme: ChartTheme | None = None,
) -> go.Figure:
    """Scatter assets by beta and expected return over the security market line.

    Args:
        betas: Market beta of each asset, finite.
        mean_returns: Expected (or realised mean) return of each asset, in the same
            period as ``risk_free_rate``, finite and the same length as ``betas``.
        risk_free_rate: Risk-free rate, the line's value at beta 0.
        market_premium: Expected market return minus the risk-free rate, the line's
            slope. It may be negative.
        labels: Optional name written next to each asset marker.
        theme: Chart appearance; the default theme when ``None``.

    Returns:
        A figure with the market line (first trace, named
        ``"Security market line"``) over the beta range of the assets extended to
        include 0 and 1, and the asset scatter (named ``"Assets"``).

    Raises:
        ConfigurationError: If ``theme`` is invalid.
        DataError: If the inputs are not finite numbers of matching length, or the
            labels do not match the assets.
    """
    beta_values, return_values = _asset_points(betas, mean_returns)
    risk_free = as_finite_scalar(risk_free_rate, "risk_free_rate")
    premium = as_finite_scalar(market_premium, "market_premium")
    names = as_labels(labels, beta_values.size)
    chart_theme = resolve_theme(theme)
    figure = go.Figure(
        [
            _market_line(beta_values, risk_free, premium),
            _asset_scatter(beta_values, return_values, names),
        ]
    )
    figure.update_layout(
        title="Security market line", xaxis_title="Beta", yaxis_title="Expected return"
    )
    chart_theme.apply(figure)
    return figure


def _asset_points(
    betas: npt.ArrayLike, mean_returns: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the validated beta and mean-return vectors of the assets."""
    beta_values = as_finite_vector(betas, "betas")
    return_values = as_finite_vector(mean_returns, "mean_returns")
    if beta_values.size != return_values.size:
        msg = (
            "betas and mean_returns must have the same length, got "
            f"{beta_values.size} and {return_values.size}"
        )
        raise DataError(msg)
    return beta_values, return_values


def _market_line(
    beta_values: npt.NDArray[np.float64], risk_free: float, premium: float
) -> go.Scatter:
    """Return the line ``rf + beta * premium`` over the betas extended to 0 and 1."""
    # 0 and 1 are listed explicitly so the line has a vertex at the risk-free rate
    # and at the market portfolio whatever range the asset betas cover.
    line_betas = np.unique(
        [min(0.0, beta_values.min()), 0.0, 1.0, max(1.0, beta_values.max())]
    )
    return go.Scatter(
        x=line_betas,
        y=risk_free + line_betas * premium,
        mode="lines",
        name="Security market line",
    )


def _asset_scatter(
    beta_values: npt.NDArray[np.float64],
    return_values: npt.NDArray[np.float64],
    names: list[str] | None,
) -> go.Scatter:
    """Return the marker trace of the assets, with their names when given."""
    return go.Scatter(
        x=beta_values,
        y=return_values,
        mode="markers" if names is None else "markers+text",
        text=names,
        textposition="top center",
        name="Assets",
    )
