"""Returns, rolling volatility and drawdown chart.

Three stacked panels share one time axis:

* the cumulative return, compounding ``(1 + r)`` period by period;
* the rolling annualised volatility, the sample standard deviation (``ddof=1``) of
  the last ``window`` returns times ``sqrt(periods_per_year)``;
* the drawdown, the fall of the compounded wealth from the best level it has
  reached so far. The starting capital counts as a level, so a loss in the very
  first period already registers as a drawdown.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd
import plotly.colors as plotly_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizer.exceptions import DataError
from optimizer.plotting._config import ChartTheme, resolve_theme
from optimizer.plotting._validation import require_integer

_MIN_WINDOW: Final = 2
_MIN_PERIODS_PER_YEAR: Final = 1
_MIN_COLUMNS: Final = 1
_DEFAULT_NAME: Final = "returns"
_PERCENT_FORMAT: Final = ".0%"
_VERTICAL_SPACING: Final = 0.06
_PALETTE: Final = plotly_colors.qualitative.Plotly


def returns_volatility_figure(
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = 21,
    periods_per_year: int = 252,
    theme: ChartTheme | None = None,
) -> go.Figure:
    """Plot cumulative return, rolling volatility and drawdown in three panels.

    Args:
        returns: Simple (linear) period returns on an increasing index, usually
            dates. A Series draws one line per panel and a DataFrame one line per
            column.
        window: Number of periods in the rolling volatility, a whole number of at
            least 2.
        periods_per_year: Periods in a year, a whole number of at least 1, used to
            annualise the volatility.
        theme: Chart appearance; the default theme when ``None``.

    Returns:
        A figure with three rows sharing the x axis: cumulative return, annualised
        rolling volatility and drawdown. The volatility is a gap for the first
        ``window - 1`` periods.

    Raises:
        ConfigurationError: If ``window``, ``periods_per_year`` or ``theme`` is
            invalid.
        DataError: If ``returns`` is not a Series or DataFrame of finite numbers
            above -1 on an increasing index, or has fewer rows than ``window``.
    """
    window, periods_per_year = _require_settings(window, periods_per_year)
    chart_theme = resolve_theme(theme)
    frame = _validated_returns(returns, window)
    panels = _panels(frame, window, periods_per_year)
    figure = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=_VERTICAL_SPACING
    )
    for row, (axis_title, panel) in enumerate(panels, start=1):
        _add_panel(figure, panel, row)
        figure.update_yaxes(
            title_text=axis_title, tickformat=_PERCENT_FORMAT, row=row, col=1
        )
    figure.update_layout(title=f"Returns and {window}-period rolling volatility")
    chart_theme.apply(figure)
    return figure


def _require_settings(window: int, periods_per_year: int) -> tuple[int, int]:
    """Return the rolling window and the annualisation factor as validated ints."""
    return (
        require_integer("window", window, _MIN_WINDOW),
        require_integer("periods_per_year", periods_per_year, _MIN_PERIODS_PER_YEAR),
    )


def _as_frame(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Return ``returns`` as a frame, naming an unnamed series."""
    if isinstance(returns, pd.DataFrame):
        return returns
    if isinstance(returns, pd.Series):
        name = _DEFAULT_NAME if returns.name is None else returns.name
        return returns.to_frame(name=name)
    msg = f"returns must be a pandas Series or DataFrame, got {type(returns).__name__}"
    raise DataError(msg)


def _validated_returns(returns: pd.Series | pd.DataFrame, window: int) -> pd.DataFrame:
    """Return the returns as a float frame after checking shape, index and values."""
    frame = _as_frame(returns)
    if frame.shape[1] < _MIN_COLUMNS:
        msg = f"returns needs at least {_MIN_COLUMNS} column, got {frame.shape[1]}"
        raise DataError(msg)
    if frame.shape[0] < window:
        msg = f"returns needs at least window={window} rows, got {frame.shape[0]}"
        raise DataError(msg)
    if not frame.index.is_monotonic_increasing:
        msg = "the index of returns must be increasing (dates in time order)"
        raise DataError(msg)
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "returns must be numeric"
        raise DataError(msg) from error
    if not np.isfinite(values).all():
        msg = "returns must be finite: drop or fill missing values first"
        raise DataError(msg)
    if (values <= -1.0).any():
        msg = "returns must be greater than -1 (a loss cannot exceed 100%)"
        raise DataError(msg)
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def _panels(
    frame: pd.DataFrame, window: int, periods_per_year: int
) -> list[tuple[str, pd.DataFrame]]:
    """Return the axis title and the series of each of the three panels."""
    wealth = (1.0 + frame).cumprod()
    # The initial capital of 1 is a level too, so a first-period loss is a drawdown.
    peak = wealth.cummax().clip(lower=1.0)
    volatility = frame.rolling(window).std(ddof=1) * math.sqrt(periods_per_year)
    return [
        ("Cumulative return", wealth - 1.0),
        ("Volatility (annualised)", volatility),
        ("Drawdown", wealth / peak - 1.0),
    ]


def _add_panel(figure: go.Figure, panel: pd.DataFrame, row: int) -> None:
    """Add one line per column of ``panel`` to the given row of ``figure``."""
    for position in range(panel.shape[1]):
        name = str(panel.columns[position])
        figure.add_trace(
            go.Scatter(
                x=panel.index,
                y=panel.iloc[:, position].to_numpy(),
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=row == 1,
                line={"color": _PALETTE[position % len(_PALETTE)]},
            ),
            row=row,
            col=1,
        )
