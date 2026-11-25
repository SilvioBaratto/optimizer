"""Tests for the returns, rolling volatility and drawdown figure.

The expected series are computed here with plain loops over the returns and
``numpy.std``, not with the pandas rolling and cumulative functions the builder uses.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.plotting import returns_volatility_figure

_WINDOW = 5
_YEAR = 252
_PANELS = 3


def _series(rows: int = 30, seed: int = 3) -> pd.Series:
    """Return reproducible daily returns on a business-day index."""
    values = np.random.default_rng(seed).normal(0.001, 0.02, rows)
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=rows), name="s")


def _panel(figure: go.Figure, panel: int, position: int = 0) -> np.ndarray:
    """Return the y values of the ``position``-th trace of a 1-based panel."""
    traces = [t for t in figure.data if t.yaxis == ("y" if panel == 1 else f"y{panel}")]
    return np.asarray(traces[position].y, dtype=float)


def _hand_cumulative(values: np.ndarray) -> list[float]:
    """Return the compounded return after each period."""
    wealth, out = 1.0, []
    for value in values:
        wealth *= 1.0 + value
        out.append(wealth - 1.0)
    return out


def _hand_volatility(values: np.ndarray, window: int, periods: int) -> list[float]:
    """Return the annualised sample standard deviation of each full window."""
    out = [math.nan] * (window - 1)
    for end in range(window, len(values) + 1):
        out.append(
            float(np.std(values[end - window : end], ddof=1)) * math.sqrt(periods)
        )
    return out


def _hand_drawdown(values: np.ndarray) -> list[float]:
    """Return the fall from the best wealth reached so far, starting from 1."""
    wealth, peak, out = 1.0, 1.0, []
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        out.append(wealth / peak - 1.0)
    return out


class TestPanels:
    """Three stacked panels sharing one time axis."""

    def test_three_panels_share_the_x_axis(self) -> None:
        """Cumulative return, volatility and drawdown on y, y2 and y3."""
        figure = returns_volatility_figure(_series(), window=_WINDOW)
        assert [t.yaxis for t in figure.data] == ["y", "y2", "y3"]
        assert [t.xaxis for t in figure.data] == ["x", "x2", "x3"]
        assert figure.layout.xaxis.matches == "x3"
        assert figure.layout.xaxis2.matches == "x3"

    def test_every_panel_has_its_own_percent_axis_title(self) -> None:
        """Each panel says what it plots, in percent."""
        figure = returns_volatility_figure(_series(), window=_WINDOW)
        titles = [
            figure.layout.yaxis.title.text,
            figure.layout.yaxis2.title.text,
            figure.layout.yaxis3.title.text,
        ]
        assert titles == ["Cumulative return", "Volatility (annualised)", "Drawdown"]
        assert figure.layout.yaxis.tickformat == ".0%"

    def test_the_window_is_named_in_the_title(self) -> None:
        """The chart title says how many periods the volatility spans."""
        figure = returns_volatility_figure(_series(), window=_WINDOW)
        assert f"{_WINDOW}" in figure.layout.title.text

    def test_the_cumulative_return_is_compounded(self) -> None:
        """Hand loop: wealth multiplied by (1 + r) each period, minus one."""
        series = _series()
        figure = returns_volatility_figure(series, window=_WINDOW)
        np.testing.assert_allclose(
            _panel(figure, 1), _hand_cumulative(series.to_numpy()), atol=1e-12
        )

    def test_rolling_volatility_matches_numpy(self) -> None:
        """Sample std (ddof=1) over each window, times sqrt(252)."""
        series = _series()
        figure = returns_volatility_figure(series, window=_WINDOW)
        np.testing.assert_allclose(
            _panel(figure, 2),
            _hand_volatility(series.to_numpy(), _WINDOW, _YEAR),
            atol=1e-12,
            equal_nan=True,
        )

    def test_the_first_window_minus_one_volatilities_are_gaps(self) -> None:
        """A full window is needed before a volatility is drawn."""
        volatility = _panel(returns_volatility_figure(_series(), window=_WINDOW), 2)
        assert np.isnan(volatility[: _WINDOW - 1]).all()
        assert not np.isnan(volatility[_WINDOW - 1 :]).any()

    def test_annualisation_scales_with_the_square_root_of_the_periods(self) -> None:
        """Monthly data: 12 periods per year instead of 252."""
        series = _series()
        daily = _panel(returns_volatility_figure(series, window=_WINDOW), 2)
        monthly = _panel(
            returns_volatility_figure(series, window=_WINDOW, periods_per_year=12), 2
        )
        np.testing.assert_allclose(
            monthly[_WINDOW - 1 :],
            daily[_WINDOW - 1 :] * math.sqrt(12 / _YEAR),
            atol=1e-12,
        )

    def test_the_drawdown_is_the_fall_from_the_running_maximum(self) -> None:
        """Hand loop over the wealth path; it is never above zero."""
        series = _series()
        drawdown = _panel(returns_volatility_figure(series, window=_WINDOW), 3)
        np.testing.assert_allclose(
            drawdown, _hand_drawdown(series.to_numpy()), atol=1e-12
        )
        assert (drawdown <= 0.0).all()

    def test_a_loss_in_the_first_period_is_already_a_drawdown(self) -> None:
        """The peak starts at the initial capital, so -10% first reads -10%."""
        values = np.array([-0.10, 0.20, -0.05, 0.02, 0.01, 0.03])
        series = pd.Series(values, index=pd.RangeIndex(6))
        drawdown = _panel(returns_volatility_figure(series, window=2), 3)
        # Hand: 0.9 / 1 - 1 = -0.1, then wealth 1.08 sets a new peak.
        assert drawdown[0] == pytest.approx(-0.10, abs=1e-15)
        assert drawdown[1] == 0.0
        # Wealth 1.08 * 0.95 = 1.026 against the peak 1.08: -5%.
        assert drawdown[2] == pytest.approx(0.95 - 1.0, abs=1e-12)


class TestInputs:
    """One trace per column, and the index and name handling."""

    def test_a_frame_draws_one_trace_per_column_in_every_panel(self) -> None:
        """Two assets give six traces, named after the columns."""
        frame = pd.DataFrame({"A": _series(seed=1), "B": _series(seed=2)})
        figure = returns_volatility_figure(frame, window=_WINDOW)
        assert len(figure.data) == _PANELS * 2
        assert [t.name for t in figure.data] == ["A", "B"] * _PANELS
        np.testing.assert_allclose(
            _panel(figure, 1, position=1),
            _hand_cumulative(frame["B"].to_numpy()),
            atol=1e-12,
        )

    def test_a_series_draws_one_trace_per_panel(self) -> None:
        """The series name titles the single line."""
        figure = returns_volatility_figure(_series(), window=_WINDOW)
        assert [t.name for t in figure.data] == ["s", "s", "s"]

    def test_an_unnamed_series_is_called_returns(self) -> None:
        """A series without a name still gets a legend entry."""
        series = _series()
        series.name = None
        figure = returns_volatility_figure(series, window=_WINDOW)
        assert figure.data[0].name == "returns"

    def test_each_asset_keeps_one_colour_and_one_legend_entry(self) -> None:
        """Panels share colour and legend group; only the first panel lists it."""
        frame = pd.DataFrame({"A": _series(seed=1), "B": _series(seed=2)})
        figure = returns_volatility_figure(frame, window=_WINDOW)
        assert figure.data[0].line.color == figure.data[2].line.color
        assert figure.data[0].line.color != figure.data[1].line.color
        assert figure.data[0].legendgroup == figure.data[2].legendgroup == "A"
        assert [t.showlegend for t in figure.data] == [True, True] + [False] * 4

    def test_a_datetime_index_is_the_x_axis(self) -> None:
        """Dates are passed through to plotly unchanged."""
        series = _series()
        figure = returns_volatility_figure(series, window=_WINDOW)
        assert pd.DatetimeIndex(figure.data[0].x).equals(series.index)

    def test_any_increasing_index_is_accepted(self) -> None:
        """A plain integer index is fine as long as it only goes up."""
        series = _series().reset_index(drop=True)
        figure = returns_volatility_figure(series, window=_WINDOW)
        assert list(figure.data[0].x) == list(range(len(series)))

    def test_the_input_is_not_modified(self) -> None:
        """The builder works on a copy."""
        series = _series()
        before = series.copy()
        returns_volatility_figure(series, window=_WINDOW)
        pd.testing.assert_series_equal(series, before)


class TestValidation:
    """Every rejected input names what is wrong."""

    @pytest.mark.parametrize("window", [1, 0, -3])
    def test_a_window_below_two_is_rejected(self, window: int) -> None:
        """A sample standard deviation needs two observations."""
        with pytest.raises(ConfigurationError, match="window"):
            returns_volatility_figure(_series(), window=window)

    @pytest.mark.parametrize("window", [5.0, 5.5, math.nan, True, "5", None])
    def test_a_window_that_is_not_an_integer_is_rejected(self, window: object) -> None:
        """Floats (even 5.0), NaN, bool and text are caught before pandas sees them."""
        with pytest.raises(ConfigurationError, match="window"):
            returns_volatility_figure(_series(), window=window)  # type: ignore[arg-type]

    @pytest.mark.parametrize("periods", [0, -252])
    def test_periods_per_year_below_one_is_rejected(self, periods: int) -> None:
        """There is at least one period in a year."""
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            returns_volatility_figure(
                _series(), window=_WINDOW, periods_per_year=periods
            )

    @pytest.mark.parametrize("periods", [252.0, 12.5, math.nan, math.inf, True, "252"])
    def test_periods_per_year_that_is_not_an_integer_is_rejected(
        self, periods: object
    ) -> None:
        """NaN would otherwise draw an all-NaN volatility panel without an error."""
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            returns_volatility_figure(
                _series(),
                window=_WINDOW,
                periods_per_year=periods,  # type: ignore[arg-type]
            )

    def test_numpy_integers_are_accepted_for_both_settings(self) -> None:
        """Values coming out of a numpy computation are whole numbers too."""
        series = _series()
        plain = returns_volatility_figure(series, window=_WINDOW)
        numpy_typed = returns_volatility_figure(
            series, window=np.int64(_WINDOW), periods_per_year=np.int32(_YEAR)
        )
        np.testing.assert_allclose(
            _panel(numpy_typed, 2), _panel(plain, 2), atol=1e-15, equal_nan=True
        )

    def test_fewer_rows_than_the_window_is_rejected(self) -> None:
        """No volatility could be drawn, so the call fails instead of showing gaps."""
        with pytest.raises(DataError, match="at least"):
            returns_volatility_figure(_series(rows=10), window=21)

    @pytest.mark.parametrize(
        "bad",
        [
            np.ones((30, 2)),
            [0.01, 0.02],
            "returns",
        ],
    )
    def test_only_series_and_frames_are_accepted(self, bad: object) -> None:
        """Arrays and lists have no index to put on the time axis."""
        with pytest.raises(DataError, match="Series or DataFrame"):
            returns_volatility_figure(bad, window=_WINDOW)  # type: ignore[arg-type]

    def test_a_frame_without_columns_is_rejected(self) -> None:
        """There is nothing to draw."""
        empty = pd.DataFrame(index=pd.bdate_range("2024-01-01", periods=30))
        with pytest.raises(DataError, match="at least 1 column"):
            returns_volatility_figure(empty, window=_WINDOW)

    def test_a_shuffled_datetime_index_is_rejected(self) -> None:
        """Compounding follows row order, so dates must increase."""
        series = _series()
        shuffled = series.iloc[[1, 0, *range(2, len(series))]]
        with pytest.raises(DataError, match="increasing"):
            returns_volatility_figure(shuffled, window=_WINDOW)

    def test_a_decreasing_integer_index_is_rejected(self) -> None:
        """A monotonic index must be increasing to read left to right in time."""
        series = _series().reset_index(drop=True).iloc[::-1]
        with pytest.raises(DataError, match="increasing"):
            returns_volatility_figure(series, window=_WINDOW)

    def test_a_missing_return_is_rejected(self) -> None:
        """A gap would break the compounding and every window containing it."""
        series = _series()
        series.iloc[4] = np.nan
        with pytest.raises(DataError, match="finite"):
            returns_volatility_figure(series, window=_WINDOW)

    @pytest.mark.parametrize("loss", [-1.0, -1.5])
    def test_a_return_of_minus_one_or_less_is_rejected(self, loss: float) -> None:
        """Losing more than everything has no compounded path."""
        series = _series()
        series.iloc[4] = loss
        with pytest.raises(DataError, match="greater than -1"):
            returns_volatility_figure(series, window=_WINDOW)

    def test_a_non_numeric_column_is_rejected(self) -> None:
        """Text cannot be compounded."""
        frame = pd.DataFrame({"A": _series(), "B": "x"})
        with pytest.raises(DataError, match="numeric"):
            returns_volatility_figure(frame, window=_WINDOW)
