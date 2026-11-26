"""Tests for the security market line figure.

The line is the CAPM relation ``E[r] = rf + beta * premium``, so its value at beta 0
is the risk-free rate and at beta 1 the risk-free rate plus the market premium.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from optimizer.exceptions import DataError
from optimizer.plotting import security_market_line_figure

_RF = 0.02
_PREMIUM = 0.05
_BETAS = [0.5, 0.9, 1.2]
_RETURNS = [0.04, 0.07, 0.09]


def _build(**overrides: object) -> go.Figure:
    """Build the figure for the standard inputs, with keyword overrides."""
    arguments: dict[str, object] = {
        "betas": _BETAS,
        "mean_returns": _RETURNS,
        "risk_free_rate": _RF,
        "market_premium": _PREMIUM,
    }
    arguments.update(overrides)
    return security_market_line_figure(**arguments)  # type: ignore[arg-type]


def _trace(figure: go.Figure, name: str) -> go.Scatter:
    """Return the scatter trace called ``name``."""
    for trace in figure.data:
        if isinstance(trace, go.Scatter) and trace.name == name:
            return trace
    pytest.fail(f"no trace named {name!r}")


def _line(figure: go.Figure) -> go.Scatter:
    """Return the trace that draws the market line."""
    return _trace(figure, "Security market line")


def _assets(figure: go.Figure) -> go.Scatter:
    """Return the trace that scatters the assets."""
    return _trace(figure, "Assets")


class TestLine:
    """The market line and the range it covers."""

    def test_the_line_passes_through_the_risk_free_rate_at_beta_zero(self) -> None:
        """The line at beta 0 equals the risk-free rate."""
        line = _line(_build())
        x = np.asarray(line.x)
        assert np.asarray(line.y)[np.flatnonzero(x == 0.0)[0]] == pytest.approx(_RF)

    def test_the_line_passes_through_the_market_at_beta_one(self) -> None:
        """The line at beta 1 is the risk-free rate plus the market premium."""
        line = _line(_build())
        x = np.asarray(line.x)
        value = np.asarray(line.y)[np.flatnonzero(x == 1.0)[0]]
        assert value == pytest.approx(_RF + _PREMIUM)

    def test_the_slope_is_the_market_premium(self) -> None:
        """Every segment of the line rises by the premium per unit of beta."""
        line = _line(_build(betas=[-0.4, 0.5, 1.8]))
        slopes = np.diff(np.asarray(line.y)) / np.diff(np.asarray(line.x))
        np.testing.assert_allclose(slopes, _PREMIUM, atol=1e-12)

    def test_a_negative_premium_gives_a_falling_line(self) -> None:
        """A negative premium is allowed and tilts the line down."""
        line = _line(_build(market_premium=-0.03))
        assert np.asarray(line.y)[-1] < np.asarray(line.y)[0]

    def test_the_range_reaches_zero_and_one_for_betas_inside_them(self) -> None:
        """Betas between 0 and 1 still draw the line from 0 to 1."""
        x = np.asarray(_line(_build(betas=[0.4, 0.6], mean_returns=[0.04, 0.05])).x)
        assert x.min() == 0.0
        assert x.max() == 1.0

    def test_the_range_is_extended_to_the_extreme_betas(self) -> None:
        """Betas of -0.3 and 1.6 stretch the line to exactly those values."""
        x = np.asarray(_line(_build(betas=[-0.3, 0.8, 1.6])).x)
        assert x.min() == -0.3
        assert x.max() == 1.6
        assert {0.0, 1.0} <= set(x.tolist())

    def test_the_line_is_sorted_left_to_right(self) -> None:
        """Points are in increasing beta order with no duplicates."""
        figure = _build(
            betas=[-0.3, 0.0, 1.0, 1.6], mean_returns=[0.0, 0.02, 0.07, 0.1]
        )
        x = np.asarray(_line(figure).x)
        assert (np.diff(x) > 0).all()


class TestAssets:
    """The scatter of the assets."""

    def test_the_assets_are_drawn_at_their_beta_and_return(self) -> None:
        """One marker per asset at (beta, mean return)."""
        trace = _assets(_build())
        assert trace.mode == "markers"
        assert np.asarray(trace.x).tolist() == _BETAS
        assert np.asarray(trace.y).tolist() == _RETURNS

    def test_labels_are_written_next_to_the_markers(self) -> None:
        """With labels the markers carry their names."""
        trace = _assets(_build(labels=["AAA", "BBB", "CCC"]))
        assert trace.mode == "markers+text"
        assert np.asarray(trace.text).tolist() == ["AAA", "BBB", "CCC"]

    def test_the_axes_are_titled(self) -> None:
        """Beta on x, expected return on y."""
        figure = _build()
        assert figure.layout.xaxis.title.text == "Beta"
        assert figure.layout.yaxis.title.text == "Expected return"

    def test_the_figure_has_exactly_two_traces(self) -> None:
        """The line and the assets, nothing else."""
        assert [t.name for t in _build().data] == ["Security market line", "Assets"]


class TestValidation:
    """Every rejected input names what is wrong."""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"betas": [0.5, 1.0], "mean_returns": [0.05]}, "same length"),
            ({"betas": [], "mean_returns": []}, "at least 1"),
            (
                {"betas": [[0.5, 1.0]], "mean_returns": [[0.05, 0.06]]},
                "one-dimensional",
            ),
            ({"betas": [0.5, np.nan, 1.0]}, "finite"),
            ({"mean_returns": [0.04, np.inf, 0.09]}, "finite"),
            ({"betas": ["a", "b", "c"]}, "numeric"),
            ({"risk_free_rate": np.nan}, "risk_free_rate"),
            ({"market_premium": np.inf}, "market_premium"),
            ({"risk_free_rate": "high"}, "risk_free_rate"),
            ({"labels": ["only", "two"]}, "labels"),
        ],
    )
    def test_invalid_inputs_are_rejected(
        self, overrides: dict[str, object], message: str
    ) -> None:
        """Lengths, finiteness, shape and the scalar inputs are all checked."""
        with pytest.raises(DataError, match=message):
            _build(**overrides)
