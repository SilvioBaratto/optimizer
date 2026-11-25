"""Tests for the efficient frontier figure and ``frontier_from_weights``.

The portfolio risk and return expectations are worked out by hand from the
formulas ``sqrt(w' S w)`` and ``w' mu``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from optimizer.exceptions import DataError
from optimizer.plotting import efficient_frontier_figure, frontier_from_weights

_MU = [0.10, 0.20]
_COVARIANCE = [[0.04, 0.006], [0.006, 0.09]]
_WEIGHTS = pd.DataFrame(
    [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
    index=["all_a", "half", "all_b"],
    columns=["A", "B"],
)


class TestFrontierFigure:
    """The frontier line, its markers and its highlighted portfolios."""

    def test_the_frontier_is_a_line_with_markers(self) -> None:
        """One scatter trace draws the frontier."""
        figure = efficient_frontier_figure([0.1, 0.2, 0.3], [0.04, 0.07, 0.08])
        trace = figure.data[0]
        assert isinstance(trace, go.Scatter)
        assert trace.mode == "lines+markers"
        assert trace.name == "Efficient frontier"

    def test_points_are_sorted_by_risk_and_keep_their_returns(self) -> None:
        """Unsorted input is drawn left to right with returns following their risk."""
        figure = efficient_frontier_figure([0.3, 0.1, 0.2], [0.08, 0.04, 0.07])
        assert list(figure.data[0].x) == [0.1, 0.2, 0.3]
        assert list(figure.data[0].y) == [0.04, 0.07, 0.08]

    def test_labels_follow_their_points_through_the_sort(self) -> None:
        """Hover text is reordered together with the coordinates."""
        figure = efficient_frontier_figure(
            [0.3, 0.1, 0.2], [0.08, 0.04, 0.07], labels=["high", "low", "mid"]
        )
        assert list(figure.data[0].text) == ["low", "mid", "high"]

    def test_without_labels_there_is_no_hover_text(self) -> None:
        """Labels are optional."""
        figure = efficient_frontier_figure([0.1, 0.2], [0.04, 0.07])
        assert figure.data[0].text is None

    def test_axes_carry_the_default_and_the_custom_titles(self) -> None:
        """The risk and return axes are titled, with overridable names."""
        default = efficient_frontier_figure([0.1, 0.2], [0.04, 0.07])
        assert default.layout.xaxis.title.text == "Volatility"
        assert default.layout.yaxis.title.text == "Expected return"
        custom = efficient_frontier_figure(
            [0.1, 0.2], [0.04, 0.07], risk_label="CVaR", return_label="Mean"
        )
        assert custom.layout.xaxis.title.text == "CVaR"
        assert custom.layout.yaxis.title.text == "Mean"

    def test_each_highlight_is_its_own_named_marker_trace(self) -> None:
        """The maximum Sharpe and minimum variance portfolios get their own traces."""
        figure = efficient_frontier_figure(
            [0.1, 0.2, 0.3],
            [0.04, 0.07, 0.08],
            highlight={"Max Sharpe": (0.2, 0.07), "Min variance": (0.1, 0.04)},
        )
        assert len(figure.data) == 3
        named = {trace.name: trace for trace in figure.data[1:]}
        assert set(named) == {"Max Sharpe", "Min variance"}
        assert list(named["Max Sharpe"].x) == [0.2]
        assert list(named["Max Sharpe"].y) == [0.07]
        assert named["Min variance"].mode == "markers"

    def test_arrays_pandas_and_lists_are_all_accepted(self) -> None:
        """The inputs may be any array-like of floats."""
        risks = pd.Series([0.1, 0.2, 0.3])
        returns = np.array([0.04, 0.07, 0.08])
        figure = efficient_frontier_figure(risks, returns)
        assert list(figure.data[0].x) == [0.1, 0.2, 0.3]

    @pytest.mark.parametrize(
        ("risks", "returns", "message"),
        [
            ([0.1, 0.2, 0.3], [0.04, 0.07], "same length"),
            ([0.1], [0.04], "at least 2"),
            ([0.1, np.nan], [0.04, 0.07], "finite"),
            ([0.1, 0.2], [0.04, np.inf], "finite"),
            ([[0.1, 0.2]], [[0.04, 0.07]], "one-dimensional"),
            ([0.1, -0.2], [0.04, 0.07], "non-negative"),
            (["a", "b"], [0.04, 0.07], "numeric"),
        ],
    )
    def test_invalid_points_are_rejected(
        self, risks: object, returns: object, message: str
    ) -> None:
        """Equal length, at least two points, finite, non-negative risk, 1-D."""
        with pytest.raises(DataError, match=message):
            efficient_frontier_figure(risks, returns)  # type: ignore[arg-type]

    def test_labels_of_the_wrong_length_are_rejected(self) -> None:
        """Every point needs exactly one label."""
        with pytest.raises(DataError, match="labels"):
            efficient_frontier_figure([0.1, 0.2], [0.04, 0.07], labels=["only one"])

    @pytest.mark.parametrize(
        ("highlight", "message"),
        [
            ({"x": (0.1,)}, "highlight"),
            ({"x": (0.1, 0.05, 0.2)}, "pair"),
            ({"x": (0.1, np.nan)}, "finite"),
            ({"x": (-0.1, 0.05)}, "non-negative"),
            ([("x", (0.1, 0.05))], "mapping"),
        ],
    )
    def test_a_malformed_highlight_is_rejected(
        self, highlight: object, message: str
    ) -> None:
        """A highlight is a mapping of name to a finite (risk, return) pair."""
        with pytest.raises(DataError, match=message):
            efficient_frontier_figure(
                [0.1, 0.2],
                [0.04, 0.07],
                highlight=highlight,  # type: ignore[arg-type]
            )


class TestFrontierFromWeights:
    """Risk and return of each row of a weights table."""

    def test_the_numbers_match_the_hand_calculation(self) -> None:
        """Hand calculation: 50/50 variance = .25*.04 + .25*.09 + 2*.25*.006 = .0355."""
        frame = frontier_from_weights(_WEIGHTS, _MU, _COVARIANCE)
        assert list(frame.columns) == ["risk", "expected_return"]
        assert list(frame.index) == ["all_a", "half", "all_b"]
        # A alone: sqrt(0.04); B alone: sqrt(0.09); mix: sqrt(0.0355).
        expected_risk = [math.sqrt(0.04), math.sqrt(0.0355), math.sqrt(0.09)]
        np.testing.assert_allclose(frame["risk"], expected_risk, atol=1e-15)
        # Returns: 1*.10, .5*.10 + .5*.20, 1*.20.
        np.testing.assert_allclose(frame["expected_return"], [0.10, 0.15, 0.20])

    def test_a_short_position_is_priced_with_its_sign(self) -> None:
        """Weights (1.5, -0.5): variance = 2.25*.04 + .25*.09 - 2*.75*.006 = .1035."""
        weights = pd.DataFrame([[1.5, -0.5]], columns=["A", "B"])
        frame = frontier_from_weights(weights, _MU, _COVARIANCE)
        assert frame["risk"].iloc[0] == pytest.approx(math.sqrt(0.1035), abs=1e-15)
        # Return: 1.5*.10 - .5*.20 = .05.
        assert frame["expected_return"].iloc[0] == pytest.approx(0.05, abs=1e-15)

    def test_labelled_inputs_are_aligned_by_asset_name(self) -> None:
        """Series and DataFrame inputs in a different order give the same answer."""
        mu = pd.Series({"B": 0.20, "A": 0.10, "C": 0.5})
        covariance = pd.DataFrame(
            [[0.09, 0.006, 0.0], [0.006, 0.04, 0.0], [0.0, 0.0, 0.01]],
            index=["B", "A", "C"],
            columns=["B", "A", "C"],
        )
        reference = frontier_from_weights(_WEIGHTS, _MU, _COVARIANCE)
        aligned = frontier_from_weights(_WEIGHTS, mu, covariance)
        pd.testing.assert_frame_equal(aligned, reference)

    def test_a_rounding_error_below_zero_variance_gives_zero_risk(self) -> None:
        """A singular matrix off by 2e-13 would give a variance of -4e-13."""
        covariance = [[1.0, 1.0 + 2e-13], [1.0 + 2e-13, 1.0]]
        weights = pd.DataFrame([[1.0, -1.0]], columns=["A", "B"])
        frame = frontier_from_weights(weights, _MU, covariance)
        assert frame["risk"].iloc[0] == 0.0

    def test_the_result_feeds_the_frontier_figure(self) -> None:
        """The two functions are meant to be chained."""
        frame = frontier_from_weights(_WEIGHTS, _MU, _COVARIANCE)
        figure = efficient_frontier_figure(frame["risk"], frame["expected_return"])
        assert len(figure.data[0].x) == 3

    @pytest.mark.parametrize(
        ("weights", "mu", "covariance", "message"),
        [
            ([[1.0, 0.0]], _MU, _COVARIANCE, "DataFrame"),
            (
                pd.DataFrame(columns=["A", "B"]),
                _MU,
                _COVARIANCE,
                "at least 1 portfolio",
            ),
            (
                pd.DataFrame([[np.nan, 1.0]], columns=["A", "B"]),
                _MU,
                _COVARIANCE,
                "finite",
            ),
            (
                pd.DataFrame([["x", 1.0]], columns=["A", "B"]),
                _MU,
                _COVARIANCE,
                "numeric",
            ),
            (_WEIGHTS, [0.1, 0.2, 0.3], _COVARIANCE, "mu"),
            (_WEIGHTS, [0.1, np.nan], _COVARIANCE, "finite"),
            (_WEIGHTS, pd.Series({"A": 0.1, "C": 0.2}), _COVARIANCE, "mu"),
            (_WEIGHTS, _MU, [[0.04, 0.0, 0.0]], "covariance"),
            (_WEIGHTS, _MU, [[0.04, np.nan], [np.nan, 0.09]], "finite"),
            (
                _WEIGHTS,
                _MU,
                pd.DataFrame(np.eye(2), index=["A", "C"], columns=["A", "C"]),
                "covariance",
            ),
            (_WEIGHTS, _MU, [["a", "b"], ["c", "d"]], "numeric"),
            (_WEIGHTS, _MU, [[0.04, 0.5], [0.0, 0.09]], "symmetric"),
            # Eigenvalues of [[1, 2], [2, 1]] are 3 and -1.
            (_WEIGHTS, _MU, [[1.0, 2.0], [2.0, 1.0]], "positive semi-definite"),
        ],
    )
    def test_invalid_inputs_are_rejected(
        self, weights: object, mu: object, covariance: object, message: str
    ) -> None:
        """Shapes, labels, finiteness and the covariance sign are all checked."""
        with pytest.raises(DataError, match=message):
            frontier_from_weights(weights, mu, covariance)  # type: ignore[arg-type]

    def test_duplicate_asset_labels_are_rejected(self) -> None:
        """Alignment by name is ambiguous when a name appears twice."""
        weights = pd.DataFrame([[0.5, 0.5]], columns=["A", "A"])
        with pytest.raises(DataError, match="unique"):
            frontier_from_weights(weights, _MU, _COVARIANCE)
