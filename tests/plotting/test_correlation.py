"""Tests for the correlation heatmap.

Every expected matrix is computed in the test with numpy or scipy, never with the
``DataFrame.corr`` call the builder itself relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy import stats

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.plotting import correlation_heatmap


def _returns(seed: int = 7, rows: int = 80, columns: int = 4) -> pd.DataFrame:
    """Return reproducible, mutually dependent returns with named columns."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(rows, columns))
    if columns >= 4:
        base[:, 1] += 0.6 * base[:, 0]
        base[:, 3] -= 0.4 * base[:, 2]
    return pd.DataFrame(base, columns=[f"A{i}" for i in range(columns)])


def _z(figure: go.Figure) -> np.ndarray:
    """Return the matrix drawn by the single heatmap trace."""
    return np.asarray(figure.data[0].z, dtype=float)


def _kendall_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return the Kendall tau matrix, pair by pair, with scipy."""
    values = frame.to_numpy()
    size = values.shape[1]
    matrix = np.ones((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            tau = stats.kendalltau(values[:, i], values[:, j]).statistic
            matrix[i, j] = matrix[j, i] = tau
    return matrix


class TestMatrix:
    """The z matrix is the correlation matrix of the returns."""

    def test_the_figure_holds_one_heatmap_trace(self) -> None:
        """The chart is a single heatmap."""
        figure = correlation_heatmap(_returns())
        assert len(figure.data) == 1
        assert isinstance(figure.data[0], go.Heatmap)

    def test_pearson_matches_numpy(self) -> None:
        """Pearson is the default and equals ``numpy.corrcoef``."""
        frame = _returns()
        expected = np.corrcoef(frame.to_numpy(), rowvar=False)
        np.testing.assert_allclose(_z(correlation_heatmap(frame)), expected, atol=1e-12)

    def test_spearman_matches_the_correlation_of_ranks(self) -> None:
        """Spearman is Pearson on the ranks of every column."""
        frame = _returns()
        ranks = np.column_stack(
            [stats.rankdata(frame[column]) for column in frame.columns]
        )
        expected = np.corrcoef(ranks, rowvar=False)
        figure = correlation_heatmap(frame, method="spearman")
        np.testing.assert_allclose(_z(figure), expected, atol=1e-12)

    def test_kendall_matches_scipy(self) -> None:
        """Kendall tau is taken pair by pair with scipy."""
        frame = _returns(rows=40)
        figure = correlation_heatmap(frame, method="kendall")
        np.testing.assert_allclose(_z(figure), _kendall_matrix(frame), atol=1e-12)

    def test_perfectly_dependent_columns_give_exact_plus_and_minus_one(self) -> None:
        """B = 2A is correlation 1 and C = -A is correlation -1, by definition."""
        base = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
        frame = pd.DataFrame({"A": base, "B": 2.0 * base, "C": -base})
        expected = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
        np.testing.assert_allclose(_z(correlation_heatmap(frame)), expected, atol=1e-12)

    def test_missing_values_are_dropped_pairwise(self) -> None:
        """A gap in one column does not blank the whole matrix."""
        frame = _returns()
        frame.loc[3, "A0"] = np.nan
        keep = frame["A0"].notna().to_numpy()
        pair = np.corrcoef(frame.loc[keep, "A0"], frame.loc[keep, "A1"])[0, 1]
        assert _z(correlation_heatmap(frame))[0, 1] == pytest.approx(pair, abs=1e-12)


class TestAxesAndColours:
    """Labels, colour scale and annotations."""

    def test_axis_labels_are_the_column_names(self) -> None:
        """Both axes carry the asset names, in column order."""
        frame = _returns()
        trace = correlation_heatmap(frame).data[0]
        assert list(trace.x) == list(frame.columns)
        assert list(trace.y) == list(frame.columns)

    def test_numeric_looking_names_are_kept_as_labels(self) -> None:
        """Integer column names are drawn on a category axis, not as numbers."""
        frame = _returns().rename(columns=lambda name: 100 + int(name[1:]))
        figure = correlation_heatmap(frame)
        assert list(figure.data[0].x) == ["100", "101", "102", "103"]
        assert figure.layout.xaxis.type == "category"
        assert figure.layout.yaxis.type == "category"

    def test_the_rows_read_top_down_like_the_matrix(self) -> None:
        """The first asset is on the top row so the diagonal runs downwards."""
        assert correlation_heatmap(_returns()).layout.yaxis.autorange == "reversed"

    def test_the_colour_scale_is_diverging_and_centred_on_zero(self) -> None:
        """The scale spans -1 to 1 and its midpoint is no correlation."""
        trace = correlation_heatmap(_returns()).data[0]
        assert trace.zmin == -1.0
        assert trace.zmax == 1.0
        assert trace.zmid == 0.0

    def test_values_are_annotated_to_two_decimals(self) -> None:
        """The cell labels are the values rounded to two decimals."""
        frame = pd.DataFrame(
            [[1.0, 0.123456], [0.123456, 1.0]], columns=["X", "Y"], index=["X", "Y"]
        )
        trace = correlation_heatmap(frame, is_correlation=True).data[0]
        assert [list(row) for row in trace.text] == [["1.00", "0.12"], ["0.12", "1.00"]]
        assert trace.texttemplate == "%{text}"

    def test_a_negative_value_keeps_its_sign_and_a_tiny_one_is_zero(self) -> None:
        """-0.4567 reads -0.46 and -0.001 reads 0.00, never -0.00."""
        frame = pd.DataFrame(
            [[1.0, -0.4567, -0.001], [-0.4567, 1.0, 0.2], [-0.001, 0.2, 1.0]],
            columns=["X", "Y", "Z"],
            index=["X", "Y", "Z"],
        )
        text = correlation_heatmap(frame, is_correlation=True).data[0].text
        assert list(text[0]) == ["1.00", "-0.46", "0.00"]

    def test_annotations_can_be_switched_off(self) -> None:
        """Without annotations the cells carry no text."""
        trace = correlation_heatmap(_returns(), annotate=False).data[0]
        assert trace.text is None
        assert trace.texttemplate is None

    def test_the_title_names_the_method(self) -> None:
        """A computed heatmap says which coefficient it shows."""
        figure = correlation_heatmap(_returns(), method="spearman")
        assert "spearman" in figure.layout.title.text.lower()


class TestPrecomputedMatrix:
    """A correlation matrix passed with ``is_correlation=True``."""

    def test_the_matrix_is_drawn_as_given(self) -> None:
        """No recomputation: the z matrix is the input, and so are the labels."""
        matrix = _returns().corr()
        figure = correlation_heatmap(matrix, is_correlation=True)
        np.testing.assert_allclose(_z(figure), matrix.to_numpy(), atol=1e-15)
        assert list(figure.data[0].x) == list(matrix.columns)

    def test_rounding_noise_on_the_diagonal_is_tolerated(self) -> None:
        """A matrix off by float noise from 1.0 is still a correlation matrix."""
        matrix = pd.DataFrame([[1.0 + 1e-12, 0.3], [0.3, 1.0 - 1e-12]])
        figure = correlation_heatmap(matrix, is_correlation=True)
        assert _z(figure).shape == (2, 2)

    @pytest.mark.parametrize(
        ("matrix", "message"),
        [
            (pd.DataFrame(np.ones((2, 3))), "square"),
            (pd.DataFrame([[1.0, 0.2], [0.4, 1.0]]), "symmetric"),
            (pd.DataFrame([[0.9, 0.2], [0.2, 1.0]]), "diagonal"),
            (pd.DataFrame([[1.0, 1.5], [1.5, 1.0]]), "between -1 and 1"),
            (pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
            (pd.DataFrame([[1.0]]), "at least two"),
        ],
    )
    def test_an_invalid_matrix_is_rejected(
        self, matrix: pd.DataFrame, message: str
    ) -> None:
        """Square, symmetric, unit diagonal, bounded, finite and at least 2x2."""
        with pytest.raises(DataError, match=message):
            correlation_heatmap(matrix, is_correlation=True)


class TestClusterOrder:
    """``order="cluster"`` puts similar assets next to each other."""

    @staticmethod
    def _frame(columns: list[str], duplicates: dict[str, str]) -> pd.DataFrame:
        """Return independent noise columns, with some columns copied from others."""
        rng = np.random.default_rng(11)
        noise = {name: rng.normal(size=60) for name in columns}
        for copy, source in duplicates.items():
            noise[copy] = noise[source].copy()
        return pd.DataFrame(noise, columns=columns)

    def test_a_perfectly_correlated_pair_becomes_adjacent(self) -> None:
        """A and C are identical but start two columns apart."""
        frame = self._frame(["A", "B", "C", "D"], {"C": "A"})
        labels = list(correlation_heatmap(frame, order="cluster").data[0].x)
        assert abs(labels.index("A") - labels.index("C")) == 1

    def test_two_correlated_pairs_both_become_adjacent(self) -> None:
        """A-C and B-D are each a copy; the interleaved input is untangled."""
        frame = self._frame(["A", "B", "C", "D"], {"C": "A", "D": "B"})
        labels = list(correlation_heatmap(frame, order="cluster").data[0].x)
        assert abs(labels.index("A") - labels.index("C")) == 1
        assert abs(labels.index("B") - labels.index("D")) == 1

    def test_the_original_order_is_kept_by_default(self) -> None:
        """Without clustering the columns stay where the caller put them."""
        frame = self._frame(["A", "B", "C", "D"], {"C": "A"})
        assert list(correlation_heatmap(frame).data[0].x) == ["A", "B", "C", "D"]

    def test_the_matrix_is_permuted_consistently(self) -> None:
        """Rows, columns and values all follow the same new order."""
        frame = self._frame(["A", "B", "C", "D"], {"C": "A"})
        trace = correlation_heatmap(frame, order="cluster").data[0]
        labels = list(trace.x)
        assert list(trace.y) == labels
        expected = np.corrcoef(frame[labels].to_numpy(), rowvar=False)
        np.testing.assert_allclose(np.asarray(trace.z), expected, atol=1e-12)

    def test_two_assets_are_clustered_without_error(self) -> None:
        """The smallest matrix has a single merge and keeps both labels."""
        frame = self._frame(["A", "B"], {})
        labels = list(correlation_heatmap(frame, order="cluster").data[0].x)
        assert sorted(labels) == ["A", "B"]

    def test_a_given_correlation_matrix_can_be_clustered(self) -> None:
        """Clustering also works on a matrix passed as is."""
        frame = self._frame(["A", "B", "C", "D"], {"C": "A"})
        figure = correlation_heatmap(frame.corr(), is_correlation=True, order="cluster")
        labels = list(figure.data[0].x)
        assert abs(labels.index("A") - labels.index("C")) == 1


class TestValidation:
    """Every rejected input names what is wrong."""

    def test_an_unknown_method_is_rejected(self) -> None:
        """Only pearson, spearman and kendall exist."""
        with pytest.raises(ConfigurationError, match="method"):
            correlation_heatmap(_returns(), method="cosine")

    def test_an_unknown_order_is_rejected(self) -> None:
        """Only none and cluster exist."""
        with pytest.raises(ConfigurationError, match="order"):
            correlation_heatmap(_returns(), order="alphabetical")

    def test_a_non_frame_is_rejected(self) -> None:
        """A bare array has no asset names to draw."""
        with pytest.raises(DataError, match="DataFrame"):
            correlation_heatmap(np.ones((5, 3)))  # type: ignore[arg-type]

    def test_a_single_asset_is_rejected(self) -> None:
        """A correlation matrix needs at least two assets."""
        with pytest.raises(DataError, match="at least two"):
            correlation_heatmap(_returns(columns=1))

    def test_a_single_observation_is_rejected(self) -> None:
        """A correlation needs at least two observations."""
        with pytest.raises(DataError, match="observations"):
            correlation_heatmap(_returns(rows=1))

    def test_a_non_numeric_column_is_rejected(self) -> None:
        """Text columns cannot be correlated."""
        frame = _returns()
        frame["name"] = "x"
        with pytest.raises(DataError, match="numeric"):
            correlation_heatmap(frame)

    def test_a_constant_column_is_rejected(self) -> None:
        """A column without variance has an undefined correlation."""
        frame = _returns()
        frame["A2"] = 0.01
        with pytest.raises(DataError, match="undefined"):
            correlation_heatmap(frame)
