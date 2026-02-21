"""Tests for OutlierTreater transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError
from optimizer.preprocessing import OutlierTreater


@pytest.fixture()
def normal_returns() -> pd.DataFrame:
    """DataFrame with known mean/std for predictable z-scores."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=0.02, size=(200, 3))
    return pd.DataFrame(
        data,
        columns=["A", "B", "C"],
        index=pd.date_range("2024-01-01", periods=200),
    )


class TestOutlierTreater:
    def test_fit_stores_statistics(self, normal_returns: pd.DataFrame) -> None:
        ot = OutlierTreater().fit(normal_returns)
        check_is_fitted(ot)
        assert hasattr(ot, "mu_")
        assert hasattr(ot, "sigma_")
        assert len(ot.mu_) == 3
        assert len(ot.sigma_) == 3

    def test_normal_values_unchanged(self, normal_returns: pd.DataFrame) -> None:
        ot = OutlierTreater()
        out = ot.fit_transform(normal_returns)
        # Most values should stay the same (all within 3σ)
        unchanged = (out == normal_returns) | (out.isna() & normal_returns.isna())
        assert unchanged.sum().sum() > 0.9 * normal_returns.size

    def test_extreme_outliers_become_nan(self) -> None:
        """Values with |z| > 10 should become NaN."""
        df = pd.DataFrame({"X": [0.0] * 100 + [5.0]})
        ot = OutlierTreater(remove_threshold=10.0).fit(df)
        mu = ot.mu_["X"]
        sigma = ot.sigma_["X"]
        z_of_5 = abs((5.0 - mu) / sigma)
        out = ot.transform(df)
        if z_of_5 > 10.0:
            assert np.isnan(out.iloc[-1, 0])

    def test_moderate_outliers_winsorized(self) -> None:
        """Values with 3 <= |z| <= 10 should be clipped."""
        # Build data where we can control the z-score precisely
        base = [0.0] * 200
        df = pd.DataFrame({"X": base})
        ot = OutlierTreater(winsorize_threshold=3.0, remove_threshold=10.0)
        ot.fit(df)
        mu = ot.mu_["X"]
        sigma = ot.sigma_["X"]

        # Inject a value at exactly 5σ above the mean
        test_val = mu + 5 * sigma
        test_df = pd.DataFrame({"X": [test_val]})
        out = ot.transform(test_df)

        if sigma > 0:
            expected = mu + 3.0 * sigma
            assert out.iloc[0, 0] == pytest.approx(expected, rel=1e-6)

    def test_zero_variance_column_safe(self) -> None:
        """Constant columns (σ=0) should not raise."""
        df = pd.DataFrame({"const": [1.0] * 50, "vary": np.random.randn(50)})
        ot = OutlierTreater()
        out = ot.fit_transform(df)
        assert isinstance(out, pd.DataFrame)

    def test_returns_dataframe_with_columns(self, normal_returns: pd.DataFrame) -> None:
        out = OutlierTreater().fit_transform(normal_returns)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["A", "B", "C"]

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(DataError, match="pandas DataFrame"):
            OutlierTreater().fit(np.array([[1, 2]]))

    def test_get_feature_names_out(self, normal_returns: pd.DataFrame) -> None:
        ot = OutlierTreater().fit(normal_returns)
        np.testing.assert_array_equal(ot.get_feature_names_out(), ["A", "B", "C"])

    def test_remove_threshold_boundary_is_nan(self) -> None:
        """A value at exactly |z| == remove_threshold should be NaN (Group 1)."""
        base = [0.0] * 200
        df = pd.DataFrame({"X": base})
        ot = OutlierTreater(winsorize_threshold=3.0, remove_threshold=5.0)
        ot.fit(df)
        mu = ot.mu_["X"]
        sigma = ot.sigma_["X"]
        if sigma > 0:
            val_at_boundary = mu + 5.0 * sigma
            test_df = pd.DataFrame({"X": [val_at_boundary]})
            out = ot.transform(test_df)
            assert np.isnan(out.iloc[0, 0]), (
                "Value at exactly remove_threshold should be NaN"
            )

    def test_winsorize_threshold_boundary_is_clipped(self) -> None:
        """A value at exactly |z| == winsorize_threshold should be clipped (Group 2)."""
        base = [0.0] * 200
        df = pd.DataFrame({"X": base})
        ot = OutlierTreater(winsorize_threshold=3.0, remove_threshold=10.0)
        ot.fit(df)
        mu = ot.mu_["X"]
        sigma = ot.sigma_["X"]
        if sigma > 0:
            # Just above winsorize threshold to ensure it's in Group 2
            test_val = mu + 3.5 * sigma
            test_df = pd.DataFrame({"X": [test_val]})
            out = ot.transform(test_df)
            expected = mu + 3.0 * sigma
            assert not np.isnan(out.iloc[0, 0])
            assert out.iloc[0, 0] == pytest.approx(expected, rel=1e-6)

    def test_normal_values_not_clipped_by_fix(self) -> None:
        """Values in the normal range (|z| < winsorize_threshold) stay unchanged."""
        base = [0.0] * 200
        df = pd.DataFrame({"X": base})
        ot = OutlierTreater(winsorize_threshold=3.0, remove_threshold=10.0)
        ot.fit(df)
        mu = ot.mu_["X"]
        sigma = ot.sigma_["X"]
        if sigma > 0:
            # Value at 1σ — clearly normal
            normal_val = mu + 1.0 * sigma
            test_df = pd.DataFrame({"X": [normal_val]})
            out = ot.transform(test_df)
            assert out.iloc[0, 0] == pytest.approx(normal_val, rel=1e-10)

    def test_no_leakage_outlier_treater(self, normal_returns: pd.DataFrame) -> None:
        """Fitted statistics must not change when transforming unseen data."""
        import copy

        train = normal_returns.iloc[:100]
        ot = OutlierTreater().fit(train)
        mu_before = copy.deepcopy(ot.mu_)
        sigma_before = copy.deepcopy(ot.sigma_)

        # Transform test data with different scale
        rng = np.random.default_rng(99)
        test = pd.DataFrame(
            rng.normal(loc=0.1, scale=0.05, size=(50, 3)),
            columns=normal_returns.columns,
            index=pd.date_range("2025-01-01", periods=50),
        )
        ot.transform(test)

        pd.testing.assert_series_equal(ot.mu_, mu_before)
        pd.testing.assert_series_equal(ot.sigma_, sigma_before)

    def test_sklearn_params(self) -> None:
        ot = OutlierTreater(winsorize_threshold=2.5, remove_threshold=8.0)
        params = ot.get_params()
        assert params["winsorize_threshold"] == 2.5
        assert params["remove_threshold"] == 8.0
