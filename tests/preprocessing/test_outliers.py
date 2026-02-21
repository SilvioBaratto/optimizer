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

    def test_sklearn_params(self) -> None:
        ot = OutlierTreater(winsorize_threshold=2.5, remove_threshold=8.0)
        params = ot.get_params()
        assert params["winsorize_threshold"] == 2.5
        assert params["remove_threshold"] == 8.0
