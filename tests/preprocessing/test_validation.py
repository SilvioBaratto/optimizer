"""Tests for DataValidator transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from optimizer.preprocessing import DataValidator


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Small return DataFrame with edge cases."""
    return pd.DataFrame(
        {
            "A": [0.01, np.inf, -0.02, 0.03, 15.0],
            "B": [-np.inf, 0.02, 0.0, np.nan, 0.01],
            "C": [0.05, -0.03, 0.04, -0.01, 0.02],
        },
        index=pd.date_range("2024-01-01", periods=5),
    )


class TestDataValidator:
    def test_fit_stores_metadata(self, returns_df: pd.DataFrame) -> None:
        v = DataValidator().fit(returns_df)
        check_is_fitted(v)
        assert v.n_features_in_ == 3
        np.testing.assert_array_equal(v.feature_names_in_, ["A", "B", "C"])

    def test_inf_replaced_with_nan(self, returns_df: pd.DataFrame) -> None:
        out = DataValidator().fit_transform(returns_df)
        assert not np.isinf(out.values[~np.isnan(out.values)]).any()

    def test_extreme_returns_replaced(self, returns_df: pd.DataFrame) -> None:
        out = DataValidator(max_abs_return=10.0).fit_transform(returns_df)
        # 15.0 in column A should be NaN
        assert np.isnan(out.loc[out.index[4], "A"])

    def test_normal_values_preserved(self, returns_df: pd.DataFrame) -> None:
        out = DataValidator().fit_transform(returns_df)
        assert out.loc[out.index[0], "A"] == pytest.approx(0.01)
        assert out.loc[out.index[2], "C"] == pytest.approx(0.04)

    def test_existing_nan_preserved(self, returns_df: pd.DataFrame) -> None:
        out = DataValidator().fit_transform(returns_df)
        assert np.isnan(out.loc[out.index[3], "B"])

    def test_returns_dataframe(self, returns_df: pd.DataFrame) -> None:
        out = DataValidator().fit_transform(returns_df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["A", "B", "C"]

    def test_custom_threshold(self) -> None:
        df = pd.DataFrame({"X": [0.5, 1.5, 2.5]})
        out = DataValidator(max_abs_return=1.0).fit_transform(df)
        assert np.isnan(out.loc[1, "X"])
        assert np.isnan(out.loc[2, "X"])
        assert out.loc[0, "X"] == pytest.approx(0.5)

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pandas DataFrame"):
            DataValidator().fit(np.array([[1, 2], [3, 4]]))

    def test_get_feature_names_out(self, returns_df: pd.DataFrame) -> None:
        v = DataValidator().fit(returns_df)
        np.testing.assert_array_equal(
            v.get_feature_names_out(), ["A", "B", "C"]
        )

    def test_sklearn_get_params(self) -> None:
        v = DataValidator(max_abs_return=5.0)
        params = v.get_params()
        assert params == {"max_abs_return": 5.0}

    def test_sklearn_set_params(self) -> None:
        v = DataValidator()
        v.set_params(max_abs_return=2.0)
        assert v.max_abs_return == 2.0
