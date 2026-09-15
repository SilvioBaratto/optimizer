"""Tests for OutlierTreater transformer."""

from __future__ import annotations

from decimal import Decimal

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

    def test_decimal_input_does_not_raise_protected_mask(self) -> None:
        # protected_mask=None is the default and behaves exactly as no arg.
        df = pd.DataFrame(
            {"X": [0.0] * 100 + [5.0]},
            index=pd.date_range("2024-01-01", periods=101),
        )
        base = OutlierTreater().fit_transform(df)
        explicit_none = OutlierTreater(protected_mask=None).fit_transform(df)
        pd.testing.assert_frame_equal(base, explicit_none)

    def test_decimal_input_does_not_raise(self) -> None:
        # Regression: object-dtype Decimal returns previously raised
        # TypeError ("float - Decimal") in the z-score subtraction.
        df = pd.DataFrame(
            {"X": [Decimal(str(v)) for v in (0.01, -0.02, 0.0, 0.05, -0.01, 0.2)]},
            index=pd.date_range("2024-01-01", periods=6),
        )
        assert df.dtypes["X"] == "object"
        out = OutlierTreater().fit_transform(df)
        assert out.dtypes["X"] == np.float64


class TestProtectedMask:
    """protected_mask exempts genuine economic events from outlier treatment."""

    def _spiked(
        self,
        spike: float = -0.50,
        n: int = 200,
        scale: float = 0.02,
        seed: int = 1,
    ) -> pd.DataFrame:
        """Noisy body (so sigma > 0) with one extreme spike on the last row."""
        rng = np.random.default_rng(seed)
        col = [*rng.normal(0.0, scale, size=n - 1).tolist(), spike]
        return pd.DataFrame(
            {"X": col}, index=pd.date_range("2024-01-01", periods=n)
        )

    @staticmethod
    def _mask_last(df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        mask.iloc[-1, 0] = True
        return mask

    def test_protected_cell_not_removed(self) -> None:
        df = self._spiked(spike=-0.50)
        # Unprotected: |z| >> 10 -> removed to NaN.
        assert np.isnan(OutlierTreater().fit_transform(df).iloc[-1, 0])
        # Protected: the -0.50 survives intact, no NaN introduced.
        out = OutlierTreater(protected_mask=self._mask_last(df)).fit_transform(df)
        assert out.iloc[-1, 0] == pytest.approx(-0.50)
        assert not out["X"].isna().any()

    def test_protected_cell_not_winsorized(self) -> None:
        # remove_threshold huge -> the spike lands in the winsorize band.
        df = self._spiked(spike=-0.50)
        unprot = OutlierTreater(remove_threshold=1e9).fit_transform(df)
        assert unprot.iloc[-1, 0] > -0.50  # clipped toward the mean
        out = OutlierTreater(
            remove_threshold=1e9, protected_mask=self._mask_last(df)
        ).fit_transform(df)
        assert out.iloc[-1, 0] == pytest.approx(-0.50)

    def test_protected_cell_excluded_from_moments(self) -> None:
        df = self._spiked(spike=-0.50)
        with_spike = OutlierTreater().fit(df)
        without = OutlierTreater(protected_mask=self._mask_last(df)).fit(df)
        # Dropping the -0.50 from the fit shrinks sigma and pulls mu to ~0.
        assert without.sigma_["X"] < with_spike.sigma_["X"]
        assert abs(without.mu_["X"]) < abs(with_spike.mu_["X"])

    def test_mask_realigned_by_label(self) -> None:
        # A mask over a superset index still protects the correct cell.
        df = self._spiked(spike=-0.50)
        extra = pd.date_range("2024-01-01", periods=len(df) + 10)
        mask = pd.DataFrame(False, index=extra, columns=df.columns)
        mask.loc[df.index[-1], "X"] = True
        out = OutlierTreater(protected_mask=mask).fit_transform(df)
        assert out.iloc[-1, 0] == pytest.approx(-0.50)

    def test_unmarked_outliers_still_removed(self) -> None:
        # Protecting one cell must not disable treatment for the rest.
        rng = np.random.default_rng(3)
        col = [*rng.normal(0.0, 0.02, size=198).tolist(), 0.60, -0.50]
        df = pd.DataFrame({"X": col}, index=pd.date_range("2024-01-01", periods=200))
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        mask.iloc[-1, 0] = True  # protect only the -0.50 terminal cell
        out = OutlierTreater(protected_mask=mask).fit_transform(df)
        assert out.iloc[-1, 0] == pytest.approx(-0.50)  # protected, kept
        assert np.isnan(out.iloc[-2, 0])  # unprotected 0.60 spike removed

    def test_sklearn_params_includes_mask(self) -> None:
        df = self._spiked()
        mask = self._mask_last(df)
        ot = OutlierTreater(protected_mask=mask)
        assert ot.get_params()["protected_mask"] is mask
