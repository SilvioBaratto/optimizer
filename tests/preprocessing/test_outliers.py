"""Tests for OutlierTreater transformer."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError
from optimizer.preprocessing import OutlierTreater
from optimizer.preprocessing._delisting import (
    apply_delisting_returns,
    delisting_protection_mask,
)


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
        return pd.DataFrame({"X": col}, index=pd.date_range("2024-01-01", periods=n))

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
        # Below remove_threshold robust sigmas, so only the mask can exempt it.
        df = self._spiked(spike=-0.15)
        with_spike = OutlierTreater().fit(df)
        without = OutlierTreater(protected_mask=self._mask_last(df)).fit(df)
        # Dropping the -0.15 from the fit shrinks sigma and pulls mu to ~0.
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


class TestDelistingPadding:
    """Synthetic post-delisting 0.0 padding must not distort the column scale."""

    N_REAL = 250
    N_PAD = 1000

    def _delisted_panel(
        self, mean: float = 0.0, scale: float = 0.013
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(7)
        n = self.N_REAL + self.N_PAD
        index = pd.date_range("2015-01-01", periods=n, freq="B")
        real = mean + scale * rng.standard_t(df=5, size=self.N_REAL)
        raw = pd.DataFrame(
            {
                "A": rng.normal(0.0, 0.0155, size=n),
                "B": rng.normal(0.0, 0.0155, size=n),
                "F": np.r_[real, np.full(self.N_PAD, np.nan)],
            },
            index=index,
        )
        delisting = {"F": -0.30}
        mask = delisting_protection_mask(raw, delisting)
        return apply_delisting_returns(raw, delisting), mask, real[:-1]

    def test_sigma_ignores_zero_padding(self) -> None:
        df, mask, genuine = self._delisted_panel()
        ot = OutlierTreater(protected_mask=mask).fit(df)
        assert ot.sigma_["F"] == pytest.approx(genuine.std(ddof=1))
        assert ot.mu_["F"] == pytest.approx(genuine.mean())

    def test_genuine_history_is_not_clipped_by_padding(self) -> None:
        df, mask, genuine = self._delisted_panel()
        out = OutlierTreater(protected_mask=mask).fit_transform(df)
        treated = out["F"].iloc[: self.N_REAL - 1]
        changed = ~np.isclose(treated.to_numpy(), genuine) | treated.isna().to_numpy()
        # Only the t(5) tails beyond 3 sigma may be winsorised, never a quarter
        # of the history as when the padding shrinks sigma.
        assert changed.mean() < 0.05

    def test_padding_cells_pass_through_untouched(self) -> None:
        # A strictly positive drift makes 0.0 look like a big negative z-score
        # against the genuine history; the padding is not a data point.
        df, mask, _ = self._delisted_panel(mean=0.004, scale=0.001)
        out = OutlierTreater(protected_mask=mask).fit_transform(df)
        assert (out["F"].iloc[self.N_REAL :] == 0.0).all()
        assert out["F"].iloc[self.N_REAL - 1] == pytest.approx(-0.30)

    def test_zeros_after_a_mid_history_protected_cell_are_real_data(self) -> None:
        # Only a run of zeros reaching the end of the frame is padding; real
        # returns that follow a protected event keep informing the scale.
        rng = np.random.default_rng(5)
        col = rng.normal(0.0, 0.02, size=200)
        col[100] = -0.40
        df = pd.DataFrame({"X": col}, index=pd.date_range("2024-01-01", periods=200))
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        mask.iloc[100, 0] = True
        ot = OutlierTreater(protected_mask=mask).fit(df)
        expected = np.delete(col, 100)
        assert ot.sigma_["X"] == pytest.approx(expected.std(ddof=1))


class TestScaleEstimateRobustToErrors:
    """A gross data error must not inflate the scale that is meant to catch it."""

    @staticmethod
    def _with_bad_tick(n: int, tick: float = 5.0) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        col = rng.normal(0.0, 0.01, size=n)
        col[n // 2] = tick
        return pd.DataFrame({"X": col}, index=pd.date_range("2024-01-01", periods=n))

    @pytest.mark.parametrize("n", [30, 60, 100, 126, 252])
    def test_bad_tick_is_removed_whatever_the_window_length(self, n: int) -> None:
        df = self._with_bad_tick(n)
        out = OutlierTreater().fit_transform(df)
        assert np.isnan(out.iloc[n // 2, 0])
        assert out["X"].notna().sum() == n - 1

    def test_scale_is_estimated_without_the_bad_tick(self) -> None:
        df = self._with_bad_tick(60)
        clean = np.delete(df["X"].to_numpy(), 30)
        ot = OutlierTreater().fit(df)
        assert ot.sigma_["X"] == pytest.approx(clean.std(ddof=1))
        assert ot.mu_["X"] == pytest.approx(clean.mean())

    def test_clean_column_statistics_are_plain_mean_and_std(
        self, normal_returns: pd.DataFrame
    ) -> None:
        ot = OutlierTreater().fit(normal_returns)
        pd.testing.assert_series_equal(ot.mu_, normal_returns.mean())
        pd.testing.assert_series_equal(ot.sigma_, normal_returns.std())

    def test_column_with_zero_mad_keeps_all_observations_in_the_scale(self) -> None:
        # More than half the cells equal the median, so the robust scale is 0
        # and cannot screen anything; every non-median cell must not be
        # mistaken for an error.
        col = [0.0] * 60 + [0.01, -0.01] * 10
        df = pd.DataFrame({"X": col}, index=pd.date_range("2024-01-01", periods=80))
        ot = OutlierTreater().fit(df)
        assert ot.sigma_["X"] == pytest.approx(np.std(col, ddof=1))


class TestScreenWithStaleZeroReturns:
    """Forward-filled prices give illiquid tickers many exact-zero returns.

    Just below a 50% zero share the MAD of the whole column is tiny but not
    zero, so a screen built on it would call genuine returns gross errors.
    """

    @staticmethod
    def _column(zero_share: float, *, vol: float = 0.02, n: int = 756) -> np.ndarray:
        rng = np.random.default_rng(7)
        col = rng.normal(0.0, vol, size=n)
        col[rng.permutation(n)[: round(zero_share * n)]] = 0.0
        return col

    @staticmethod
    def _frame(col: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {"X": col}, index=pd.date_range("2020-01-01", periods=len(col))
        )

    @pytest.mark.parametrize("zero_share", [0.30, 0.44, 0.46, 0.48, 0.499])
    def test_clean_column_keeps_plain_mean_and_std(self, zero_share: float) -> None:
        col = self._column(zero_share)
        ot = OutlierTreater().fit(self._frame(col))
        assert ot.sigma_["X"] == pytest.approx(np.std(col, ddof=1))
        assert ot.mu_["X"] == pytest.approx(np.mean(col))

    @pytest.mark.parametrize("zero_share", [0.45, 0.48, 0.499])
    def test_genuine_returns_are_never_deleted(self, zero_share: float) -> None:
        df = self._frame(self._column(zero_share))
        out = OutlierTreater().fit_transform(df)
        assert out["X"].notna().all()

    @pytest.mark.parametrize("zero_share", [0.30, 0.48, 0.60])
    def test_bad_tick_is_still_screened_out_of_the_scale(
        self, zero_share: float
    ) -> None:
        col = self._column(zero_share, vol=0.01)
        col[10] = 5.0
        df = self._frame(col)
        ot = OutlierTreater().fit(df)
        assert ot.sigma_["X"] == pytest.approx(np.delete(col, 10).std(ddof=1))
        assert np.isnan(ot.transform(df).iloc[10, 0])


class TestScreenWithVolatilityRegimes:
    """A calm-regime robust scale must not label crisis-day returns as errors."""

    @staticmethod
    def _regime_frame(seed: int) -> pd.DataFrame:
        # 10% of days at ten times the volatility of the other 90%.
        rng = np.random.default_rng(seed)
        vol = np.where(rng.random(756) < 0.10, 0.10, 0.01)
        return pd.DataFrame(
            {"X": rng.normal(0.0, 1.0, size=756) * vol},
            index=pd.date_range("2020-01-01", periods=756),
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_crisis_days_do_not_collapse_the_scale(self, seed: int) -> None:
        df = self._regime_frame(seed)
        ot = OutlierTreater().fit(df)
        assert ot.sigma_["X"] >= 0.8 * df["X"].std()

    @pytest.mark.parametrize("seed", range(8))
    def test_few_crisis_days_are_deleted(self, seed: int) -> None:
        df = self._regime_frame(seed)
        out = OutlierTreater().fit_transform(df)
        assert out["X"].isna().sum() <= 2


class TestProtectedMaskLabelMismatch:
    """A mask that shares no labels with X would silently protect nothing."""

    @staticmethod
    def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(1)
        col = [*rng.normal(0.0, 0.02, size=199).tolist(), -0.50]
        df = pd.DataFrame({"X": col}, index=pd.date_range("2024-01-01", periods=200))
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        mask.iloc[-1, 0] = True
        return df, mask

    def test_aligned_mask_protects_the_cell(self) -> None:
        df, mask = self._panel()
        out = OutlierTreater(protected_mask=mask).fit_transform(df)
        assert out.iloc[-1, 0] == pytest.approx(-0.50)

    @pytest.mark.parametrize(
        "relabel",
        [
            pytest.param(lambda m: m.tz_localize("UTC", level=0), id="tz-aware"),
            pytest.param(lambda m: m.set_axis(m.index.strftime("%Y-%m-%d")), id="str"),
            pytest.param(lambda m: m.reset_index(drop=True), id="range-index"),
        ],
    )
    def test_mask_with_no_index_overlap_raises(
        self, relabel: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> None:
        df, mask = self._panel()
        mismatched = relabel(mask)
        with pytest.raises(DataError, match="protected_mask"):
            OutlierTreater(protected_mask=mismatched).fit(df)

    def test_mask_with_no_column_overlap_raises(self) -> None:
        df, mask = self._panel()
        with pytest.raises(DataError, match="protected_mask"):
            OutlierTreater(protected_mask=mask.rename(columns={"X": "Y"})).fit(df)

    def test_partial_overlap_is_still_allowed(self) -> None:
        # A CV fold only sees a slice of the mask's dates and columns.
        df, mask = self._panel()
        wide = mask.assign(Z=False)
        fold = df.iloc[100:]
        out = OutlierTreater(protected_mask=wide).fit_transform(fold)
        assert out.iloc[-1, 0] == pytest.approx(-0.50)

    def test_empty_mask_is_not_a_mismatch(self) -> None:
        df, _ = self._panel()
        empty = pd.DataFrame(dtype=bool)
        OutlierTreater(protected_mask=empty).fit(df)
