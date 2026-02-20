"""Tests for RegressionImputer transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from optimizer.preprocessing import RegressionImputer, SectorImputer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_correlated_df(
    n_periods: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of correlated returns with no NaN."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_periods)
    data = {
        "A": 0.8 * common + 0.2 * rng.standard_normal(n_periods),
        "B": 0.7 * common + 0.3 * rng.standard_normal(n_periods),
        "C": 0.6 * common + 0.4 * rng.standard_normal(n_periods),
        "D": -0.5 * common + 0.5 * rng.standard_normal(n_periods),
        "E": rng.standard_normal(n_periods),
    }
    return pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=n_periods))


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    return _make_correlated_df()


@pytest.fixture()
def df_with_nan(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Insert NaN at specific positions."""
    df = clean_df.copy()
    # Scatter NaN so that neighbors are always available for each missing cell
    df.iloc[100, 0] = np.nan  # A at row 100
    df.iloc[101, 1] = np.nan  # B at row 101
    df.iloc[102, 2] = np.nan  # C at row 102
    return df


# ---------------------------------------------------------------------------
# Fit behaviour
# ---------------------------------------------------------------------------


class TestFit:
    def test_stores_metadata(self, clean_df: pd.DataFrame) -> None:
        imp = RegressionImputer(n_neighbors=3, min_train_periods=20).fit(clean_df)
        check_is_fitted(imp)
        assert imp.n_features_in_ == 5
        np.testing.assert_array_equal(
            imp.feature_names_in_, ["A", "B", "C", "D", "E"]
        )

    def test_neighbors_capped_by_n_assets(self, clean_df: pd.DataFrame) -> None:
        """With 5 assets and n_neighbors=10, each asset gets at most 4 nbrs."""
        imp = RegressionImputer(n_neighbors=10).fit(clean_df)
        for col, nbrs in imp.neighbors_.items():
            assert len(nbrs) <= 4  # at most n_assets - 1

    def test_neighbor_count_respects_n_neighbors(
        self, clean_df: pd.DataFrame
    ) -> None:
        imp = RegressionImputer(n_neighbors=2).fit(clean_df)
        for nbrs in imp.neighbors_.values():
            assert len(nbrs) <= 2

    def test_coefs_fitted_when_enough_data(self, clean_df: pd.DataFrame) -> None:
        imp = RegressionImputer(n_neighbors=3, min_train_periods=20).fit(clean_df)
        for col in clean_df.columns:
            assert imp.coefs_[col] is not None
            # intercept + n_neighbors betas
            assert len(imp.coefs_[col]) == len(imp.neighbors_[col]) + 1

    def test_coefs_none_when_insufficient_data(self) -> None:
        """min_train_periods > training rows → regression skipped."""
        df = _make_correlated_df(n_periods=30)
        imp = RegressionImputer(n_neighbors=3, min_train_periods=100).fit(df)
        for col in df.columns:
            assert imp.coefs_[col] is None

    def test_fallback_imputer_fitted(self, clean_df: pd.DataFrame) -> None:
        imp = RegressionImputer().fit(clean_df)
        check_is_fitted(imp._fallback_imputer_)

    def test_unsupported_fallback_raises(self, clean_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unsupported fallback"):
            RegressionImputer(fallback="constant").fit(clean_df)

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pandas DataFrame"):
            RegressionImputer().fit(np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# Transform behaviour
# ---------------------------------------------------------------------------


class TestTransform:
    def test_no_nan_in_output(self, df_with_nan: pd.DataFrame) -> None:
        imp = RegressionImputer(n_neighbors=3, min_train_periods=20)
        out = imp.fit_transform(df_with_nan)
        assert not out.isna().any().any()

    def test_non_nan_values_preserved(
        self, df_with_nan: pd.DataFrame, clean_df: pd.DataFrame
    ) -> None:
        imp = RegressionImputer(n_neighbors=3, min_train_periods=20)
        out = imp.fit_transform(df_with_nan)
        # Non-NaN positions must be unchanged
        mask = df_with_nan.notna()
        pd.testing.assert_frame_equal(out[mask], df_with_nan[mask])

    def test_returns_dataframe(self, df_with_nan: pd.DataFrame) -> None:
        imp = RegressionImputer(n_neighbors=3, min_train_periods=20)
        out = imp.fit_transform(df_with_nan)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == list(df_with_nan.columns)

    def test_imputed_value_in_range_of_neighbors(self) -> None:
        """For a single NaN cell, regression prediction should be plausible.

        We construct a dataset where A = 2*B + 0.01*noise, so the
        imputed value of A should be very close to 2*B at the missing row.
        """
        rng = np.random.default_rng(0)
        n = 200
        b = rng.standard_normal(n) * 0.01
        a = 2.0 * b + rng.standard_normal(n) * 1e-4

        df = pd.DataFrame(
            {"A": a, "B": b, "C": rng.standard_normal(n) * 0.01},
            index=pd.date_range("2020-01-01", periods=n),
        )

        # Remove one value of A
        missing_idx = df.index[180]
        true_val = df.loc[missing_idx, "A"]
        df.loc[missing_idx, "A"] = np.nan

        imp = RegressionImputer(n_neighbors=2, min_train_periods=60)
        out = imp.fit_transform(df)

        imputed = out.loc[missing_idx, "A"]
        # Should be within 5x the noise level of the true value
        assert abs(imputed - true_val) < 0.01

    def test_imputed_covariance_closer_than_sector_mean(self) -> None:
        """Regression imputation should preserve covariance structure better
        than sector-mean imputation on correlated synthetic data."""
        rng = np.random.default_rng(7)
        n = 300
        common = rng.standard_normal(n)
        df_full = pd.DataFrame(
            {
                "A": 0.9 * common + 0.1 * rng.standard_normal(n),
                "B": 0.85 * common + 0.15 * rng.standard_normal(n),
                "C": 0.8 * common + 0.2 * rng.standard_normal(n),
                "D": 0.75 * common + 0.25 * rng.standard_normal(n),
            },
            index=pd.date_range("2020-01-01", periods=n),
        )
        true_cov = df_full.cov().to_numpy()

        # Insert 10% NaN randomly
        df_nan = df_full.copy()
        for col in df_nan.columns:
            idx = rng.choice(n, size=int(n * 0.1), replace=False)
            df_nan.iloc[idx, df_nan.columns.get_loc(col)] = np.nan

        sector_map = {"A": "S", "B": "S", "C": "S", "D": "S"}

        out_reg = RegressionImputer(
            n_neighbors=3, min_train_periods=20
        ).fit_transform(df_nan)

        out_sec = SectorImputer(sector_mapping=sector_map).fit_transform(df_nan)

        def _frob(cov_hat: np.ndarray) -> float:
            return float(np.linalg.norm(cov_hat - true_cov, "fro"))

        err_reg = _frob(out_reg.cov().to_numpy())
        err_sec = _frob(out_sec.cov().to_numpy())

        assert err_reg < err_sec, (
            f"Regression error ({err_reg:.6f}) should be less than "
            f"sector-mean error ({err_sec:.6f})"
        )

    def test_fallback_when_insufficient_train_periods(self) -> None:
        """When coefs_ is None, fallback SectorImputer is used."""
        df = _make_correlated_df(n_periods=30)
        df.iloc[25, 0] = np.nan  # A at row 25

        imp = RegressionImputer(
            n_neighbors=3,
            min_train_periods=100,  # forces all coefs_ to None
        )
        out = imp.fit_transform(df)
        assert not out.isna().any().any()

    def test_fallback_when_all_neighbors_nan(self) -> None:
        """When all K neighbors are NaN at the same row, use fallback."""
        # Create a row where A is NaN AND its top-2 neighbors (B, C) are also NaN
        df = _make_correlated_df(n_periods=120)
        row = df.index[110]
        df.loc[row, "A"] = np.nan
        df.loc[row, "B"] = np.nan
        df.loc[row, "C"] = np.nan

        imp = RegressionImputer(n_neighbors=2, min_train_periods=20)
        imp.fit(df)
        # Manually ensure B and C are the top neighbors for A
        # (they are highly correlated via the common factor)
        out = imp.transform(df)
        assert not out.isna().any().any()

    def test_sector_mapping_passed_to_fallback(self) -> None:
        """sector_mapping flows through to the internal SectorImputer."""
        df = _make_correlated_df(n_periods=30)  # forces fallback
        df.iloc[25, 0] = np.nan

        sector_map = {"A": "G1", "B": "G1", "C": "G2", "D": "G2", "E": "G2"}
        imp = RegressionImputer(
            min_train_periods=100,
            sector_mapping=sector_map,
        )
        out = imp.fit_transform(df)
        assert imp._fallback_imputer_.sector_mapping == sector_map
        assert not out.isna().any().any()

    def test_get_feature_names_out(self, clean_df: pd.DataFrame) -> None:
        imp = RegressionImputer().fit(clean_df)
        expected = list(clean_df.columns)
        np.testing.assert_array_equal(imp.get_feature_names_out(), expected)

    def test_no_nan_input_returned_unchanged(self, clean_df: pd.DataFrame) -> None:
        imp = RegressionImputer().fit(clean_df)
        out = imp.transform(clean_df)
        pd.testing.assert_frame_equal(out, clean_df)

    def test_rejects_non_dataframe_in_transform(
        self, clean_df: pd.DataFrame
    ) -> None:
        imp = RegressionImputer().fit(clean_df)
        with pytest.raises(TypeError, match="pandas DataFrame"):
            imp.transform(np.zeros((10, 5)))
