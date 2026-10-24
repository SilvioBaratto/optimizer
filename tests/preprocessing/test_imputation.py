"""Tests for SectorImputer transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.preprocessing import SectorImputer


@pytest.fixture()
def sector_df() -> pd.DataFrame:
    """DataFrame with known NaN positions for sector imputation tests."""
    return pd.DataFrame(
        {
            "AAPL": [0.01, np.nan, 0.03, 0.02],
            "MSFT": [0.02, 0.04, np.nan, 0.01],
            "GOOG": [0.03, 0.02, 0.01, np.nan],
            "JPM": [np.nan, 0.01, 0.02, 0.03],
            "GS": [0.02, np.nan, 0.01, 0.02],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )


@pytest.fixture()
def sector_mapping() -> dict[str, str]:
    return {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOG": "Technology",
        "JPM": "Financials",
        "GS": "Financials",
    }


class TestSectorImputer:
    def test_fit_stores_metadata(
        self, sector_df: pd.DataFrame, sector_mapping: dict[str, str]
    ) -> None:
        imp = SectorImputer(sector_mapping=sector_mapping).fit(sector_df)
        check_is_fitted(imp)
        assert imp.n_features_in_ == 5
        assert "Technology" in imp.sector_groups_
        assert "Financials" in imp.sector_groups_

    def test_no_nan_after_transform(
        self, sector_df: pd.DataFrame, sector_mapping: dict[str, str]
    ) -> None:
        out = SectorImputer(sector_mapping=sector_mapping).fit_transform(sector_df)
        assert not out.isna().any().any()

    def test_sector_average_used(
        self, sector_df: pd.DataFrame, sector_mapping: dict[str, str]
    ) -> None:
        """AAPL is NaN at row 1; MSFT=0.04, GOOG=0.02 → sector avg = 0.03."""
        out = SectorImputer(sector_mapping=sector_mapping).fit_transform(sector_df)
        # Leave-one-out: AAPL NaN at idx 1 → mean of MSFT(0.04), GOOG(0.02) = 0.03
        assert out.loc[out.index[1], "AAPL"] == pytest.approx(0.03)

    def test_financial_sector_average(
        self, sector_df: pd.DataFrame, sector_mapping: dict[str, str]
    ) -> None:
        """JPM is NaN at row 0; GS=0.02 → sector avg = 0.02."""
        out = SectorImputer(sector_mapping=sector_mapping).fit_transform(sector_df)
        assert out.loc[out.index[0], "JPM"] == pytest.approx(0.02)

    def test_no_mapping_uses_global_mean(self, sector_df: pd.DataFrame) -> None:
        out = SectorImputer(sector_mapping=None).fit_transform(sector_df)
        assert not out.isna().any().any()

    def test_non_nan_values_preserved(
        self, sector_df: pd.DataFrame, sector_mapping: dict[str, str]
    ) -> None:
        out = SectorImputer(sector_mapping=sector_mapping).fit_transform(sector_df)
        assert out.loc[out.index[0], "AAPL"] == pytest.approx(0.01)
        assert out.loc[out.index[0], "GS"] == pytest.approx(0.02)

    def test_returns_dataframe(
        self, sector_df: pd.DataFrame, sector_mapping: dict[str, str]
    ) -> None:
        out = SectorImputer(sector_mapping=sector_mapping).fit_transform(sector_df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["AAPL", "MSFT", "GOOG", "JPM", "GS"]

    def test_unmapped_columns_in_catchall(self) -> None:
        df = pd.DataFrame({"X": [1.0, np.nan], "Y": [np.nan, 2.0]})
        imp = SectorImputer(sector_mapping={"X": "A"}).fit(df)
        assert "Y" in imp.sector_groups_["__unmapped__"]

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(DataError, match="pandas DataFrame"):
            SectorImputer().fit(np.array([[1, 2]]))

    def test_get_feature_names_out(self, sector_df: pd.DataFrame) -> None:
        imp = SectorImputer().fit(sector_df)
        expected = ["AAPL", "MSFT", "GOOG", "JPM", "GS"]
        np.testing.assert_array_equal(imp.get_feature_names_out(), expected)

    def test_no_leakage_sector_imputer(
        self, sector_df: pd.DataFrame, sector_mapping: dict[str, str]
    ) -> None:
        """Fitted sector_groups_ must not change when transforming unseen data."""
        import copy

        imp = SectorImputer(sector_mapping=sector_mapping).fit(sector_df)
        groups_before = copy.deepcopy(imp.sector_groups_)

        # Transform new data
        rng = np.random.default_rng(99)
        test = pd.DataFrame(
            rng.normal(0, 0.02, (10, 5)),
            columns=sector_df.columns,
            index=pd.date_range("2025-01-01", periods=10),
        )
        test.iloc[0, 0] = np.nan
        imp.transform(test)

        assert imp.sector_groups_ == groups_before

    def test_leave_one_out_excludes_own_value(self) -> None:
        """Imputed value should be the other asset's value, not the average
        including self (leave-one-out)."""
        df = pd.DataFrame(
            {"X": [0.10, np.nan], "Y": [0.20, 0.30]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        mapping = {"X": "S1", "Y": "S1"}
        out = SectorImputer(sector_mapping=mapping).fit_transform(df)
        # X is NaN at row 1; only Y=0.30 in same sector → imputed = 0.30
        assert out.loc[out.index[1], "X"] == pytest.approx(0.30)

    def test_entire_sector_nan_falls_back_to_global(self) -> None:
        """When an entire sector is NaN for a row, use global mean."""
        df = pd.DataFrame(
            {
                "A": [np.nan, 0.02],
                "B": [np.nan, 0.04],
                "C": [0.06, 0.08],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        mapping = {"A": "S1", "B": "S1", "C": "S2"}
        out = SectorImputer(sector_mapping=mapping).fit_transform(df)
        # Row 0: A and B are NaN, entire S1 is NaN → fall back to global
        # Global mean of row 0 = 0.06 (only C has data)
        assert out.loc[out.index[0], "A"] == pytest.approx(0.06)
        assert out.loc[out.index[0], "B"] == pytest.approx(0.06)

    def test_row_missing_for_every_asset_is_left_missing(self) -> None:
        # Standalone contract: a row with no peer observation has nothing to
        # impute from, so it stays NaN rather than failing the transform.
        df = pd.DataFrame(
            {"A": [0.01, np.nan, 0.02], "B": [0.02, np.nan, np.nan]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        out = SectorImputer().fit_transform(df)

        assert out.iloc[1].isna().all()
        assert out.iloc[2, 1] == pytest.approx(0.02)

    def test_frame_without_columns_is_returned_unchanged(self) -> None:
        df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3))
        out = SectorImputer().fit_transform(df)
        assert out.shape == (3, 0)

    @pytest.mark.parametrize("strategy", ["zero", "sector_mean", ""])
    def test_unsupported_fallback_strategy_raises(
        self, sector_df: pd.DataFrame, strategy: str
    ) -> None:
        with pytest.raises(ConfigurationError, match="fallback_strategy"):
            SectorImputer(fallback_strategy=strategy).fit(sector_df)

    def test_global_mean_fallback_strategy_accepted(
        self, sector_df: pd.DataFrame
    ) -> None:
        out = SectorImputer(fallback_strategy="global_mean").fit_transform(sector_df)
        assert not out.isna().any().any()


class TestSectorImputerEdgeGaps:
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "LATE": [np.nan, np.nan, 0.03, np.nan, 0.05],
                "GONE": [0.01, 0.02, np.nan, np.nan, np.nan],
                "FULL": [0.02, 0.04, 0.06, 0.08, 0.10],
            },
            index=pd.date_range("2024-01-01", periods=5),
        )

    def test_edges_are_filled_by_default(self) -> None:
        out = SectorImputer().fit_transform(self._frame())

        assert not out.isna().any().any()

    def test_leading_and_trailing_gaps_stay_missing_when_edges_are_not_filled(
        self,
    ) -> None:
        out = SectorImputer(fill_edges=False).fit_transform(self._frame())

        assert out["LATE"].isna().tolist() == [True, True, False, False, False]
        assert out["GONE"].isna().tolist() == [False, False, True, True, True]
        assert out.loc["2024-01-04", "LATE"] == pytest.approx(0.08)

    def test_a_column_with_no_observation_stays_missing_when_edges_are_not_filled(
        self,
    ) -> None:
        df = self._frame().assign(NONE=np.nan)

        out = SectorImputer(fill_edges=False).fit_transform(df)

        assert out["NONE"].isna().all()
