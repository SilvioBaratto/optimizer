"""Tests for RawSelectComplete, the pre-selection pipeline's first step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import EqualWeighted

from optimizer.exceptions import DataError
from optimizer.pre_selection import (
    PreSelectionConfig,
    build_portfolio_pipeline,
    build_preselection_pipeline,
)
from optimizer.pre_selection._select_complete import RawSelectComplete


@pytest.fixture()
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, (120, 4)),
        columns=list("ABCD"),
        index=pd.date_range("2023-01-02", periods=120, freq="B"),
    )


class TestInputMustBeADataFrame:
    """A bare array carries no asset labels, so a sector mapping would match
    nothing and the result could not be traced back to assets.  The first
    pipeline step has to fail fast, as the validator it replaced as first step
    did."""

    def test_fit_rejects_an_ndarray(self, returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="requires a pandas DataFrame"):
            RawSelectComplete().fit(returns.to_numpy())

    def test_transform_rejects_an_ndarray(self, returns: pd.DataFrame) -> None:
        fitted = RawSelectComplete().fit(returns)

        with pytest.raises(DataError, match="requires a pandas DataFrame"):
            fitted.transform(returns.to_numpy())

    def test_preselection_pipeline_rejects_an_ndarray(
        self, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(DataError, match="requires a pandas DataFrame"):
            build_preselection_pipeline().fit_transform(returns.to_numpy())

    def test_portfolio_pipeline_rejects_an_ndarray(self, returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="requires a pandas DataFrame"):
            build_portfolio_pipeline(EqualWeighted()).fit(returns.to_numpy())

    def test_dataframe_still_selects_complete_assets(
        self, returns: pd.DataFrame
    ) -> None:
        returns.iloc[-1, 0] = np.nan

        selector = RawSelectComplete().fit(returns)

        assert selector.get_support().tolist() == [False, True, True, True]
        assert selector.transform(returns).shape == (120, 3)


class TestDropInternalNaNJudgesRawData:
    """``drop_internal_nan`` is a statement about gaps in the supplied frame.

    Completeness is judged before cleaning, so only a ``NaN`` the caller passed
    in counts as a gap.  A cell that validation or outlier treatment blanks
    afterwards is imputed whichever way the flag is set: a flagged value is a
    bad observation of a live asset, not evidence that the history is missing.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        rng = np.random.default_rng(3)
        return pd.DataFrame(
            rng.normal(0.0, 0.01, (300, 8)),
            columns=list("ABCDEFGH"),
            index=pd.date_range("2023-01-02", periods=300, freq="B"),
        )

    @staticmethod
    def _run(df: pd.DataFrame, *, drop_internal_nan: bool) -> pd.DataFrame:
        cfg = PreSelectionConfig(
            drop_internal_nan=drop_internal_nan, correlation_threshold=1.0
        )
        return build_preselection_pipeline(cfg).fit_transform(df)

    def test_caller_supplied_interior_gap_drops_the_asset_when_enabled(self) -> None:
        df = self._frame()
        df.iloc[150, 0] = np.nan

        out = self._run(df, drop_internal_nan=True)

        assert "A" not in out.columns

    def test_caller_supplied_interior_gap_is_imputed_when_disabled(self) -> None:
        df = self._frame()
        df.iloc[150, 0] = np.nan

        out = self._run(df, drop_internal_nan=False)

        assert "A" in out.columns
        assert np.isfinite(out.to_numpy()).all()

    @pytest.mark.parametrize("value", [15.0, 0.45, np.inf])
    def test_interior_cell_blanked_by_cleaning_is_imputed_when_enabled(
        self, value: float
    ) -> None:
        df = self._frame()
        df.iloc[150, 0] = value

        out = self._run(df, drop_internal_nan=True)

        assert "A" in out.columns
        assert np.isfinite(out.to_numpy()).all()
