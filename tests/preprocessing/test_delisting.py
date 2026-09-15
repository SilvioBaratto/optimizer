"""Tests for survivorship-bias guard (delisting returns)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.preprocessing._delisting import (
    apply_delisting_returns,
    delisting_protection_mask,
)


@pytest.fixture()
def sample_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.01, np.nan, np.nan],
            "MSFT": [0.01, 0.01, 0.01, 0.01, 0.01],
            "GOOG": [np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=dates,
    )


class TestApplyDelistingReturns:
    def test_replaces_last_valid(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.50})
        # AAPL's last valid is at index 2 (value 0.01 → -0.50)
        assert result["AAPL"].iloc[2] == pytest.approx(-0.50)

    def test_preserves_others(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.50})
        assert result["AAPL"].iloc[0] == pytest.approx(0.01)
        assert result["AAPL"].iloc[1] == pytest.approx(0.02)
        assert result["MSFT"].iloc[4] == pytest.approx(0.01)

    def test_raises_on_unknown_ticker(self, sample_returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="TSLA"):
            apply_delisting_returns(sample_returns, {"TSLA": -0.30})

    def test_skips_all_nan_column(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"GOOG": -1.0})
        assert result["GOOG"].isna().all()

    def test_returns_copy(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.50})
        assert result is not sample_returns
        # Original unchanged
        assert sample_returns["AAPL"].iloc[2] == pytest.approx(0.01)

    def test_crsp_default_applied_verbatim(self, sample_returns: pd.DataFrame) -> None:
        # -0.30 CRSP-style default is written as-is (no sign flip / scaling).
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.30})
        assert result["AAPL"].iloc[2] == pytest.approx(-0.30)

    def test_bankruptcy_total_loss(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"AAPL": -1.0})
        assert result["AAPL"].iloc[2] == pytest.approx(-1.0)

    def test_post_death_filled_with_zero(self, sample_returns: pd.DataFrame) -> None:
        # After the terminal return, trailing rows are held as cash (0.0), so no
        # trailing NaN remains for a downstream SelectComplete to drop.
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.30})
        assert result["AAPL"].iloc[2] == pytest.approx(-0.30)  # terminal loss
        assert result["AAPL"].iloc[3] == pytest.approx(0.0)
        assert result["AAPL"].iloc[4] == pytest.approx(0.0)
        assert not result["AAPL"].isna().any()  # loss stays in the sample

    def test_leading_nan_not_filled(self) -> None:
        # A late listing (leading NaN) keeps its leading gap so SelectComplete
        # still drops it as short-history; only the trailing gap is cash-filled.
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"LATE": [np.nan, np.nan, 0.01, 0.02, np.nan]},
            index=dates,
        )
        result = apply_delisting_returns(df, {"LATE": -0.30})
        assert result["LATE"].iloc[3] == pytest.approx(-0.30)  # terminal loss
        assert result["LATE"].iloc[4] == pytest.approx(0.0)  # post-death cash
        assert np.isnan(result["LATE"].iloc[0])  # leading gap preserved
        assert np.isnan(result["LATE"].iloc[1])

    def test_none_delisting_return_raises(self, sample_returns: pd.DataFrame) -> None:
        # Unresolved DB NULL must not be silently written as NaN.
        with pytest.raises(DataError, match="AAPL"):
            apply_delisting_returns(sample_returns, {"AAPL": None})  # type: ignore[dict-item]

    def test_nan_delisting_return_raises(self, sample_returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="AAPL"):
            apply_delisting_returns(sample_returns, {"AAPL": float("nan")})

    def test_inf_delisting_return_raises(self, sample_returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="AAPL"):
            apply_delisting_returns(sample_returns, {"AAPL": float("inf")})


class TestDelistingProtectionMask:
    def test_marks_last_valid_cell(self, sample_returns: pd.DataFrame) -> None:
        mask = delisting_protection_mask(sample_returns, {"AAPL": -0.30})
        # AAPL's last valid is idx 2; only that cell is True.
        assert bool(mask["AAPL"].iloc[2])
        assert mask["AAPL"].sum() == 1
        # No other ticker is marked.
        assert not mask["MSFT"].any()
        assert not mask["GOOG"].any()

    def test_marks_cell_that_receives_terminal_return(
        self, sample_returns: pd.DataFrame
    ) -> None:
        # The marked cell is exactly where apply_delisting_returns writes -0.30.
        mask = delisting_protection_mask(sample_returns, {"AAPL": -0.30})
        applied = apply_delisting_returns(sample_returns, {"AAPL": -0.30})
        marked = mask["AAPL"]
        assert applied["AAPL"][marked].iloc[0] == pytest.approx(-0.30)

    def test_skips_all_nan_column(self, sample_returns: pd.DataFrame) -> None:
        mask = delisting_protection_mask(sample_returns, {"GOOG": -1.0})
        assert not mask.to_numpy().any()

    def test_skips_unknown_ticker(self, sample_returns: pd.DataFrame) -> None:
        # No raise here (apply_delisting_returns is the validator); no mark.
        mask = delisting_protection_mask(sample_returns, {"TSLA": -0.30})
        assert not mask.to_numpy().any()

    def test_aligned_to_returns(self, sample_returns: pd.DataFrame) -> None:
        mask = delisting_protection_mask(sample_returns, {"AAPL": -0.30})
        assert list(mask.columns) == list(sample_returns.columns)
        assert mask.index.equals(sample_returns.index)
        assert mask.to_numpy().dtype == bool

    def test_marks_pre_fill_position_not_padding(self) -> None:
        # With a trailing gap, the mark is the pre-fill last valid row, not the
        # 0.0 post-death padding that apply_delisting_returns writes after it.
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame({"X": [0.01, 0.02, np.nan, np.nan, np.nan]}, index=dates)
        mask = delisting_protection_mask(df, {"X": -0.30})
        assert bool(mask["X"].iloc[1])  # last valid at idx 1
        assert mask["X"].sum() == 1
