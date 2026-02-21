"""Tests for survivorship-bias guard (delisting returns)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.preprocessing._delisting import apply_delisting_returns


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
