"""Tests for the serialisable prices->returns wrapper (to_returns)."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from skfolio.preprocessing import prices_to_returns

from optimizer.exceptions import DataError
from optimizer.preprocessing import JoinMethod, ReturnsConfig, to_returns


@pytest.fixture()
def prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    return pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 99.0, 102.0, 104.0, 103.0],
            "MSFT": [50.0, 50.5, 51.0, 50.0, 49.5, 50.0],
        },
        index=idx,
    )


class TestReturnsConfig:
    def test_defaults_are_linear(self) -> None:
        cfg = ReturnsConfig()
        assert cfg.log_returns is False
        assert cfg.join is JoinMethod.OUTER

    def test_is_frozen(self) -> None:
        cfg = ReturnsConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.log_returns = True  # type: ignore[misc]

    def test_hashable(self) -> None:
        assert hash(ReturnsConfig()) == hash(ReturnsConfig())


class TestToReturns:
    def test_matches_skfolio_default(self, prices: pd.DataFrame) -> None:
        out = to_returns(prices)
        expected = prices_to_returns(prices)
        pd.testing.assert_frame_equal(out, expected)

    def test_linear_returns_values(self, prices: pd.DataFrame) -> None:
        out = to_returns(prices)
        # First AAPL return: 101/100 - 1 = 0.01
        assert out["AAPL"].iloc[0] == pytest.approx(0.01)

    def test_log_returns_option(self, prices: pd.DataFrame) -> None:
        cfg = ReturnsConfig(log_returns=True)
        out = to_returns(prices, config=cfg)
        assert out["AAPL"].iloc[0] == pytest.approx(np.log(101.0 / 100.0))

    def test_join_value_forwarded(self, prices: pd.DataFrame) -> None:
        cfg = ReturnsConfig(join=JoinMethod.INNER)
        y_prices = prices[["MSFT"]].rename(columns={"MSFT": "BENCH"})
        x_ret, y_ret = to_returns(prices, y_prices, config=cfg)
        assert list(x_ret.columns) == ["AAPL", "MSFT"]
        assert list(y_ret.columns) == ["BENCH"]

    def test_returns_tuple_with_benchmark(self, prices: pd.DataFrame) -> None:
        y_prices = prices[["AAPL"]].rename(columns={"AAPL": "SPY"})
        result = to_returns(prices, y_prices)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_rejects_non_dataframe_prices(self) -> None:
        with pytest.raises(TypeError, match="prices"):
            to_returns(np.zeros((5, 2)))  # type: ignore[arg-type]

    def test_rejects_non_dataframe_benchmark(self, prices: pd.DataFrame) -> None:
        with pytest.raises(TypeError, match="y_prices"):
            to_returns(prices, np.zeros((6, 1)))  # type: ignore[arg-type]


class TestDecimalCoercion:
    """DB price_history is Numeric(20, 6) -> Decimal (object dtype)."""

    @pytest.fixture()
    def decimal_prices(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=6, freq="B")
        return pd.DataFrame(
            {
                "AAPL": [Decimal(str(v)) for v in (100, 101, 99, 102, 104, 103)],
                "MSFT": [Decimal(str(v)) for v in (50, 50.5, 51, 50, 49.5, 50)],
            },
            index=idx,
        )

    def test_decimal_prices_yield_float_returns(
        self, decimal_prices: pd.DataFrame
    ) -> None:
        assert (decimal_prices.dtypes == "object").all()  # sanity: DB-shaped input
        out = to_returns(decimal_prices)
        # skfolio silently emits object-dtype Decimal without the boundary cast.
        assert (out.dtypes == np.float64).all()

    def test_decimal_returns_match_float(
        self, decimal_prices: pd.DataFrame, prices: pd.DataFrame
    ) -> None:
        out = to_returns(decimal_prices)
        expected = to_returns(prices.astype(float))
        pd.testing.assert_frame_equal(out, expected)

    def test_decimal_benchmark_coerced(self, decimal_prices: pd.DataFrame) -> None:
        y = decimal_prices[["MSFT"]].rename(columns={"MSFT": "BENCH"})
        x_ret, y_ret = to_returns(decimal_prices, y)
        assert (x_ret.dtypes == np.float64).all()
        assert (y_ret.dtypes == np.float64).all()

    def test_non_numeric_object_column_raises(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        bad = pd.DataFrame({"AAPL": ["a", "b", "c"]}, index=idx)
        with pytest.raises(DataError, match="prices"):
            to_returns(bad)

    def test_float_prices_unaffected(self, prices: pd.DataFrame) -> None:
        # Already-float frames are a no-op: identical to raw skfolio output.
        pd.testing.assert_frame_equal(to_returns(prices), prices_to_returns(prices))
