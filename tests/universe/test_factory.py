"""Tests for universe screening factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.universe import (
    InvestabilityScreenConfig,
    build_investability_screen,
    screen_universe,
)


class TestScreenUniverse:
    def test_returns_index(self) -> None:
        rng = np.random.default_rng(42)
        tickers = ["A", "B", "C"]
        dates = pd.bdate_range("2022-01-01", periods=300)

        fundamentals = pd.DataFrame(
            {"market_cap": [1e9, 2e9, 3e9], "current_price": [50.0, 100.0, 200.0]},
            index=pd.Index(tickers, name="ticker"),
        )
        prices = pd.DataFrame(
            100 + rng.normal(0, 1, (300, 3)).cumsum(axis=0),
            index=dates,
            columns=tickers,
        )
        volumes = pd.DataFrame(
            rng.integers(500_000, 5_000_000, (300, 3)),
            index=dates,
            columns=tickers,
        )

        result = screen_universe(fundamentals, prices, volumes)
        assert isinstance(result, pd.Index)
        assert len(result) <= len(tickers)

    def test_default_config_used(self) -> None:
        rng = np.random.default_rng(42)
        tickers = ["X", "Y"]
        dates = pd.bdate_range("2022-01-01", periods=300)

        fundamentals = pd.DataFrame(
            {"market_cap": [5e9, 5e9], "current_price": [100.0, 100.0]},
            index=pd.Index(tickers, name="ticker"),
        )
        prices = pd.DataFrame(
            100 + rng.normal(0, 1, (300, 2)).cumsum(axis=0),
            index=dates,
            columns=tickers,
        )
        volumes = pd.DataFrame(
            rng.integers(1_000_000, 10_000_000, (300, 2)),
            index=dates,
            columns=tickers,
        )

        # With default config, both should pass
        result = screen_universe(fundamentals, prices, volumes)
        assert set(result) == {"X", "Y"}

    def test_custom_config_passed(self) -> None:
        rng = np.random.default_rng(42)
        tickers = ["A"]
        dates = pd.bdate_range("2022-01-01", periods=300)

        fundamentals = pd.DataFrame(
            {"market_cap": [1e9], "current_price": [100.0]},
            index=pd.Index(tickers, name="ticker"),
        )
        prices = pd.DataFrame(
            100 + rng.normal(0, 1, (300, 1)).cumsum(axis=0),
            index=dates,
            columns=tickers,
        )
        volumes = pd.DataFrame(
            rng.integers(1_000_000, 10_000_000, (300, 1)),
            index=dates,
            columns=tickers,
        )

        config = InvestabilityScreenConfig.for_small_cap()
        result = screen_universe(fundamentals, prices, volumes, config=config)
        assert isinstance(result, pd.Index)


class TestRequiredInputsReachTheWrappers:
    """The DataError contract of the screens must hold through every entry point."""

    @staticmethod
    def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dates = pd.bdate_range("2023-01-01", periods=300)
        prices = pd.DataFrame({"A": np.full(300, 50.0)}, index=dates)
        volume = pd.DataFrame({"A": np.full(300, 1_000_000.0)}, index=dates)
        fundamentals = pd.DataFrame(
            {"market_cap": [5e9], "current_price": [50.0]},
            index=pd.Index(["A"], name="ticker"),
        )
        return fundamentals, prices, volume

    @pytest.mark.parametrize("empty", ["price_history", "volume_history"])
    def test_screen_universe_raises_on_empty_history(self, empty: str) -> None:
        fundamentals, prices, volume = self._inputs()
        frames = {"price_history": prices, "volume_history": volume}
        frames[empty] = pd.DataFrame()
        with pytest.raises(DataError, match=empty):
            screen_universe(fundamentals, **frames)

    @pytest.mark.parametrize("column", ["market_cap", "current_price"])
    def test_screen_universe_raises_on_missing_column(self, column: str) -> None:
        fundamentals, prices, volume = self._inputs()
        with pytest.raises(DataError, match=column):
            screen_universe(fundamentals.drop(columns=column), prices, volume)

    @pytest.mark.parametrize("column", ["market_cap", "current_price"])
    def test_selector_fit_raises_on_missing_column(self, column: str) -> None:
        fundamentals, prices, volume = self._inputs()
        selector = build_investability_screen(
            fundamentals.drop(columns=column), prices, volume
        )
        with pytest.raises(DataError, match=column):
            selector.fit(prices.pct_change().iloc[1:])
