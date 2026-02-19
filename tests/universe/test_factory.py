"""Tests for universe screening factory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer.universe import InvestabilityScreenConfig, screen_universe


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
