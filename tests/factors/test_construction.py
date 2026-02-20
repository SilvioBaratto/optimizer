"""Tests for factor construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorConstructionConfig,
    FactorType,
    compute_all_factors,
    compute_factor,
)


@pytest.fixture()
def fundamentals() -> pd.DataFrame:
    """Synthetic fundamentals for 10 tickers."""
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(10)]
    return pd.DataFrame(
        {
            "market_cap": rng.uniform(1e9, 50e9, 10),
            "book_value": rng.uniform(1e8, 10e9, 10),
            "net_income": rng.uniform(-1e8, 5e9, 10),
            "total_equity": rng.uniform(1e8, 10e9, 10),
            "total_revenue": rng.uniform(1e9, 50e9, 10),
            "total_assets": rng.uniform(5e9, 100e9, 10),
            "gross_profit": rng.uniform(5e8, 20e9, 10),
            "operating_income": rng.uniform(-5e8, 10e9, 10),
            "operating_cashflow": rng.uniform(-1e9, 10e9, 10),
            "ebitda": rng.uniform(1e8, 10e9, 10),
            "enterprise_value": rng.uniform(2e9, 80e9, 10),
            "asset_growth": rng.uniform(-0.1, 0.3, 10),
            "dividend_yield": rng.uniform(0, 0.05, 10),
            "current_price": rng.uniform(10, 500, 10),
        },
        index=pd.Index(tickers, name="ticker"),
    )


@pytest.fixture()
def price_history() -> pd.DataFrame:
    """300 days of synthetic prices for 10 tickers."""
    rng = np.random.default_rng(42)
    n_days, n_tickers = 300, 10
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    returns = rng.normal(0.0005, 0.02, (n_days, n_tickers))
    prices = 100 * np.exp(returns.cumsum(axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture()
def volume_history(price_history: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.integers(100_000, 5_000_000, size=price_history.shape),
        index=price_history.index,
        columns=price_history.columns,
    )


class TestIndividualFactors:
    def test_book_to_price(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.BOOK_TO_PRICE, fundamentals, price_history)
        assert isinstance(result, pd.Series)
        assert len(result) == len(fundamentals)
        assert result.notna().any()

    def test_earnings_yield(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.EARNINGS_YIELD, fundamentals, price_history)
        assert len(result) == len(fundamentals)

    def test_gross_profitability(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(
            FactorType.GROSS_PROFITABILITY, fundamentals, price_history
        )
        assert len(result) == len(fundamentals)

    def test_roe(self, fundamentals: pd.DataFrame, price_history: pd.DataFrame) -> None:
        result = compute_factor(FactorType.ROE, fundamentals, price_history)
        assert len(result) == len(fundamentals)

    def test_asset_growth(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.ASSET_GROWTH, fundamentals, price_history)
        # Sign is flipped: negative growth -> positive score
        assert (result == -fundamentals["asset_growth"]).all()

    def test_momentum(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.MOMENTUM_12_1, fundamentals, price_history)
        assert isinstance(result, pd.Series)
        assert len(result) == price_history.shape[1]

    def test_volatility(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.VOLATILITY, fundamentals, price_history)
        # Sign flipped: lower vol -> higher score
        assert (result <= 0).all()

    def test_beta(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.BETA, fundamentals, price_history)
        assert isinstance(result, pd.Series)
        assert len(result) == price_history.shape[1]

    def test_amihud(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        result = compute_factor(
            FactorType.AMIHUD_ILLIQUIDITY,
            fundamentals,
            price_history,
            volume_history=volume_history,
        )
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()

    def test_dividend_yield(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.DIVIDEND_YIELD, fundamentals, price_history)
        assert len(result) == len(fundamentals)

    def test_recommendation_change_none(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(
            FactorType.RECOMMENDATION_CHANGE, fundamentals, price_history
        )
        assert len(result) == 0  # No analyst data

    def test_net_insider_buying_none(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(
            FactorType.NET_INSIDER_BUYING, fundamentals, price_history
        )
        assert len(result) == 0  # No insider data


class TestComputeAllFactors:
    def test_core_factors(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        result = compute_all_factors(
            fundamentals, price_history, volume_history
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(fundamentals)
        assert result.shape[1] > 0

    def test_all_factors(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        config = FactorConstructionConfig.for_all_factors()
        result = compute_all_factors(
            fundamentals, price_history, volume_history, config=config
        )
        assert isinstance(result, pd.DataFrame)
        # Should have columns for computable factors
        assert result.shape[1] >= 8

    def test_custom_lookbacks(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> None:
        config = FactorConstructionConfig(
            factors=(FactorType.MOMENTUM_12_1,),
            momentum_lookback=126,
            momentum_skip=10,
        )
        result = compute_all_factors(fundamentals, price_history, config=config)
        assert "momentum_12_1" in result.columns
