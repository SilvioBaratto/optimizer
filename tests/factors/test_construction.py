"""Tests for factor construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorConstructionConfig,
    FactorType,
    align_to_pit,
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


# ---------------------------------------------------------------------------
# Point-in-time alignment tests
# ---------------------------------------------------------------------------


def _make_annual_data(
    tickers: list[str],
    period_dates: list[str],
    values: list[float],
) -> pd.DataFrame:
    """Build a minimal time-series fundamentals DataFrame."""
    rows = []
    for ticker, date, val in zip(tickers, period_dates, values):
        rows.append({"ticker": ticker, "period_date": date, "earnings": val})
    return pd.DataFrame(rows)


class TestAlignToPit:
    """Acceptance criteria: correct lag applied per source type."""

    def test_annual_data_unavailable_before_90_days(self) -> None:
        """Annual Dec 31 data is NOT available before March 31 (90 days)."""
        data = _make_annual_data(
            tickers=["AAPL"],
            period_dates=["2023-12-31"],
            values=[5.0],
        )
        # 89 days after Dec 31 → not yet available
        as_of_date = pd.Timestamp("2023-12-31") + pd.Timedelta(days=89)
        result = align_to_pit(data, "period_date", as_of_date, lag_days=90)
        assert result.empty

    def test_annual_data_available_from_march_31(self) -> None:
        """Annual Dec 31 data IS available on March 31 (exactly 90 days)."""
        data = _make_annual_data(
            tickers=["AAPL"],
            period_dates=["2023-12-31"],
            values=[5.0],
        )
        # Exactly 90 days after Dec 31 = March 31
        as_of_date = pd.Timestamp("2023-12-31") + pd.Timedelta(days=90)
        result = align_to_pit(data, "period_date", as_of_date, lag_days=90)
        assert len(result) == 1
        assert result.loc["AAPL", "earnings"] == 5.0

    def test_analyst_data_unavailable_before_5_days(self) -> None:
        """Analyst revision on Monday is NOT available before Friday (5 days)."""
        monday = pd.Timestamp("2024-01-08")  # a Monday
        data = _make_annual_data(
            tickers=["GOOG"],
            period_dates=[str(monday.date())],
            values=[3.0],
        )
        # 4 days later (Friday) → not yet available
        as_of_friday = monday + pd.Timedelta(days=4)
        result = align_to_pit(data, "period_date", as_of_friday, lag_days=5)
        assert result.empty

    def test_analyst_data_available_after_5_days(self) -> None:
        """Analyst revision on Monday IS available 5 days later."""
        monday = pd.Timestamp("2024-01-08")
        data = _make_annual_data(
            tickers=["GOOG"],
            period_dates=[str(monday.date())],
            values=[3.0],
        )
        as_of = monday + pd.Timedelta(days=5)
        result = align_to_pit(data, "period_date", as_of, lag_days=5)
        assert len(result) == 1
        assert result.loc["GOOG", "earnings"] == 3.0

    def test_most_recent_record_returned_per_ticker(self) -> None:
        """When multiple records exist, the most recent available is returned."""
        data = pd.DataFrame([
            {"ticker": "AAPL", "period_date": "2023-03-31", "earnings": 1.0},
            {"ticker": "AAPL", "period_date": "2023-06-30", "earnings": 2.0},
            {"ticker": "AAPL", "period_date": "2023-09-30", "earnings": 3.0},
        ])
        # as_of = 2023-11-15, lag = 45 days → cutoff = 2023-10-01
        # Sep 30 + 45 = Nov 14 ≤ Nov 15: available
        as_of = pd.Timestamp("2023-11-15")
        result = align_to_pit(data, "period_date", as_of, lag_days=45)
        assert len(result) == 1
        assert result.loc["AAPL", "earnings"] == 3.0  # most recent

    def test_multiple_tickers_independent(self) -> None:
        """Each ticker gets its own most-recent available record."""
        data = pd.DataFrame([
            {"ticker": "AAPL", "period_date": "2023-06-30", "earnings": 2.0},
            {"ticker": "AAPL", "period_date": "2023-09-30", "earnings": 3.0},
            {"ticker": "MSFT", "period_date": "2023-06-30", "earnings": 4.0},
            # MSFT Q3 not yet available (released too recently)
            {"ticker": "MSFT", "period_date": "2023-10-15", "earnings": 5.0},
        ])
        # as_of = Nov 15; lag = 45; cutoff = Oct 1
        # AAPL Sep 30 available; MSFT Oct 15 not (Oct 15 + 45 > Nov 15)
        as_of = pd.Timestamp("2023-11-15")
        result = align_to_pit(data, "period_date", as_of, lag_days=45)
        assert result.loc["AAPL", "earnings"] == 3.0
        assert result.loc["MSFT", "earnings"] == 4.0  # falls back to Q2

    def test_empty_result_when_all_too_recent(self) -> None:
        """Returns empty DataFrame when no records pass the cutoff."""
        data = _make_annual_data(
            tickers=["AAPL"],
            period_dates=["2023-12-31"],
            values=[5.0],
        )
        # Only 10 days after Dec 31 → not available with 90-day lag
        as_of = pd.Timestamp("2024-01-10")
        result = align_to_pit(data, "period_date", as_of, lag_days=90)
        assert result.empty

    def test_accepts_string_as_of_date(self) -> None:
        """align_to_pit accepts a string for as_of_date."""
        data = _make_annual_data(
            tickers=["AAPL"],
            period_dates=["2023-12-31"],
            values=[5.0],
        )
        result = align_to_pit(
            data, "period_date", "2024-04-01", lag_days=90
        )
        assert len(result) == 1
