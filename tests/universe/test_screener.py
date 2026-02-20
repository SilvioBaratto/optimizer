"""Tests for universe screening logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.universe import (
    ExchangeRegion,
    HysteresisConfig,
    InvestabilityScreenConfig,
    apply_investability_screens,
    apply_screen,
    compute_addv,
    compute_listing_age,
    compute_trading_frequency,
    count_financial_statements,
)


@pytest.fixture()
def price_history() -> pd.DataFrame:
    """300 trading days of synthetic prices for 5 tickers."""
    rng = np.random.default_rng(42)
    n_days, n_tickers = 300, 5
    tickers = ["AAPL", "MSFT", "GOOG", "TINY", "NEW"]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    prices = 100.0 + rng.normal(0, 1, (n_days, n_tickers)).cumsum(axis=0)
    prices = np.abs(prices)  # keep positive
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    # NEW ticker: only 50 days of data (recent IPO)
    df.loc[df.index[:-50], "NEW"] = np.nan
    return df


@pytest.fixture()
def volume_history(price_history: pd.DataFrame) -> pd.DataFrame:
    """Synthetic volume aligned with price_history."""
    rng = np.random.default_rng(99)
    vol = rng.integers(100_000, 5_000_000, size=price_history.shape)
    df = pd.DataFrame(
        vol, index=price_history.index, columns=price_history.columns
    )
    # TINY has zero volume on 20% of days
    mask = rng.random(len(df)) < 0.20
    df.loc[mask, "TINY"] = 0
    # NEW aligns with price NaN
    df.loc[price_history["NEW"].isna(), "NEW"] = 0
    return df


@pytest.fixture()
def fundamentals() -> pd.DataFrame:
    """Cross-sectional fundamentals for 5 tickers."""
    return pd.DataFrame(
        {
            "market_cap": [2e9, 1.5e9, 3e9, 50e6, 500e6],
            "current_price": [150.0, 300.0, 100.0, 0.5, 25.0],
        },
        index=pd.Index(["AAPL", "MSFT", "GOOG", "TINY", "NEW"], name="ticker"),
    )


@pytest.fixture()
def financial_statements() -> pd.DataFrame:
    """Financial statement records for testing."""
    records = []
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        for i in range(4):
            records.append({
                "ticker": ticker,
                "period_type": "annual",
                "period_date": f"202{i}-12-31",
            })
        for i in range(10):
            records.append({
                "ticker": ticker,
                "period_type": "quarterly",
                "period_date": f"2023-{(i % 4 + 1) * 3:02d}-30",
            })
    # TINY has only 1 annual report
    records.append({
        "ticker": "TINY",
        "period_type": "annual",
        "period_date": "2023-12-31",
    })
    # NEW has no statements
    return pd.DataFrame(records)


class TestApplyScreen:
    def test_no_current_members(self) -> None:
        values = pd.Series([100, 200, 300], index=["A", "B", "C"])
        result = apply_screen(values, HysteresisConfig(entry=150, exit_=100))
        assert set(result) == {"B", "C"}

    def test_hysteresis_retains_existing(self) -> None:
        values = pd.Series([120, 200, 300], index=["A", "B", "C"])
        current = pd.Index(["A"])  # A is existing member
        result = apply_screen(
            values,
            HysteresisConfig(entry=150, exit_=100),
            current_members=current,
        )
        # A (120) is above exit (100), so retained; B, C above entry
        assert set(result) == {"A", "B", "C"}

    def test_hysteresis_removes_below_exit(self) -> None:
        values = pd.Series([80, 200, 300], index=["A", "B", "C"])
        current = pd.Index(["A"])
        result = apply_screen(
            values,
            HysteresisConfig(entry=150, exit_=100),
            current_members=current,
        )
        # A (80) is below exit (100), so removed
        assert set(result) == {"B", "C"}

    def test_empty_values(self) -> None:
        values = pd.Series([], dtype=float)
        result = apply_screen(values, HysteresisConfig(entry=100, exit_=50))
        assert len(result) == 0

    def test_equal_entry_exit(self) -> None:
        values = pd.Series([100, 50, 200], index=["A", "B", "C"])
        result = apply_screen(values, HysteresisConfig(entry=100, exit_=100))
        assert set(result) == {"A", "C"}


class TestComputeAddv:
    def test_basic(
        self, price_history: pd.DataFrame, volume_history: pd.DataFrame
    ) -> None:
        addv = compute_addv(price_history, volume_history, window=252)
        assert isinstance(addv, pd.Series)
        assert len(addv) == price_history.shape[1]
        assert (addv >= 0).all()

    def test_short_window(
        self, price_history: pd.DataFrame, volume_history: pd.DataFrame
    ) -> None:
        addv_short = compute_addv(price_history, volume_history, window=20)
        addv_long = compute_addv(price_history, volume_history, window=252)
        # Both should produce values for all tickers
        assert len(addv_short) == len(addv_long)


class TestComputeTradingFrequency:
    def test_basic(self, volume_history: pd.DataFrame) -> None:
        freq = compute_trading_frequency(volume_history, window=252)
        assert isinstance(freq, pd.Series)
        assert (freq >= 0).all()
        assert (freq <= 1).all()

    def test_tiny_lower_frequency(self, volume_history: pd.DataFrame) -> None:
        freq = compute_trading_frequency(volume_history, window=252)
        # TINY has 20% zero-volume days
        assert freq["TINY"] < freq["AAPL"]


class TestComputeListingAge:
    def test_basic(self, price_history: pd.DataFrame) -> None:
        age = compute_listing_age(price_history)
        assert age["AAPL"] == 300  # full history
        assert age["NEW"] == 50  # only 50 non-NaN days


class TestCountFinancialStatements:
    def test_annual_count(self, financial_statements: pd.DataFrame) -> None:
        counts = count_financial_statements(financial_statements, period_type="annual")
        assert counts["AAPL"] == 4
        assert counts["TINY"] == 1

    def test_quarterly_count(self, financial_statements: pd.DataFrame) -> None:
        counts = count_financial_statements(
            financial_statements, period_type="quarterly"
        )
        assert counts["AAPL"] == 10
        assert "TINY" not in counts  # no quarterly reports


class TestApplyInvestabilityScreens:
    def test_screens_filter_correctly(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
        financial_statements: pd.DataFrame,
    ) -> None:
        result = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
            financial_statements=financial_statements,
        )
        # TINY fails market cap and price; NEW fails listing age
        assert "TINY" not in result
        assert "NEW" not in result
        # AAPL, MSFT, GOOG should pass all screens
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOG" in result

    def test_broad_universe_more_inclusive(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        strict = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
        )
        broad = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
            config=InvestabilityScreenConfig.for_broad_universe(),
        )
        assert len(broad) >= len(strict)

    def test_no_financial_statements(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        # Without financial statements, that screen is skipped
        result = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
            financial_statements=None,
        )
        assert isinstance(result, pd.Index)

    def test_europe_region(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        config = InvestabilityScreenConfig(exchange_region=ExchangeRegion.EUROPE)
        result = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
            config=config,
        )
        assert isinstance(result, pd.Index)

    def test_hysteresis_retains_members(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        # Run once without current members
        first_pass = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
        )
        # Run again with current members — should be stable
        second_pass = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
            current_members=first_pass,
        )
        assert set(first_pass) == set(second_pass)
