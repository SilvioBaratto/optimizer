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
    compute_exchange_mcap_percentile_thresholds,
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


# ---------------------------------------------------------------------------
# Exchange percentile threshold tests
# ---------------------------------------------------------------------------

def _make_exchange_fundamentals(
    tickers: list[str],
    mcaps: list[float],
    exchanges: list[str],
) -> pd.DataFrame:
    """Build a minimal fundamentals DataFrame with exchange column."""
    return pd.DataFrame(
        {
            "market_cap": mcaps,
            "current_price": [10.0] * len(tickers),
            "exchange": exchanges,
        },
        index=pd.Index(tickers, name="ticker"),
    )


class TestComputeExchangeMcapPercentileThresholds:
    def test_returns_correct_percentile_per_exchange(self) -> None:
        # 10 stocks on NYSE with mcaps 100–1000 (step 100)
        tickers = [f"T{i}" for i in range(10)]
        mcaps = pd.Series(
            [100.0, 200.0, 300.0, 400.0, 500.0,
             600.0, 700.0, 800.0, 900.0, 1000.0],
            index=tickers,
        )
        exchanges = pd.Series(["NYSE"] * 10, index=tickers)

        thresholds = compute_exchange_mcap_percentile_thresholds(
            mcaps, exchanges, percentile=0.10
        )

        expected = float(np.percentile(mcaps.values, 10))
        assert (thresholds == expected).all()

    def test_small_exchange_defaults_to_zero(self) -> None:
        # 5 stocks — below default min_exchange_size of 10
        tickers = [f"T{i}" for i in range(5)]
        mcaps = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], index=tickers)
        exchanges = pd.Series(["AIM"] * 5, index=tickers)

        thresholds = compute_exchange_mcap_percentile_thresholds(
            mcaps, exchanges, percentile=0.10
        )

        assert (thresholds == 0.0).all()

    def test_two_exchanges_get_independent_thresholds(self) -> None:
        nyse_tickers = [f"N{i}" for i in range(10)]
        aim_tickers = [f"A{i}" for i in range(10)]
        all_tickers = nyse_tickers + aim_tickers

        nyse_mcaps = list(range(100, 1100, 100))   # 100, 200, ..., 1000
        aim_mcaps = list(range(10, 110, 10))        # 10, 20, ..., 100

        mcaps = pd.Series(nyse_mcaps + aim_mcaps, index=all_tickers, dtype=float)
        exchanges = pd.Series(
            ["NYSE"] * 10 + ["AIM"] * 10, index=all_tickers
        )

        thresholds = compute_exchange_mcap_percentile_thresholds(
            mcaps, exchanges, percentile=0.10
        )

        nyse_thresh = float(np.percentile(nyse_mcaps, 10))
        aim_thresh = float(np.percentile(aim_mcaps, 10))

        assert thresholds.loc[nyse_tickers].eq(nyse_thresh).all()
        assert thresholds.loc[aim_tickers].eq(aim_thresh).all()
        assert nyse_thresh != aim_thresh

    def test_returns_series_indexed_by_ticker(self) -> None:
        tickers = [f"T{i}" for i in range(10)]
        mcaps = pd.Series(range(100, 1100, 100), index=tickers, dtype=float)
        exchanges = pd.Series(["NYSE"] * 10, index=tickers)

        thresholds = compute_exchange_mcap_percentile_thresholds(
            mcaps, exchanges, percentile=0.10
        )

        assert isinstance(thresholds, pd.Series)
        assert set(thresholds.index) == set(tickers)

    def test_custom_min_exchange_size(self) -> None:
        # 8 stocks; with min_exchange_size=5, percentile IS computed
        tickers = [f"T{i}" for i in range(8)]
        mcaps = pd.Series(range(100, 900, 100), index=tickers, dtype=float)
        exchanges = pd.Series(["XETRA"] * 8, index=tickers)

        thresholds = compute_exchange_mcap_percentile_thresholds(
            mcaps, exchanges, percentile=0.10, min_exchange_size=5
        )

        expected = float(np.percentile(mcaps.values, 10))
        assert (thresholds == expected).all()


class TestMcapPercentileScreenInApplyInvestabilityScreens:
    """Acceptance criteria from issue #22."""

    def _make_large_exchange_fundamentals(self) -> pd.DataFrame:
        """12 NYSE stocks; one stock at 250M is ~8th percentile."""
        tickers = ["TARGET"] + [f"BIG{i}" for i in range(11)]
        # TARGET: 250M; 11 bigger stocks: 300M, 400M, ..., 1400M
        mcaps = [250e6] + [300e6 + i * 100e6 for i in range(11)]
        # 10th percentile of [250, 300, 400, ..., 1400] ≈ 310M > 250M
        prices = [10.0] * len(tickers)
        exchanges = ["NYSE"] * len(tickers)
        return pd.DataFrame(
            {
                "market_cap": mcaps,
                "current_price": prices,
                "exchange": exchanges,
            },
            index=pd.Index(tickers, name="ticker"),
        )

    def test_passes_absolute_but_below_exchange_percentile_excluded(
        self,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        """Stock above 200M floor but below 10th exchange percentile is excluded."""
        fundamentals = self._make_large_exchange_fundamentals()

        # Extend price_history / volume_history for new tickers
        n_days = len(price_history)
        extra_tickers = [
            t for t in fundamentals.index if t not in price_history.columns
        ]
        rng = np.random.default_rng(7)
        extra_prices = pd.DataFrame(
            np.abs(100 + rng.normal(0, 1, (n_days, len(extra_tickers))).cumsum(axis=0)),
            index=price_history.index,
            columns=extra_tickers,
        )
        extra_vols = pd.DataFrame(
            rng.integers(1_000_000, 5_000_000, (n_days, len(extra_tickers))),
            index=volume_history.index,
            columns=extra_tickers,
        )
        ph = pd.concat([price_history[["AAPL"]], extra_prices], axis=1)
        ph.columns = ["AAPL"] + extra_tickers
        vh = pd.concat([volume_history[["AAPL"]], extra_vols], axis=1)
        vh.columns = ["AAPL"] + extra_tickers

        # Only test fundamentals (no financial statements, simpler)
        result = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=ph,
            volume_history=vh,
        )

        # TARGET (250M) passes absolute floor (200M) but is below the
        # 10th percentile of the NYSE exchange → must be excluded
        assert "TARGET" not in result

    def test_stock_below_absolute_floor_excluded_regardless_of_percentile(
        self,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        """Stock at 150M (below 200M absolute floor) is excluded regardless."""
        # Single stock exchange: percentile threshold = 0 (< 10 stocks)
        fundamentals = pd.DataFrame(
            {
                "market_cap": [150e6],
                "current_price": [5.0],
                "exchange": ["NYSE"],
            },
            index=pd.Index(["CHEAP"], name="ticker"),
        )
        ph = price_history[["AAPL"]].rename(columns={"AAPL": "CHEAP"})
        vh = volume_history[["AAPL"]].rename(columns={"AAPL": "CHEAP"})

        result = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=ph,
            volume_history=vh,
        )

        assert "CHEAP" not in result

    def test_no_exchange_column_skips_percentile_screen(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
        financial_statements: pd.DataFrame,
    ) -> None:
        """Without exchange column the percentile screen is skipped."""
        # fundamentals fixture has no 'exchange' column
        assert "exchange" not in fundamentals.columns

        result_without = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
            financial_statements=financial_statements,
        )
        assert "AAPL" in result_without
        assert "MSFT" in result_without
        assert "GOOG" in result_without

    def test_small_exchange_passes_percentile_screen(
        self,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        """Exchange with < 10 stocks: percentile threshold = 0, all pass."""
        # 3 stocks, absolute floor = 200M (all qualify)
        tickers = ["A1", "A2", "A3"]
        fundamentals = pd.DataFrame(
            {
                "market_cap": [300e6, 400e6, 500e6],
                "current_price": [10.0, 10.0, 10.0],
                "exchange": ["AIM", "AIM", "AIM"],
            },
            index=pd.Index(tickers, name="ticker"),
        )
        n_days = len(price_history)
        rng = np.random.default_rng(11)
        ph = pd.DataFrame(
            np.abs(50 + rng.normal(0, 1, (n_days, 3)).cumsum(axis=0)),
            index=price_history.index,
            columns=tickers,
        )
        vh = pd.DataFrame(
            rng.integers(2_000_000, 5_000_000, (n_days, 3)),
            index=price_history.index,
            columns=tickers,
        )

        result = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=ph,
            volume_history=vh,
        )

        # All 3 stocks pass (small exchange → no percentile filter)
        assert set(result) == {"A1", "A2", "A3"}

    def test_exactly_at_percentile_threshold_accepted(self) -> None:
        """Stock at exactly the 10th percentile is accepted (>= boundary).

        With 11 values, np.percentile at 10% lands on index 1.0 exactly,
        giving the second-smallest value.  "EXACT" holds that value and
        must pass the screen.
        """
        rng = np.random.default_rng(42)
        n_days = 300
        dates = pd.bdate_range("2023-01-01", periods=n_days)

        # 11 stocks: "EXACT" at 250M, 10 bigger stocks at 300M–1200M
        tickers = ["EXACT"] + [f"BIG{i}" for i in range(10)]
        mcaps = [250e6] + [300e6 + i * 100e6 for i in range(10)]
        # With 11 values, 10th percentile = index 1.0 = value[1] = 300M
        # but EXACT is sorted to index 0 (smallest); we need to verify
        pct_10 = float(np.percentile(sorted(mcaps), 10))
        # pct_10 == 300M; EXACT (250M) is below it — redesign so EXACT IS at pct_10

        # Rebuild: "EXACT" at 300M (which IS the 10th percentile)
        mcaps = [300e6] + [300e6 + i * 100e6 for i in range(10)]
        pct_10 = float(np.percentile(mcaps, 10))

        fundamentals = pd.DataFrame(
            {
                "market_cap": mcaps,
                "current_price": [10.0] * 11,
                "exchange": ["NYSE"] * 11,
            },
            index=pd.Index(tickers, name="ticker"),
        )

        prices = np.abs(50 + rng.normal(0, 1, (n_days, 11)).cumsum(axis=0))
        ph = pd.DataFrame(prices, index=dates, columns=tickers)
        vols = rng.integers(2_000_000, 5_000_000, (n_days, 11))
        vh = pd.DataFrame(vols, index=dates, columns=tickers)

        result = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=ph,
            volume_history=vh,
        )

        # EXACT (300M) >= pct_10 and >= 200M abs floor → must be included
        assert mcaps[0] >= pct_10, "EXACT must be at or above 10th percentile"
        assert "EXACT" in result
