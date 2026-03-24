"""Tests for forward-return boundary semantics in build_factor_scores_history().

Verifies that the forward-return window ``(returns.index > dt) &
(returns.index <= next_dt)`` correctly excludes the rebalancing-day return
(no look-ahead bias) and includes the last holding-period day.

See: GitHub issue #295.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from skfolio.preprocessing import prices_to_returns

_skip_no_research = pytest.mark.skipif(
    importlib.util.find_spec("research") is None,
    reason="research package not available in CI",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fundamentals(
    tickers: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a fundamentals snapshot with all required columns."""
    n = len(tickers)
    return pd.DataFrame(
        {
            "market_cap": rng.uniform(1e9, 50e9, n),
            "enterprise_value": rng.uniform(1e9, 60e9, n),
            "net_income": rng.uniform(1e7, 5e8, n),
            "total_assets": rng.uniform(1e9, 5e9, n),
            "total_equity": rng.uniform(5e8, 2e9, n),
            "book_value": rng.uniform(5e8, 2e9, n),
            "dividend_yield": rng.uniform(0.0, 0.05, n),
        },
        index=pd.Index(tickers, name="ticker"),
    )


def _make_prices(
    n_dates: int = 20,
    n_tickers: int = 3,
    start: str = "2023-01-02",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a simple cumulative price DataFrame on business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    raw = rng.uniform(0.5, 2.0, (n_dates, n_tickers))
    return pd.DataFrame(raw.cumsum(axis=0) + 10, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# 1. prices_to_returns index alignment
# ---------------------------------------------------------------------------


class TestPricesToReturnsIndexAlignment:
    """Verify that prices_to_returns produces the expected index."""

    def test_drops_first_row(self) -> None:
        prices = _make_prices(n_dates=10)
        returns = prices_to_returns(prices)

        assert len(returns) == len(prices) - 1
        assert prices.index[0] not in returns.index

    def test_all_subsequent_dates_present(self) -> None:
        prices = _make_prices(n_dates=10)
        returns = prices_to_returns(prices)

        for dt in prices.index[1:]:
            assert dt in returns.index

    def test_return_value_matches_pct_change(self) -> None:
        """Return at date T = (price[T] - price[T-1]) / price[T-1]."""
        prices = _make_prices(n_dates=5, n_tickers=1)
        returns = prices_to_returns(prices)

        for i in range(1, len(prices)):
            dt = prices.index[i]
            expected = (prices.iloc[i, 0] - prices.iloc[i - 1, 0]) / prices.iloc[
                i - 1, 0
            ]
            assert np.isclose(returns.loc[dt].iloc[0], expected), (
                f"Return at {dt.date()} does not match pct_change"
            )

    def test_rebal_date_present_in_returns(self) -> None:
        """Every interior price date has a return entry."""
        prices = _make_prices(n_dates=20)
        returns = prices_to_returns(prices)

        # Simulate rebal dates from the 5th date onwards
        for idx in range(5, len(prices) - 1):
            dt = prices.index[idx]
            assert dt in returns.index


# ---------------------------------------------------------------------------
# 2. Forward-return boundary exclusion / inclusion
# ---------------------------------------------------------------------------


class TestForwardReturnBoundary:
    """Verify the mask ``(returns.index > dt) & (returns.index <= next_dt)``."""

    @pytest.fixture()
    def returns(self) -> pd.DataFrame:
        return prices_to_returns(_make_prices(n_dates=30))

    def test_day0_excluded(self, returns: pd.DataFrame) -> None:
        """The return at ``dt`` must NOT appear in the forward window."""
        dt = returns.index[5]
        next_dt = returns.index[10]

        mask = (returns.index > dt) & (returns.index <= next_dt)
        window = returns.loc[mask]

        assert dt not in window.index

    def test_next_dt_included(self, returns: pd.DataFrame) -> None:
        """The return at ``next_dt`` must appear in the forward window."""
        dt = returns.index[5]
        next_dt = returns.index[10]

        mask = (returns.index > dt) & (returns.index <= next_dt)
        window = returns.loc[mask]

        assert next_dt in window.index

    def test_all_intermediate_dates_included(
        self,
        returns: pd.DataFrame,
    ) -> None:
        dt = returns.index[5]
        next_dt = returns.index[10]

        mask = (returns.index > dt) & (returns.index <= next_dt)
        window = returns.loc[mask]

        expected_dates = returns.index[6:11]  # indices 6,7,8,9,10
        pd.testing.assert_index_equal(window.index, expected_dates)

    def test_consecutive_dates_yield_single_return(
        self,
        returns: pd.DataFrame,
    ) -> None:
        """When dt and next_dt are adjacent, window has exactly one row."""
        dt = returns.index[5]
        next_dt = returns.index[6]

        mask = (returns.index > dt) & (returns.index <= next_dt)
        window = returns.loc[mask]

        assert len(window) == 1
        assert window.index[0] == next_dt

    def test_window_mean_excludes_day0_value(self) -> None:
        """Mean forward return must NOT include the day-0 return value."""
        dates = pd.bdate_range("2023-01-02", periods=6)
        # Construct returns where day-0 has a large outlier value
        data = {
            "A": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
        returns = pd.DataFrame(data, index=dates[1:])

        dt = dates[1]  # return = 0.01
        next_dt = dates[3]  # return = 0.03

        mask = (returns.index > dt) & (returns.index <= next_dt)
        fwd = returns.loc[mask, "A"].mean()

        # Window should include dates[2] (0.02) and dates[3] (0.03)
        expected = (0.02 + 0.03) / 2
        assert np.isclose(fwd, expected)


# ---------------------------------------------------------------------------
# 3. Integration test with build_factor_scores_history
# ---------------------------------------------------------------------------


@_skip_no_research
class TestForwardReturnIntegration:
    """End-to-end boundary verification via build_factor_scores_history."""

    def test_assertion_does_not_fire(self) -> None:
        """The runtime assertion in _factors.py must not trigger."""
        import warnings

        from optimizer.factors import FactorConstructionConfig, StandardizationConfig
        from research._factors import build_factor_scores_history

        rng = np.random.default_rng(42)
        n_dates, n_tickers = 300, 5
        dates = pd.bdate_range("2023-01-01", periods=n_dates)
        tickers = [f"T{i}" for i in range(n_tickers)]

        prices = pd.DataFrame(
            rng.uniform(10, 100, (n_dates, n_tickers)),
            index=dates,
            columns=tickers,
        ).cumsum()

        volumes = pd.DataFrame(
            rng.uniform(1e5, 1e7, (n_dates, n_tickers)),
            index=dates,
            columns=tickers,
        )

        fundamentals = _make_fundamentals(tickers, rng)
        sector_mapping = dict.fromkeys(tickers, "Technology")

        class MockAssembly:
            analyst_data = pd.DataFrame()
            insider_data = pd.DataFrame()

        # Should complete without AssertionError
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = build_factor_scores_history(
                investable_prices=prices,
                investable_volumes=volumes,
                investable_fundamentals=fundamentals,
                assembly=MockAssembly(),  # type: ignore[arg-type]
                factor_config=FactorConstructionConfig(),
                std_config=StandardizationConfig(),
                sector_mapping=sector_mapping,
                rebalance_freq=63,
                fundamental_history=None,
            )

        # Verify we got results
        _, returns_history, health = result
        assert health.succeeded_dates > 0
        assert not returns_history.empty

    def test_returns_history_dates_exclude_rebal_day_returns(self) -> None:
        """For each rebal date in returns_history, verify the values
        correspond to post-rebal returns, not the rebal-day return."""
        import warnings

        from optimizer.factors import FactorConstructionConfig, StandardizationConfig
        from research._factors import build_factor_scores_history

        rng = np.random.default_rng(99)
        n_dates, n_tickers = 300, 3
        dates = pd.bdate_range("2023-01-01", periods=n_dates)
        tickers = [f"T{i}" for i in range(n_tickers)]

        prices = pd.DataFrame(
            rng.uniform(10, 100, (n_dates, n_tickers)),
            index=dates,
            columns=tickers,
        ).cumsum()

        volumes = pd.DataFrame(
            rng.uniform(1e5, 1e7, (n_dates, n_tickers)),
            index=dates,
            columns=tickers,
        )

        fundamentals = _make_fundamentals(tickers, rng)
        sector_mapping = dict.fromkeys(tickers, "Technology")

        class MockAssembly:
            analyst_data = pd.DataFrame()
            insider_data = pd.DataFrame()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, returns_history, _ = build_factor_scores_history(
                investable_prices=prices,
                investable_volumes=volumes,
                investable_fundamentals=fundamentals,
                assembly=MockAssembly(),  # type: ignore[arg-type]
                factor_config=FactorConstructionConfig(),
                std_config=StandardizationConfig(),
                sector_mapping=sector_mapping,
                rebalance_freq=63,
                fundamental_history=None,
            )

        # Manually verify: for each rebal date in returns_history,
        # the mean return should match the window (dt, next_dt]
        all_returns = prices_to_returns(prices)
        rebal_dates = list(returns_history.index)

        for i, dt in enumerate(rebal_dates[:-1]):
            next_dt = rebal_dates[i + 1]
            mask = (all_returns.index > dt) & (all_returns.index <= next_dt)
            expected_mean = all_returns.loc[mask, tickers].mean()

            actual = returns_history.loc[dt, tickers]
            pd.testing.assert_series_equal(
                actual,
                expected_mean,
                check_names=False,
            )
