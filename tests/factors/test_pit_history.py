"""Tests for point-in-time fundamental history slicing (issue #245)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    PublicationLagConfig,
    align_to_pit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_history_panel(
    tickers: list[str],
    dates: list[str],
    period_type: str = "annual",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic fundamental history panel.

    Returns a MultiIndex (period_date, ticker) DataFrame with columns:
    net_income, gross_profit, operating_income, total_assets,
    total_equity, period_type, asset_growth.
    """
    rng = np.random.default_rng(seed)
    records = []
    for ticker in tickers:
        prev_assets = None
        for date_str in dates:
            total_assets = rng.uniform(1e9, 5e9)
            growth = np.nan
            if prev_assets is not None and prev_assets != 0:
                growth = (total_assets - prev_assets) / abs(prev_assets)
            records.append({
                "period_date": pd.Timestamp(date_str),
                "ticker": ticker,
                "net_income": rng.uniform(1e7, 5e8),
                "gross_profit": rng.uniform(5e7, 1e9),
                "operating_income": rng.uniform(2e7, 4e8),
                "total_assets": total_assets,
                "total_equity": rng.uniform(5e8, 2e9),
                "period_type": period_type,
                "asset_growth": growth,
            })
            prev_assets = total_assets

    df = pd.DataFrame(records)
    df = df.set_index(["period_date", "ticker"]).sort_index()
    return df


def _build_snapshot(tickers: list[str], seed: int = 99) -> pd.DataFrame:
    """Build a static fundamentals snapshot indexed by ticker."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "market_cap": rng.uniform(1e9, 50e9, len(tickers)),
            "enterprise_value": rng.uniform(1e9, 60e9, len(tickers)),
            "net_income": rng.uniform(1e7, 5e8, len(tickers)),
            "total_assets": rng.uniform(1e9, 5e9, len(tickers)),
            "total_equity": rng.uniform(5e8, 2e9, len(tickers)),
            "book_value": rng.uniform(5e8, 2e9, len(tickers)),
            "dividend_yield": rng.uniform(0.0, 0.05, len(tickers)),
        },
        index=pd.Index(tickers, name="ticker"),
    )


# ---------------------------------------------------------------------------
# Tests: align_to_pit
# ---------------------------------------------------------------------------


class TestAlignToPitWithPanel:
    """Verify that align_to_pit correctly filters by publication lag."""

    def test_annual_lag_blocks_recent_data(self) -> None:
        """Annual data with 90-day lag should not be visible within 90 days."""
        panel = _build_history_panel(
            ["AAPL", "MSFT"],
            ["2023-12-31", "2024-12-31"],
            period_type="annual",
        )
        hist_reset = panel.reset_index()

        # Query as of 2025-02-01 — only 32 days after 2024-12-31
        # With 90-day lag, 2024-12-31 should NOT be available
        result = align_to_pit(
            hist_reset,
            period_date_col="period_date",
            as_of_date=pd.Timestamp("2025-02-01"),
            lag_days=90,
            ticker_col="ticker",
        )

        # Should get 2023-12-31 data (>90 days ago)
        assert not result.empty
        for ticker in ["AAPL", "MSFT"]:
            row = result.loc[ticker]
            assert pd.Timestamp(row["period_date"]) == pd.Timestamp("2023-12-31")

    def test_annual_lag_allows_old_data(self) -> None:
        """Data older than lag_days should be available."""
        panel = _build_history_panel(
            ["AAPL"],
            ["2023-12-31", "2024-12-31"],
            period_type="annual",
        )
        hist_reset = panel.reset_index()

        # Query as of 2025-04-15 — 105 days after 2024-12-31
        result = align_to_pit(
            hist_reset,
            period_date_col="period_date",
            as_of_date=pd.Timestamp("2025-04-15"),
            lag_days=90,
            ticker_col="ticker",
        )

        assert not result.empty
        row = result.loc["AAPL"]
        assert pd.Timestamp(row["period_date"]) == pd.Timestamp("2024-12-31")

    def test_quarterly_lag_uses_more_recent(self) -> None:
        """Quarterly lag (45 days) should pick up more recent data than annual."""
        panel = _build_history_panel(
            ["GOOG"],
            ["2024-09-30", "2024-12-31"],
            period_type="quarterly",
        )
        hist_reset = panel.reset_index()

        # Query as of 2025-03-01 — 60 days after Q4 end
        result_q = align_to_pit(
            hist_reset,
            period_date_col="period_date",
            as_of_date=pd.Timestamp("2025-03-01"),
            lag_days=45,
            ticker_col="ticker",
        )

        # Should get 2024-12-31 (60 days > 45 day lag)
        assert pd.Timestamp(result_q.loc["GOOG", "period_date"]) == pd.Timestamp(
            "2024-12-31"
        )

    def test_empty_when_no_data_before_cutoff(self) -> None:
        """Should return empty when all data is too recent."""
        panel = _build_history_panel(["AAPL"], ["2025-01-01"])
        hist_reset = panel.reset_index()

        result = align_to_pit(
            hist_reset,
            period_date_col="period_date",
            as_of_date=pd.Timestamp("2025-02-01"),
            lag_days=90,
            ticker_col="ticker",
        )
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: _slice_fundamentals_at (imported from research module)
# ---------------------------------------------------------------------------


class TestSliceFundamentalsAt:
    """Verify the PIT cross-section slicer."""

    @pytest.fixture()
    def _import_slicer(self):
        """Import the private helper from research._factors."""
        from research._factors import _slice_fundamentals_at
        return _slice_fundamentals_at

    def test_returns_snapshot_when_history_empty(self, _import_slicer) -> None:
        tickers = ["AAPL", "MSFT"]
        snapshot = _build_snapshot(tickers)
        empty_hist = pd.DataFrame()

        result = _import_slicer(
            pd.Timestamp("2025-01-01"),
            empty_hist,
            snapshot,
            PublicationLagConfig(),
        )
        assert result.equals(snapshot)

    def test_pit_slicing_uses_historical_values(self, _import_slicer) -> None:
        """Values at early date should differ from late date."""
        tickers = ["AAPL", "MSFT"]
        panel = _build_history_panel(
            tickers,
            ["2022-12-31", "2023-12-31", "2024-12-31"],
            period_type="annual",
        )
        snapshot = _build_snapshot(tickers)

        early = _import_slicer(
            pd.Timestamp("2024-04-01"),  # sees 2023-12-31
            panel,
            snapshot,
            PublicationLagConfig(annual_days=90),
        )
        late = _import_slicer(
            pd.Timestamp("2025-04-01"),  # sees 2024-12-31
            panel,
            snapshot,
            PublicationLagConfig(annual_days=90),
        )

        # net_income should differ between the two dates
        assert not np.allclose(
            early.loc["AAPL", "net_income"],
            late.loc["AAPL", "net_income"],
        )

    def test_snapshot_columns_merged(self, _import_slicer) -> None:
        """Snapshot-only columns like market_cap should be present."""
        tickers = ["AAPL"]
        panel = _build_history_panel(tickers, ["2023-12-31"])
        snapshot = _build_snapshot(tickers)

        result = _import_slicer(
            pd.Timestamp("2024-04-15"),
            panel,
            snapshot,
            PublicationLagConfig(annual_days=90),
        )
        assert "market_cap" in result.columns

    def test_quarterly_overrides_annual(self, _import_slicer) -> None:
        """When both annual and quarterly data exist, quarterly takes precedence."""
        tickers = ["AAPL"]

        # Annual data for 2023
        annual = _build_history_panel(
            tickers, ["2023-12-31"], period_type="annual", seed=10,
        )
        # Quarterly data for Q1 2024 with different values
        quarterly = _build_history_panel(
            tickers, ["2024-03-31"], period_type="quarterly", seed=20,
        )

        panel = pd.concat([annual, quarterly]).sort_index()
        snapshot = _build_snapshot(tickers)

        # As of 2024-06-01: annual 2023-12-31 (lag=90, available after 2024-03-31)
        # and quarterly 2024-03-31 (lag=45, available after 2024-05-15)
        result = _import_slicer(
            pd.Timestamp("2024-06-01"),
            panel,
            snapshot,
            PublicationLagConfig(annual_days=90, quarterly_days=45),
        )

        # net_income should match the quarterly seed=20 value, not annual seed=10
        q_ni = quarterly.loc[(pd.Timestamp("2024-03-31"), "AAPL"), "net_income"]
        assert np.isclose(result.loc["AAPL", "net_income"], q_ni)


# ---------------------------------------------------------------------------
# Tests: build_factor_scores_history warning
# ---------------------------------------------------------------------------


class TestBuildFactorScoresHistoryWarning:
    """Verify that the degraded-mode warning is emitted."""

    def test_warns_when_no_fundamental_history(self) -> None:
        """Should emit UserWarning when fundamental_history is None."""
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
        )
        # Make prices cumulative so they look like real prices
        prices = prices.cumsum()

        volumes = pd.DataFrame(
            rng.uniform(1e5, 1e7, (n_dates, n_tickers)),
            index=dates,
            columns=tickers,
        )

        fundamentals = _build_snapshot(tickers)
        sector_mapping = dict.fromkeys(tickers, "Technology")

        # Minimal DataAssembly mock
        class MockAssembly:
            analyst_data = pd.DataFrame()
            insider_data = pd.DataFrame()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_factor_scores_history(
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
            # Check warning was raised
            bias_warnings = [
                x for x in w if "look-ahead bias" in str(x.message)
            ]
            assert len(bias_warnings) >= 1

    def test_no_warning_when_history_provided(self) -> None:
        """Should NOT warn when fundamental_history is provided."""
        from optimizer.factors import FactorConstructionConfig, StandardizationConfig
        from research._factors import build_factor_scores_history

        rng = np.random.default_rng(42)
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

        fundamentals = _build_snapshot(tickers)
        sector_mapping = dict.fromkeys(tickers, "Technology")

        # Build a minimal fundamental_history
        panel = _build_history_panel(
            tickers,
            ["2022-12-31", "2023-12-31"],
            period_type="annual",
        )

        class MockAssembly:
            analyst_data = pd.DataFrame()
            insider_data = pd.DataFrame()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_factor_scores_history(
                investable_prices=prices,
                investable_volumes=volumes,
                investable_fundamentals=fundamentals,
                assembly=MockAssembly(),  # type: ignore[arg-type]
                factor_config=FactorConstructionConfig(),
                std_config=StandardizationConfig(),
                sector_mapping=sector_mapping,
                rebalance_freq=63,
                fundamental_history=panel,
            )
            bias_warnings = [
                x for x in w if "look-ahead bias" in str(x.message)
            ]
            assert len(bias_warnings) == 0
