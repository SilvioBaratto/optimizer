"""Tests for point-in-time fundamental history slicing (issue #245)."""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorConstructionConfig,
    FactorType,
    PublicationLagConfig,
    StandardizationConfig,
    align_to_pit,
)

_skip_no_research = pytest.mark.skipif(
    importlib.util.find_spec("research") is None,
    reason="research package not available in CI",
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
            records.append(
                {
                    "period_date": pd.Timestamp(date_str),
                    "ticker": ticker,
                    "net_income": rng.uniform(1e7, 5e8),
                    "gross_profit": rng.uniform(5e7, 1e9),
                    "operating_income": rng.uniform(2e7, 4e8),
                    "total_assets": total_assets,
                    "total_equity": rng.uniform(5e8, 2e9),
                    "period_type": period_type,
                    "asset_growth": growth,
                }
            )
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


@_skip_no_research
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
            tickers,
            ["2023-12-31"],
            period_type="annual",
            seed=10,
        )
        # Quarterly data for Q1 2024 with different values
        quarterly = _build_history_panel(
            tickers,
            ["2024-03-31"],
            period_type="quarterly",
            seed=20,
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


@_skip_no_research
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
            bias_warnings = [x for x in w if "look-ahead bias" in str(x.message)]
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
            bias_warnings = [x for x in w if "look-ahead bias" in str(x.message)]
            assert len(bias_warnings) == 0


# ---------------------------------------------------------------------------
# Tests: PIT correctness end-to-end (issue #273)
# ---------------------------------------------------------------------------


def _build_pit_test_data(
    n_dates: int = 500,
    n_tickers: int = 3,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
]:
    """Build synthetic data for PIT correctness tests.

    Returns (prices, volumes, snapshot_fundamentals, fundamental_history,
    sector_mapping).

    The key design:
    - snapshot_fundamentals uses book_value=5e9 (the 2022-12-31 value)
    - fundamental_history has 2021-12-31 (book_value=1e9) and 2022-12-31
      (book_value=5e9)
    - With 90-day annual lag, rebalancing dates before 2023-03-31 should see
      only 2021-12-31 data in PIT mode, but snapshot mode always uses 5e9.
    """
    rng = np.random.default_rng(42)
    tickers = [f"T{i}" for i in range(n_tickers)]
    dates = pd.bdate_range("2022-01-03", periods=n_dates)

    # Cumulative prices so they look realistic
    prices = pd.DataFrame(
        rng.uniform(0.5, 2.0, (n_dates, n_tickers)),
        index=dates,
        columns=tickers,
    ).cumsum() + 50.0

    volumes = pd.DataFrame(
        rng.uniform(1e5, 1e7, (n_dates, n_tickers)),
        index=dates,
        columns=tickers,
    )

    # Snapshot: uses the "future" book_value (2022-12-31 fiscal year)
    # Values differ across tickers so z-score standardization produces
    # different rankings than the 2021 data.
    snapshot = pd.DataFrame(
        {
            "market_cap": [10e9, 15e9, 8e9],
            "enterprise_value": [12e9, 18e9, 10e9],
            "net_income": [5e8, 3e8, 7e8],
            "gross_profit": [8e8, 6e8, 1e9],
            "operating_income": [4e8, 2e8, 5e8],
            "total_assets": [20e9, 25e9, 15e9],
            "total_equity": [8e9, 10e9, 6e9],
            "book_value": [5e9, 8e9, 3e9],
            "dividend_yield": [0.02, 0.01, 0.03],
        },
        index=pd.Index(tickers, name="ticker"),
    )

    # PIT history: two annual reporting periods with very different values
    # 2021 data has *reversed* equity ranking vs 2022 data so that
    # standardized factor scores (z-score across tickers) differ.
    records = []
    equity_2021 = [6e9, 2e9, 4e9]  # T0 highest equity in 2021
    equity_2022 = [8e9, 10e9, 6e9]  # T1 highest equity in 2022
    for i, t in enumerate(tickers):
        records.append(
            {
                "period_date": pd.Timestamp("2021-12-31"),
                "ticker": t,
                "net_income": 1e8 + i * 5e7,
                "gross_profit": 2e8 + i * 1e8,
                "operating_income": 1e8 + i * 5e7,
                "total_assets": 5e9 + i * 2e9,
                "total_equity": equity_2021[i],
                "period_type": "annual",
                "asset_growth": np.nan,
            }
        )
        records.append(
            {
                "period_date": pd.Timestamp("2022-12-31"),
                "ticker": t,
                "net_income": 5e8 + i * 1e8,
                "gross_profit": 8e8 + i * 2e8,
                "operating_income": 4e8 + i * 1e8,
                "total_assets": 20e9 + i * 5e9,
                "total_equity": equity_2022[i],
                "period_type": "annual",
                "asset_growth": (20e9 - 5e9) / 5e9,
            }
        )

    history = pd.DataFrame(records).set_index(["period_date", "ticker"]).sort_index()
    sector_mapping = dict.fromkeys(tickers, "Technology")

    return prices, volumes, snapshot, history, sector_mapping


@_skip_no_research
class TestBuildFactorScoresHistoryPITCorrectness:
    """Verify that PIT fundamentals produce different scores than snapshot.

    Uses synthetic data where 2021-12-31 and 2022-12-31 annual reports have
    very different values. With a 90-day publication lag, rebalancing dates
    before 2023-03-31 should see only the 2021-12-31 data in PIT mode, but
    snapshot mode always sees the 2022-12-31 values.
    """

    def test_pit_fundamentals_differ_from_snapshot_at_early_dates(self) -> None:
        """PIT mode should produce different scores than snapshot at early dates."""
        from research._factors import build_factor_scores_history

        prices, volumes, snapshot, history, sector_mapping = _build_pit_test_data()

        class MockAssembly:
            analyst_data = pd.DataFrame()
            insider_data = pd.DataFrame()

        # ROE = net_income / total_equity — both columns exist in the
        # history panel, so PIT slicing produces genuinely different raw
        # values (not filled from snapshot).
        factor_config = FactorConstructionConfig(
            factors=(FactorType.ROE,),
            publication_lag=PublicationLagConfig(annual_days=90),
        )
        std_config = StandardizationConfig()

        # Run 1: snapshot mode (no fundamental_history)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snap_scores, _, _ = build_factor_scores_history(
                investable_prices=prices,
                investable_volumes=volumes,
                investable_fundamentals=snapshot,
                assembly=MockAssembly(),  # type: ignore[arg-type]
                factor_config=factor_config,
                std_config=std_config,
                sector_mapping=sector_mapping,
                rebalance_freq=63,
                fundamental_history=None,
            )

        # Run 2: PIT mode (with fundamental_history)
        pit_scores, _, _ = build_factor_scores_history(
            investable_prices=prices,
            investable_volumes=volumes,
            investable_fundamentals=snapshot,
            assembly=MockAssembly(),  # type: ignore[arg-type]
            factor_config=factor_config,
            std_config=std_config,
            sector_mapping=sector_mapping,
            rebalance_freq=63,
            fundamental_history=history,
        )

        factor_name = "roe"
        assert factor_name in snap_scores
        assert factor_name in pit_scores

        snap_f = snap_scores[factor_name]
        pit_f = pit_scores[factor_name]

        # Find early dates (before 2023-03-31 — when 2022-12-31 data
        # becomes available with 90-day lag)
        cutoff = pd.Timestamp("2023-03-31")
        early_dates = [d for d in pit_f.index if d < cutoff]

        assert len(early_dates) > 0, "Expected at least one early rebalancing date"

        # At early dates, PIT (sees 2021 data) should differ from
        # snapshot (sees 2022 data)
        any_different = False
        for dt in early_dates:
            if dt in snap_f.index:
                snap_row = snap_f.loc[dt].dropna()
                pit_row = pit_f.loc[dt].dropna()
                common = snap_row.index.intersection(pit_row.index)
                if len(common) > 0 and not np.allclose(
                    snap_row[common].values,
                    pit_row[common].values,
                    atol=1e-8,
                ):
                    any_different = True
                    break

        assert any_different, (
            "PIT and snapshot scores should differ at early dates "
            "where only 2021 fundamentals are available in PIT mode"
        )

    def test_pit_mode_does_not_use_future_data(self) -> None:
        """At early dates, PIT factor scores must reflect 2021 data, not 2022."""
        from research._factors import _slice_fundamentals_at

        prices, _, snapshot, history, _ = _build_pit_test_data()

        # Identify rebalancing dates before the 2022-12-31 data becomes
        # available (i.e., before 2023-03-31 with 90-day lag)
        dates = prices.index
        rebal_indices = list(range(len(dates) - 1, 63, -63))
        rebal_indices.reverse()
        rebal_dates = [dates[i] for i in rebal_indices]
        cutoff = pd.Timestamp("2023-03-31")
        early_dates = [d for d in rebal_dates if d < cutoff]

        assert len(early_dates) > 0, "Expected at least one early rebalancing date"

        lag_config = PublicationLagConfig(annual_days=90)

        for dt in early_dates:
            sliced = _slice_fundamentals_at(
                as_of_date=dt,
                fundamental_history=history,
                snapshot_fundamentals=snapshot,
                lag_config=lag_config,
            )

            # The sliced fundamentals should reflect 2021-12-31 values,
            # NOT 2022-12-31 values.
            expected_equity = {"T0": 6e9, "T1": 2e9, "T2": 4e9}
            for ticker in ["T0", "T1", "T2"]:
                if ticker in sliced.index and "total_equity" in sliced.columns:
                    equity = sliced.loc[ticker, "total_equity"]
                    assert np.isclose(equity, expected_equity[ticker]), (
                        f"At {dt.date()}, {ticker} total_equity={equity:.0f} "
                        f"but expected {expected_equity[ticker]:.0f} "
                        f"(2021-12-31 value). "
                        f"Got future 2022-12-31 data — look-ahead bias!"
                    )
