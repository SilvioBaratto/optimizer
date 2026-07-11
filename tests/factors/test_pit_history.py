"""Tests for point-in-time fundamental history slicing (issue #245)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from optimizer.factors import (
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
            as_of_date=cast(pd.Timestamp, pd.Timestamp("2025-02-01")),
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
            as_of_date=cast(pd.Timestamp, pd.Timestamp("2025-04-15")),
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
            as_of_date=cast(pd.Timestamp, pd.Timestamp("2025-03-01")),
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
            as_of_date=cast(pd.Timestamp, pd.Timestamp("2025-02-01")),
            lag_days=90,
            ticker_col="ticker",
        )
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: _slice_fundamentals_at (imported from research module)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: build_factor_scores_history warning
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: PIT correctness end-to-end (issue #273)
# ---------------------------------------------------------------------------
