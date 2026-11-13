"""Mock repository and client for tests that drive the yfinance sweep.

``make_repo`` answers every upsert with a row count, and ``make_silent_yf_client``
answers every request with nothing, so a test configures only the categories it
is about.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def make_repo(*, staleness: dict[str, Any] | None = None) -> MagicMock:
    """Return a mock ``YFinanceRepository`` whose writes store nothing but news.

    Args:
        staleness: What ``get_staleness_info`` answers.

    Returns:
        A mock whose upserts return 0, except ``upsert_news``, which returns 3.
    """
    repo = MagicMock(name="repo")
    repo.get_staleness_info.return_value = staleness
    repo.upsert_profile.return_value = 0
    repo.upsert_price_history.return_value = 0
    repo.upsert_financial_statements.return_value = 0
    repo.upsert_dividends.return_value = 0
    repo.upsert_splits.return_value = 0
    repo.upsert_shares_outstanding.return_value = 0
    repo.upsert_recommendations.return_value = 0
    repo.upsert_price_targets.return_value = 0
    repo.upsert_institutional_holders.return_value = 0
    repo.upsert_mutualfund_holders.return_value = 0
    repo.upsert_insider_transactions.return_value = 0
    repo.upsert_major_holders.return_value = 0
    repo.upsert_insider_purchases.return_value = 0
    repo.upsert_insider_roster.return_value = 0
    repo.upsert_profile_extras.return_value = 0
    repo.upsert_news.return_value = 3
    return repo


def make_silent_yf_client() -> MagicMock:
    """Return a mock ``YFinanceClient`` that answers every request with nothing.

    Returns:
        A mock whose requests return ``None`` and whose ticker has no news.
    """
    yf_client = MagicMock(name="yf_client")
    yf_client.fetch_info.return_value = None
    yf_client.fetch_history.return_value = None
    yf_client.financials.fetch_income_stmt.return_value = None
    yf_client.financials.fetch_balance_sheet.return_value = None
    yf_client.financials.fetch_cashflow.return_value = None
    yf_client.corporate_actions.fetch_dividends.return_value = None
    yf_client.corporate_actions.fetch_splits.return_value = None
    yf_client.corporate_actions.fetch_shares_full.return_value = None
    yf_client.analysis.fetch_recommendations_summary.return_value = None
    yf_client.analysis.fetch_upgrades_downgrades.return_value = None
    yf_client.financials.fetch_sec_filings.return_value = None
    yf_client.analysis.fetch_analyst_price_targets.return_value = None
    yf_client.analysis.fetch_eps_trend.return_value = None
    yf_client.analysis.fetch_eps_revisions.return_value = None
    yf_client.holders.fetch_institutional_holders.return_value = None
    yf_client.holders.fetch_mutualfund_holders.return_value = None
    yf_client.holders.fetch_insider_transactions.return_value = None
    yf_client.holders.fetch_major_holders.return_value = None
    yf_client.holders.fetch_insider_purchases.return_value = None
    yf_client.holders.fetch_insider_roster_holders.return_value = None
    yf_client.metadata.fetch_valuation_measures.return_value = None
    yf_client.metadata.fetch_isin.return_value = None
    yf_client.analysis.fetch_earnings_estimate.return_value = None
    yf_client.analysis.fetch_revenue_estimate.return_value = None
    yf_client.analysis.fetch_growth_estimates.return_value = None
    yf_client.analysis.fetch_earnings_history.return_value = None
    yf_client.metadata.fetch_earnings_dates.return_value = None
    ticker_mock = MagicMock()
    ticker_mock.news = []
    yf_client.get_ticker.return_value = ticker_mock
    return yf_client
