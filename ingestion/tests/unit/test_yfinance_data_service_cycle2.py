"""Service-level tests for cycle 2 wiring: valuation / eps_trend / eps_revisions.

Asserts that ``YFinanceDataService.fetch_and_store`` routes the three fetchers to
the right repository methods:

* ``valuation_measures`` → ``repo.upsert_financial_statements`` (date-keyed EAV
  table) with the documented ``statement_type`` / ``period_type`` /
  ``currency_code`` triple, and
* ``eps_trend`` / ``eps_revisions`` → the dedicated typed
  ``repo.upsert_eps_trend`` / ``repo.upsert_eps_revisions`` methods, untouched.

The eps panels are indexed by forward-period label ("0q"/"+1q"/"0y"/"+1y") with
metric-name columns — neither axis is a date, so the old ``pd.to_datetime`` column
coercion (which dropped every column and stored nothing) is gone.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd

from app.services.market_data.yfinance_data_service import YFinanceDataService


def _valuation_fixture() -> pd.DataFrame:
    columns = [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-03-31")]
    rows = {
        "PeRatio": [28.5, 29.1],
        "PbRatio": [45.0, 46.5],
        "MarketCap": [3_000_000_000_000.0, 3_100_000_000_000.0],
    }
    return pd.DataFrame(rows, index=columns).T


def _eps_trend_fixture() -> pd.DataFrame:
    """Real yfinance shape: index = period labels, columns = snapshot ages."""
    return pd.DataFrame(
        {
            "current": [1.50, 1.65, 6.10, 6.80],
            "7daysAgo": [1.51, 1.66, 6.11, 6.82],
            "30daysAgo": [1.52, 1.67, 6.14, 6.88],
            "60daysAgo": [1.55, 1.70, 6.20, 6.95],
            "90daysAgo": [1.58, 1.72, 6.25, 7.00],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


def _eps_revisions_fixture() -> pd.DataFrame:
    """Real yfinance shape: index = period labels, columns = revision counts."""
    return pd.DataFrame(
        {
            "upLast7days": [2, 1, 3, 0],
            "upLast30days": [5, 4, 7, 2],
            "downLast7days": [0, 1, 0, 1],
            "downLast30days": [1, 2, 1, 3],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


def _build_yf_client(
    *,
    valuation: pd.DataFrame | None = None,
    eps_trend: pd.DataFrame | None = None,
    eps_revisions: pd.DataFrame | None = None,
) -> MagicMock:
    yf_client = MagicMock(name="yf_client")
    yf_client.fetch_info.return_value = None
    yf_client.fetch_history.return_value = None
    yf_client.financials.fetch_income_stmt.return_value = None
    yf_client.financials.fetch_balance_sheet.return_value = None
    yf_client.financials.fetch_cashflow.return_value = None
    yf_client.financials.fetch_sec_filings.return_value = None
    yf_client.corporate_actions.fetch_dividends.return_value = None
    yf_client.corporate_actions.fetch_splits.return_value = None
    yf_client.analysis.fetch_recommendations_summary.return_value = None
    yf_client.analysis.fetch_analyst_price_targets.return_value = None
    yf_client.holders.fetch_institutional_holders.return_value = None
    yf_client.holders.fetch_mutualfund_holders.return_value = None
    yf_client.holders.fetch_insider_transactions.return_value = None
    yf_client.metadata.fetch_valuation_measures.return_value = valuation
    yf_client.analysis.fetch_eps_trend.return_value = eps_trend
    yf_client.analysis.fetch_eps_revisions.return_value = eps_revisions
    yf_client.get_ticker.return_value = MagicMock(news=[])
    return yf_client


def _build_repo() -> MagicMock:
    repo = MagicMock(name="repo")
    repo.get_staleness_info.return_value = None
    repo.upsert_financial_statements.return_value = 0
    repo.upsert_eps_trend.return_value = 0
    repo.upsert_eps_revisions.return_value = 0
    repo.upsert_profile.return_value = 0
    repo.upsert_price_history.return_value = 0
    repo.upsert_dividends.return_value = 0
    repo.upsert_splits.return_value = 0
    repo.upsert_recommendations.return_value = 0
    repo.upsert_price_targets.return_value = 0
    repo.upsert_institutional_holders.return_value = 0
    repo.upsert_mutualfund_holders.return_value = 0
    repo.upsert_insider_transactions.return_value = 0
    repo.upsert_news.return_value = 0
    return repo


def _run(
    yf_client: MagicMock,
    repo: MagicMock,
    *,
    currency_code: str | None = "USD",
) -> dict[str, Any]:
    service = YFinanceDataService(repo=repo, yf_client=yf_client)
    return service.fetch_and_store(
        instrument_id=uuid4(),
        yfinance_ticker="AAPL",
        period="5y",
        mode="full",
        currency_code=currency_code,
    )


def _statement_type_calls(repo: MagicMock) -> list[tuple[str, str, str | None]]:
    """Extract ``(statement_type, period_type, currency_code)`` for every upsert."""
    triples: list[tuple[str, str, str | None]] = []
    for call in repo.upsert_financial_statements.call_args_list:
        stmt_type = (
            call.args[2] if len(call.args) >= 3 else call.kwargs["statement_type"]
        )
        period_type = (
            call.args[3] if len(call.args) >= 4 else call.kwargs["period_type"]
        )
        currency = call.kwargs.get("currency_code")
        triples.append((stmt_type, period_type, currency))
    return triples


def test_when_fetchers_return_data_then_routed_to_correct_upserts() -> None:
    yf_client = _build_yf_client(
        valuation=_valuation_fixture(),
        eps_trend=_eps_trend_fixture(),
        eps_revisions=_eps_revisions_fixture(),
    )
    repo = _build_repo()

    result = _run(yf_client, repo)

    # valuation still uses the EAV financial_statements table...
    triples = _statement_type_calls(repo)
    assert ("valuation_measures", "point_in_time", "USD") in triples
    # ...but the eps panels route to their own typed tables, never the EAV one.
    assert all(t[0] != "eps_trend" for t in triples)
    assert all(t[0] != "eps_revisions" for t in triples)
    repo.upsert_eps_trend.assert_called_once()
    repo.upsert_eps_revisions.assert_called_once()
    assert "valuation_measures" in result["counts"]
    assert "eps_trend" in result["counts"]
    assert "eps_revisions" in result["counts"]


def test_when_fetchers_return_none_then_no_upsert_and_no_errors() -> None:
    yf_client = _build_yf_client(valuation=None, eps_trend=None, eps_revisions=None)
    repo = _build_repo()

    result = _run(yf_client, repo)

    triples = _statement_type_calls(repo)
    assert all(t[0] != "valuation_measures" for t in triples)
    repo.upsert_eps_trend.assert_not_called()
    repo.upsert_eps_revisions.assert_not_called()
    assert not any(
        e.startswith(("valuation_measures", "eps_trend", "eps_revisions"))
        for e in result["errors"]
    )


def test_when_eps_panels_present_then_passed_through_unchanged() -> None:
    # The real period-labelled panels must reach the typed upserts with their index
    # and columns intact — no date coercion mangling the metric-name axis.
    trend = _eps_trend_fixture()
    revisions = _eps_revisions_fixture()
    yf_client = _build_yf_client(
        valuation=None, eps_trend=trend, eps_revisions=revisions
    )
    repo = _build_repo()

    _run(yf_client, repo)

    trend_df = repo.upsert_eps_trend.call_args.args[1]
    assert list(trend_df.index) == ["0q", "+1q", "0y", "+1y"]
    assert "current" in trend_df.columns

    rev_df = repo.upsert_eps_revisions.call_args.args[1]
    assert list(rev_df.index) == ["0q", "+1q", "0y", "+1y"]
    assert "upLast7days" in rev_df.columns
