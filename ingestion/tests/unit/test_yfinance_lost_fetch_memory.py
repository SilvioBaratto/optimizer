"""A request Yahoo never answered must not be remembered as "Yahoo has no data".

``EmptyFetchMemory`` settles a category for ``no_data_hours`` after a clean empty
answer. The yfinance client returns ``None`` both for that answer and for a
request that was lost (5xx, 401/403, DNS), and only a rate limit opens the
circuit breaker, so the service has to consult the request-outcome tally to tell
them apart.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.services.market_data.empty_fetch_memory import EmptyFetchMemory
from app.services.market_data.yfinance._base import fetch_with_resilience
from app.services.market_data.yfinance.infrastructure import (
    CircuitBreaker,
    YahooTransientError,
)
from app.services.market_data.yfinance_data_service import YFinanceDataService

_STAMPED = (
    "profile",
    "income_statement_quarterly",
    "balance_sheet_annual",
    "balance_sheet_quarterly",
    "cashflow_annual",
    "cashflow_quarterly",
    "valuation_measures",
    "eps_trend",
    "eps_revisions",
    "earnings_estimate",
    "revenue_estimate",
    "growth_estimates",
    "earnings_history",
    "earnings_dates",
    "sec_filings",
    "shares_outstanding",
    "dividends",
    "splits",
    "recommendations",
    "price_targets",
    "institutional_holders",
    "mutualfund_holders",
    "insider_transactions",
)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the backoff pause between attempts of a failing request."""
    monkeypatch.setattr("app.services.infrastructure.retry.time.sleep", lambda s: None)


def _staleness(now: datetime) -> dict[str, Any]:
    """Everything stored just now except the annual income statement."""
    staleness: dict[str, Any] = {f"{name}_updated_at": now for name in _STAMPED}
    staleness["price_max_date"] = date.today()
    return staleness


def _client(breaker: CircuitBreaker, *, income_error: Exception | None) -> MagicMock:
    """A client whose annual income statement either gets lost or comes back empty."""
    client = MagicMock(name="yf_client")
    client.circuit_breaker = breaker
    client.fetch_history.return_value = None
    client.fetch_info.return_value = None

    def _income(symbol: str, quarterly: bool = False) -> Any:
        def _action() -> Any:
            if income_error is not None:
                raise income_error
            return None

        return fetch_with_resilience(
            _action,
            3,
            circuit_breaker=breaker,
            is_valid=lambda df: df is not None and not df.empty,
        )

    client.financials.fetch_income_stmt.side_effect = _income
    for name in ("fetch_balance_sheet", "fetch_cashflow"):
        getattr(client.financials, name).return_value = None
    news_ticker = MagicMock()
    news_ticker.news = []
    client.get_ticker.return_value = news_ticker
    return client


def _sweep(
    client: MagicMock, memory: EmptyFetchMemory, instrument_id: Any, now: datetime
) -> dict[str, Any]:
    service = YFinanceDataService(MagicMock(), client, empty_fetches=memory)
    return service.fetch_and_store(
        instrument_id, "TEST", mode="incremental", staleness=_staleness(now)
    )


class TestLostRequestIsNotNoData:
    """Only a request Yahoo answered may settle a category as empty."""

    def test_a_lost_statement_request_is_not_remembered_as_empty(self) -> None:
        """Retry a statement whose request got a 503 on the next run."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        breaker = CircuitBreaker()
        instrument_id = uuid4()
        client = _client(breaker, income_error=YahooTransientError(503))

        result = _sweep(client, memory, instrument_id, now)

        assert breaker.is_active is False
        assert result["transport_failures"] == 1
        assert not memory.is_recent(
            instrument_id, "income_statement_annual_updated_at", 72, now
        )

    def test_a_lost_request_keeps_the_other_categories_retryable(self) -> None:
        """Do not settle categories that came back empty in a ticker that lost a request."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        instrument_id = uuid4()
        staleness = _staleness(now)
        staleness["eps_trend_updated_at"] = None
        client = _client(CircuitBreaker(), income_error=YahooTransientError(503))

        YFinanceDataService(MagicMock(), client, empty_fetches=memory).fetch_and_store(
            instrument_id, "TEST", mode="incremental", staleness=staleness
        )

        assert not memory.is_recent(instrument_id, "eps_trend_updated_at", 72, now)

    def test_a_genuinely_empty_answer_is_still_remembered(self) -> None:
        """A clean empty answer keeps settling the category."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        instrument_id = uuid4()
        client = _client(CircuitBreaker(), income_error=None)

        result = _sweep(client, memory, instrument_id, now)

        assert result["transport_failures"] == 0
        assert memory.is_recent(
            instrument_id, "income_statement_annual_updated_at", 72, now
        )
