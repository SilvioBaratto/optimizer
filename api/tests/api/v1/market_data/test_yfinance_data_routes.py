"""Route tests for the yfinance-data endpoints (issue #815).

Covers all 14 endpoints in ``app.api.v1.market_data.yfinance_data`` at success
plus their documented error paths.  ``YFinanceRepository`` is patched at the
import site (the ``_get_repo`` dependency constructs it per request, so the
patch is honoured); the module-level ``_job_service`` is mocked via
``mock_job_service``; ``_get_yf_client`` is swapped through
``dependency_overrides`` so no real yfinance client is built.

Fakes are plain ``SimpleNamespace`` objects carrying only each response model's
required attributes — ``from_attributes`` fills the optional fields from their
defaults.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.market_data.yfinance_data import _get_yf_client
from app.main import app
from app.schemas.market_data.yfinance_data import (
    AnalystPriceTargetResponse,
    AnalystRecommendationResponse,
    DividendResponse,
    FinancialStatementResponse,
    InsiderTransactionResponse,
    InstitutionalHolderResponse,
    MutualFundHolderResponse,
    PriceHistoryResponse,
    StockSplitResponse,
    TickerNewsResponse,
    TickerProfileResponse,
    YFinanceFetchJobResponse,
    YFinanceFetchProgress,
    YFinanceSingleFetchResponse,
)
from tests._fixtures.assertions import assert_json_safe, assert_response_matches_schema
from tests._fixtures.job_service_mock import mock_job_service

BASE_URL = "/api/v1/yfinance-data"
_REPO = "app.api.v1.market_data.yfinance_data.YFinanceRepository"
_SERVICE = "app.api.v1.market_data.yfinance_data.YFinanceDataService"
_JOB = "app.api.v1.market_data.yfinance_data._job_service"
_INSTRUMENT_ID = str(uuid.uuid4())


def _now() -> datetime:
    return datetime(2024, 1, 2, tzinfo=timezone.utc)


def _base(**extra: Any) -> SimpleNamespace:
    """Fake row carrying the identity fields every response model declares."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        instrument_id=uuid.UUID(_INSTRUMENT_ID),
        created_at=_now(),
        updated_at=_now(),
        **extra,
    )


@pytest.fixture
def override_yf_client() -> Iterator[None]:
    app.dependency_overrides[_get_yf_client] = lambda: MagicMock()
    try:
        yield
    finally:
        app.dependency_overrides.pop(_get_yf_client, None)


# ---------------------------------------------------------------------------
# Bulk fetch: POST /fetch + GET /fetch/{job_id}
# ---------------------------------------------------------------------------


class TestBulkFetch:
    def test_when_started_then_202_with_job_id(
        self, client: TestClient, override_yf_client: None
    ) -> None:
        with mock_job_service(_JOB, job_id="yf-1") as job:
            resp = client.post(f"{BASE_URL}/fetch")

        assert resp.status_code == 202
        body = resp.json()
        assert_response_matches_schema(body, YFinanceFetchJobResponse)
        assert body["job_id"] == "yf-1"
        assert body["status"] == "pending"
        job.start_background.assert_called_once()

    def test_when_already_running_then_409(
        self, client: TestClient, override_yf_client: None
    ) -> None:
        with mock_job_service(_JOB, raise_already_running=True):
            resp = client.post(f"{BASE_URL}/fetch")

        assert resp.status_code == 409

    def test_when_job_pending_then_200_progress(self, client: TestClient) -> None:
        with mock_job_service(_JOB, job_id="yf-2", initial_status="pending"):
            resp = client.get(f"{BASE_URL}/fetch/yf-2")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, YFinanceFetchProgress)
        assert_json_safe(body)

    def test_when_job_unknown_then_404(self, client: TestClient) -> None:
        with patch(f"{_JOB}.get_job", return_value=None):
            resp = client.get(f"{BASE_URL}/fetch/ghost")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Single-ticker fetch: POST /fetch/ticker/{ticker}
# ---------------------------------------------------------------------------


class TestSingleTickerFetch:
    def test_when_instrument_found_then_200_with_counts(
        self, client: TestClient, override_yf_client: None
    ) -> None:
        instrument = SimpleNamespace(
            id=uuid.UUID(_INSTRUMENT_ID),
            exchange_name="NASDAQ",
            currency_code="USD",
        )
        fetch_result = {"counts": {"prices": 10}, "errors": [], "skipped": []}
        with patch(_REPO) as MockRepo, patch(_SERVICE) as MockSvc:
            MockRepo.return_value.get_instrument_by_yfinance_ticker.return_value = (
                instrument
            )
            MockSvc.return_value.fetch_and_store.return_value = fetch_result
            resp = client.post(f"{BASE_URL}/fetch/ticker/AAPL")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, YFinanceSingleFetchResponse)
        assert body["counts"] == {"prices": 10}
        assert body["errors"] == []
        assert body["skipped"] == []
        assert_json_safe(body)

    def test_when_instrument_missing_then_404(
        self, client: TestClient, override_yf_client: None
    ) -> None:
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_instrument_by_yfinance_ticker.return_value = None
            resp = client.post(f"{BASE_URL}/fetch/ticker/GHOST")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# List read endpoints (parametrized)
# ---------------------------------------------------------------------------

_ReadCase = tuple[str, str, type, Callable[[], SimpleNamespace]]

READ_CASES: list[_ReadCase] = [
    ("prices", "get_price_history", PriceHistoryResponse,
     lambda: _base(date=date(2024, 1, 2))),
    ("financials", "get_financial_statements", FinancialStatementResponse,
     lambda: _base(statement_type="income_statement", period_type="annual",
                   period_date=date(2024, 1, 2), line_item="Revenue")),
    ("dividends", "get_dividends", DividendResponse,
     lambda: _base(date=date(2024, 1, 2), amount=0.5)),
    ("splits", "get_splits", StockSplitResponse,
     lambda: _base(date=date(2024, 1, 2), ratio=2.0)),
    ("recommendations", "get_recommendations", AnalystRecommendationResponse,
     lambda: _base(period="0m")),
    ("institutional-holders", "get_institutional_holders",
     InstitutionalHolderResponse, lambda: _base(holder_name="Vanguard")),
    ("mutualfund-holders", "get_mutualfund_holders", MutualFundHolderResponse,
     lambda: _base(holder_name="Fidelity Fund")),
    ("insider-transactions", "get_insider_transactions",
     InsiderTransactionResponse,
     lambda: _base(insider_name="Jane Doe", transaction_type="Buy",
                   start_date=date(2024, 1, 2))),
    ("news", "get_news", TickerNewsResponse,
     lambda: _base(news_uuid="news-abc-123")),
]
_READ_IDS = [c[0] for c in READ_CASES]


class TestListReadEndpoints:
    @pytest.mark.parametrize("suffix, method, model, builder", READ_CASES,
                             ids=_READ_IDS)
    def test_when_rows_exist_then_200_validates_model(
        self, suffix: str, method: str, model: type,
        builder: Callable[[], SimpleNamespace], client: TestClient,
    ) -> None:
        with patch(_REPO) as MockRepo:
            getattr(MockRepo.return_value, method).return_value = [builder()]
            resp = client.get(
                f"{BASE_URL}/instruments/{_INSTRUMENT_ID}/{suffix}"
            )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert_response_matches_schema(body[0], model)
        assert_json_safe(body)


# ---------------------------------------------------------------------------
# Single-object read endpoints with 404: profile + price-targets
# ---------------------------------------------------------------------------


class TestProfile:
    def test_when_profile_exists_then_200(self, client: TestClient) -> None:
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_profile.return_value = _base()
            resp = client.get(f"{BASE_URL}/instruments/{_INSTRUMENT_ID}/profile")

        assert resp.status_code == 200
        assert_response_matches_schema(resp.json(), TickerProfileResponse)

    def test_when_profile_missing_then_404(self, client: TestClient) -> None:
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_profile.return_value = None
            resp = client.get(f"{BASE_URL}/instruments/{_INSTRUMENT_ID}/profile")

        assert resp.status_code == 404


class TestPriceTargets:
    def test_when_targets_exist_then_200(self, client: TestClient) -> None:
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_price_targets.return_value = _base(mean=180.0)
            resp = client.get(
                f"{BASE_URL}/instruments/{_INSTRUMENT_ID}/price-targets"
            )

        assert resp.status_code == 200
        assert_response_matches_schema(resp.json(), AnalystPriceTargetResponse)

    def test_when_targets_missing_then_404(self, client: TestClient) -> None:
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_price_targets.return_value = None
            resp = client.get(
                f"{BASE_URL}/instruments/{_INSTRUMENT_ID}/price-targets"
            )

        assert resp.status_code == 404
