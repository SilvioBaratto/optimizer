"""Route-level tests for the Trading212 universe endpoints (issue #812).

Covers all 7 endpoints in ``app.api.v1.universe.trading212`` at success +
their documented error paths.  Repositories are patched at the **import
site**; the module-level ``_job_service`` is mocked via ``mock_job_service``;
``get_trading212_client`` / ``get_ticker_cache`` are swapped through FastAPI
``dependency_overrides``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.universe.trading212 import (
    get_ticker_cache,
    get_trading212_client,
)
from app.main import app
from app.schemas.universe.trading212 import (
    BuildJobResponse,
    BuildProgressResponse,
    CacheStatsResponse,
    ExchangeResponse,
    InstrumentListResponse,
    UniverseStatsResponse,
)
from tests._fixtures.assertions import assert_response_matches_schema
from tests._fixtures.job_service_mock import mock_job_service

BASE_URL = "/api/v1/universe"
_REPO = "app.api.v1.universe.trading212.UniverseRepository"
_JOB = "app.api.v1.universe.trading212._job_service"
_SETTINGS = "app.api.v1.universe.trading212.settings"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2024, 1, 2, tzinfo=timezone.utc)


def _fake_exchange(name: str = "NYSE") -> MagicMock:
    ex = MagicMock()
    ex.id = uuid.uuid4()
    ex.name = name
    ex.t212_id = 1
    ex.created_at = _now()
    ex.updated_at = _now()
    return ex


def _fake_instrument(ticker: str = "AAPL_US_EQ") -> MagicMock:
    inst = MagicMock()
    inst.id = uuid.uuid4()
    inst.ticker = ticker
    inst.short_name = ticker.split("_")[0]
    inst.name = "Apple Inc."
    inst.isin = "US0378331005"
    inst.instrument_type = "STOCK"
    inst.currency_code = "USD"
    inst.yfinance_ticker = "AAPL"
    inst.exchange_name = "NYSE"
    inst.created_at = _now()
    inst.updated_at = _now()
    return inst


def _cache_stats() -> dict[str, Any]:
    return {
        "total": 10,
        "fresh": 8,
        "expired": 2,
        "cache_file": "cache/ticker_map.json",
        "cache_file_size": 2048,
    }


@pytest.fixture
def override_build_deps() -> Iterator[MagicMock]:
    """Swap the T212 client + ticker cache deps; yields the cache mock."""
    cache = MagicMock()
    cache.get_stats.return_value = _cache_stats()
    app.dependency_overrides[get_trading212_client] = lambda: MagicMock()
    app.dependency_overrides[get_ticker_cache] = lambda: cache
    try:
        yield cache
    finally:
        app.dependency_overrides.pop(get_trading212_client, None)
        app.dependency_overrides.pop(get_ticker_cache, None)


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_when_called_then_counts_returned(self, client: TestClient) -> None:
        with patch(_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_exchange_count.return_value = 3
            repo.get_instrument_count.return_value = 42
            resp = client.get(f"{BASE_URL}/stats")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, UniverseStatsResponse)
        assert body["exchange_count"] == 3
        assert body["instrument_count"] == 42


# ---------------------------------------------------------------------------
# GET /exchanges
# ---------------------------------------------------------------------------


class TestExchanges:
    def test_when_called_then_exchange_list_returned(
        self, client: TestClient
    ) -> None:
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_exchanges.return_value = [_fake_exchange()]
            resp = client.get(f"{BASE_URL}/exchanges")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert_response_matches_schema(body[0], ExchangeResponse)


# ---------------------------------------------------------------------------
# GET /instruments
# ---------------------------------------------------------------------------


class TestInstruments:
    def test_when_called_then_list_response_returned(
        self, client: TestClient
    ) -> None:
        with patch(_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_instruments.return_value = [_fake_instrument()]
            repo.get_instrument_count.return_value = 1
            resp = client.get(f"{BASE_URL}/instruments")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, InstrumentListResponse)
        assert body["total"] == 1

    def test_when_query_params_set_then_forwarded_to_repo(
        self, client: TestClient
    ) -> None:
        with patch(_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_instruments.return_value = []
            repo.get_instrument_count.return_value = 0
            client.get(
                f"{BASE_URL}/instruments"
                "?exchange=NASDAQ&search=app&skip=5&limit=25"
            )

        repo.get_instruments.assert_called_once_with(
            exchange_name="NASDAQ", search="app", skip=5, limit=25
        )


# ---------------------------------------------------------------------------
# POST /build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_when_started_then_202_with_job_id(
        self, client: TestClient, override_build_deps: MagicMock
    ) -> None:
        with mock_job_service(_JOB, job_id="bid-1") as job:
            resp = client.post(f"{BASE_URL}/build", json={})

        assert resp.status_code == 202
        body = resp.json()
        assert_response_matches_schema(body, BuildJobResponse)
        assert body["job_id"] == "bid-1"
        assert body["status"] == "pending"
        job.start_background.assert_called_once()

    def test_when_already_running_then_409(
        self, client: TestClient, override_build_deps: MagicMock
    ) -> None:
        with mock_job_service(_JOB, raise_already_running=True):
            resp = client.post(f"{BASE_URL}/build", json={})

        assert resp.status_code == 409

    def test_when_api_key_empty_then_503(self, client: TestClient) -> None:
        with patch(_SETTINGS) as mock_settings:
            mock_settings.trading_212_api_key = ""
            resp = client.post(f"{BASE_URL}/build", json={})

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /build/{build_id}
# ---------------------------------------------------------------------------


class TestBuildProgress:
    def test_when_job_exists_then_progress_returned(
        self, client: TestClient
    ) -> None:
        with mock_job_service(_JOB, job_id="bid-2", initial_status="running"):
            resp = client.get(f"{BASE_URL}/build/bid-2")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, BuildProgressResponse)
        assert body["build_id"] == "bid-2"
        assert body["status"] == "running"

    def test_when_job_missing_then_404(self, client: TestClient) -> None:
        with patch(f"{_JOB}.get_job", return_value=None):
            resp = client.get(f"{BASE_URL}/build/ghost")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /cache/stats + DELETE /cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_when_stats_requested_then_cache_stats_returned(
        self, client: TestClient, override_build_deps: MagicMock
    ) -> None:
        resp = client.get(f"{BASE_URL}/cache/stats")

        assert resp.status_code == 200
        assert_response_matches_schema(resp.json(), CacheStatsResponse)

    def test_when_cleared_then_204_and_cache_cleared(
        self, client: TestClient, override_build_deps: MagicMock
    ) -> None:
        resp = client.delete(f"{BASE_URL}/cache")

        assert resp.status_code == 204
        override_build_deps.clear.assert_called_once()
