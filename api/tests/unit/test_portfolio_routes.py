"""Unit tests for portfolio CRUD and broker sync endpoints.

All repositories and the module-level _sync_job_service are mocked.
No real database is touched.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch targets
_PORTFOLIO_REPO = "app.api.v1.portfolio.PortfolioRepository"
_SYNC_JOB_SVC = "app.api.v1.portfolio._sync_job_service"

BASE_URL = "/api/v1/portfolio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(name: str = "test") -> MagicMock:
    p = MagicMock()
    p.id = uuid.uuid4()
    p.name = name
    p.description = "A test portfolio"
    p.currency = "EUR"
    p.benchmark_ticker = "SPY"
    p.is_active = True
    p.created_at = datetime.now(timezone.utc)
    p.updated_at = datetime.now(timezone.utc)
    return p


def _make_snapshot(portfolio_id: uuid.UUID | None = None) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.portfolio_id = portfolio_id or uuid.uuid4()
    s.snapshot_date = date.today()
    s.snapshot_type = "manual"
    s.weights = {"AAPL": 0.5, "MSFT": 0.5}
    s.sector_mapping = {"AAPL": "Technology", "MSFT": "Technology"}
    s.summary = None
    s.optimizer_config = None
    s.turnover = None
    s.holding_count = 2
    s.created_at = datetime.now(timezone.utc)
    return s


def _make_position() -> MagicMock:
    p = MagicMock()
    p.id = uuid.uuid4()
    p.ticker = "AAPL_US_EQ"
    p.yfinance_ticker = "AAPL"
    p.name = "Apple Inc."
    p.quantity = 10.0
    p.average_price = 150.0
    p.current_price = 175.0
    p.ppl = 250.0
    p.fx_ppl = None
    p.initial_fill_date = date.today()
    p.synced_at = datetime.now(timezone.utc)
    return p


def _make_account() -> MagicMock:
    a = MagicMock()
    a.id = uuid.uuid4()
    a.total = 10000.0
    a.free = 2000.0
    a.invested = 8000.0
    a.blocked = None
    a.result = 500.0
    a.currency = "EUR"
    a.synced_at = datetime.now(timezone.utc)
    return a


def _mock_t212_client_dep() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# GET /portfolio/
# ---------------------------------------------------------------------------


class TestListPortfolios:
    def test_returns_list(self, client: TestClient) -> None:
        portfolio = _make_portfolio("myport")
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_all_active.return_value = [portfolio]
            resp = client.get(f"{BASE_URL}/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["name"] == "myport"

    def test_empty_list(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_all_active.return_value = []
            resp = client.get(f"{BASE_URL}/")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# POST /portfolio/
# ---------------------------------------------------------------------------


class TestCreatePortfolio:
    def test_creates_successfully(self, client: TestClient) -> None:
        portfolio = _make_portfolio("newport")
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_or_create.return_value = portfolio
            resp = client.post(
                f"{BASE_URL}/",
                json={"name": "newport", "currency": "EUR", "benchmark_ticker": "SPY"},
            )
        assert resp.status_code == 201
        assert resp.json()["name"] == "newport"

    def test_missing_name_rejected(self, client: TestClient) -> None:
        resp = client.post(f"{BASE_URL}/", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /portfolio/{name}
# ---------------------------------------------------------------------------


class TestGetPortfolio:
    def test_found(self, client: TestClient) -> None:
        portfolio = _make_portfolio("myport")
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = portfolio
            resp = client.get(f"{BASE_URL}/myport")
        assert resp.status_code == 200
        assert resp.json()["name"] == "myport"

    def test_not_found(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{BASE_URL}/missing")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /portfolio/{name}/snapshots
# ---------------------------------------------------------------------------


class TestCreateSnapshot:
    def test_created(self, client: TestClient) -> None:
        portfolio = _make_portfolio()
        snap = _make_snapshot(portfolio.id)
        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.create_snapshot.return_value = snap
            resp = client.post(
                f"{BASE_URL}/test/snapshots",
                json={
                    "snapshot_date": str(date.today()),
                    "snapshot_type": "manual",
                    "weights": {"AAPL": 0.5, "MSFT": 0.5},
                },
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["snapshot_type"] == "manual"
        assert "weights" in body

    def test_portfolio_not_found(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.post(
                f"{BASE_URL}/missing/snapshots",
                json={
                    "snapshot_date": str(date.today()),
                    "snapshot_type": "manual",
                    "weights": {"AAPL": 1.0},
                },
            )
        assert resp.status_code == 404

    def test_invalid_snapshot_type_rejected(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE_URL}/test/snapshots",
            json={
                "snapshot_date": str(date.today()),
                "snapshot_type": "invalid_type",
                "weights": {"AAPL": 1.0},
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /portfolio/{name}/snapshots
# ---------------------------------------------------------------------------


class TestListSnapshots:
    def test_returns_list(self, client: TestClient) -> None:
        portfolio = _make_portfolio()
        snap = _make_snapshot(portfolio.id)
        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_snapshots.return_value = [snap]
            resp = client.get(f"{BASE_URL}/test/snapshots")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_portfolio_not_found(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{BASE_URL}/missing/snapshots")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /portfolio/{name}/snapshots/latest
# ---------------------------------------------------------------------------


class TestGetLatestSnapshot:
    def test_found(self, client: TestClient) -> None:
        portfolio = _make_portfolio()
        snap = _make_snapshot(portfolio.id)
        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snap
            resp = client.get(f"{BASE_URL}/test/snapshots/latest")
        assert resp.status_code == 200
        assert "weights" in resp.json()

    def test_no_snapshots(self, client: TestClient) -> None:
        portfolio = _make_portfolio()
        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = None
            resp = client.get(f"{BASE_URL}/test/snapshots/latest")
        assert resp.status_code == 404

    def test_portfolio_not_found(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{BASE_URL}/missing/snapshots/latest")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /portfolio/{name}/sync
# ---------------------------------------------------------------------------


class TestTriggerSync:
    def test_returns_202_and_job_id(self, client: TestClient) -> None:
        from app.api.v1.portfolio import get_t212_client
        from app.main import app

        portfolio = _make_portfolio()
        try:
            app.dependency_overrides[get_t212_client] = _mock_t212_client_dep
            with (
                patch(_PORTFOLIO_REPO) as MockRepo,
                patch(f"{_SYNC_JOB_SVC}.create_job", return_value="job-abc"),
                patch(f"{_SYNC_JOB_SVC}.start_background"),
            ):
                MockRepo.return_value.get_by_name.return_value = portfolio
                resp = client.post(f"{BASE_URL}/test/sync")
        finally:
            app.dependency_overrides.pop(get_t212_client, None)

        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "job-abc"
        assert body["status"] == "pending"

    def test_409_when_job_already_running(self, client: TestClient) -> None:
        from app.api.v1.portfolio import get_t212_client
        from app.main import app
        from app.services.background_job import JobAlreadyRunningError

        portfolio = _make_portfolio()
        try:
            app.dependency_overrides[get_t212_client] = _mock_t212_client_dep
            with (
                patch(_PORTFOLIO_REPO) as MockRepo,
                patch(
                    f"{_SYNC_JOB_SVC}.create_job",
                    side_effect=JobAlreadyRunningError("existing-id"),
                ),
            ):
                MockRepo.return_value.get_by_name.return_value = portfolio
                resp = client.post(f"{BASE_URL}/test/sync")
        finally:
            app.dependency_overrides.pop(get_t212_client, None)

        assert resp.status_code == 409

    def test_portfolio_not_found(self, client: TestClient) -> None:
        from app.api.v1.portfolio import get_t212_client
        from app.main import app

        try:
            app.dependency_overrides[get_t212_client] = _mock_t212_client_dep
            with patch(_PORTFOLIO_REPO) as MockRepo:
                MockRepo.return_value.get_by_name.return_value = None
                resp = client.post(f"{BASE_URL}/missing/sync")
        finally:
            app.dependency_overrides.pop(get_t212_client, None)

        assert resp.status_code == 404

    def test_503_when_no_api_key(self, client: TestClient) -> None:
        """get_t212_client raises 503 when API key is absent."""
        with patch("app.api.v1.portfolio.settings") as mock_settings:
            mock_settings.trading_212_api_key = ""
            resp = client.post(f"{BASE_URL}/test/sync")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /portfolio/{name}/sync/{job_id}
# ---------------------------------------------------------------------------


class TestGetSyncStatus:
    def test_pending_job(self, client: TestClient) -> None:
        with patch(f"{_SYNC_JOB_SVC}.get_job", return_value={
            "job_id": "abc",
            "status": "pending",
            "current": 0,
            "total": 4,
            "result": None,
            "error": None,
        }):
            resp = client.get(f"{BASE_URL}/test/sync/abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "pending"
        assert body["job_id"] == "abc"

    def test_completed_job_has_result(self, client: TestClient) -> None:
        with patch(f"{_SYNC_JOB_SVC}.get_job", return_value={
            "job_id": "abc",
            "status": "completed",
            "current": 4,
            "total": 4,
            "result": {"positions_synced": 10},
            "error": None,
        }):
            resp = client.get(f"{BASE_URL}/test/sync/abc")
        assert resp.status_code == 200
        assert resp.json()["result"]["positions_synced"] == 10

    def test_job_not_found_returns_404(self, client: TestClient) -> None:
        with patch(f"{_SYNC_JOB_SVC}.get_job", return_value=None):
            resp = client.get(f"{BASE_URL}/test/sync/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /portfolio/{name}/positions
# ---------------------------------------------------------------------------


class TestGetPositions:
    def test_returns_positions(self, client: TestClient) -> None:
        portfolio = _make_portfolio()
        position = _make_position()
        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_positions.return_value = [position]
            resp = client.get(f"{BASE_URL}/test/positions")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["ticker"] == "AAPL_US_EQ"

    def test_portfolio_not_found(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{BASE_URL}/missing/positions")
        assert resp.status_code == 404

    def test_empty_positions(self, client: TestClient) -> None:
        portfolio = _make_portfolio()
        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_positions.return_value = []
            resp = client.get(f"{BASE_URL}/test/positions")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /portfolio/{name}/account
# ---------------------------------------------------------------------------


class TestGetAccount:
    def test_returns_account(self, client: TestClient) -> None:
        portfolio = _make_portfolio()
        account = _make_account()
        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_account_snapshot.return_value = account
            resp = client.get(f"{BASE_URL}/test/account")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 10000.0
        assert body["currency"] == "EUR"

    def test_no_account_snapshot(self, client: TestClient) -> None:
        portfolio = _make_portfolio()
        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_account_snapshot.return_value = None
            resp = client.get(f"{BASE_URL}/test/account")
        assert resp.status_code == 404

    def test_portfolio_not_found(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{BASE_URL}/missing/account")
        assert resp.status_code == 404
