"""Unit tests for POST /api/v1/backtest and GET /api/v1/backtest/{job_id}.

Covers:
  - POST: returns 202 + job_id + run_id
  - POST: empty tickers returns 422
  - POST: JobAlreadyRunningError returns 409
  - GET: existing job returns progress
  - GET: missing job returns 404

Note on test isolation
----------------------
The POST endpoint calls ``db.commit()`` to persist the pending BacktestRun before
handing off to the background worker.  SQLAlchemy's SAVEPOINT-based test isolation
(conftest.py) does not always roll back rows that were committed within a savepoint
when using SQLite + StaticPool.  The ``cleanup_backtest_runs`` autouse fixture
explicitly deletes and commits the deletion so rows never leak into other test
modules regardless of SQLAlchemy's rollback behaviour.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

BASE_URL = "/api/v1/backtest"

_JOB_SVC_CREATE = "app.api.v1.backtest._job_service.create_job"
_JOB_SVC_START = "app.api.v1.backtest._job_service.start_background"
_JOB_SVC_GET = "app.api.v1.backtest._job_service.get_job"

_MOCK_JOB_ID = "00000000-0000-0000-0000-000000000001"

_VALID_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "pipeline_config": {"optimizer_type": "hrp"},
}


# ---------------------------------------------------------------------------
# Isolation fixture
# ---------------------------------------------------------------------------


import pytest  # noqa: E402 — placed here so module-level constants are first


@pytest.fixture(autouse=True)
def cleanup_backtest_runs(db_session: Session) -> Generator[None, None, None]:
    """Delete all BacktestRun rows after each test.

    The POST endpoint calls ``db.commit()`` which can persist rows even when the
    outer SAVEPOINT is later rolled back (SQLite + StaticPool edge case).  This
    fixture commits the deletion so no rows leak into subsequent test modules.
    """
    yield
    from app.models.execution import BacktestRun

    db_session.query(BacktestRun).delete()
    db_session.commit()


class TestPostBacktest:
    """POST /api/v1/backtest always launches a background job, returns 202."""

    def test_valid_request_returns_202(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.status_code == 202

    def test_valid_request_body_contains_job_id(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        body = resp.json()
        assert body["jobId"] == _MOCK_JOB_ID

    def test_valid_request_body_contains_run_id(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        body = resp.json()
        assert "runId" in body
        # run_id must be a valid UUID string
        uuid.UUID(body["runId"])

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_URL, json={**_VALID_REQUEST, "tickers": []})
        assert resp.status_code == 422

    def test_missing_tickers_field_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_REQUEST.items() if k != "tickers"}
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_job_already_running_returns_409(self, client: TestClient) -> None:
        from app.services.background_job import JobAlreadyRunningError

        with patch(_JOB_SVC_CREATE, side_effect=JobAlreadyRunningError(_MOCK_JOB_ID)):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.status_code == 409

    def test_background_worker_started(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START) as mock_start,
        ):
            client.post(BASE_URL, json=_VALID_REQUEST)

        mock_start.assert_called_once()


class TestGetBacktestJob:
    """GET /api/v1/backtest/{job_id} polls progress for a background job."""

    def test_existing_pending_job_returns_200(self, client: TestClient) -> None:
        mock_job = _pending_job(_MOCK_JOB_ID)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.status_code == 200

    def test_existing_job_returns_job_id(self, client: TestClient) -> None:
        mock_job = _pending_job(_MOCK_JOB_ID)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        # AsyncJobProgress uses BaseModel (snake_case), not CamelCaseModel
        assert resp.json()["job_id"] == _MOCK_JOB_ID

    def test_existing_job_returns_status(self, client: TestClient) -> None:
        mock_job = {**_pending_job(_MOCK_JOB_ID), "status": "running"}
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.json()["status"] == "running"

    def test_missing_job_returns_404(self, client: TestClient) -> None:
        with patch(_JOB_SVC_GET, return_value=None):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.status_code == 404


class TestGetBacktestRun:
    """GET /api/v1/backtest/runs/{run_id} returns the persisted BacktestRun (issue #464)."""

    def test_existing_run_returns_200(self, client: TestClient, db_session: Session) -> None:
        run = _create_backtest_run(db_session)

        resp = client.get(f"{BASE_URL}/runs/{run.id}")

        assert resp.status_code == 200

    def test_response_contains_run_id_and_status(
        self, client: TestClient, db_session: Session
    ) -> None:
        run = _create_backtest_run(db_session, status="completed")

        resp = client.get(f"{BASE_URL}/runs/{run.id}")

        body = resp.json()
        assert body["id"] == str(run.id)
        assert body["status"] == "completed"

    def test_response_contains_camel_case_result_fields(
        self, client: TestClient, db_session: Session
    ) -> None:
        run = _create_backtest_run(
            db_session,
            equity_curve={"2024-01-02": 100.0},
            summary_stats={"sharpe": 1.25},
        )

        resp = client.get(f"{BASE_URL}/runs/{run.id}")

        body = resp.json()
        assert body["equityCurve"] == {"2024-01-02": 100.0}
        assert body["summaryStats"] == {"sharpe": 1.25}

    def test_missing_run_returns_404(self, client: TestClient) -> None:
        random_uuid = str(uuid.uuid4())

        resp = client.get(f"{BASE_URL}/runs/{random_uuid}")

        assert resp.status_code == 404

    def test_invalid_uuid_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{BASE_URL}/runs/not-a-uuid")

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pending_job(job_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "pending",
        "current": 0,
        "total": 0,
        "errors": [],
        "result": None,
        "error": None,
    }


def _create_backtest_run(
    session: Session,
    status: str = "completed",
    equity_curve: dict[str, Any] | None = None,
    summary_stats: dict[str, Any] | None = None,
) -> Any:
    """Insert a minimal BacktestRun row covering every NOT NULL column."""
    from app.models.execution import BacktestRun

    run = BacktestRun(
        status=status,
        config={"optimizer_type": "hrp"},
        equity_curve=equity_curve or {},
        drawdowns={},
        monthly_returns={},
        yearly_returns={},
        rolling_metrics={},
        turnover_history={},
        summary_stats=summary_stats or {},
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run
