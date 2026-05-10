"""Tests for GET /api/v1/scheduler/status endpoint (issue #375).

Tests written FIRST (TDD — Red phase).  They cover:
  1. Returns full job list when scheduler is running.
  2. Returns scheduler_running=False with empty jobs when app.state.scheduler is None.
  3. Returns null last_run_time / last_status when no background_jobs rows exist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

BASE_URL = "/api/v1/scheduler/status"

_REPO_PATH = "app.api.v1.jobs.scheduler.BackgroundJobRepository"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_apscheduler_job(
    job_id: str,
    name: str,
    next_run_time: datetime | None = None,
    trigger_str: str = "cron[hour=7, minute=0]",
) -> MagicMock:
    job = MagicMock()
    job.id = job_id
    job.name = name
    job.next_run_time = next_run_time or datetime(
        2026, 4, 16, 7, 0, 0, tzinfo=timezone.utc
    )
    trigger = MagicMock()
    trigger.__str__ = lambda _: trigger_str
    job.trigger = trigger
    return job


def _make_bg_job(
    job_type: str = "yfinance_fetch",
    status: str = "completed",
    finished_at: datetime | None = None,
) -> MagicMock:
    row = MagicMock()
    row.job_type = job_type
    row.status = status
    row.finished_at = finished_at or datetime(
        2026, 4, 15, 7, 30, 0, tzinfo=timezone.utc
    )
    return row


def _make_scheduler(jobs: list[MagicMock]) -> MagicMock:
    scheduler = MagicMock()
    scheduler.running = True
    scheduler.get_jobs.return_value = jobs
    return scheduler


# ---------------------------------------------------------------------------
# Tests: scheduler running
# ---------------------------------------------------------------------------


class TestSchedulerStatusRunning:
    """Endpoint returns job statuses when scheduler is active."""

    def test_returns_200(self, client: TestClient) -> None:
        apsjobs = [
            _make_apscheduler_job("daily_pipeline", "Daily data pipeline"),
        ]
        scheduler = _make_scheduler(apsjobs)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_latest_by_type.return_value = _make_bg_job("yfinance_fetch")

            with patch.object(
                cast(FastAPI, client.app).state, "scheduler", scheduler, create=True
            ):
                response = client.get(BASE_URL)

        assert response.status_code == 200

    def test_scheduler_running_true(self, client: TestClient) -> None:
        apsjobs = [_make_apscheduler_job("daily_pipeline", "Daily data pipeline")]
        scheduler = _make_scheduler(apsjobs)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_latest_by_type.return_value = None

            with patch.object(
                cast(FastAPI, client.app).state, "scheduler", scheduler, create=True
            ):
                response = client.get(BASE_URL)

        data = response.json()
        assert data["schedulerRunning"] is True

    def test_jobs_list_contains_all_apscheduler_jobs(self, client: TestClient) -> None:
        apsjobs = [
            _make_apscheduler_job("daily_pipeline", "Daily data pipeline"),
            _make_apscheduler_job("midday_news", "Midday news refresh"),
            _make_apscheduler_job("weekly_refetch", "Weekly full data refetch"),
        ]
        scheduler = _make_scheduler(apsjobs)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_latest_by_type.return_value = None

            with patch.object(
                cast(FastAPI, client.app).state, "scheduler", scheduler, create=True
            ):
                response = client.get(BASE_URL)

        data = response.json()
        assert len(data["jobs"]) == 3

    def test_job_entry_contains_required_fields(self, client: TestClient) -> None:
        next_run = datetime(2026, 4, 16, 7, 0, 0, tzinfo=timezone.utc)
        apsjobs = [
            _make_apscheduler_job("daily_pipeline", "Daily data pipeline", next_run)
        ]
        scheduler = _make_scheduler(apsjobs)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_latest_by_type.return_value = None

            with patch.object(
                cast(FastAPI, client.app).state, "scheduler", scheduler, create=True
            ):
                response = client.get(BASE_URL)

        job = response.json()["jobs"][0]
        assert "jobId" in job
        assert "name" in job
        assert "nextRunTime" in job
        assert "lastRunTime" in job
        assert "lastStatus" in job
        assert "trigger" in job

    def test_job_id_and_name_match_apscheduler(self, client: TestClient) -> None:
        apsjobs = [_make_apscheduler_job("fred_monthly", "Monthly FRED data fetch")]
        scheduler = _make_scheduler(apsjobs)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_latest_by_type.return_value = None

            with patch.object(
                cast(FastAPI, client.app).state, "scheduler", scheduler, create=True
            ):
                response = client.get(BASE_URL)

        job = response.json()["jobs"][0]
        assert job["jobId"] == "fred_monthly"
        assert job["name"] == "Monthly FRED data fetch"

    def test_last_run_populated_from_background_jobs(self, client: TestClient) -> None:
        finished = datetime(2026, 4, 15, 7, 45, 0, tzinfo=timezone.utc)
        apsjobs = [_make_apscheduler_job("daily_pipeline", "Daily data pipeline")]
        scheduler = _make_scheduler(apsjobs)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_latest_by_type.return_value = _make_bg_job(
                "yfinance_fetch", "completed", finished
            )

            with patch.object(
                cast(FastAPI, client.app).state, "scheduler", scheduler, create=True
            ):
                response = client.get(BASE_URL)

        job = response.json()["jobs"][0]
        assert job["lastStatus"] == "completed"
        assert job["lastRunTime"] is not None


# ---------------------------------------------------------------------------
# Tests: scheduler not running
# ---------------------------------------------------------------------------


class TestSchedulerStatusNotRunning:
    """Endpoint returns safe defaults when scheduler is None or not running."""

    def test_returns_200_when_scheduler_none(self, client: TestClient) -> None:
        with patch.object(
            cast(FastAPI, client.app).state, "scheduler", None, create=True
        ):
            response = client.get(BASE_URL)
        assert response.status_code == 200

    def test_scheduler_running_false_when_none(self, client: TestClient) -> None:
        with patch.object(
            cast(FastAPI, client.app).state, "scheduler", None, create=True
        ):
            response = client.get(BASE_URL)
        data = response.json()
        assert data["schedulerRunning"] is False

    def test_empty_jobs_list_when_scheduler_none(self, client: TestClient) -> None:
        with patch.object(
            cast(FastAPI, client.app).state, "scheduler", None, create=True
        ):
            response = client.get(BASE_URL)
        data = response.json()
        assert data["jobs"] == []

    def test_scheduler_running_false_when_not_started(self, client: TestClient) -> None:
        scheduler = MagicMock()
        scheduler.running = False
        scheduler.get_jobs.return_value = []

        with patch.object(
            cast(FastAPI, client.app).state, "scheduler", scheduler, create=True
        ):
            response = client.get(BASE_URL)

        data = response.json()
        assert data["schedulerRunning"] is False
        assert data["jobs"] == []


# ---------------------------------------------------------------------------
# Tests: missing job history (first startup)
# ---------------------------------------------------------------------------


class TestSchedulerStatusMissingHistory:
    """last_run_time and last_status are null when no background_jobs rows exist."""

    def test_last_run_null_when_no_history(self, client: TestClient) -> None:
        apsjobs = [_make_apscheduler_job("fred_monthly", "Monthly FRED data fetch")]
        scheduler = _make_scheduler(apsjobs)

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_latest_by_type.return_value = None

            with patch.object(
                cast(FastAPI, client.app).state, "scheduler", scheduler, create=True
            ):
                response = client.get(BASE_URL)

        job = response.json()["jobs"][0]
        assert job["lastRunTime"] is None
        assert job["lastStatus"] is None
