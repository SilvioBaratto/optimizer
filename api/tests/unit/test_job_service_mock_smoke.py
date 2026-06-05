"""Smoke tests for the reusable BackgroundJobService mock helper (issue #806).

Proves the helper patches a live module-level instance's
create_job/get_job/start_background, returns a complete job dict, drives a
real route to 202 + job_id, and the 409 conflict path.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.services.jobs.background_job import JobAlreadyRunningError
from tests._fixtures.job_service_mock import JobServiceStub, mock_job_service

_OPTIMIZE_INSTANCE = "app.api.v1.optimization.optimize._job_service"
_REQUIRED_KEYS = frozenset(
    {
        "job_id",
        "status",
        "current",
        "total",
        "errors",
        "result",
        "error",
        "started_at",
        "finished_at",
    }
)
_PAYLOAD = {
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "optimizer_type": "mean_risk",
    "config": {},
}


class TestMockJobServiceHelper:
    def test_when_create_job_mocked_default_job_id_is_returned(self):
        with mock_job_service(_OPTIMIZE_INSTANCE) as stub:
            assert isinstance(stub, JobServiceStub)
            assert stub.create_job() == "test-job-id"

    def test_when_get_job_mocked_complete_dict_is_returned(self):
        with mock_job_service(_OPTIMIZE_INSTANCE) as stub:
            job = stub.get_job("test-job-id")
            assert job.keys() >= _REQUIRED_KEYS
            assert job["status"] == "pending"

    def test_when_extra_passed_get_job_merges_domain_keys(self):
        with mock_job_service(_OPTIMIZE_INSTANCE, extra={"current_country": "IT"}) as s:
            job = s.get_job("test-job-id")
            assert job["current_country"] == "IT"
            assert job.keys() >= _REQUIRED_KEYS

    def test_when_raise_already_running_set_create_job_raises(self):
        with mock_job_service(_OPTIMIZE_INSTANCE, raise_already_running=True) as stub:
            with pytest.raises(JobAlreadyRunningError):
                stub.create_job()

    def test_when_custom_job_id_passed_create_job_returns_it(self):
        with mock_job_service(_OPTIMIZE_INSTANCE, job_id="custom-id") as stub:
            assert stub.create_job() == "custom-id"


class TestJobServiceMockFixture:
    def test_when_fixture_used_it_yields_the_context_manager(self, job_service_mock):
        with job_service_mock(_OPTIMIZE_INSTANCE) as stub:
            assert isinstance(stub, JobServiceStub)


class TestMockJobServiceAgainstRoute:
    def test_when_optimize_posted_async_202_and_job_id_returned(
        self, client: TestClient
    ):
        with (
            patch_sync_threshold_zero(),
            mock_job_service(_OPTIMIZE_INSTANCE) as stub,
        ):
            resp = client.post("/api/v1/optimize", json=_PAYLOAD)
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "test-job-id"
        stub.start_background.assert_called_once()

    def test_when_job_already_running_409_is_returned(self, client: TestClient):
        with (
            patch_sync_threshold_zero(),
            mock_job_service(_OPTIMIZE_INSTANCE, raise_already_running=True),
        ):
            resp = client.post("/api/v1/optimize", json=_PAYLOAD)
        assert resp.status_code == 409


def patch_sync_threshold_zero():
    """Force the optimize route's async path regardless of ticker count."""
    from unittest.mock import patch

    return patch.dict(os.environ, {"OPTIMIZE_SYNC_THRESHOLD": "0"})
