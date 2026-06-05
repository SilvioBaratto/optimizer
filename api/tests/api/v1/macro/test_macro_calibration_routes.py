"""Route tests for the macro-calibration batch job endpoints (issue #814).

Scoped to the two batch endpoints in ``app.api.v1.macro.macro_calibration``:
``POST /views/macro-calibration/batch`` and its poll
``GET /views/macro-calibration/batch/{job_id}``.  The synchronous
``GET /views/macro-calibration`` endpoint is already covered by
``tests/unit/test_macro_calibration.py::TestMacroCalibrationEndpoint`` (200,
422, 502, force_refresh) and is intentionally NOT re-tested here.

The module-level ``_calibrate_job_service`` is mocked via ``mock_job_service``
(patches ``create_job`` / ``get_job`` / ``start_background`` on the instance),
so no daemon worker launches and ``run_bulk_calibrate`` never runs.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.schemas.macro.macro_regime import (
    MacroCalibrateBatchJobResponse,
    MacroCalibrateBatchProgress,
)
from app.services.macro.scrapers import PORTFOLIO_COUNTRIES
from tests._fixtures.assertions import assert_json_safe, assert_response_matches_schema
from tests._fixtures.job_service_mock import mock_job_service

BASE_URL = "/api/v1/views/macro-calibration/batch"
_JOB = "app.api.v1.macro.macro_calibration._calibrate_job_service"


# ---------------------------------------------------------------------------
# POST /views/macro-calibration/batch
# ---------------------------------------------------------------------------


class TestStartBatchCalibration:
    def test_when_started_then_202_with_pending_job_id(
        self, client: TestClient
    ) -> None:
        with mock_job_service(_JOB, job_id="cal-1") as job:
            resp = client.post(BASE_URL)

        assert resp.status_code == 202
        body = resp.json()
        assert_response_matches_schema(body, MacroCalibrateBatchJobResponse)
        assert body["job_id"] == "cal-1"
        assert body["status"] == "pending"
        job.start_background.assert_called_once()

    def test_when_already_running_then_409(self, client: TestClient) -> None:
        with mock_job_service(_JOB, raise_already_running=True):
            resp = client.post(BASE_URL)

        assert resp.status_code == 409

    def test_when_no_countries_then_defaults_to_portfolio_countries(
        self, client: TestClient
    ) -> None:
        with mock_job_service(_JOB) as job:
            resp = client.post(BASE_URL, json={})

        assert resp.status_code == 202
        forwarded_countries = job.start_background.call_args.kwargs["args"][1]
        assert forwarded_countries == list(PORTFOLIO_COUNTRIES)


# ---------------------------------------------------------------------------
# GET /views/macro-calibration/batch/{job_id}
# ---------------------------------------------------------------------------


class TestPollBatchCalibration:
    def test_when_job_pending_then_200_with_progress(
        self, client: TestClient
    ) -> None:
        with mock_job_service(_JOB, job_id="cal-2", initial_status="pending"):
            resp = client.get(f"{BASE_URL}/cal-2")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, MacroCalibrateBatchProgress)
        assert body["job_id"] == "cal-2"
        assert body["status"] == "pending"
        assert_json_safe(body)

    def test_when_job_unknown_then_404(self, client: TestClient) -> None:
        with patch(f"{_JOB}.get_job", return_value=None):
            resp = client.get(f"{BASE_URL}/ghost")

        assert resp.status_code == 404
