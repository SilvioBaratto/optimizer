"""Unit tests for pipeline_builder steps router (issue #714)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.pipeline_builder import pipeline_builder_router
from app.services.jobs.background_job import JobAlreadyRunningError
from app.services.pipeline_builder.session_store import _PipelineSession
from optimizer.exceptions import FactorCoverageError


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(pipeline_builder_router, prefix="/api/v1")
    return a


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _session(sid: str, **step_status_overrides: str) -> _PipelineSession:
    base_status = dict.fromkeys(
        (
            "load",
            "screen",
            "clean_returns",
            "build_history",
            "validate_is",
            "validate_oos",
            "coverage_gate",
            "regime",
            "optimize",
            "rebalance_decision",
            "cost",
            "report",
            "persist",
        ),
        "pending",
    )
    base_status.update(step_status_overrides)
    return _PipelineSession(
        session_id=sid,
        created_at=datetime.now(timezone.utc),
        run_config={},
        assembly=None,
        investable=None,
        clean_returns=None,
        factor_scores_dict=None,
        returns_history=None,
        is_result=None,
        oos_result=None,
        coverage_result=None,
        regime_result=None,
        optimize_result=None,
        rebalance_result=None,
        cost_result=None,
        report_result=None,
        current_step="idle",
        step_status=base_status,
        step_results={},
    )


@pytest.fixture
def mock_job_svc() -> MagicMock:
    svc = MagicMock()
    svc.create_job.return_value = "job-uuid-1"
    svc.start_background.return_value = (MagicMock(), MagicMock())
    svc.get_job.return_value = None
    return svc


class TestPostStep:
    def test_when_invalid_step_id_then_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/pipeline-builder/sessions/abc/steps/not_a_step",
            json={},
        )
        assert response.status_code == 400

    def test_when_session_missing_then_returns_404(self, client: TestClient) -> None:
        with patch("app.api.v1.pipeline_builder.steps.get_session", return_value=None):
            response = client.post(
                "/api/v1/pipeline-builder/sessions/abc/steps/load",
                json={},
            )
        assert response.status_code == 404

    def test_when_job_already_running_then_returns_409(
        self, client: TestClient, mock_job_svc: MagicMock
    ) -> None:
        mock_job_svc.create_job.side_effect = JobAlreadyRunningError("existing-job-id")
        with (
            patch(
                "app.api.v1.pipeline_builder.steps.get_session",
                return_value=_session("abc"),
            ),
            patch(
                "app.api.v1.pipeline_builder.steps.get_job_service",
                return_value=mock_job_svc,
            ),
        ):
            response = client.post(
                "/api/v1/pipeline-builder/sessions/abc/steps/load",
                json={},
            )
        assert response.status_code == 409
        assert "existing-job-id" in response.text

    def test_when_happy_path_then_returns_202_with_job_id(
        self, client: TestClient, mock_job_svc: MagicMock
    ) -> None:
        with (
            patch(
                "app.api.v1.pipeline_builder.steps.get_session",
                return_value=_session("abc"),
            ),
            patch(
                "app.api.v1.pipeline_builder.steps.get_job_service",
                return_value=mock_job_svc,
            ),
            patch(
                "app.api.v1.pipeline_builder.steps.update_session"
            ) as mock_update_session,
        ):
            response = client.post(
                "/api/v1/pipeline-builder/sessions/abc/steps/load",
                json={},
            )
        assert response.status_code == 202
        body = response.json()
        assert body["job_id"] == "job-uuid-1"
        assert body["status"] == "pending"
        # Route writes job_id into step_status before start_background
        mock_update_session.assert_called()
        kwargs = mock_update_session.call_args.kwargs
        assert "step_status" in kwargs
        assert kwargs["step_status"]["load"] == "job-uuid-1"
        # Background dispatch fired
        mock_job_svc.start_background.assert_called_once()

    def test_when_body_invalid_for_step_then_returns_422(
        self, client: TestClient, mock_job_svc: MagicMock
    ) -> None:
        with (
            patch(
                "app.api.v1.pipeline_builder.steps.get_session",
                return_value=_session("abc"),
            ),
            patch(
                "app.api.v1.pipeline_builder.steps.get_job_service",
                return_value=mock_job_svc,
            ),
        ):
            response = client.post(
                "/api/v1/pipeline-builder/sessions/abc/steps/screen",
                json={"preset": "bogus"},
            )
        assert response.status_code == 422


class TestGetStep:
    def test_when_invalid_step_id_then_returns_400(self, client: TestClient) -> None:
        response = client.get("/api/v1/pipeline-builder/sessions/abc/steps/not_a_step")
        assert response.status_code == 400

    def test_when_session_missing_then_returns_404(self, client: TestClient) -> None:
        with patch("app.api.v1.pipeline_builder.steps.get_session", return_value=None):
            response = client.get("/api/v1/pipeline-builder/sessions/abc/steps/load")
        assert response.status_code == 404

    def test_when_step_status_pending_then_returns_not_started(
        self, client: TestClient
    ) -> None:
        with patch(
            "app.api.v1.pipeline_builder.steps.get_session",
            return_value=_session("abc"),
        ):
            response = client.get("/api/v1/pipeline-builder/sessions/abc/steps/load")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "not_started"
        assert body["progress"] == {}
        assert body["result"] is None
        assert body["error"] is None

    def test_when_job_running_then_returns_step_poll_response(
        self, client: TestClient, mock_job_svc: MagicMock
    ) -> None:
        mock_job_svc.get_job.return_value = {
            "job_id": "j1",
            "status": "running",
            "current": 3,
            "total": 10,
            "result": None,
            "error": None,
        }
        sess = _session("abc", load="j1")
        with (
            patch("app.api.v1.pipeline_builder.steps.get_session", return_value=sess),
            patch(
                "app.api.v1.pipeline_builder.steps.get_job_service",
                return_value=mock_job_svc,
            ),
        ):
            response = client.get("/api/v1/pipeline-builder/sessions/abc/steps/load")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "running"
        assert body["progress"]["current"] == 3
        assert body["progress"]["total"] == 10

    def test_when_job_completed_with_gate_reason_then_surfaced(
        self, client: TestClient, mock_job_svc: MagicMock
    ) -> None:
        mock_job_svc.get_job.return_value = {
            "job_id": "j1",
            "status": "failed",
            "current": 0,
            "total": 0,
            "result": None,
            "error": None,
            "gate_reason": "Drift below threshold",
        }
        sess = _session("abc", rebalance_decision="j1")
        with (
            patch("app.api.v1.pipeline_builder.steps.get_session", return_value=sess),
            patch(
                "app.api.v1.pipeline_builder.steps.get_job_service",
                return_value=mock_job_svc,
            ),
        ):
            response = client.get(
                "/api/v1/pipeline-builder/sessions/abc/steps/rebalance_decision"
            )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed"
        assert body["gateReason"] == "Drift below threshold"


class TestRunStepWorker:
    def test_when_handler_succeeds_then_status_completed_and_session_updated(
        self,
    ) -> None:
        from app.api.v1.pipeline_builder.steps import _run_step

        svc = MagicMock()
        sess = _session("abc")
        handler = MagicMock(return_value={"weights": {"AAPL": 0.5}})
        with (
            patch("app.api.v1.pipeline_builder.steps.get_session", return_value=sess),
            patch(
                "app.api.v1.pipeline_builder.steps.get_job_service",
                return_value=svc,
            ),
            patch(
                "app.api.v1.pipeline_builder.steps._STEP_HANDLER_MAP",
                {"load": (MagicMock, handler)},
            ),
            patch("app.api.v1.pipeline_builder.steps.update_session") as mock_us,
        ):
            _run_step("abc", "load", "job1", {})

        handler.assert_called_once()
        # Final terminal update
        update_calls = svc.update_job.call_args_list
        terminal = update_calls[-1].kwargs
        assert terminal["status"] == "completed"
        assert terminal["result"] == {"weights": {"AAPL": 0.5}}
        # Session step_status updated to completed
        last_us = mock_us.call_args.kwargs["step_status"]
        assert last_us["load"] == "completed"

    def test_when_handler_raises_runtime_error_then_gate_reason_set(self) -> None:
        from app.api.v1.pipeline_builder.steps import _run_step

        svc = MagicMock()
        sess = _session("abc")
        handler = MagicMock(side_effect=RuntimeError("Drift below 5%"))
        with (
            patch("app.api.v1.pipeline_builder.steps.get_session", return_value=sess),
            patch(
                "app.api.v1.pipeline_builder.steps.get_job_service",
                return_value=svc,
            ),
            patch(
                "app.api.v1.pipeline_builder.steps._STEP_HANDLER_MAP",
                {"rebalance_decision": (MagicMock, handler)},
            ),
        ):
            _run_step("abc", "rebalance_decision", "job1", {})

        terminal = svc.update_job.call_args_list[-1].kwargs
        assert terminal["status"] == "failed"
        assert terminal["gate_reason"] == "Drift below 5%"

    def test_when_handler_raises_factor_coverage_then_error_set(self) -> None:
        from app.api.v1.pipeline_builder.steps import _run_step

        svc = MagicMock()
        sess = _session("abc")
        handler = MagicMock(side_effect=FactorCoverageError("1 < 2"))
        with (
            patch("app.api.v1.pipeline_builder.steps.get_session", return_value=sess),
            patch(
                "app.api.v1.pipeline_builder.steps.get_job_service",
                return_value=svc,
            ),
            patch(
                "app.api.v1.pipeline_builder.steps._STEP_HANDLER_MAP",
                {"coverage_gate": (MagicMock, handler)},
            ),
        ):
            _run_step("abc", "coverage_gate", "job1", {})

        terminal = svc.update_job.call_args_list[-1].kwargs
        assert terminal["status"] == "failed"
        assert "1 < 2" in terminal["error"]


class TestHandlerMap:
    def test_when_map_loaded_then_all_step_ids_present(self) -> None:
        from app.api.v1.pipeline_builder.steps import _STEP_HANDLER_MAP
        from app.services.pipeline_builder.session_store import STEP_IDS

        assert set(_STEP_HANDLER_MAP.keys()) == set(STEP_IDS)
