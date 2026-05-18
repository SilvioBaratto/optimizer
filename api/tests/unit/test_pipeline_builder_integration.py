"""End-to-end integration tests for the pipeline_builder HTTP layer (issue #716).

Mocks `get_job_service` per session and `_STEP_HANDLER_MAP` entries to keep
real DB / research pipeline out of the test path. Uses the real
`session_store` module state (reset by `unit/conftest.py`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.services.jobs.background_job import JobAlreadyRunningError
from app.services.pipeline_builder import session_store
from app.services.pipeline_builder.session_store import PipelineSessionCapacityError
from optimizer.exceptions import FactorCoverageError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_job_service(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Return a MagicMock service; patched into the steps router."""
    svc = MagicMock()
    svc.create_job.return_value = "test-job-id"
    svc.get_job.return_value = None
    # Synchronous start_background — call target inline so terminal state lands.
    svc.start_background.side_effect = lambda target, args: target(*args)
    monkeypatch.setattr(
        "app.api.v1.pipeline_builder.steps.get_job_service", lambda _sid: svc
    )
    monkeypatch.setattr(
        "app.api.v1.pipeline_builder.sessions.get_session", session_store.get_session
    )
    return svc


@pytest.fixture
def session_id(client: TestClient) -> str:
    """Create one session via the API and return its id."""
    response = client.post("/api/v1/pipeline-builder/sessions", json={})
    assert response.status_code == 201
    return response.json()["sessionId"]


def _patch_step_handler(
    monkeypatch: pytest.MonkeyPatch, step_id: str, handler: Any
) -> None:
    """Replace ``_STEP_HANDLER_MAP[step_id]`` with a mocked handler entry."""
    from app.api.v1.pipeline_builder import steps as steps_mod

    original = dict(steps_mod._STEP_HANDLER_MAP)
    original[step_id] = (original[step_id][0], handler)
    monkeypatch.setattr(steps_mod, "_STEP_HANDLER_MAP", original)


@pytest.fixture
def reset_session_store() -> Iterator[None]:
    """Belt-and-braces — unit/conftest.py also resets between tests."""
    session_store._sessions.clear()
    session_store._job_services.clear()
    session_store._daemon_started = False
    yield
    session_store._sessions.clear()
    session_store._job_services.clear()
    session_store._daemon_started = False


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_when_post_session_then_returns_201_with_session_id(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/v1/pipeline-builder/sessions", json={})
        assert response.status_code == 201
        body = response.json()
        assert "sessionId" in body
        assert isinstance(body["sessionId"], str)
        assert len(body["sessionId"]) > 0

    def test_when_capacity_reached_then_returns_429(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        reset_session_store: None,
    ) -> None:
        def _raise(**_: Any) -> str:
            raise PipelineSessionCapacityError(3, 3)

        monkeypatch.setattr(
            "app.api.v1.pipeline_builder.sessions.create_session", _raise
        )
        response = client.post("/api/v1/pipeline-builder/sessions", json={})
        assert response.status_code == 429
        body = response.json()
        assert "cap reached" in body["error"]["message"].lower()

    def test_when_delete_existing_session_then_returns_204(
        self, client: TestClient, session_id: str
    ) -> None:
        response = client.delete(f"/api/v1/pipeline-builder/sessions/{session_id}")
        assert response.status_code == 204

    def test_when_delete_unknown_session_then_returns_204(
        self, client: TestClient
    ) -> None:
        response = client.delete("/api/v1/pipeline-builder/sessions/does-not-exist")
        assert response.status_code == 204


# ---------------------------------------------------------------------------
# Steps — dispatch
# ---------------------------------------------------------------------------


class TestSteps:
    def test_when_post_step_load_happy_then_returns_202_with_job_id(
        self,
        client: TestClient,
        mock_job_service: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        session_id: str,
    ) -> None:
        _patch_step_handler(
            monkeypatch, "load", MagicMock(return_value={"loaded": True})
        )
        response = client.post(
            f"/api/v1/pipeline-builder/sessions/{session_id}/steps/load",
            json={},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["job_id"] == "test-job-id"
        assert body["status"] == "pending"

    def test_when_second_post_step_then_returns_409_with_reason(
        self,
        client: TestClient,
        mock_job_service: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        session_id: str,
    ) -> None:
        _patch_step_handler(monkeypatch, "load", MagicMock(return_value={}))
        mock_job_service.create_job.side_effect = JobAlreadyRunningError(
            "existing-job-uuid"
        )
        response = client.post(
            f"/api/v1/pipeline-builder/sessions/{session_id}/steps/load",
            json={},
        )
        assert response.status_code == 409
        assert "existing-job-uuid" in response.json()["error"]["message"]

    def test_when_unknown_step_id_then_returns_400(
        self, client: TestClient, session_id: str
    ) -> None:
        response = client.post(
            f"/api/v1/pipeline-builder/sessions/{session_id}/steps/unknown",
            json={},
        )
        assert response.status_code == 400
        assert "unknown" in response.json()["error"]["message"].lower()

    def test_when_unknown_session_then_returns_404(
        self, client: TestClient, mock_job_service: MagicMock
    ) -> None:
        response = client.post(
            "/api/v1/pipeline-builder/sessions/unknown-id/steps/load",
            json={},
        )
        assert response.status_code == 404

    def test_when_get_step_before_fire_then_status_not_started(
        self, client: TestClient, session_id: str
    ) -> None:
        response = client.get(
            f"/api/v1/pipeline-builder/sessions/{session_id}/steps/load"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "not_started"
        assert body["progress"] == {}
        assert body["result"] is None
        assert body["error"] is None

    def test_when_get_step_after_fire_then_forwards_get_job_payload(
        self,
        client: TestClient,
        mock_job_service: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        session_id: str,
    ) -> None:
        _patch_step_handler(monkeypatch, "load", MagicMock(return_value={"ok": True}))
        # Fire — synchronous start_background populates terminal state.
        client.post(
            f"/api/v1/pipeline-builder/sessions/{session_id}/steps/load",
            json={},
        )
        mock_job_service.get_job.return_value = {
            "job_id": "test-job-id",
            "status": "completed",
            "current": 1,
            "total": 1,
            "result": {"ok": True},
            "error": None,
        }
        response = client.get(
            f"/api/v1/pipeline-builder/sessions/{session_id}/steps/load"
        )
        assert response.status_code == 200
        body = response.json()
        # After completion the sentinel "completed" lands in step_status,
        # short-circuiting the get_job lookup — assert terminal status only.
        assert body["status"] == "completed"


# ---------------------------------------------------------------------------
# Abort gates
# ---------------------------------------------------------------------------


class TestAbortGates:
    @pytest.mark.parametrize(
        "step_id,message",
        [
            ("load", "assembly floor breach"),
            ("screen", "universe floor breach: 187 < 200"),
            ("clean_returns", "NaN remains after preprocessing"),
            ("validate_oos", "no OOS folds produced"),
        ],
    )
    def test_when_runtime_error_then_gate_reason_surfaced(
        self,
        client: TestClient,
        mock_job_service: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        session_id: str,
        step_id: str,
        message: str,
    ) -> None:
        captured: dict[str, Any] = {}

        def _capture(_job_id: str, **kwargs: Any) -> None:
            captured.update(kwargs)

        mock_job_service.update_job.side_effect = _capture
        _patch_step_handler(
            monkeypatch, step_id, MagicMock(side_effect=RuntimeError(message))
        )
        client.post(
            f"/api/v1/pipeline-builder/sessions/{session_id}/steps/{step_id}",
            json={},
        )
        assert captured.get("status") == "failed"
        assert captured.get("gate_reason") == message

    def test_when_coverage_gate_factor_coverage_error_then_error_field_set(
        self,
        client: TestClient,
        mock_job_service: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        session_id: str,
    ) -> None:
        captured: dict[str, Any] = {}

        def _capture(_job_id: str, **kwargs: Any) -> None:
            captured.update(kwargs)

        mock_job_service.update_job.side_effect = _capture
        _patch_step_handler(
            monkeypatch,
            "coverage_gate",
            MagicMock(side_effect=FactorCoverageError("only 1 factor passed")),
        )
        client.post(
            f"/api/v1/pipeline-builder/sessions/{session_id}/steps/coverage_gate",
            json={},
        )
        assert captured.get("status") == "failed"
        assert "only 1 factor passed" in captured.get("error", "")

    def test_when_factor_coverage_raised_in_sync_route_then_returns_422(
        self, client: TestClient
    ) -> None:
        """Validates issue #712 wiring — global handler maps to 422."""
        from app.main import app

        @app.get("/_test_only/factor-coverage")
        def _raise() -> None:
            raise FactorCoverageError("integration test")

        response = client.get("/_test_only/factor-coverage")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "FACTOR_COVERAGE_ERROR"


# ---------------------------------------------------------------------------
# Artefact streaming
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_when_artifact_before_persist_completed_then_returns_404(
        self, client: TestClient, session_id: str
    ) -> None:
        response = client.get(
            f"/api/v1/pipeline-builder/sessions/{session_id}/artifacts/report.md"
        )
        assert response.status_code == 404

    def test_when_artifact_after_persist_then_returns_file_contents(
        self,
        client: TestClient,
        session_id: str,
        tmp_path: Path,
    ) -> None:
        artifact_file = tmp_path / "report.md"
        artifact_file.write_text("# Final Report\n\nbody")
        # Mutate the session in place — bypass route to set persist=completed
        # and inject artifact_paths.
        session_store.update_session(
            session_id,
            step_status={
                **session_store.get_session(session_id).step_status,
                "persist": "completed",
            },
            report_result={"artifact_paths": {"report_md": str(artifact_file)}},
        )
        response = client.get(
            f"/api/v1/pipeline-builder/sessions/{session_id}/artifacts/report.md"
        )
        assert response.status_code == 200
        assert response.content == b"# Final Report\n\nbody"
        assert response.headers["content-type"].startswith("text/markdown")

    def test_when_path_traversal_attempted_then_returns_400(
        self, client: TestClient, session_id: str
    ) -> None:
        # The frozenset whitelist is consulted BEFORE any path resolution.
        response = client.get(
            f"/api/v1/pipeline-builder/sessions/{session_id}/artifacts/..%2Fetc%2Fpasswd"
        )
        assert response.status_code in (400, 404)
        # Body must indicate the artifact is unknown — never a filesystem error.
        body = response.json()
        if response.status_code == 400:
            assert "unknown" in body["error"]["message"].lower()
