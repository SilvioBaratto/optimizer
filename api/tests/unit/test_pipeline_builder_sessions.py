"""Unit tests for pipeline_builder sessions router (issue #713)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.pipeline_builder import pipeline_builder_router
from app.services.pipeline_builder.session_store import (
    PipelineSessionCapacityError,
    _PipelineSession,
)


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(pipeline_builder_router, prefix="/api/v1")
    return a


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _make_session(
    sid: str,
    *,
    persist_status: str = "pending",
    artifact_paths: dict[str, str] | None = None,
) -> _PipelineSession:
    step_status = {"persist": persist_status}
    report_result = {"artifact_paths": artifact_paths} if artifact_paths else None
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
        report_result=report_result,
        current_step="idle",
        step_status=step_status,
        step_results={},
    )


class TestCreateSession:
    def test_when_valid_run_config_then_returns_201_with_session_id(
        self, client: TestClient
    ) -> None:
        with patch(
            "app.api.v1.pipeline_builder.sessions.create_session",
            return_value="abc123",
        ) as mock_create:
            response = client.post("/api/v1/pipeline-builder/sessions", json={})
        assert response.status_code == 201
        assert response.json() == {"sessionId": "abc123"}
        mock_create.assert_called_once()

    def test_when_capacity_reached_then_returns_429(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.pipeline_builder.sessions.create_session",
            side_effect=PipelineSessionCapacityError(3, 3),
        ):
            response = client.post("/api/v1/pipeline-builder/sessions", json={})
        assert response.status_code == 429

    def test_when_invalid_n_selected_then_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/pipeline-builder/sessions", json={"n_selected": 10}
        )
        assert response.status_code == 422


class TestDeleteSession:
    def test_when_session_exists_then_returns_204(self, client: TestClient) -> None:
        with patch("app.api.v1.pipeline_builder.sessions.delete_session") as mock_del:
            response = client.delete("/api/v1/pipeline-builder/sessions/abc123")
        assert response.status_code == 204
        mock_del.assert_called_once_with("abc123")

    def test_when_session_missing_then_returns_204_idempotent(
        self, client: TestClient
    ) -> None:
        with patch("app.api.v1.pipeline_builder.sessions.delete_session"):
            response = client.delete("/api/v1/pipeline-builder/sessions/missing")
        assert response.status_code == 204


class TestArtifactDownload:
    def test_when_invalid_name_then_returns_400(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/pipeline-builder/sessions/abc/artifacts/evil.exe"
        )
        assert response.status_code == 400

    def test_when_path_traversal_attempted_then_returns_400(
        self, client: TestClient
    ) -> None:
        response = client.get(
            "/api/v1/pipeline-builder/sessions/abc/artifacts/..%2Fetc%2Fpasswd"
        )
        assert response.status_code in (400, 404)

    def test_when_session_missing_then_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.pipeline_builder.sessions.get_session", return_value=None
        ):
            response = client.get(
                "/api/v1/pipeline-builder/sessions/abc/artifacts/report.md"
            )
        assert response.status_code == 404

    def test_when_persist_not_completed_then_returns_404(
        self, client: TestClient
    ) -> None:
        session = _make_session("abc", persist_status="pending")
        with patch(
            "app.api.v1.pipeline_builder.sessions.get_session", return_value=session
        ):
            response = client.get(
                "/api/v1/pipeline-builder/sessions/abc/artifacts/report.md"
            )
        assert response.status_code == 404

    def test_when_path_missing_on_disk_then_returns_404(
        self, client: TestClient
    ) -> None:
        session = _make_session(
            "abc",
            persist_status="completed",
            artifact_paths={"report_md": "/nonexistent/path/report.md"},
        )
        with patch(
            "app.api.v1.pipeline_builder.sessions.get_session", return_value=session
        ):
            response = client.get(
                "/api/v1/pipeline-builder/sessions/abc/artifacts/report.md"
            )
        assert response.status_code == 404

    def test_when_artifact_exists_then_returns_file(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "report.md"
        artifact.write_text("# Report\n\nContents")
        session = _make_session(
            "abc",
            persist_status="completed",
            artifact_paths={"report_md": str(artifact)},
        )
        with patch(
            "app.api.v1.pipeline_builder.sessions.get_session", return_value=session
        ):
            response = client.get(
                "/api/v1/pipeline-builder/sessions/abc/artifacts/report.md"
            )
        assert response.status_code == 200
        assert response.content == b"# Report\n\nContents"
        assert response.headers["content-type"].startswith("text/markdown")
        assert 'attachment; filename="report.md"' in response.headers.get(
            "content-disposition", ""
        )

    def test_when_weights_csv_exists_then_csv_media_type(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "weights.csv"
        artifact.write_text("ticker,weight\nAAPL,0.5\n")
        session = _make_session(
            "abc",
            persist_status="completed",
            artifact_paths={"weights_csv": str(artifact)},
        )
        with patch(
            "app.api.v1.pipeline_builder.sessions.get_session", return_value=session
        ):
            response = client.get(
                "/api/v1/pipeline-builder/sessions/abc/artifacts/weights.csv"
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")

    def test_when_metrics_json_exists_then_json_media_type(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "metrics.json"
        artifact.write_text('{"sharpe": 1.5}')
        session = _make_session(
            "abc",
            persist_status="completed",
            artifact_paths={"metrics_json": str(artifact)},
        )
        with patch(
            "app.api.v1.pipeline_builder.sessions.get_session", return_value=session
        ):
            response = client.get(
                "/api/v1/pipeline-builder/sessions/abc/artifacts/metrics.json"
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
