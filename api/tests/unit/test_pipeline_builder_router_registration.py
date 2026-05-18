"""Smoke test: pipeline_builder routes registered on the main API router (issue #715)."""

from __future__ import annotations

from fastapi.testclient import TestClient

_EXPECTED_PATHS = {
    "/api/v1/pipeline-builder/sessions",
    "/api/v1/pipeline-builder/sessions/{session_id}",
    "/api/v1/pipeline-builder/sessions/{session_id}/artifacts/{name}",
    "/api/v1/pipeline-builder/sessions/{session_id}/steps/{step_id}",
}


class TestPipelineBuilderRouterRegistration:
    def test_when_openapi_loaded_then_pipeline_builder_paths_listed(
        self, client: TestClient
    ) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        for expected in _EXPECTED_PATHS:
            assert expected in paths, f"Missing route: {expected}"

    def test_when_openapi_loaded_then_post_create_session_present(
        self, client: TestClient
    ) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "post" in paths["/api/v1/pipeline-builder/sessions"]

    def test_when_openapi_loaded_then_post_and_get_steps_present(
        self, client: TestClient
    ) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        steps = paths["/api/v1/pipeline-builder/sessions/{session_id}/steps/{step_id}"]
        assert "post" in steps
        assert "get" in steps

    def test_when_openapi_loaded_then_delete_session_present(
        self, client: TestClient
    ) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "delete" in paths["/api/v1/pipeline-builder/sessions/{session_id}"]

    def test_when_openapi_loaded_then_get_artifact_present(
        self, client: TestClient
    ) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        artifact_path = (
            "/api/v1/pipeline-builder/sessions/{session_id}/artifacts/{name}"
        )
        assert "get" in paths[artifact_path]
