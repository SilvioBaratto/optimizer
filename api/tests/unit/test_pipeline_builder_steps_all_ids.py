"""Parametrized step-route coverage across ALL 13 step IDs (issue #819).

Gap-fill for ``test_pipeline_builder_steps.py``, which spot-checks the
POST/GET contract on a single step only.  This module sweeps every entry in
``STEP_IDS`` so a newly-added or renamed step cannot silently lose route
coverage.  The worker error branches (FactorCoverageError, RuntimeError →
gate_reason), 400/404/409/422 paths, job_type/caching, and TTL eviction are
already covered elsewhere and are NOT duplicated here.

Lives under ``tests/unit/`` so the autouse ``_reset_pipeline_builder_state``
fixture clears the module-level ``_sessions`` / ``_job_services`` registries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.pipeline_builder import pipeline_builder_router
from app.schemas.pipeline_builder import StepPollResponse, StepRunResponse
from app.services.pipeline_builder.session_store import STEP_IDS, _PipelineSession
from tests._fixtures.assertions import assert_response_matches_schema

_STEPS = "app.api.v1.pipeline_builder.steps"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(pipeline_builder_router, prefix="/api/v1")
    return TestClient(app)


@pytest.fixture
def mock_job_svc() -> MagicMock:
    svc = MagicMock()
    svc.create_job.return_value = "job-uuid-1"
    svc.start_background.return_value = (MagicMock(), MagicMock())
    svc.get_job.return_value = None
    return svc


def _session(
    sid: str,
    *,
    step_status: dict[str, str] | None = None,
    step_results: dict[str, Any] | None = None,
) -> _PipelineSession:
    base = dict.fromkeys(STEP_IDS, "pending")
    if step_status:
        base.update(step_status)
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
        step_status=base,
        step_results=step_results or {},
    )


def _url(sid: str, step_id: str) -> str:
    return f"/api/v1/pipeline-builder/sessions/{sid}/steps/{step_id}"


# ---------------------------------------------------------------------------
# POST — every step dispatches with an empty body (all request models default)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_when_step_posted_then_202_with_job_id(
    step_id: str, client: TestClient, mock_job_svc: MagicMock
) -> None:
    with (
        patch(f"{_STEPS}.get_session", return_value=_session("sid")),
        patch(f"{_STEPS}.get_job_service", return_value=mock_job_svc),
        patch(f"{_STEPS}.update_session"),
    ):
        resp = client.post(_url("sid", step_id), json={})

    assert resp.status_code == 202
    body = resp.json()
    assert_response_matches_schema(body, StepRunResponse)
    assert body["job_id"] == "job-uuid-1"
    assert body["status"] == "pending"
    mock_job_svc.start_background.assert_called_once()


# ---------------------------------------------------------------------------
# GET — not_started before any job, completed (with result) after
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_when_step_pending_then_get_returns_not_started(
    step_id: str, client: TestClient
) -> None:
    with patch(f"{_STEPS}.get_session", return_value=_session("sid")):
        resp = client.get(_url("sid", step_id))

    assert resp.status_code == 200
    body = resp.json()
    assert_response_matches_schema(body, StepPollResponse)
    assert body["status"] == "not_started"


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_when_step_completed_then_get_returns_result(
    step_id: str, client: TestClient
) -> None:
    result = {"ok": True, "step": step_id}
    session = _session(
        "sid",
        step_status={step_id: "completed"},
        step_results={step_id: result},
    )
    with patch(f"{_STEPS}.get_session", return_value=session):
        resp = client.get(_url("sid", step_id))

    assert resp.status_code == 200
    body = resp.json()
    assert_response_matches_schema(body, StepPollResponse)
    assert body["status"] == "completed"
    assert body["result"] == result
