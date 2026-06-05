"""Route tests for macro-regime background-job endpoints (issue #813).

Covers the four fire-and-poll job pairs and the single-country sync endpoint
in ``app.api.v1.macro.macro_regime``.  The four module-level
``BackgroundJobService`` instances are mocked via ``mock_job_service`` (which
patches ``create_job`` / ``get_job`` / ``start_background`` on the live
instance).  Because ``start_background`` is mocked, no daemon worker launches,
so the bulk runners (``run_bulk_macro_fetch`` etc.) never execute real work —
patching them is unnecessary.

Read-endpoint coverage lives in ``tests/unit/test_macro_regime_routes.py`` and
is left untouched.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.schemas.macro.macro_regime import (
    MacroFetchJobResponse,
    MacroFetchProgress,
    MacroNewsSummarizeJobResponse,
    MacroNewsSummarizeProgress,
)
from tests._fixtures.assertions import assert_json_safe, assert_response_matches_schema
from tests._fixtures.job_service_mock import mock_job_service

BASE_URL = "/api/v1/macro-data"
_MODULE = "app.api.v1.macro.macro_regime"
_SERVICE = f"{_MODULE}.MacroRegimeService"


class _JobCase(BaseModel):
    """One fire-and-poll job endpoint pair."""

    path: str  # shared POST + poll prefix, e.g. "/fetch"
    attr: str  # module-level instance name, e.g. "_job_service"
    create_model: type[BaseModel]
    progress_model: type[BaseModel]


JOB_CASES = [
    _JobCase(
        path="/fetch",
        attr="_job_service",
        create_model=MacroFetchJobResponse,
        progress_model=MacroFetchProgress,
    ),
    _JobCase(
        path="/fred/fetch",
        attr="_fred_job_service",
        create_model=MacroFetchJobResponse,
        progress_model=MacroFetchProgress,
    ),
    _JobCase(
        path="/news/fetch",
        attr="_news_job_service",
        create_model=MacroFetchJobResponse,
        progress_model=MacroFetchProgress,
    ),
    _JobCase(
        path="/news/summarize",
        attr="_summarize_job_service",
        create_model=MacroNewsSummarizeJobResponse,
        progress_model=MacroNewsSummarizeProgress,
    ),
]
_IDS = [c.attr for c in JOB_CASES]


def _module_path(case: _JobCase) -> str:
    return f"{_MODULE}.{case.attr}"


# ---------------------------------------------------------------------------
# POST job-start endpoints
# ---------------------------------------------------------------------------


class TestJobStart:
    @pytest.mark.parametrize("case", JOB_CASES, ids=_IDS)
    def test_when_started_then_202_pending_job_id(
        self, case: _JobCase, client: TestClient
    ) -> None:
        with mock_job_service(_module_path(case), job_id="jid-1") as job:
            resp = client.post(f"{BASE_URL}{case.path}")

        assert resp.status_code == 202
        body = resp.json()
        assert_response_matches_schema(body, case.create_model)
        assert body["job_id"] == "jid-1"
        assert body["status"] == "pending"
        job.start_background.assert_called_once()

    @pytest.mark.parametrize("case", JOB_CASES, ids=_IDS)
    def test_when_already_running_then_409(
        self, case: _JobCase, client: TestClient
    ) -> None:
        with mock_job_service(_module_path(case), raise_already_running=True):
            resp = client.post(f"{BASE_URL}{case.path}")

        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET poll endpoints
# ---------------------------------------------------------------------------


class TestJobPoll:
    @pytest.mark.parametrize("case", JOB_CASES, ids=_IDS)
    def test_when_job_pending_then_200_with_progress(
        self, case: _JobCase, client: TestClient
    ) -> None:
        with mock_job_service(
            _module_path(case), job_id="jid-2", initial_status="pending"
        ):
            resp = client.get(f"{BASE_URL}{case.path}/jid-2")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, case.progress_model)
        assert body["job_id"] == "jid-2"
        assert body["status"] == "pending"
        assert "current" in body
        assert "total" in body
        assert_json_safe(body)

    @pytest.mark.parametrize("case", JOB_CASES, ids=_IDS)
    def test_when_job_unknown_then_404(
        self, case: _JobCase, client: TestClient
    ) -> None:
        with patch(f"{_module_path(case)}.get_job", return_value=None):
            resp = client.get(f"{BASE_URL}{case.path}/ghost")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Single-country sync fetch
# ---------------------------------------------------------------------------


class TestSingleCountryFetch:
    def test_when_fetch_succeeds_then_200_with_counts_and_errors(
        self, client: TestClient
    ) -> None:
        result = {"counts": {"economic_indicators": 3}, "errors": []}
        with patch(_SERVICE) as MockSvc:
            MockSvc.return_value.fetch_country.return_value = result
            resp = client.post(f"{BASE_URL}/fetch/USA")

        assert resp.status_code == 200
        body = resp.json()
        assert body["country"] == "USA"
        assert body["counts"] == {"economic_indicators": 3}
        assert body["errors"] == []
        assert_json_safe(body)

    def test_when_service_raises_then_error_surfaced_not_dropped(
        self, client: TestClient
    ) -> None:
        # The route has no try/except: a service failure propagates instead of
        # being swallowed into a misleading 200.  TestClient re-raises server
        # errors, so the surfacing manifests as the exception escaping.
        with patch(_SERVICE) as MockSvc:
            MockSvc.return_value.fetch_country.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                client.post(f"{BASE_URL}/fetch/USA")
