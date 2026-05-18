"""Unit tests for app-level exception handlers (issue #712)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.exceptions import setup_exception_handlers
from optimizer.exceptions import FactorCoverageError


@pytest.fixture
def app_with_handlers() -> FastAPI:
    """Minimal FastAPI app with exception handlers wired up."""
    app = FastAPI()
    setup_exception_handlers(app)

    @app.get("/_test/factor-coverage")
    def _raise_coverage() -> None:
        raise FactorCoverageError("IS ∩ OOS factor set below min_factors (1 < 2)")

    return app


class TestFactorCoverageHandler:
    def test_when_factor_coverage_error_raised_then_returns_422(
        self, app_with_handlers: FastAPI
    ) -> None:
        client = TestClient(app_with_handlers, raise_server_exceptions=False)
        response = client.get("/_test/factor-coverage")
        assert response.status_code == 422

    def test_when_factor_coverage_error_raised_then_body_has_structured_error(
        self, app_with_handlers: FastAPI
    ) -> None:
        client = TestClient(app_with_handlers, raise_server_exceptions=False)
        response = client.get("/_test/factor-coverage")
        body = response.json()
        assert body["error"]["code"] == "FACTOR_COVERAGE_ERROR"
        assert "IS ∩ OOS factor set below min_factors" in body["error"]["message"]
