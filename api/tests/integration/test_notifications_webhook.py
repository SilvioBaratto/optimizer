"""Integration tests: webhook fires on background-job failure across routers.

Covers the four Cycle-2 execution routers:
  - /api/v1/optimize             → app.api.v1.optimize._job_service
  - /api/v1/backtest             → app.api.v1.backtest._job_service
  - /api/v1/validate/cross-val.. → app.api.v1.validate._job_service
  - /api/v1/tune                 → app.api.v1.tune._tune_job_service

Strategy
--------
Each router instantiates a module-level ``BackgroundJobService`` whose
``update_job(status="failed", error=...)`` call triggers the webhook via
``settings.notification_webhook_url`` → ``notifications.notify_failure``
→ ``httpx.post``. These tests:

1. Patch ``settings.notification_webhook_url`` to a test URL (monkeypatch).
2. Patch ``settings.enable_metrics`` off to avoid Prometheus side-effects.
3. Replace each service's ``_session_factory`` with a stub so ``update_job``
   does not touch PostgreSQL (the module-level service bypasses FastAPI DI
   per CLAUDE.md, so the SQLite test session does not reach it).
4. Capture ``httpx.post`` at ``app.services.notifications.httpx.post``
   (the import site used by the notifications module).
5. Exercise both terminal statuses (``failed`` and ``completed``) and assert
   the webhook fires exactly when it should.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.api.v1.backtest import _job_service as backtest_job_service
from app.api.v1.optimize import _job_service as optimize_job_service
from app.api.v1.tune import _tune_job_service
from app.api.v1.validate import _job_service as validate_job_service
from app.config import settings
from app.services.background_job import BackgroundJobService

_TEST_WEBHOOK_URL = "https://discord.com/api/webhooks/test"

_HTTPX_POST_PATH = "app.services.notifications.httpx.post"


# ---------------------------------------------------------------------------
# Router fixtures
# ---------------------------------------------------------------------------


ROUTER_SERVICES: list[tuple[str, BackgroundJobService]] = [
    ("optimize", optimize_job_service),
    ("backtest", backtest_job_service),
    ("validate", validate_job_service),
    ("tune", _tune_job_service),
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route webhook dispatch to the test URL and silence Prometheus calls."""
    monkeypatch.setattr(settings, "notification_webhook_url", _TEST_WEBHOOK_URL)
    monkeypatch.setattr(settings, "enable_metrics", False)


@pytest.fixture
def stub_session_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap each service's ``_session_factory`` for a DB-less stub."""

    @contextmanager
    def _noop_session() -> Any:
        session = MagicMock()
        yield session

    for _, service in ROUTER_SERVICES:
        monkeypatch.setattr(service, "_session_factory", _noop_session)

    # Prevent the repository's .update from running against the MagicMock.
    monkeypatch.setattr(
        "app.repositories.background_job_repository.BackgroundJobRepository.update",
        lambda self, *args, **kwargs: None,
    )


# ===========================================================================
# Failure path: webhook is dispatched
# ===========================================================================


class TestWebhookFiresOnJobFailure:
    """Each of the four router services posts to the webhook on ``failed``."""

    @pytest.mark.parametrize("job_type,service", ROUTER_SERVICES)
    def test_posts_to_configured_webhook_url(
        self,
        job_type: str,
        service: BackgroundJobService,
        stub_session_factories: None,
    ) -> None:
        job_id = str(uuid.uuid4())

        with patch(_HTTPX_POST_PATH) as mock_post:
            service.update_job(job_id, status="failed", error="boom")

        mock_post.assert_called_once()
        (url,) = mock_post.call_args.args
        assert url == _TEST_WEBHOOK_URL

    @pytest.mark.parametrize("job_type,service", ROUTER_SERVICES)
    def test_payload_contains_job_type_and_job_id(
        self,
        job_type: str,
        service: BackgroundJobService,
        stub_session_factories: None,
    ) -> None:
        job_id = str(uuid.uuid4())

        with patch(_HTTPX_POST_PATH) as mock_post:
            service.update_job(job_id, status="failed", error="boom")

        payload = mock_post.call_args.kwargs["json"]
        assert job_type in payload["content"]
        assert job_id[:8] in payload["content"]

    @pytest.mark.parametrize("job_type,service", ROUTER_SERVICES)
    def test_payload_contains_error_message(
        self,
        job_type: str,
        service: BackgroundJobService,
        stub_session_factories: None,
    ) -> None:
        job_id = str(uuid.uuid4())
        error_text = f"{job_type}-specific failure"

        with patch(_HTTPX_POST_PATH) as mock_post:
            service.update_job(job_id, status="failed", error=error_text)

        payload = mock_post.call_args.kwargs["json"]
        assert error_text in payload["content"]


# ===========================================================================
# Success path: webhook stays silent
# ===========================================================================


class TestWebhookSilentOnJobSuccess:
    """No webhook is dispatched when a job completes successfully."""

    @pytest.mark.parametrize("job_type,service", ROUTER_SERVICES)
    def test_completed_status_does_not_post(
        self,
        job_type: str,
        service: BackgroundJobService,
        stub_session_factories: None,
    ) -> None:
        job_id = str(uuid.uuid4())

        with patch(_HTTPX_POST_PATH) as mock_post:
            service.update_job(job_id, status="completed")

        mock_post.assert_not_called()

    @pytest.mark.parametrize("job_type,service", ROUTER_SERVICES)
    def test_running_status_does_not_post(
        self,
        job_type: str,
        service: BackgroundJobService,
        stub_session_factories: None,
    ) -> None:
        job_id = str(uuid.uuid4())

        with patch(_HTTPX_POST_PATH) as mock_post:
            service.update_job(job_id, status="running")

        mock_post.assert_not_called()


# ===========================================================================
# Webhook configuration is respected
# ===========================================================================


class TestWebhookDisabledWhenUnconfigured:
    """When ``NOTIFICATION_WEBHOOK_URL`` is unset, failure does not post."""

    @pytest.mark.parametrize("job_type,service", ROUTER_SERVICES)
    def test_unset_webhook_url_suppresses_post_on_failure(
        self,
        job_type: str,
        service: BackgroundJobService,
        stub_session_factories: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "notification_webhook_url", None)
        job_id = str(uuid.uuid4())

        with patch(_HTTPX_POST_PATH) as mock_post:
            service.update_job(job_id, status="failed", error="boom")

        mock_post.assert_not_called()
