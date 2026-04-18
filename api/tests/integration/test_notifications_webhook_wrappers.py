"""Integration tests: webhook fires when a route's job-wrapper raises.

Sibling to ``test_notifications_webhook.py`` (issue #443).  That file
exercises ``BackgroundJobService.update_job(status="failed")`` directly.
*This* file goes one layer up: it invokes each route's actual ``_run_*_bg``
wrapper function with inputs crafted to raise inside the wrapper's
``try:`` body, then asserts the wrapper's own ``except`` block dispatches
the webhook via the real ``update_job → notifications → httpx.post`` chain.

What the wrappers look like
---------------------------

Each of the four execution routes defines a private background-worker
function that follows the same shape::

    def _run_<route>_bg(job_id, ..., request):
        _job_service.update_job(job_id, status="running")
        try:
            with database_manager.get_session() as session:
                <inner service call>(session, request, ...)
            _job_service.update_job(..., status="completed", ...)
        except Exception as exc:
            _job_service.update_job(..., status="failed", error=str(exc))
            <maybe _fail_run(...)>

We force an exception inside the inner service call; the rest of the
chain (update_job → _emit_transition → notifications.notify_failure →
httpx.post) runs for real.

Stubs in play
-------------

* ``BackgroundJobService._session_factory`` — yields a MagicMock so
  ``update_job`` does not touch PostgreSQL.
* ``BackgroundJobRepository.update`` — no-op; we don't care about
  persistence, only the webhook side-effect.
* ``database_manager.get_session`` — replaced with a no-op context
  manager so the wrapper's own session block does not blow up
  before reaching the patched inner call.
* ``app.api.v1.optimize._fail_run`` and ``app.api.v1.backtest._fail_run``
  — no-op (those helpers open a fresh session of their own *after* the
  webhook has already fired; testing them is out of scope for #443).
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from app.api.v1 import backtest as backtest_module
from app.api.v1 import optimize as optimize_module
from app.api.v1 import tune as tune_module
from app.api.v1 import validate as validate_module
from app.config import settings
from app.schemas.backtest import BacktestRequest
from app.schemas.optimization import OptimizeRequest
from app.schemas.tuning import TuneRequest
from app.schemas.validation import ValidateRequest
from app.services.background_job import BackgroundJobService

from tests.integration.test_notifications_webhook import (
    _HTTPX_POST_PATH,
    _TEST_WEBHOOK_URL,
)

_FORCED_ERROR_MESSAGE = "forced wrapper failure"


# ---------------------------------------------------------------------------
# Per-wrapper metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WrapperCase:
    """Self-contained description of one wrapper for parametrized tests."""

    job_type: str                                       # "optimize", etc.
    module: Any                                         # the route module
    wrapper: Callable[..., None]                        # _run_*_bg
    inner_attr: str                                     # name of the inner call
    fail_run_attr: str | None                           # optional _fail_run name
    service: BackgroundJobService                       # wrapper's job service
    build_args: Callable[[str], tuple[Any, ...]]        # job_id → wrapper args


def _new_run_id() -> str:
    return str(uuid.uuid4())


def _build_optimize_args(job_id: str) -> tuple[Any, ...]:
    return (
        job_id,
        _new_run_id(),
        OptimizeRequest(
            tickers=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            optimizer_type="hrp",
        ),
    )


def _build_backtest_args(job_id: str) -> tuple[Any, ...]:
    return (
        job_id,
        _new_run_id(),
        BacktestRequest(
            tickers=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        ),
    )


def _build_validate_args(job_id: str) -> tuple[Any, ...]:
    return (
        job_id,
        ValidateRequest(
            tickers=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        ),
    )


def _build_tune_args(job_id: str) -> tuple[Any, ...]:
    return (
        job_id,
        TuneRequest(
            tickers=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            optimizer_type="hrp",
            param_grid={"l2_coef": [0.1, 0.2]},
        ),
    )


WRAPPER_CASES: list[WrapperCase] = [
    WrapperCase(
        job_type="optimize",
        module=optimize_module,
        wrapper=optimize_module._run_optimize_bg,
        inner_attr="_execute_and_persist",
        fail_run_attr="_fail_run",
        service=optimize_module._job_service,
        build_args=_build_optimize_args,
    ),
    WrapperCase(
        job_type="backtest",
        module=backtest_module,
        wrapper=backtest_module._run_backtest_bg,
        inner_attr="run_and_persist",
        fail_run_attr="_fail_run",
        service=backtest_module._job_service,
        build_args=_build_backtest_args,
    ),
    WrapperCase(
        job_type="validate",
        module=validate_module,
        wrapper=validate_module._run_walk_forward_bg,
        inner_attr="run_validation",
        fail_run_attr=None,
        service=validate_module._job_service,
        build_args=_build_validate_args,
    ),
    WrapperCase(
        job_type="tune",
        module=tune_module,
        wrapper=tune_module._run_tune_bg,
        inner_attr="run_tune",
        fail_run_attr=None,
        service=tune_module._tune_job_service,
        build_args=_build_tune_args,
    ),
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
def stub_all_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every database touch-point exercised by the wrappers."""

    @contextmanager
    def _noop_session() -> Any:
        yield MagicMock()

    # 1) Each BackgroundJobService.update_job opens its own session.
    for case in WRAPPER_CASES:
        monkeypatch.setattr(case.service, "_session_factory", _noop_session)

    # 2) BackgroundJobRepository.update — make persistence a no-op.
    monkeypatch.setattr(
        "app.repositories.background_job_repository.BackgroundJobRepository.update",
        lambda self, *args, **kwargs: None,
    )

    # 3) The wrapper's own `with database_manager.get_session() as s:` block
    #    runs *before* the patched inner call.  The four route modules share
    #    the same `database_manager` singleton — patching its method here is
    #    enough to make every wrapper's session block a no-op.
    monkeypatch.setattr(
        optimize_module.database_manager, "get_session", _noop_session
    )

    # 4) Optimize and backtest call _fail_run *after* the failed update_job;
    #    those helpers open another session of their own.  No-op them.
    monkeypatch.setattr(optimize_module, "_fail_run", lambda *a, **kw: None)
    monkeypatch.setattr(backtest_module, "_fail_run", lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_inner_to_raise(case: WrapperCase) -> Any:
    """Patch the wrapper's inner call to raise ``RuntimeError(_FORCED_ERROR_MESSAGE)``."""
    return patch.object(
        case.module,
        case.inner_attr,
        side_effect=RuntimeError(_FORCED_ERROR_MESSAGE),
    )


def _patch_inner_to_succeed(case: WrapperCase) -> Any:
    """Patch the wrapper's inner call to return a benign value."""
    # MagicMock auto-provides .model_dump() for the tune wrapper's success path.
    return patch.object(case.module, case.inner_attr, return_value=MagicMock())


# ===========================================================================
# Failure path: wrapper raises → webhook fires
# ===========================================================================


class TestWrapperFailureDispatchesWebhook:
    """Each wrapper's own ``except`` block must trigger the webhook."""

    @pytest.mark.parametrize(
        "case", WRAPPER_CASES, ids=[c.job_type for c in WRAPPER_CASES]
    )
    def test_inner_exception_posts_to_webhook(
        self, case: WrapperCase, stub_all_sessions: None
    ) -> None:
        job_id = str(uuid.uuid4())

        with _patch_inner_to_raise(case), patch(_HTTPX_POST_PATH) as mock_post:
            case.wrapper(*case.build_args(job_id))

        mock_post.assert_called_once()
        (url,) = mock_post.call_args.args
        assert url == _TEST_WEBHOOK_URL

    @pytest.mark.parametrize(
        "case", WRAPPER_CASES, ids=[c.job_type for c in WRAPPER_CASES]
    )
    def test_payload_contains_job_type_and_error(
        self, case: WrapperCase, stub_all_sessions: None
    ) -> None:
        job_id = str(uuid.uuid4())

        with _patch_inner_to_raise(case), patch(_HTTPX_POST_PATH) as mock_post:
            case.wrapper(*case.build_args(job_id))

        payload = mock_post.call_args.kwargs["json"]
        assert case.job_type in payload["content"]
        assert _FORCED_ERROR_MESSAGE in payload["content"]


# ===========================================================================
# Success path: wrapper completes → webhook stays silent
# ===========================================================================


class TestWrapperSuccessKeepsWebhookSilent:
    """When the inner call returns normally the wrapper must NOT post."""

    @pytest.mark.parametrize(
        "case", WRAPPER_CASES, ids=[c.job_type for c in WRAPPER_CASES]
    )
    def test_normal_return_does_not_post(
        self, case: WrapperCase, stub_all_sessions: None
    ) -> None:
        job_id = str(uuid.uuid4())

        with _patch_inner_to_succeed(case), patch(_HTTPX_POST_PATH) as mock_post:
            case.wrapper(*case.build_args(job_id))

        mock_post.assert_not_called()
