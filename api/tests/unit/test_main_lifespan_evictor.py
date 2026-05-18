"""Tests for FastAPI lifespan eviction daemon wiring (issue #693)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from app import main as main_module


@pytest.fixture(autouse=True)
def _reset_sentinel():
    main_module._reconciled_this_process = False
    yield
    main_module._reconciled_this_process = False


@contextmanager
def _patched_startup(ensure_exc: Exception | None = None):
    fake_session = MagicMock()

    @contextmanager
    def fake_get_session():
        yield fake_session

    scheduler_mock = MagicMock()
    ensure_mock = MagicMock()
    if ensure_exc is not None:
        ensure_mock.side_effect = ensure_exc

    with (
        patch.object(main_module, "init_db"),
        patch.object(main_module.database_manager, "health_check", return_value=True),
        patch.object(
            main_module.database_manager, "get_session", side_effect=fake_get_session
        ),
        patch.object(main_module, "create_scheduler", return_value=scheduler_mock),
        patch(
            "app.services._shared._benchmark_bootstrap.bootstrap_benchmarks",
            return_value=None,
        ),
        patch(
            "app.repositories.jobs.background_job_repository.BackgroundJobRepository",
            return_value=MagicMock(reconcile_orphans=MagicMock(return_value=0)),
        ),
        patch(
            "app.services.pipeline_builder.session_store._ensure_daemon_running",
            ensure_mock,
        ),
    ):
        yield {"ensure": ensure_mock, "scheduler": scheduler_mock}


@pytest.mark.asyncio
async def test_when_lifespan_starts_then_ensure_daemon_running_invoked() -> None:
    app = FastAPI()
    with _patched_startup() as m:
        async with main_module.lifespan(app):
            pass
    m["ensure"].assert_called_once()


@pytest.mark.asyncio
async def test_when_lifespan_starts_then_ensure_daemon_called_before_scheduler_start() -> None:
    app = FastAPI()
    call_order: list[str] = []

    with _patched_startup() as m:
        m["ensure"].side_effect = lambda: call_order.append("ensure")
        m["scheduler"].start.side_effect = lambda: call_order.append("scheduler_start")

        async with main_module.lifespan(app):
            pass

    assert call_order.index("ensure") < call_order.index("scheduler_start")


@pytest.mark.asyncio
async def test_when_ensure_daemon_raises_then_startup_continues_and_scheduler_starts() -> None:
    app = FastAPI()
    boom = RuntimeError("evictor-boom")
    with _patched_startup(ensure_exc=boom) as m:
        async with main_module.lifespan(app):
            pass
    m["ensure"].assert_called_once()
    m["scheduler"].start.assert_called_once()


@pytest.mark.asyncio
async def test_when_ensure_daemon_raises_then_error_logged_with_exc_info(caplog) -> None:
    app = FastAPI()
    boom = RuntimeError("evictor-boom-log")
    with caplog.at_level(logging.ERROR, logger=main_module.__name__):
        with _patched_startup(ensure_exc=boom):
            async with main_module.lifespan(app):
                pass

    matching = [
        r for r in caplog.records
        if r.levelno == logging.ERROR and r.exc_info is not None
    ]
    assert matching, "expected an ERROR log with exc_info on daemon-start failure"


@pytest.mark.asyncio
async def test_when_daemon_starts_then_info_log_emitted(caplog) -> None:
    app = FastAPI()
    with caplog.at_level(logging.INFO, logger=main_module.__name__):
        with _patched_startup():
            async with main_module.lifespan(app):
                pass

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        m == "Pipeline session-store eviction daemon started" for m in msgs
    )
