"""Tests for FastAPI lifespan reconcile_orphans wiring (issue #559)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from app import main as main_module


@contextmanager
def _patched_startup(reconcile_return: int = 0, reconcile_exc: Exception | None = None):
    """Patch lifespan side-effects so only reconcile_orphans is exercised."""
    repo_instance = MagicMock()
    if reconcile_exc is not None:
        repo_instance.reconcile_orphans.side_effect = reconcile_exc
    else:
        repo_instance.reconcile_orphans.return_value = reconcile_return

    fake_session = MagicMock()

    @contextmanager
    def fake_get_session():
        yield fake_session

    scheduler_mock = MagicMock()

    with patch.object(main_module, "init_db"), \
         patch.object(
             main_module.database_manager, "health_check", return_value=True
         ), \
         patch.object(
             main_module.database_manager, "get_session", side_effect=fake_get_session
         ), \
         patch.object(
             main_module, "create_scheduler", return_value=scheduler_mock
         ), \
         patch(
             "app.services._benchmark_bootstrap.bootstrap_benchmarks",
             return_value=None,
         ), \
         patch(
             "app.repositories.background_job_repository.BackgroundJobRepository",
             return_value=repo_instance,
         ) as repo_cls:
        yield {
            "repo_cls": repo_cls,
            "repo": repo_instance,
            "scheduler": scheduler_mock,
            "session": fake_session,
        }


@pytest.mark.asyncio
async def test_when_lifespan_starts_reconcile_orphans_invoked() -> None:
    app = FastAPI()
    with _patched_startup(reconcile_return=2) as m:
        async with main_module.lifespan(app):
            pass
    m["repo"].reconcile_orphans.assert_called_once_with(
        "orphaned at startup — process restarted"
    )
    m["session"].commit.assert_called()


@pytest.mark.asyncio
async def test_when_reconcile_raises_startup_continues_and_scheduler_starts() -> None:
    app = FastAPI()
    boom = RuntimeError("boom")
    with _patched_startup(reconcile_exc=boom) as m:
        async with main_module.lifespan(app):
            pass
    m["scheduler"].start.assert_called_once()


@pytest.mark.asyncio
async def test_when_reconcile_succeeds_count_logged(caplog) -> None:
    app = FastAPI()
    caplog.set_level("INFO", logger="app.main")
    with _patched_startup(reconcile_return=5):
        async with main_module.lifespan(app):
            pass
    assert any(
        "Reconciled 5 orphan job" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_when_reconcile_raises_warning_logged(caplog) -> None:
    app = FastAPI()
    caplog.set_level("WARNING", logger="app.main")
    with _patched_startup(reconcile_exc=RuntimeError("db down")):
        async with main_module.lifespan(app):
            pass
    assert any(
        "Orphan reconciliation failed" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_when_reconciliation_runs_it_runs_before_scheduler() -> None:
    """Order guarantee: reconcile_orphans completes before scheduler.start."""
    app = FastAPI()
    call_order: list[str] = []

    with _patched_startup(reconcile_return=1) as m:
        m["repo"].reconcile_orphans.side_effect = (
            lambda *a, **kw: call_order.append("reconcile") or 1
        )
        m["scheduler"].start.side_effect = lambda *a, **kw: call_order.append("scheduler")
        async with main_module.lifespan(app):
            pass

    assert call_order == ["reconcile", "scheduler"]
