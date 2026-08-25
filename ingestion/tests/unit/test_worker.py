"""Tests for the daemon entrypoint (``python -m app.worker``).

``main()`` blocks on a shutdown event, so every test drives it with the event
pre-set: startup runs to completion, then the wait returns immediately and
shutdown proceeds. Nothing here opens a socket, a database, or a scheduler.

The startup ordering is the contract: metrics are served before the DB is
touched (so the healthcheck answers during a slow connect), orphans are reaped
before the scheduler starts (so a reaped job's type is free when its cron
fires), and the benchmark bootstrap runs off-thread (a cold fetch takes
minutes and must not delay the first scheduled run).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app import worker as worker_module


@contextmanager
def _patched(**overrides):
    """Patch every side-effecting call in ``main()``; yield the mock namespace."""
    scheduler = overrides.get("scheduler", MagicMock())
    health = overrides.get("health_check", True)

    with (
        patch.object(worker_module, "_configure_logging"),
        patch.object(worker_module, "start_http_server") as serve,
        patch.object(worker_module, "init_db") as init,
        patch.object(worker_module, "close_db") as close,
        patch.object(worker_module, "_reconcile_orphans") as reconcile,
        patch.object(worker_module, "_install_signal_handlers"),
        patch.object(worker_module, "create_scheduler", return_value=scheduler),
        patch.object(
            worker_module.database_manager, "health_check", return_value=health
        ),
    ):
        yield MagicMock(
            serve=serve,
            init=init,
            close=close,
            reconcile=reconcile,
            scheduler=scheduler,
        )


@pytest.fixture(autouse=True)
def _preset_shutdown():
    """Make ``_shutdown.wait()`` return immediately, and reset for the next test."""
    worker_module._shutdown.set()
    yield
    worker_module._shutdown.clear()


class TestStartup:
    def test_starts_scheduler_and_shuts_it_down(self) -> None:
        with _patched() as m, patch.object(worker_module, "_force_exit") as fx:
            worker_module.main()

        m.scheduler.start.assert_called_once()
        # §5.5 hardened shutdown: pause (stop claiming) then drain (wait=True).
        m.scheduler.pause.assert_called_once()
        m.scheduler.shutdown.assert_called_once_with(wait=True)
        m.close.assert_called_once()
        fx.assert_not_called()

    def test_serves_metrics_before_touching_the_database(self) -> None:
        """The healthcheck probes /metrics — it must answer during a slow DB connect."""
        order: list[str] = []

        with _patched() as m:
            m.serve.side_effect = lambda *_a, **_k: order.append("metrics")
            m.init.side_effect = lambda *_a, **_k: order.append("init_db")
            worker_module.main()

        assert order == ["metrics", "init_db"]

    def test_reaps_orphans_before_starting_the_scheduler(self) -> None:
        """A job left 'running' by a crash would otherwise block its next cron fire."""
        order: list[str] = []

        with _patched() as m:
            m.reconcile.side_effect = lambda: order.append("reconcile")
            m.scheduler.start.side_effect = lambda: order.append("scheduler")
            worker_module.main()

        assert order == ["reconcile", "scheduler"]

    def test_continues_when_health_check_fails(self) -> None:
        """A red health check is logged, not fatal — the DB may still come up."""
        with _patched(health_check=False) as m:
            worker_module.main()

        m.scheduler.start.assert_called_once()


class TestServeMetrics:
    def test_when_metrics_disabled_then_no_server_is_started(self) -> None:
        with (
            patch.object(worker_module.settings, "enable_metrics", False),
            patch.object(worker_module, "start_http_server") as serve,
        ):
            worker_module._serve_metrics()

        serve.assert_not_called()

    def test_when_metrics_enabled_then_served_on_configured_port(self) -> None:
        with (
            patch.object(worker_module.settings, "enable_metrics", True),
            patch.object(worker_module.settings, "metrics_port", 9123),
            patch.object(worker_module, "start_http_server") as serve,
        ):
            worker_module._serve_metrics()

        serve.assert_called_once_with(9123)


class TestReconcileOrphans:
    def test_commits_and_forwards_the_timeout(self) -> None:
        repo = MagicMock()
        repo.reconcile_orphans.return_value = 2
        session = MagicMock()

        @contextmanager
        def _session_cm():
            yield session

        with (
            patch.object(
                worker_module.database_manager, "get_session", side_effect=_session_cm
            ),
            patch(
                "app.repositories.jobs.background_job_repository.BackgroundJobRepository",
                return_value=repo,
            ),
        ):
            worker_module._reconcile_orphans()

        assert repo.reconcile_orphans.call_args.kwargs["heartbeat_timeout_seconds"] == (
            worker_module.settings.scheduler_orphan_heartbeat_timeout_seconds
        )
        session.commit.assert_called_once()

    def test_failure_is_swallowed_so_the_scheduler_still_starts(self) -> None:
        with patch.object(
            worker_module.database_manager,
            "get_session",
            side_effect=RuntimeError("db down"),
        ):
            worker_module._reconcile_orphans()  # must not raise


class TestSignalHandling:
    def test_sigterm_sets_the_shutdown_event(self) -> None:
        worker_module._shutdown.clear()
        captured: dict = {}

        with patch.object(
            worker_module.signal,
            "signal",
            side_effect=lambda sig, fn: captured.__setitem__(sig, fn),
        ):
            worker_module._install_signal_handlers()

        assert not worker_module._shutdown.is_set()
        captured[worker_module.signal.SIGTERM](15, None)
        assert worker_module._shutdown.is_set()

    def test_sigint_is_also_handled(self) -> None:
        captured: dict = {}

        with patch.object(
            worker_module.signal,
            "signal",
            side_effect=lambda sig, fn: captured.__setitem__(sig, fn),
        ):
            worker_module._install_signal_handlers()

        assert worker_module.signal.SIGINT in captured


class TestGracefulShutdown:
    """§5.5 — SIGTERM stops new claims, drains in-flight, force-exits if stuck."""

    def test_drains_cleanly_then_closes_db(self) -> None:
        scheduler = MagicMock()  # shutdown(wait=True) returns instantly
        order: list[str] = []
        scheduler.pause.side_effect = lambda: order.append("pause")
        scheduler.shutdown.side_effect = lambda **_k: order.append("shutdown")

        with (
            _patched(scheduler=scheduler) as m,
            patch.object(worker_module, "_force_exit") as fx,
        ):
            worker_module.main()

        assert order == ["pause", "shutdown"]  # stop claiming, then drain
        scheduler.shutdown.assert_called_once_with(wait=True)
        m.close.assert_called_once()
        fx.assert_not_called()

    def test_force_exits_when_drain_times_out(self) -> None:
        # A stuck in-flight job: shutdown(wait=True) blocks past the deadline.
        block = threading.Event()

        def _blocking_shutdown(wait: bool = True) -> None:
            if wait:
                block.wait(timeout=5)

        scheduler = MagicMock()
        scheduler.shutdown.side_effect = _blocking_shutdown

        with (
            _patched(scheduler=scheduler) as m,
            patch.object(
                worker_module.settings,
                "scheduler_shutdown_drain_timeout_seconds",
                0.05,
            ),
            patch.object(worker_module, "_force_exit") as fx,
        ):
            worker_module.main()

        fx.assert_called_once()
        m.close.assert_not_called()  # skipped on the forced path
        block.set()  # release the daemon drain thread
