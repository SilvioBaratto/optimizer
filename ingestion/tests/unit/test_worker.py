"""Tests for the daemon entrypoint (``python -m app.worker``).

``main()`` blocks on a shutdown event, so every test but the supervision ones
replaces that wait with an immediate "shutdown requested": startup runs to
completion, then shutdown proceeds. Nothing here opens a socket, a database, or a
scheduler. The event itself stays clear, because a set event during boot is exactly
the signal-arrived-early case the worker must honour before it starts the scheduler.

The startup ordering is the contract: signal handlers come first (a container's PID 1
ignores SIGTERM until it has one), metrics are served before the DB is touched (so
the healthcheck answers during a slow connect), orphans are reaped before the
scheduler starts (so a reaped job's type is free when its cron fires), and the
benchmark bootstrap runs off-thread (a cold fetch takes minutes and must not delay
the first scheduled run).
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app import worker as worker_module


@contextmanager
def _patched(*, real_wait: bool = False, **overrides):
    """Patch every side-effecting call in ``main()``; yield the mock namespace.

    Args:
        real_wait: Keep the real shutdown wait, for tests that drive supervision.
        **overrides: ``scheduler`` and ``health_check`` replace the defaults.
    """
    scheduler = overrides.get("scheduler", MagicMock())
    health = overrides.get("health_check", True)

    with ExitStack() as stack:
        enter = stack.enter_context
        enter(patch.object(worker_module, "_configure_logging"))
        serve = enter(patch.object(worker_module, "start_http_server"))
        init = enter(patch.object(worker_module, "init_db"))
        close = enter(patch.object(worker_module, "close_db"))
        reconcile = enter(patch.object(worker_module, "_reconcile_orphans"))
        signals = enter(patch.object(worker_module, "_install_signal_handlers"))
        enter(patch.object(worker_module, "create_scheduler", return_value=scheduler))
        enter(
            patch.object(
                worker_module.database_manager, "health_check", return_value=health
            )
        )
        if not real_wait:
            enter(
                patch.object(
                    worker_module,
                    "_wait_for_shutdown_or_scheduler_death",
                    return_value=False,
                )
            )
        yield MagicMock(
            serve=serve,
            init=init,
            close=close,
            reconcile=reconcile,
            signals=signals,
            scheduler=scheduler,
        )


@pytest.fixture(autouse=True)
def _clean_shutdown_event():
    """Start and leave every test with the shutdown event clear."""
    worker_module._shutdown.clear()
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

    def test_installs_signal_handlers_before_any_boot_step(self) -> None:
        """As a container's PID 1 the process ignores SIGTERM until it has a
        handler, so a stop during the slow boot would run out the grace period."""
        order: list[str] = []

        with _patched() as m:
            m.signals.side_effect = lambda: order.append("signals")
            m.serve.side_effect = lambda *_a, **_k: order.append("metrics")
            m.init.side_effect = lambda *_a, **_k: order.append("init_db")
            m.reconcile.side_effect = lambda: order.append("reconcile")
            m.scheduler.start.side_effect = lambda: order.append("scheduler")
            worker_module.main()

        assert order[0] == "signals"
        assert order.index("signals") < order.index("scheduler")

    def test_a_shutdown_requested_during_boot_never_starts_the_scheduler(
        self,
    ) -> None:
        """Starting it would fire misfire catch-up jobs that the drain then has to
        wait for or abandon."""
        with _patched() as m, patch.object(worker_module, "_force_exit") as fx:
            m.init.side_effect = lambda *_a, **_k: worker_module._shutdown.set()
            worker_module.main()

        m.scheduler.start.assert_not_called()
        m.close.assert_called_once()
        fx.assert_not_called()

    def test_continues_when_health_check_fails(self) -> None:
        """A red health check is logged, not fatal — the DB may still come up."""
        with _patched(health_check=False) as m:
            worker_module.main()

        m.scheduler.start.assert_called_once()


class TestConfigureLogging:
    """``LOG_FORMAT=json`` (the default) must emit one valid JSON object per record."""

    @staticmethod
    @contextmanager
    def _configured(log_format: str):
        """Run ``_configure_logging`` against a bare root logger, then restore it.

        pytest attaches its capture handlers to the root during each phase and
        ``basicConfig`` is a no-op while the root has handlers, so the swap has to
        happen inside the test body.
        """
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        root.handlers = []
        try:
            with (
                patch.object(worker_module.settings, "log_format", log_format),
                patch.object(worker_module.settings, "log_level", "INFO"),
                patch("app.services.market_data.yfinance.configure_yfinance"),
            ):
                worker_module._configure_logging()
            yield
        finally:
            root.handlers = handlers
            root.setLevel(level)

    def test_json_lines_survive_quotes_and_newlines_in_the_message(
        self, capsys
    ) -> None:
        message = (
            'Orphan reconciliation failed: relation "background_jobs" does not exist'
            "\nsecond line"
        )
        with self._configured("json"):
            logging.getLogger("app.probe").warning(message)

        (line,) = capsys.readouterr().err.splitlines()
        record = json.loads(line)
        assert record["level"] == "WARNING"
        assert record["name"] == "app.probe"
        assert record["message"] == message
        assert record["timestamp"]

    def test_json_traceback_stays_inside_the_record(self, capsys) -> None:
        with self._configured("json"):
            probe = logging.getLogger("app.probe")
            try:
                raise ValueError('bad "value"')
            except ValueError:
                probe.exception("step failed")
            probe.info("after")

        lines = capsys.readouterr().err.splitlines()
        failed, after = (json.loads(line) for line in lines)
        assert failed["message"] == "step failed"
        assert "Traceback" in failed["exception"]
        assert "ValueError" in failed["exception"]
        assert after["message"] == "after"

    def test_text_format_is_unchanged(self, capsys) -> None:
        with self._configured("text"):
            logging.getLogger("app.probe").warning("plain")

        assert capsys.readouterr().err.rstrip().endswith("app.probe - WARNING - plain")


class TestSchedulerSupervision:
    """A dead APScheduler thread must take the process down, not idle it green."""

    @staticmethod
    def _run_main(
        scheduler: MagicMock, *, join_seconds: float
    ) -> tuple[bool, dict[str, object]]:
        """Run ``main()`` off-thread; return (still_blocked, outcome).

        Without supervision ``main()`` waits on ``_shutdown`` forever, so the
        test must never call it on the pytest thread.
        """
        worker_module._shutdown.clear()
        outcome: dict[str, object] = {}

        def _run() -> None:
            try:
                worker_module.main()
            except SystemExit as exc:
                outcome["code"] = exc.code

        runner = threading.Thread(target=_run, daemon=True)
        runner.start()
        runner.join(timeout=join_seconds)
        still_blocked = runner.is_alive()
        worker_module._shutdown.set()  # release a main() that never noticed
        runner.join(timeout=5)
        return still_blocked, outcome

    def test_exits_nonzero_when_the_scheduler_thread_dies(self) -> None:
        """The metrics thread keeps the healthcheck green, so only a restart helps."""
        scheduler = MagicMock()
        scheduler.loop_alive.return_value = False

        with (
            _patched(scheduler=scheduler, real_wait=True),
            patch.object(worker_module, "_SUPERVISION_INTERVAL_SECONDS", 0.01),
            patch.object(worker_module, "_force_exit"),
        ):
            still_blocked, outcome = self._run_main(scheduler, join_seconds=5)

        assert not still_blocked, "main() kept waiting although the scheduler died"
        assert outcome.get("code") not in (None, 0)
        scheduler.shutdown.assert_called_once_with(wait=True)

    def test_keeps_running_while_the_scheduler_thread_is_alive(self) -> None:
        scheduler = MagicMock()
        scheduler.loop_alive.return_value = True

        with (
            _patched(scheduler=scheduler, real_wait=True),
            patch.object(worker_module, "_SUPERVISION_INTERVAL_SECONDS", 0.01),
            patch.object(worker_module, "_force_exit"),
        ):
            still_blocked, outcome = self._run_main(scheduler, join_seconds=0.2)

        assert still_blocked
        assert "code" not in outcome  # a requested shutdown still exits cleanly

    def test_forced_exit_after_scheduler_death_is_nonzero(self) -> None:
        block = threading.Event()
        scheduler = MagicMock()
        scheduler.loop_alive.return_value = False
        scheduler.shutdown.side_effect = lambda wait=True: block.wait(timeout=5)

        with (
            _patched(scheduler=scheduler, real_wait=True),
            patch.object(worker_module, "_SUPERVISION_INTERVAL_SECONDS", 0.01),
            patch.object(
                worker_module.settings,
                "scheduler_shutdown_drain_timeout_seconds",
                0.05,
            ),
            patch.object(worker_module, "_force_exit") as fx,
        ):
            self._run_main(scheduler, join_seconds=5)
            block.set()

        fx.assert_called_once_with(1)


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


class TestLoggingSetup:
    def test_masks_api_key_query_params_in_log_records(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """urllib3 logs the FRED request path (key included) at DEBUG."""
        leaked = "FREDSECRETKEY0123456789abcdef"
        with patch("app.services.market_data.yfinance.configure_yfinance"):
            worker_module._configure_logging()

        logging.getLogger("urllib3.connectionpool").warning(
            '"GET /fred/series?series_id=T10Y2Y&api_key=%s HTTP/1.1" 200', leaked
        )

        assert "api_key=***" in caplog.text
        assert leaked not in caplog.text
