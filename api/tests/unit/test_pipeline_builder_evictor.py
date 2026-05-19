"""Tests for ``pipeline_builder`` TTL eviction daemon (issue #692)."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest

from app.services.pipeline_builder import session_store
from app.services.pipeline_builder.session_store import (
    SESSION_TTL_SECONDS,
    _ensure_daemon_running,
    _evict_stale,
    _eviction_loop,
    create_session,
)


@pytest.fixture(autouse=True)
def _reset_registries() -> Iterator[None]:
    with session_store._sessions_lock:
        session_store._sessions.clear()
        session_store._daemon_started = False
    with session_store._job_services_lock:
        session_store._job_services.clear()
    yield
    with session_store._sessions_lock:
        session_store._sessions.clear()
        session_store._daemon_started = False
    with session_store._job_services_lock:
        session_store._job_services.clear()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestEvictStale:
    def test_when_snapshot_empty_then_returns_empty_list(self):
        assert _evict_stale(_now(), {}, 60) == []

    def test_when_all_sessions_young_then_returns_empty_list(self):
        now = _now()
        snapshot = {"a": now - timedelta(seconds=10), "b": now - timedelta(seconds=20)}

        assert _evict_stale(now, snapshot, 60) == []

    def test_when_mixed_ages_then_only_stale_ids_returned(self):
        now = _now()
        snapshot = {
            "young": now - timedelta(seconds=10),
            "old_1": now - timedelta(seconds=120),
            "old_2": now - timedelta(seconds=600),
        }

        result = _evict_stale(now, snapshot, 60)

        assert set(result) == {"old_1", "old_2"}

    def test_when_age_exactly_equals_ttl_then_evicted(self):
        now = _now()
        snapshot = {"edge": now - timedelta(seconds=60)}

        assert _evict_stale(now, snapshot, 60) == ["edge"]

    def test_when_called_then_pure_no_mutation_of_snapshot(self):
        now = _now()
        snapshot = {"a": now - timedelta(seconds=200)}
        before = dict(snapshot)

        _evict_stale(now, snapshot, 60)

        assert snapshot == before

    def test_when_imported_from_module_then_accessible(self):
        from app.services.pipeline_builder.session_store import _evict_stale as fn

        assert callable(fn)


class TestEnsureDaemonRunning:
    def test_when_first_call_then_thread_spawned(self, monkeypatch):
        spawned: list[threading.Thread] = []
        original_thread = threading.Thread

        def capture_thread(*args, **kwargs):
            t = original_thread(*args, **kwargs)
            spawned.append(t)
            return t

        monkeypatch.setattr(
            "app.services.pipeline_builder.session_store.threading.Thread",
            capture_thread,
        )

        _ensure_daemon_running()

        assert len(spawned) == 1
        assert spawned[0].daemon is True
        assert spawned[0].name == "pipeline-session-evictor"

    def test_when_called_then_daemon_started_flag_set(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.pipeline_builder.session_store.threading.Thread",
            lambda *a, **kw: _FakeThread(),
        )

        _ensure_daemon_running()

        assert session_store._daemon_started is True

    def test_when_called_twice_then_only_one_thread_spawned(self, monkeypatch):
        spawn_count = [0]

        def fake_thread(*args, **kwargs):
            spawn_count[0] += 1
            return _FakeThread()

        monkeypatch.setattr(
            "app.services.pipeline_builder.session_store.threading.Thread",
            fake_thread,
        )

        _ensure_daemon_running()
        _ensure_daemon_running()

        assert spawn_count[0] == 1

    def test_when_daemon_starts_then_info_log_emitted(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "app.services.pipeline_builder.session_store.threading.Thread",
            lambda *a, **kw: _FakeThread(),
        )

        with caplog.at_level(logging.INFO, logger=session_store.__name__):
            _ensure_daemon_running()

        assert any(
            "daemon" in rec.message.lower() or "evictor" in rec.message.lower()
            for rec in caplog.records
        )


class TestCreateSessionStartsDaemon:
    def test_when_create_session_called_then_ensure_daemon_running_invoked(
        self, monkeypatch
    ):
        called = [0]

        def fake_ensure() -> None:
            called[0] += 1

        monkeypatch.setattr(
            "app.services.pipeline_builder.session_store._ensure_daemon_running",
            fake_ensure,
        )

        create_session({})

        assert called[0] == 1


class TestEvictionLoop:
    def test_when_loop_iterates_then_stale_sessions_evicted(self, monkeypatch):
        sid = create_session({})
        with session_store._sessions_lock:
            session_store._sessions[sid] = dataclasses.replace(
                session_store._sessions[sid],
                created_at=_now() - timedelta(seconds=SESSION_TTL_SECONDS + 1),
            )

        _drive_loop_once(monkeypatch)

        assert sid not in session_store._sessions
        assert sid not in session_store._job_services

    def test_when_loop_iterates_with_no_stale_then_session_kept(self, monkeypatch):
        sid = create_session({})

        _drive_loop_once(monkeypatch)

        assert sid in session_store._sessions

    def test_when_exception_in_iteration_then_logged_and_loop_survives(
        self, monkeypatch, caplog
    ):
        def boom(*_args, **_kwargs):
            raise RuntimeError("eviction-test-boom")

        monkeypatch.setattr(
            "app.services.pipeline_builder.session_store._evict_stale", boom
        )

        with caplog.at_level(logging.WARNING, logger=session_store.__name__):
            _drive_loop_n_times(monkeypatch, iterations=2)

        assert any(
            rec.levelno == logging.WARNING and rec.exc_info for rec in caplog.records
        )

    def test_when_session_evicted_then_info_log_with_session_id(
        self, monkeypatch, caplog
    ):
        sid = create_session({})
        with session_store._sessions_lock:
            session_store._sessions[sid] = dataclasses.replace(
                session_store._sessions[sid],
                created_at=_now() - timedelta(seconds=SESSION_TTL_SECONDS + 5),
            )

        with caplog.at_level(logging.INFO, logger=session_store.__name__):
            _drive_loop_once(monkeypatch)

        eviction_logs = [r for r in caplog.records if sid in r.getMessage()]
        assert eviction_logs, "expected an INFO log mentioning the evicted session id"

    def test_when_loop_uses_module_level_constants_then_monkeypatch_picked_up(
        self, monkeypatch
    ):
        sid = create_session({})
        monkeypatch.setattr(
            "app.services.pipeline_builder.session_store.SESSION_TTL_SECONDS", 0
        )

        _drive_loop_once(monkeypatch)

        assert sid not in session_store._sessions


class TestLoopSignature:
    def test_when_inspecting_signature_then_interval_seconds_param_with_default(self):
        sig = inspect.signature(_eviction_loop)

        assert "interval_seconds" in sig.parameters
        assert sig.parameters["interval_seconds"].default == 60


class _FakeThread:
    daemon = True
    name = "fake"

    def start(self) -> None:
        pass


def _drive_loop_n_times(monkeypatch, iterations: int) -> None:
    counter = [0]

    def fake_sleep(_seconds: float) -> None:
        counter[0] += 1
        if counter[0] >= iterations + 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "app.services.pipeline_builder.session_store.time.sleep", fake_sleep
    )
    with pytest.raises(KeyboardInterrupt):
        _eviction_loop(interval_seconds=0)


def _drive_loop_once(monkeypatch) -> None:
    _drive_loop_n_times(monkeypatch, iterations=1)
