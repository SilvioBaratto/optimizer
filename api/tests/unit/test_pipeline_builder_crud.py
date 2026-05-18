"""Tests for ``pipeline_builder`` CRUD functions (issue #690)."""

from __future__ import annotations

import inspect
import re
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pytest

from app.services.pipeline_builder import session_store
from app.services.pipeline_builder.session_store import (
    MAX_CONCURRENT_SESSIONS,
    STEP_IDS,
    PipelineSessionCapacityError,
    _PipelineSession,
    create_session,
    delete_session,
    get_session,
    update_session,
)


@pytest.fixture(autouse=True)
def _reset_registries() -> Iterator[None]:
    """Empty the module-level registries before and after each test."""
    with session_store._sessions_lock:
        session_store._sessions.clear()
    with session_store._job_services_lock:
        session_store._job_services.clear()
    yield
    with session_store._sessions_lock:
        session_store._sessions.clear()
    with session_store._job_services_lock:
        session_store._job_services.clear()


class TestCreateSession:
    def test_when_called_then_returns_uuid4_hex_session_id(self):
        sid = create_session({"k": "v"})

        assert isinstance(sid, str)
        assert len(sid) == 32
        uuid.UUID(hex=sid)

    def test_when_called_then_inserts_session_into_registry(self):
        sid = create_session({"k": "v"})

        assert sid in session_store._sessions
        assert isinstance(session_store._sessions[sid], _PipelineSession)

    def test_when_called_then_all_result_fields_are_none(self):
        sid = create_session({"k": "v"})
        session = session_store._sessions[sid]

        result_fields = (
            "assembly",
            "investable",
            "clean_returns",
            "factor_scores_dict",
            "returns_history",
            "is_result",
            "oos_result",
            "coverage_result",
            "regime_result",
            "optimize_result",
            "rebalance_result",
            "cost_result",
            "report_result",
        )
        for field in result_fields:
            assert getattr(session, field) is None, f"{field!r} should default to None"

    def test_when_called_then_current_step_is_idle(self):
        sid = create_session({})

        assert session_store._sessions[sid].current_step == "idle"

    def test_when_called_then_step_status_initialised_to_pending_for_all_step_ids(self):
        sid = create_session({})

        status = session_store._sessions[sid].step_status
        assert set(status.keys()) == set(STEP_IDS)
        assert all(v == "pending" for v in status.values())

    def test_when_called_then_run_config_preserved(self):
        cfg = {"universe": "sp500", "horizon": 252}

        sid = create_session(cfg)

        assert session_store._sessions[sid].run_config == cfg

    def test_when_called_then_created_at_is_timezone_aware_utc(self):
        sid = create_session({})

        created = session_store._sessions[sid].created_at
        assert isinstance(created, datetime)
        assert created.tzinfo is not None
        assert created.utcoffset() == timezone.utc.utcoffset(created)

    def test_when_cap_reached_then_raises_pipeline_session_capacity_error(self):
        for _ in range(MAX_CONCURRENT_SESSIONS):
            create_session({})

        with pytest.raises(PipelineSessionCapacityError) as exc:
            create_session({})

        assert exc.value.current == MAX_CONCURRENT_SESSIONS
        assert exc.value.cap == MAX_CONCURRENT_SESSIONS

    def test_when_cap_reached_then_no_new_session_added(self):
        for _ in range(MAX_CONCURRENT_SESSIONS):
            create_session({})

        with pytest.raises(PipelineSessionCapacityError):
            create_session({})

        assert len(session_store._sessions) == MAX_CONCURRENT_SESSIONS


class TestGetSession:
    def test_when_session_exists_then_returns_dataclass(self):
        sid = create_session({"k": "v"})

        result = get_session(sid)

        assert isinstance(result, _PipelineSession)
        assert result.session_id == sid

    def test_when_session_missing_then_returns_none(self):
        assert get_session("nonexistent") is None


class TestUpdateSession:
    def test_when_called_with_valid_field_then_replaces_session(self):
        sid = create_session({})

        update_session(sid, current_step="load")

        assert session_store._sessions[sid].current_step == "load"

    def test_when_called_then_returns_none(self):
        sid = create_session({})

        result = update_session(sid, current_step="screen")

        assert result is None

    def test_when_session_missing_then_no_op(self):
        update_session("nonexistent", current_step="load")

        assert "nonexistent" not in session_store._sessions

    def test_when_unknown_field_then_raises_value_error(self):
        sid = create_session({})

        with pytest.raises(ValueError, match="Unknown _PipelineSession field"):
            update_session(sid, totally_not_a_field=42)

    def test_when_replacing_step_status_then_old_instance_unchanged(self):
        sid = create_session({})
        original = session_store._sessions[sid]

        update_session(
            sid,
            step_status={**original.step_status, "load": "running"},
            current_step="load",
        )

        assert original.step_status["load"] == "pending"
        assert session_store._sessions[sid].step_status["load"] == "running"


class TestDeleteSession:
    def test_when_session_exists_then_removed_from_registry(self):
        sid = create_session({})

        delete_session(sid)

        assert sid not in session_store._sessions

    def test_when_session_missing_then_idempotent_no_raise(self):
        delete_session("nonexistent")

    def test_when_job_service_present_then_also_removed(self):
        sid = create_session({})
        with session_store._job_services_lock:
            session_store._job_services[sid] = object()  # type: ignore[assignment]

        delete_session(sid)

        assert sid not in session_store._job_services

    def test_when_job_service_absent_then_no_raise(self):
        sid = create_session({})

        delete_session(sid)


class TestModuleLevelInvariants:
    def test_when_inspecting_module_then_lock_order_documented(self):
        source = inspect.getsource(session_store)

        assert re.search(
            r"_sessions_lock.+_job_services_lock",
            source,
            re.DOTALL,
        ), "lock-order rule must be documented in a module-level comment"

    def test_when_importing_facade_then_crud_functions_re_exported(self):
        from app.services.pipeline_builder import (
            create_session as facade_create,
        )
        from app.services.pipeline_builder import (
            delete_session as facade_delete,
        )
        from app.services.pipeline_builder import (
            get_session as facade_get,
        )
        from app.services.pipeline_builder import (
            update_session as facade_update,
        )

        assert facade_create is create_session
        assert facade_get is get_session
        assert facade_update is update_session
        assert facade_delete is delete_session


class TestThreadSafety:
    def test_when_concurrent_creates_then_cap_strictly_enforced(self):
        attempts = 20
        successes: list[str] = []
        capacity_errors = 0
        errors_lock = threading.Lock()

        def worker() -> None:
            nonlocal capacity_errors
            try:
                sid = create_session({})
                with errors_lock:
                    successes.append(sid)
            except PipelineSessionCapacityError:
                with errors_lock:
                    capacity_errors += 1

        with ThreadPoolExecutor(max_workers=attempts) as pool:
            list(pool.map(lambda _: worker(), range(attempts)))

        assert len(successes) == MAX_CONCURRENT_SESSIONS
        assert capacity_errors == attempts - MAX_CONCURRENT_SESSIONS
        assert len(session_store._sessions) == MAX_CONCURRENT_SESSIONS

    def test_when_concurrent_update_and_delete_then_no_keyerror(self):
        sids = [create_session({}) for _ in range(MAX_CONCURRENT_SESSIONS)]
        errors: list[BaseException] = []

        def update_worker(sid: str) -> None:
            try:
                for step in STEP_IDS:
                    update_session(sid, current_step=step)
            except BaseException as exc:
                errors.append(exc)

        def delete_worker(sid: str) -> None:
            try:
                delete_session(sid)
                delete_session(sid)
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=len(sids) * 2) as pool:
            futures = [pool.submit(update_worker, sid) for sid in sids]
            futures += [pool.submit(delete_worker, sid) for sid in sids]
            for f in as_completed(futures):
                f.result()

        assert errors == [], f"thread-safety violation: {errors!r}"
