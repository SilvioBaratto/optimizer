"""Tests for ``pipeline_builder.get_job_service`` factory (issue #691)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.config import settings
from app.services.jobs.background_job import BackgroundJobService
from app.services.pipeline_builder import session_store
from app.services.pipeline_builder.session_store import (
    create_session,
    delete_session,
    get_job_service,
)


@pytest.fixture(autouse=True)
def _reset_registries() -> Iterator[None]:
    with session_store._sessions_lock:
        session_store._sessions.clear()
    with session_store._job_services_lock:
        session_store._job_services.clear()
    yield
    with session_store._sessions_lock:
        session_store._sessions.clear()
    with session_store._job_services_lock:
        session_store._job_services.clear()


class TestGetJobService:
    def test_when_called_then_returns_background_job_service_instance(self):
        sid = create_session({})

        svc = get_job_service(sid)

        assert isinstance(svc, BackgroundJobService)

    def test_when_called_twice_with_same_sid_then_same_instance(self):
        sid = create_session({})

        first = get_job_service(sid)
        second = get_job_service(sid)

        assert first is second

    def test_when_called_with_two_sids_then_distinct_instances(self):
        sid_a = create_session({})
        sid_b = create_session({})

        svc_a = get_job_service(sid_a)
        svc_b = get_job_service(sid_b)

        assert svc_a is not svc_b

    def test_when_called_then_job_type_namespaced_with_session_prefix(self):
        sid = create_session({})

        svc = get_job_service(sid)

        assert svc._job_type == f"pipeline_{sid[:8]}"

    def test_when_called_with_two_sids_then_job_types_distinct(self):
        sid_a = create_session({})
        sid_b = create_session({})

        svc_a = get_job_service(sid_a)
        svc_b = get_job_service(sid_b)

        assert svc_a._job_type != svc_b._job_type

    def test_when_called_then_job_type_within_database_column_limit(self):
        sid = create_session({})

        svc = get_job_service(sid)

        assert len(svc._job_type) <= 100

    def test_when_called_then_heartbeat_cadence_from_settings(self):
        sid = create_session({})

        svc = get_job_service(sid)

        assert svc._heartbeat_cadence == settings.scheduler_heartbeat_cadence_seconds

    def test_when_called_then_session_factory_is_database_manager_get_session(self):
        from app.database import database_manager

        sid = create_session({})

        svc = get_job_service(sid)

        assert svc._session_factory == database_manager.get_session
        assert svc._session_factory.__self__ is database_manager

    def test_when_session_cached_then_present_in_registry(self):
        sid = create_session({})

        svc = get_job_service(sid)

        assert session_store._job_services[sid] is svc


class TestDeleteSessionEviction:
    def test_when_session_deleted_then_job_service_evicted(self):
        sid = create_session({})
        get_job_service(sid)
        assert sid in session_store._job_services

        delete_session(sid)

        assert sid not in session_store._job_services


class TestFacadeReExport:
    def test_when_importing_facade_then_get_job_service_exposed(self):
        from app.services.pipeline_builder import get_job_service as facade_factory

        assert facade_factory is get_job_service


class TestModuleLevelImport:
    def test_when_inspecting_session_store_then_background_job_imported_at_module_level(
        self,
    ):
        import app.services.pipeline_builder.session_store as ss

        assert ss.BackgroundJobService is BackgroundJobService


class TestThreadSafety:
    def test_when_concurrent_first_calls_for_same_sid_then_single_instance_cached(self):
        sid = create_session({})
        results: list[BackgroundJobService] = []
        results_lock = threading.Lock()

        def worker() -> None:
            svc = get_job_service(sid)
            with results_lock:
                results.append(svc)

        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(lambda _: worker(), range(20)))

        first = results[0]
        for svc in results[1:]:
            assert svc is first
