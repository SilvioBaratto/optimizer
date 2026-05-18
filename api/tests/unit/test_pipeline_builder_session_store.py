"""Tests for the ``pipeline_builder`` package skeleton (issue #689)."""

from __future__ import annotations

import dataclasses
import inspect
import sys
import threading
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone

import pytest

EXPECTED_STEP_IDS: tuple[str, ...] = (
    "load",
    "screen",
    "clean_returns",
    "build_history",
    "validate_is",
    "validate_oos",
    "coverage_gate",
    "regime",
    "optimize",
    "rebalance_decision",
    "cost",
    "report",
    "persist",
)

EXPECTED_FIELDS: tuple[str, ...] = (
    "session_id",
    "created_at",
    "run_config",
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
    "current_step",
    "step_status",
    "step_results",
)


def _build_session():
    from app.services.pipeline_builder.session_store import _PipelineSession

    return _PipelineSession(
        session_id="abc",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        run_config={"k": "v"},
        assembly=None,
        investable=None,
        clean_returns=None,
        factor_scores_dict=None,
        returns_history=None,
        is_result=None,
        oos_result=None,
        coverage_result=None,
        regime_result=None,
        optimize_result=None,
        rebalance_result=None,
        cost_result=None,
        report_result=None,
        current_step="load",
        step_status={"load": "pending"},
        step_results={},
    )


class TestPublicReExports:
    def test_when_importing_facade_then_public_symbols_are_available(self):
        from app.services.pipeline_builder import (
            STEP_IDS,
            PipelineSessionCapacityError,
            _PipelineSession,
        )

        assert _PipelineSession is not None
        assert PipelineSessionCapacityError is not None
        assert STEP_IDS is not None

    def test_when_inspecting_facade_then_module_private_state_is_not_exported(self):
        import app.services.pipeline_builder as facade

        for hidden in (
            "_sessions",
            "_job_services",
            "_sessions_lock",
            "_job_services_lock",
            "_daemon_started",
        ):
            assert not hasattr(facade, hidden), (
                f"{hidden!r} must remain module-private to session_store"
            )


class TestStepIds:
    def test_when_reading_step_ids_then_thirteen_entries_in_pipeline_order(self):
        from app.services.pipeline_builder.session_store import STEP_IDS

        assert isinstance(STEP_IDS, tuple)
        assert STEP_IDS == EXPECTED_STEP_IDS
        assert len(STEP_IDS) == 13


class TestConstants:
    def test_when_reading_ttl_constant_then_one_day_in_seconds(self):
        from app.services.pipeline_builder.session_store import SESSION_TTL_SECONDS

        assert SESSION_TTL_SECONDS == 86400

    def test_when_reading_capacity_constant_then_three(self):
        from app.services.pipeline_builder.session_store import (
            MAX_CONCURRENT_SESSIONS,
        )

        assert MAX_CONCURRENT_SESSIONS == 3


class TestCapacityError:
    def test_when_raised_then_carries_current_and_cap_attrs_and_message(self):
        from app.services.pipeline_builder.session_store import (
            PipelineSessionCapacityError,
        )

        err = PipelineSessionCapacityError(current=3, cap=3)

        assert err.current == 3
        assert err.cap == 3
        assert str(err) == "Pipeline session cap reached (3/3)"
        assert isinstance(err, Exception)


class TestPipelineSessionDataclass:
    def test_when_inspecting_dataclass_then_frozen_true(self):
        from app.services.pipeline_builder.session_store import _PipelineSession

        assert is_dataclass(_PipelineSession)
        params = _PipelineSession.__dataclass_params__
        assert params.frozen is True

    def test_when_inspecting_dataclass_then_slots_disabled(self):
        from app.services.pipeline_builder.session_store import _PipelineSession

        assert "__slots__" not in _PipelineSession.__dict__, (
            "slots=True breaks dataclasses.replace with mutable dict fields"
        )

    def test_when_inspecting_dataclass_then_nineteen_fields_in_exact_order(self):
        from app.services.pipeline_builder.session_store import _PipelineSession

        names = tuple(f.name for f in fields(_PipelineSession))
        assert names == EXPECTED_FIELDS
        assert len(names) == 19

    def test_when_constructed_then_instance_holds_supplied_values(self):
        session = _build_session()

        assert session.session_id == "abc"
        assert session.current_step == "load"
        assert session.step_status == {"load": "pending"}

    def test_when_setting_attribute_then_frozen_instance_error_raised(self):
        session = _build_session()

        with pytest.raises(dataclasses.FrozenInstanceError):
            session.current_step = "screen"  # type: ignore[misc]

    def test_when_using_replace_with_mutable_dict_then_new_instance_returned(self):
        session = _build_session()

        replaced = dataclasses.replace(
            session,
            current_step="screen",
            step_status={"load": "completed", "screen": "running"},
        )

        assert replaced.current_step == "screen"
        assert session.current_step == "load"
        assert replaced.step_status == {"load": "completed", "screen": "running"}


class TestModulePrivatePlaceholders:
    def test_when_inspecting_module_then_session_registry_starts_empty(self):
        from app.services.pipeline_builder import session_store

        assert isinstance(session_store._sessions, dict)
        assert session_store._sessions == {}

    def test_when_inspecting_module_then_job_service_registry_starts_empty(self):
        from app.services.pipeline_builder import session_store

        assert isinstance(session_store._job_services, dict)
        assert session_store._job_services == {}

    def test_when_inspecting_module_then_locks_are_threading_lock_instances(self):
        from app.services.pipeline_builder import session_store

        lock_type = type(threading.Lock())
        assert isinstance(session_store._sessions_lock, lock_type)
        assert isinstance(session_store._job_services_lock, lock_type)

    def test_when_inspecting_module_source_then_daemon_flag_initialised_false(self):
        from app.services.pipeline_builder import session_store

        source = inspect.getsource(session_store)
        assert "_daemon_started: bool = False" in source


class TestImportBoundaries:
    def test_when_importing_session_store_then_no_runtime_pull_of_research(self):
        # Subprocess avoids polluting the parent interpreter's sys.modules,
        # which would break `is`-identity assertions in sibling tests.
        import json
        import subprocess

        script = (
            "import sys, json; "
            "import app.services.pipeline_builder.session_store; "
            "print(json.dumps([m for m in sys.modules if m.startswith('research')]))"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        leaked = json.loads(result.stdout.strip().splitlines()[-1])
        assert leaked == [], f"session_store leaked research imports: {leaked}"

    def test_when_inspecting_session_store_then_background_job_imported_at_module_level(
        self,
    ):
        import app.services.pipeline_builder.session_store as ss
        from app.services.jobs.background_job import BackgroundJobService

        assert getattr(ss, "BackgroundJobService", None) is BackgroundJobService

    def test_when_reading_session_store_source_then_future_annotations_enabled(self):
        import app.services.pipeline_builder.session_store as ss

        source = inspect.getsource(ss)
        assert "from __future__ import annotations" in source

    def test_when_reading_session_store_source_then_no_research_or_pandas_import(self):
        import app.services.pipeline_builder.session_store as ss

        source = inspect.getsource(ss)
        for forbidden in (
            "import pandas",
            "from pandas",
            "import research",
            "from research",
        ):
            assert forbidden not in source, (
                f"session_store must not import {forbidden!r} at module load"
            )
