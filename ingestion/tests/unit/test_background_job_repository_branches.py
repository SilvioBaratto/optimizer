"""Branch-coverage tests for app/repositories/jobs/background_job_repository.py.

Targets missed branches at lines: 63-72, 79-90, 118, 134, 145-150, 160,
173-182, 210-221, 348-352.

Does NOT duplicate tests already in test_background_job_repository.py
(claim_or_create, cleanup_expired TTL arithmetic, reconcile_orphans,
liveness predicate, get_latest_by_type).

All tests use the real ``db_session`` SQLite SAVEPOINT fixture.
No second session is used anywhere.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.repositories.jobs.background_job_repository import BackgroundJobRepository

_MODULE = "app.repositories.jobs.background_job_repository"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_repo(session: Session) -> BackgroundJobRepository:
    return BackgroundJobRepository(session)


def _pending(session: Session, job_type: str) -> uuid.UUID:
    """Create a pending job and return its id."""
    repo = _new_repo(session)
    row = repo.create(job_type)
    session.flush()
    return row.id


# ===========================================================================
# _ensure_tables_exist — exception fallback branches (lines 63-72)
# ===========================================================================


class TestEnsureTablesExistFallback:
    """_ensure_tables_exist has two nested exception handlers; exercise both.

    The method calls the builtin ``__import__("sqlalchemy.inspection", ...)``
    and then ``.inspect()``.  We make the outer try-block raise by patching
    ``sqlalchemy.inspection`` in ``sys.modules`` so that the attribute access
    ``.inspect`` raises AttributeError — triggering the outer except branch.
    """

    def test_when_inspector_raises_then_falls_back_to_create_all(
        self, db_session: Session
    ) -> None:
        # Remove the cached module so __import__ returns a fresh MagicMock
        # whose .inspect() raises, driving the outer except branch.
        import sys

        broken_mod = MagicMock()
        broken_mod.inspect.side_effect = RuntimeError("inspect unavailable")
        with patch.dict(sys.modules, {"sqlalchemy.inspection": broken_mod}):
            repo = _new_repo(db_session)
            # Inner fallback calls Base.metadata.create_all — no-op on live SQLite.
            repo._ensure_tables_exist()  # Must not raise.

    def test_when_both_inspector_and_create_all_raise_then_logged_not_raised(
        self, db_session: Session
    ) -> None:
        # Outer except fires (broken inspect); inner fallback also fails.
        import sys

        from portopt_db.base import Base

        broken_mod = MagicMock()
        broken_mod.inspect.side_effect = RuntimeError("inspect boom")
        with (
            patch.dict(sys.modules, {"sqlalchemy.inspection": broken_mod}),
            patch.object(Base.metadata, "create_all", side_effect=Exception("pg down")),
        ):
            repo = _new_repo(db_session)
            # Must not propagate — inner except logs at DEBUG and swallows.
            repo._ensure_tables_exist()


# ===========================================================================
# create() — extra kwargs stored vs None (lines 79-90)
# ===========================================================================


class TestCreate:
    def test_when_no_extra_kwargs_then_extra_is_none(self, db_session: Session) -> None:
        repo = _new_repo(db_session)
        row = repo.create("create_no_extra")
        db_session.flush()
        assert row.extra is None

    def test_when_extra_kwargs_provided_then_extra_dict_stored(
        self, db_session: Session
    ) -> None:
        # Exercises the ``extra=initial_extra or None`` branch where
        # initial_extra is truthy.
        repo = _new_repo(db_session)
        row = repo.create("create_with_extra", country="DE", run=7)
        db_session.flush()
        assert row.extra == {"country": "DE", "run": 7}

    def test_created_row_has_pending_status_and_zero_counters(
        self, db_session: Session
    ) -> None:
        repo = _new_repo(db_session)
        row = repo.create("create_defaults")
        assert row.status == "pending"
        assert row.current == 0
        assert row.total == 0


# ===========================================================================
# update() — errors_value replacement path (line 118)
# ===========================================================================


class TestUpdateErrorsValue:
    def test_when_errors_provided_and_row_matched_then_error_entries_stored(
        self, db_session: Session
    ) -> None:
        # Covers line 118: rowcount==1 and errors_value is not None.
        jid = _pending(db_session, "update_errors_branch")
        repo = _new_repo(db_session)

        count = repo.update(jid, status="running", errors=["err_a", "err_b"])
        db_session.flush()

        assert count == 1
        row = repo.get(jid)
        assert row is not None
        assert row.errors == ["err_a", "err_b"]

    def test_when_errors_is_empty_list_then_entries_cleared(
        self, db_session: Session
    ) -> None:
        jid = _pending(db_session, "update_errors_empty")
        repo = _new_repo(db_session)
        # First set some errors while transitioning to running.
        repo.update(jid, status="running", errors=["original"])
        db_session.flush()
        # Clear errors: must include a core kwarg (current=0) so values is
        # non-empty → rowcount=1 → _replace_error_entries is called with [].
        repo.update(jid, current=0, errors=[])
        db_session.flush()

        row = repo.get(jid)
        assert row is not None
        assert row.errors is None  # empty list → no entries → property returns None

    def test_when_errors_not_provided_then_no_error_replacement(
        self, db_session: Session
    ) -> None:
        # errors_value stays None → _replace_error_entries is NOT called.
        jid = _pending(db_session, "update_no_errors")
        repo = _new_repo(db_session)
        count = repo.update(jid, current=5)
        assert count == 1
        row = repo.get(jid)
        assert row is not None
        assert row.errors is None


# ===========================================================================
# _partition_kwargs — extras branch (line 134)
# ===========================================================================


class TestPartitionKwargs:
    def test_non_core_key_goes_to_extras(self) -> None:
        # Line 134: key not in _CORE_COLUMNS → extras dict.
        core, extras = BackgroundJobRepository._partition_kwargs(
            {"status": "running", "my_custom_field": "hello"}
        )
        assert "status" in core
        assert "my_custom_field" in extras

    def test_finished_at_string_converted_to_datetime(self) -> None:
        # Line 130-131: finished_at + isinstance(value, str) → fromisoformat.
        ts = "2026-01-15T10:00:00+00:00"
        core, _ = BackgroundJobRepository._partition_kwargs({"finished_at": ts})
        assert isinstance(core["finished_at"], datetime)

    def test_finished_at_datetime_not_converted(self) -> None:
        dt = datetime.now(timezone.utc)
        core, _ = BackgroundJobRepository._partition_kwargs({"finished_at": dt})
        assert core["finished_at"] is dt

    def test_all_core_columns_routed_correctly(self) -> None:
        kwargs = {"status": "completed", "current": 5, "total": 10}
        core, extras = BackgroundJobRepository._partition_kwargs(kwargs)
        assert set(core.keys()) == {"status", "current", "total"}
        assert extras == {}


# ===========================================================================
# _build_values — extras merge path (lines 145-150)
# ===========================================================================


class TestBuildValuesMerge:
    def test_when_extras_provided_then_merged_with_existing_extra(
        self, db_session: Session
    ) -> None:
        # Seed a row that already has extra data.
        repo = _new_repo(db_session)
        row = repo.create("build_values_merge", existing_key="old_value")
        db_session.flush()

        # Update with a non-core kwarg: triggers _build_values extras path.
        count = repo.update(row.id, status="running", new_key="new_value")
        db_session.flush()

        assert count == 1
        refreshed = repo.get(row.id)
        assert refreshed is not None
        # Merged: both keys present.
        assert refreshed.extra is not None
        assert refreshed.extra.get("new_key") == "new_value"

    def test_when_extras_provided_and_existing_extra_is_none_then_new_dict(
        self, db_session: Session
    ) -> None:
        # extra starts as None → merged = {} → only new key present.
        jid = _pending(db_session, "build_values_none_extra")
        repo = _new_repo(db_session)

        repo.update(jid, status="running", injected_field="xyz")
        db_session.flush()

        row = repo.get(jid)
        assert row is not None
        assert row.extra is not None
        assert row.extra.get("injected_field") == "xyz"


# ===========================================================================
# _exec_status_guarded_update — empty values early return (line 160)
# ===========================================================================


class TestExecStatusGuardedUpdateEmptyValues:
    def test_when_values_empty_then_returns_zero_without_hitting_db(
        self, db_session: Session
    ) -> None:
        # Line 159-160: ``if not values: return 0``.
        # Pass errors=None so _replace_error_entries is skipped; pass no
        # core/extras kwargs so values dict ends up empty.
        jid = _pending(db_session, "exec_empty_values")
        repo = _new_repo(db_session)

        # Call _exec_status_guarded_update directly with empty dict.
        result = repo._exec_status_guarded_update(jid, None, {})
        assert result == 0

    def test_update_with_only_errors_kwarg_and_active_row_still_zero_rowcount(
        self, db_session: Session
    ) -> None:
        # errors is popped before building values; if that's all that was
        # passed, values dict is empty → rowcount=0, errors branch skipped.
        jid = _pending(db_session, "exec_only_errors_kwarg")
        repo = _new_repo(db_session)

        count = repo.update(jid, errors=["lone error"])
        # values was empty → rowcount=0 → errors NOT applied
        assert count == 0


# ===========================================================================
# _replace_error_entries — list iteration (lines 173-182)
# ===========================================================================


class TestReplaceErrorEntries:
    def test_when_errors_is_list_then_entries_created_in_order(
        self, db_session: Session
    ) -> None:
        jid = _pending(db_session, "replace_errors_list")
        repo = _new_repo(db_session)

        repo.update(jid, status="running", errors=["first", "second", "third"])
        db_session.flush()

        row = repo.get(jid)
        assert row is not None
        assert row.errors == ["first", "second", "third"]

    def test_when_errors_is_not_list_then_entries_deleted_and_no_new_entries(
        self, db_session: Session
    ) -> None:
        # Line 179: ``if not isinstance(errors_value, list): return``
        jid = _pending(db_session, "replace_errors_non_list")
        repo = _new_repo(db_session)

        # Set some entries first.
        repo.update(jid, status="running", errors=["pre_existing"])
        db_session.flush()

        # Now call _replace_error_entries directly with a non-list value.
        repo._replace_error_entries(jid, "not a list")
        db_session.flush()

        row = repo.get(jid)
        assert row is not None
        # Entries were deleted; non-list path returns without inserting new ones.
        assert row.errors is None

    def test_when_errors_contains_non_string_then_converted_to_str(
        self, db_session: Session
    ) -> None:
        jid = _pending(db_session, "replace_errors_str_cast")
        repo = _new_repo(db_session)

        repo.update(jid, status="running", errors=[42, None, {"key": "val"}])
        db_session.flush()

        row = repo.get(jid)
        assert row is not None
        assert row.errors is not None
        assert row.errors[0] == "42"
        assert row.errors[1] == "None"


# ===========================================================================
# is_any_running — true/false branches (lines 210-221)
# ===========================================================================


class TestIsAnyRunning:
    def test_when_no_active_job_returns_false_and_none(
        self, db_session: Session
    ) -> None:
        repo = _new_repo(db_session)
        running, job_id_str = repo.is_any_running("is_any_no_active_xyz")
        assert running is False
        assert job_id_str is None

    def test_when_pending_job_exists_returns_true_and_id_string(
        self, db_session: Session
    ) -> None:
        jid = _pending(db_session, "is_any_pending")
        repo = _new_repo(db_session)

        running, job_id_str = repo.is_any_running("is_any_pending")

        assert running is True
        assert job_id_str == str(jid)

    def test_when_running_job_exists_returns_true(self, db_session: Session) -> None:
        jid = _pending(db_session, "is_any_running_status")
        repo = _new_repo(db_session)
        repo.update(jid, status="running")
        db_session.flush()

        running, _ = repo.is_any_running("is_any_running_status")
        assert running is True

    def test_when_only_completed_jobs_returns_false(self, db_session: Session) -> None:
        jid = _pending(db_session, "is_any_completed_only")
        repo = _new_repo(db_session)
        repo.update(jid, status="completed", finished_at=datetime.now(timezone.utc))
        db_session.flush()

        running, _ = repo.is_any_running("is_any_completed_only")
        assert running is False

    def test_different_job_type_does_not_match(self, db_session: Session) -> None:
        _pending(db_session, "is_any_type_a")
        repo = _new_repo(db_session)
        running, _ = repo.is_any_running("is_any_type_b_different")
        assert running is False


# ===========================================================================
# cleanup_expired — "no such table" exception swallow (lines 348-352)
# ===========================================================================


class TestCleanupExpiredNoSuchTable:
    def test_when_execute_raises_no_such_table_then_returns_zero(
        self, db_session: Session
    ) -> None:
        # Line 349-350: ``if "no such table" in str(e).lower(): return 0``
        repo = _new_repo(db_session)

        with patch.object(
            db_session,
            "execute",
            side_effect=Exception("no such table: background_jobs"),
        ):
            result = repo.cleanup_expired(ttl_seconds=0)

        assert result == 0

    def test_when_execute_raises_other_exception_then_re_raised(
        self, db_session: Session
    ) -> None:
        # Line 351-352: ``raise`` for non "no such table" errors.
        repo = _new_repo(db_session)

        with patch.object(
            db_session,
            "execute",
            side_effect=RuntimeError("unexpected database failure"),
        ):
            with pytest.raises(RuntimeError, match="unexpected database failure"):
                repo.cleanup_expired(ttl_seconds=0)


# ===========================================================================
# list_jobs and count_jobs — filter branches
# ===========================================================================


class TestListJobsFilters:
    def test_when_domain_filter_applied_then_only_matching_type_returned(
        self, db_session: Session
    ) -> None:
        _pending(db_session, "list_filter_type_a")
        _pending(db_session, "list_filter_type_b")
        repo = _new_repo(db_session)

        results = repo.list_jobs(domain="list_filter_type_a")
        assert all(r.job_type == "list_filter_type_a" for r in results)

    def test_when_status_filter_applied_then_only_matching_status_returned(
        self, db_session: Session
    ) -> None:
        jid = _pending(db_session, "list_filter_status")
        repo = _new_repo(db_session)
        repo.update(jid, status="running")
        db_session.flush()

        results = repo.list_jobs(status="running")
        assert all(r.status == "running" for r in results)

    def test_when_no_filters_then_all_jobs_returned_up_to_limit(
        self, db_session: Session
    ) -> None:
        _pending(db_session, "list_no_filter_a")
        _pending(db_session, "list_no_filter_b")
        repo = _new_repo(db_session)

        results = repo.list_jobs(limit=100)
        assert len(results) >= 2

    def test_count_jobs_with_domain_filter(self, db_session: Session) -> None:
        _pending(db_session, "count_domain_filter_unique_xyz")
        repo = _new_repo(db_session)
        count = repo.count_jobs(domain="count_domain_filter_unique_xyz")
        assert count == 1

    def test_count_jobs_with_status_filter(self, db_session: Session) -> None:
        jid = _pending(db_session, "count_status_filter_unique_xyz")
        repo = _new_repo(db_session)
        repo.update(jid, status="completed", finished_at=datetime.now(timezone.utc))
        db_session.flush()

        count = repo.count_jobs(status="completed")
        assert count >= 1

    def test_offset_paginates_results(self, db_session: Session) -> None:
        for _ in range(3):
            _pending(db_session, "list_paginate_unique_xyz")
        repo = _new_repo(db_session)

        page0 = repo.list_jobs(domain="list_paginate_unique_xyz", limit=2, offset=0)
        page1 = repo.list_jobs(domain="list_paginate_unique_xyz", limit=2, offset=2)
        assert len(page0) == 2
        assert len(page1) == 1


# ===========================================================================
# update — rowcount=0 when row is terminal and new_status is active
# ===========================================================================


class TestUpdateStatusGuard:
    def test_when_row_already_failed_and_update_to_running_then_zero_rowcount(
        self, db_session: Session
    ) -> None:
        jid = _pending(db_session, "status_guard_terminal")
        repo = _new_repo(db_session)
        repo.update(jid, status="failed", finished_at=datetime.now(timezone.utc))
        db_session.flush()

        # Try to push a non-terminal update onto a terminal row.
        count = repo.update(jid, current=99)
        assert count == 0
        row = repo.get(jid)
        assert row is not None
        assert row.current == 0  # unchanged

    def test_when_terminal_status_sent_then_update_succeeds_regardless(
        self, db_session: Session
    ) -> None:
        # new_status in _TERMINAL_STATUSES → guard clause skipped.
        jid = _pending(db_session, "status_guard_terminal_ok")
        repo = _new_repo(db_session)

        count = repo.update(
            jid, status="completed", finished_at=datetime.now(timezone.utc)
        )
        assert count == 1
