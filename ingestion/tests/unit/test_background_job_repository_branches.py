"""Branch-coverage tests for app/repositories/jobs/background_job_repository.py.

Targets branches that the main repository suite leaves unexercised:
``create`` extras, ``update`` errors replacement, ``_partition_kwargs``,
``_build_values`` extras merge, ``_exec_status_guarded_update`` empty values,
``_replace_error_entries``, ``is_any_running``, the ``cleanup_expired``
"no such table" swallow, and the "missing table is never created on demand"
contract.

Does NOT duplicate tests already in test_background_job_repository.py
(claim_or_create, cleanup_expired TTL arithmetic, reconcile_orphans,
liveness predicate, get_latest_by_type).

Tests use the real ``db_session`` SQLite SAVEPOINT fixture, except
``TestMissingTableIsNotCreatedOnDemand``: it needs a database with no tables
at all, which the shared fixture never provides, so it builds its own bare
in-memory engine.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

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
# Schema ownership — Alembic only, the repository never creates tables
# ===========================================================================


class TestMissingTableIsNotCreatedOnDemand:
    """A DB that was never migrated must surface the error, not get a schema.

    ``Base.metadata.create_all`` from here would build every model table
    unversioned, so the next ``alembic upgrade head`` dies with DuplicateTable.
    """

    @pytest.fixture
    def bare_session(self) -> Iterator[Session]:
        engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
        with Session(engine) as session:
            yield session
        engine.dispose()

    def test_cleanup_expired_on_unmigrated_db_creates_no_tables(
        self, bare_session: Session
    ) -> None:
        repo = _new_repo(bare_session)

        assert repo.cleanup_expired(60) == 0

        assert inspect(bare_session.get_bind()).get_table_names() == []

    def test_claim_or_create_on_unmigrated_db_raises_and_creates_no_tables(
        self, bare_session: Session
    ) -> None:
        repo = _new_repo(bare_session)

        with pytest.raises(OperationalError, match="background_jobs"):
            repo.claim_or_create("unmigrated")

        assert inspect(bare_session.get_bind()).get_table_names() == []


# ===========================================================================
# create() — extra kwargs stored vs None
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
# update() — errors_value replacement path
# ===========================================================================


class TestUpdateErrorsValue:
    def test_when_errors_provided_and_row_matched_then_error_entries_stored(
        self, db_session: Session
    ) -> None:
        # Covers rowcount==1 with errors_value is not None.
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
# _partition_kwargs — extras branch
# ===========================================================================


class TestPartitionKwargs:
    def test_non_core_key_goes_to_extras(self) -> None:
        # Key not in _CORE_COLUMNS → extras dict.
        core, extras = BackgroundJobRepository._partition_kwargs(
            {"status": "running", "my_custom_field": "hello"}
        )
        assert "status" in core
        assert "my_custom_field" in extras

    def test_finished_at_string_converted_to_datetime(self) -> None:
        # finished_at + isinstance(value, str) → fromisoformat.
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
# _build_values — extras merge path
# ===========================================================================


class TestBuildValuesMerge:
    def test_when_extras_provided_then_merged_with_existing_extra(
        self, db_session: Session
    ) -> None:
        repo = _new_repo(db_session)
        row = repo.create("build_values_merge", existing_key="old_value")
        db_session.flush()

        # Triggers _build_values extras merge path.
        count = repo.update(row.id, status="running", new_key="new_value")
        db_session.flush()

        assert count == 1
        refreshed = repo.get(row.id)
        assert refreshed is not None
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
# _exec_status_guarded_update — empty values early return
# ===========================================================================


class TestExecStatusGuardedUpdateEmptyValues:
    def test_when_values_empty_then_returns_zero_without_hitting_db(
        self, db_session: Session
    ) -> None:
        # ``if not values: return 0``.
        # Pass errors=None so _replace_error_entries is skipped; pass no
        # core/extras kwargs so values dict ends up empty.
        jid = _pending(db_session, "exec_empty_values")
        repo = _new_repo(db_session)

        # Call _exec_status_guarded_update directly with empty dict.
        result = repo._exec_status_guarded_update(jid, {})
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
# _replace_error_entries — list iteration
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
        # ``if not isinstance(errors_value, list): return``
        jid = _pending(db_session, "replace_errors_non_list")
        repo = _new_repo(db_session)

        repo.update(jid, status="running", errors=["pre_existing"])
        db_session.flush()

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
# is_any_running — true/false branches
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
# cleanup_expired — "no such table" exception swallow
# ===========================================================================


class TestCleanupExpiredNoSuchTable:
    def test_when_execute_raises_no_such_table_then_returns_zero(
        self, db_session: Session
    ) -> None:
        # ``if "no such table" in str(e).lower(): return 0``
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
        # ``raise`` for non "no such table" errors.
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
        repo = _new_repo(db_session)
        for _ in range(3):
            # Finished before the next row: only one active row per type exists.
            repo.update(
                _pending(db_session, "list_paginate_unique_xyz"), status="completed"
            )

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

        count = repo.update(jid, current=99)
        assert count == 0
        row = repo.get(jid)
        assert row is not None
        assert row.current == 0  # unchanged

    def test_when_terminal_status_sent_on_active_row_then_update_succeeds(
        self, db_session: Session
    ) -> None:
        jid = _pending(db_session, "status_guard_terminal_ok")
        repo = _new_repo(db_session)

        count = repo.update(
            jid, status="completed", finished_at=datetime.now(timezone.utc)
        )
        assert count == 1

    @pytest.mark.parametrize(
        ("first", "late"), [("failed", "completed"), ("completed", "failed")]
    )
    def test_when_row_already_terminal_then_late_terminal_write_is_rejected(
        self, db_session: Session, first: str, late: str
    ) -> None:
        """A reaped row's zombie worker must not flip failed -> completed."""
        jid = _pending(db_session, f"status_guard_late_{first}")
        repo = _new_repo(db_session)
        repo.update(jid, status=first, error="orphaned - reaped")
        db_session.flush()

        count = repo.update(jid, status=late, errors=["late"])
        db_session.flush()

        assert count == 0
        row = repo.get(jid)
        assert row is not None
        assert row.status == first
        assert row.error == "orphaned - reaped"
        assert row.errors is None
