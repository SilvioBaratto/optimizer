"""Branch-coverage tests for app/repositories/_shared/base.py.

Targets BaseRepository CRUD operations and the RepositoryBase._upsert
empty-rows early-return branch.  All tests use the real ``db_session``
SQLite SAVEPOINT fixture.  The _upsert PG-only path (ON CONFLICT DO UPDATE)
is NOT tested with real SQL — only the empty-rows guard (line 41–42) is
exercised, which returns 0 without touching the DB.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import BaseModel as PydanticBase
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.jobs.background_job import BackgroundJob
from app.repositories._shared.base import BaseRepository, RepositoryBase

# ---------------------------------------------------------------------------
# Minimal Pydantic schemas that mirror the BackgroundJob create/update surface
# ---------------------------------------------------------------------------


class JobCreate(PydanticBase):
    job_type: str
    status: str = "pending"
    current: int = 0
    total: int = 0


class JobUpdate(PydanticBase):
    status: str | None = None
    current: int | None = None
    total: int | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_repo(session: Session) -> BaseRepository[BackgroundJob, JobCreate, JobUpdate]:
    return BaseRepository(BackgroundJob, session)


def _seed_job(session: Session, job_type: str = "base_branch_test") -> BackgroundJob:
    repo = _make_repo(session)
    return repo.create_from_dict(
        {
            "id": uuid.uuid4(),
            "job_type": job_type,
            "status": "pending",
            "current": 0,
            "total": 0,
        }
    )


# ===========================================================================
# RepositoryBase._upsert — empty-rows early-return (PG-only upsert NOT called)
# ===========================================================================


class TestUpsertEmptyRowsGuard:
    """_upsert with an empty rows list returns 0 without any DB interaction."""

    def test_when_rows_empty_then_returns_zero(self, db_session: Session) -> None:
        # _upsert ON CONFLICT is Postgres-only; only the early-return branch
        # (if not rows: return 0) executes on SQLite safely.
        repo = RepositoryBase(db_session)
        result = repo._upsert(BackgroundJob, [], "some_constraint")
        assert result == 0

    def test_when_rows_empty_with_update_columns_then_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = RepositoryBase(db_session)
        result = repo._upsert(
            BackgroundJob, [], "some_constraint", update_columns=["status"]
        )
        assert result == 0


# ===========================================================================
# get — found and not-found branches
# ===========================================================================


class TestGet:
    def test_when_id_exists_then_returns_instance(self, db_session: Session) -> None:
        row = _seed_job(db_session)
        repo = _make_repo(db_session)
        result = repo.get(row.id)
        assert result is not None
        assert result.id == row.id

    def test_when_id_missing_then_returns_none(self, db_session: Session) -> None:
        repo = _make_repo(db_session)
        result = repo.get(uuid.uuid4())
        assert result is None


# ===========================================================================
# get_by_field — found and not-found branches
# ===========================================================================


class TestGetByField:
    def test_when_field_matches_then_returns_instance(
        self, db_session: Session
    ) -> None:
        row = _seed_job(db_session, "gbf_match")
        repo = _make_repo(db_session)
        result = repo.get_by_field("job_type", "gbf_match")
        assert result is not None
        assert result.id == row.id

    def test_when_field_has_no_match_then_returns_none(
        self, db_session: Session
    ) -> None:
        repo = _make_repo(db_session)
        result = repo.get_by_field("job_type", "no_such_type_xyz_base_branch")
        assert result is None


# ===========================================================================
# get_multi — ordering branches
# ===========================================================================


class TestGetMulti:
    def test_default_ordering_falls_back_to_created_at(
        self, db_session: Session
    ) -> None:
        # order_by=None → created_at fallback branch (line 122)
        _seed_job(db_session, "gm_default_a")
        _seed_job(db_session, "gm_default_b")
        repo = _make_repo(db_session)
        rows = repo.get_multi()
        assert len(rows) >= 2

    def test_explicit_order_by_valid_attr(self, db_session: Session) -> None:
        # order_by present AND hasattr → normal ordering branch (line 120)
        _seed_job(db_session, "gm_ordered_a")
        _seed_job(db_session, "gm_ordered_b")
        repo = _make_repo(db_session)
        rows = repo.get_multi(order_by="status")
        assert len(rows) >= 2

    def test_order_by_nonexistent_attr_falls_to_created_at(
        self, db_session: Session
    ) -> None:
        # order_by present but NOT on model → falls to created_at branch (line 122)
        _seed_job(db_session, "gm_fallback_a")
        repo = _make_repo(db_session)
        rows = repo.get_multi(order_by="no_such_column_xyz")
        # fallback ordering still returns the seeded row (created_at branch)
        assert len(rows) >= 1

    def test_desc_false_uses_asc_ordering(self, db_session: Session) -> None:
        # desc=False branch (line 121/124)
        _seed_job(db_session, "gm_asc_a")
        _seed_job(db_session, "gm_asc_b")
        repo = _make_repo(db_session)
        rows = repo.get_multi(order_by="status", desc=False)
        assert len(rows) >= 2


# ===========================================================================
# get_all
# ===========================================================================


class TestGetAll:
    def test_returns_all_rows(self, db_session: Session) -> None:
        before = _make_repo(db_session).count()
        _seed_job(db_session, "ga_a")
        _seed_job(db_session, "ga_b")
        rows = _make_repo(db_session).get_all()
        assert len(rows) == before + 2


# ===========================================================================
# create — via Pydantic schema (model_dump path)
# ===========================================================================


class TestCreate:
    def test_when_schema_valid_then_row_persisted(self, db_session: Session) -> None:
        repo = _make_repo(db_session)
        schema = JobCreate(job_type="create_via_schema")
        row = repo.create(schema)
        assert row.id is not None
        assert row.job_type == "create_via_schema"
        assert repo.get(row.id) is not None

    def test_when_schema_update_only_changes_given_fields(
        self, db_session: Session
    ) -> None:
        # update path: model_dump(exclude_unset=True)
        existing = _seed_job(db_session, "update_schema_target")
        repo = _make_repo(db_session)
        schema = JobUpdate(current=99)
        updated = repo.update(existing.id, schema)
        assert updated is not None
        assert updated.current == 99
        # status was not in schema, should remain unchanged
        assert updated.status == "pending"

    def test_update_when_id_not_found_returns_none(self, db_session: Session) -> None:
        repo = _make_repo(db_session)
        schema = JobUpdate(current=5)
        result = repo.update(uuid.uuid4(), schema)
        assert result is None


# ===========================================================================
# create_from_dict and update_from_dict
# ===========================================================================


class TestCreateFromDict:
    def test_when_dict_valid_then_row_created(self, db_session: Session) -> None:
        repo = _make_repo(db_session)
        row = repo.create_from_dict(
            {"id": uuid.uuid4(), "job_type": "from_dict", "status": "pending",
             "current": 0, "total": 0}
        )
        assert row.job_type == "from_dict"

    def test_when_missing_required_field_then_integrity_error(
        self, db_session: Session
    ) -> None:
        # job_type is NOT NULL; omitting it triggers IntegrityError on flush.
        # This test runs in its own savepoint context via the fixture.
        repo = _make_repo(db_session)
        with pytest.raises(IntegrityError):
            repo.create_from_dict({})


class TestUpdateFromDict:
    def test_when_id_found_then_updates_fields(self, db_session: Session) -> None:
        row = _seed_job(db_session, "ufd_target")
        repo = _make_repo(db_session)
        result = repo.update_from_dict(row.id, {"current": 42})
        assert result is not None
        assert result.current == 42

    def test_when_id_not_found_returns_none(self, db_session: Session) -> None:
        repo = _make_repo(db_session)
        result = repo.update_from_dict(uuid.uuid4(), {"current": 1})
        assert result is None

    def test_when_field_not_on_model_then_hasattr_guard_skips_it(
        self, db_session: Session
    ) -> None:
        # hasattr guard (line 169): non-existent key must not raise
        row = _seed_job(db_session, "ufd_guard")
        repo = _make_repo(db_session)
        result = repo.update_from_dict(
            row.id, {"current": 7, "nonexistent_field_xyz": "ignored"}
        )
        assert result is not None
        assert result.current == 7


# ===========================================================================
# delete
# ===========================================================================


class TestDelete:
    def test_when_row_exists_then_returns_true(self, db_session: Session) -> None:
        row = _seed_job(db_session, "del_exists")
        repo = _make_repo(db_session)
        ok = repo.delete(row.id)
        assert ok is True
        assert repo.get(row.id) is None

    def test_when_row_missing_then_returns_false(self, db_session: Session) -> None:
        repo = _make_repo(db_session)
        ok = repo.delete(uuid.uuid4())
        assert ok is False


# ===========================================================================
# count, exists, exists_by_field
# ===========================================================================


class TestCountAndExists:
    def test_count_zero_then_increments_after_insert(
        self, db_session: Session
    ) -> None:
        repo = _make_repo(db_session)
        before = repo.count()
        _seed_job(db_session, "count_job")
        assert repo.count() == before + 1

    def test_exists_true_when_row_present(self, db_session: Session) -> None:
        row = _seed_job(db_session, "exists_true")
        assert _make_repo(db_session).exists(row.id) is True

    def test_exists_false_when_row_absent(self, db_session: Session) -> None:
        assert _make_repo(db_session).exists(uuid.uuid4()) is False

    def test_exists_by_field_true(self, db_session: Session) -> None:
        _seed_job(db_session, "ebf_present_unique_xyz")
        assert (
            _make_repo(db_session).exists_by_field(
                "job_type", "ebf_present_unique_xyz"
            )
            is True
        )

    def test_exists_by_field_false(self, db_session: Session) -> None:
        assert (
            _make_repo(db_session).exists_by_field(
                "job_type", "no_such_type_ebf_xyz"
            )
            is False
        )
