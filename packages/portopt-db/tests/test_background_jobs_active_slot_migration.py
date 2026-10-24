"""Tests for the ``add_background_jobs_active_slot_index`` Alembic revision.

Mirrors ``test_fund_autonomy_state_migration.py``: loads the revision module by
path and replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite
engine via a raw ``MigrationContext`` + ``Operations`` (no Postgres needed, so it
runs in CI). The table is hand-built with only the columns the revision touches,
because it must be created *without* the index so duplicate active rows can
exist beforehand, as they can on a live database that predates the index.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "e3a5c7b9d1f4_add_background_jobs_active_slot_index.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "e3a5c7b9d1f4"
EXPECTED_DOWN_REVISION = "f5a7c9e1b3d2"

TABLE = "background_jobs"
INDEX_NAME = "uq_background_jobs_active_type"


def _load_migration() -> ModuleType:
    """Import the revision module directly from its file path."""
    spec = importlib.util.spec_from_file_location(
        "background_jobs_active_slot_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(engine, fn_name: str) -> None:
    mod = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(mod, fn_name)()


@pytest.fixture
def pre_migration_engine():
    """A SQLite engine holding ``background_jobs`` as it was before the index."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE background_jobs ("
                "id CHAR(32) PRIMARY KEY, job_type VARCHAR(100) NOT NULL, "
                "status VARCHAR(20) NOT NULL, started_at DATETIME NOT NULL, "
                "finished_at DATETIME, error TEXT)"
            )
        )
    return engine


def _insert(engine, job_id: str, job_type: str, status: str, started_at: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO background_jobs (id, job_type, status, started_at) "
                "VALUES (:id, :jt, :st, :sa)"
            ),
            {"id": job_id, "jt": job_type, "st": status, "sa": started_at},
        )


def _status_by_id(engine) -> dict[str, tuple[str, str | None, str | None]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, status, error, finished_at FROM background_jobs")
        ).all()
    return {r.id: (r.status, r.error, r.finished_at) for r in rows}


def test_revision_id_is_correct() -> None:
    assert _load_migration().revision == EXPECTED_REVISION


def test_down_revision_chains_from_head() -> None:
    assert _load_migration().down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_creates_the_unique_partial_index(pre_migration_engine) -> None:
    _run(pre_migration_engine, "upgrade")

    indexes = {i["name"]: i for i in inspect(pre_migration_engine).get_indexes(TABLE)}
    assert INDEX_NAME in indexes
    assert indexes[INDEX_NAME]["unique"]
    assert indexes[INDEX_NAME]["column_names"] == ["job_type"]


def test_upgrade_enforces_one_active_row_per_job_type(pre_migration_engine) -> None:
    _run(pre_migration_engine, "upgrade")
    _insert(
        pre_migration_engine, "a" * 32, "yfinance", "running", "2026-01-01 00:00:00"
    )

    with pytest.raises(IntegrityError):
        _insert(
            pre_migration_engine, "b" * 32, "yfinance", "pending", "2026-01-01 00:00:01"
        )
    _insert(
        pre_migration_engine, "c" * 32, "yfinance", "completed", "2026-01-01 00:00:02"
    )
    _insert(pre_migration_engine, "d" * 32, "macro", "running", "2026-01-01 00:00:03")


def test_upgrade_demotes_all_but_the_newest_duplicate_active_row(
    pre_migration_engine,
) -> None:
    """Rows a past race left behind must not make the index creation fail."""
    _insert(
        pre_migration_engine, "1" * 32, "yfinance", "running", "2026-01-01 00:00:00"
    )
    _insert(
        pre_migration_engine, "2" * 32, "yfinance", "pending", "2026-01-03 00:00:00"
    )
    _insert(
        pre_migration_engine, "3" * 32, "yfinance", "running", "2026-01-02 00:00:00"
    )
    _insert(
        pre_migration_engine, "4" * 32, "yfinance", "completed", "2026-01-04 00:00:00"
    )
    _insert(pre_migration_engine, "5" * 32, "macro", "running", "2026-01-01 00:00:00")

    _run(pre_migration_engine, "upgrade")

    state = _status_by_id(pre_migration_engine)
    assert state["2" * 32][0] == "pending"
    assert state["5" * 32][0] == "running"
    assert state["4" * 32][0] == "completed"
    for demoted in ("1" * 32, "3" * 32):
        status, error, finished_at = state[demoted]
        assert status == "failed"
        assert error and "duplicate" in error
        assert finished_at is not None


def test_downgrade_drops_the_index(pre_migration_engine) -> None:
    _run(pre_migration_engine, "upgrade")
    _run(pre_migration_engine, "downgrade")

    names = {i["name"] for i in inspect(pre_migration_engine).get_indexes(TABLE)}
    assert INDEX_NAME not in names
    assert TABLE in set(inspect(pre_migration_engine).get_table_names())
