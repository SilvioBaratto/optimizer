"""Backfill revision: JSON-literal ``null`` -> SQL NULL in the job and journal columns.

The revision is replayed by path against in-memory SQLite with the tables built from
the models, so it runs in CI without PostgreSQL; the ``CAST(col AS TEXT)`` predicate
it uses reads the literal back as ``null`` on both backends.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from portopt_db.models.jobs.background_job import BackgroundJob
from portopt_db.models.orders.portfolio_journal import PortfolioJournal

MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_FILENAME = "b8d0f2a4c6e1_backfill_job_journal_json_sql_null.py"

EXPECTED_REVISION = "b8d0f2a4c6e1"
EXPECTED_DOWN_REVISION = "f4b6d8a0c2e3"

_REAL_VALUE = '{"AAA": 1.0}'
_JSON_STRING_NULL = '"null"'

_LITERAL, _SQL_NULL, _VALUE, _STRING = ("a" * 32, "b" * 32, "c" * 32, "d" * 32)


def _load() -> ModuleType:
    path = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())
    spec = importlib.util.spec_from_file_location(MIGRATION_FILENAME[:-3], path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(engine, fn) -> None:
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _insert_job(conn, job_id: str, cell: str | None) -> None:
    conn.execute(
        text(
            "INSERT INTO background_jobs (id, job_type, status, extra, result) "
            "VALUES (:id, :job_type, 'completed', :cell, :cell)"
        ),
        {"id": job_id, "job_type": f"job-{job_id[0]}", "cell": cell},
    )


def _insert_journal(conn, row_id: str, cell: str | None) -> None:
    conn.execute(
        text(
            "INSERT INTO portfolio_journal "
            "(id, portfolio_id, as_of, trades, allocation, drift, narrative) "
            "VALUES (:id, :id, :as_of, :cell, :cell, :cell, '')"
        ),
        {"id": row_id, "as_of": f"2026-01-0{ord(row_id[0]) - 96}", "cell": cell},
    )


@pytest.fixture
def engine():
    """SQLite with both tables, seeded with one row per storage shape."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    BackgroundJob.__table__.create(engine)
    PortfolioJournal.__table__.create(engine)
    shapes = {
        _LITERAL: "null",
        _SQL_NULL: None,
        _VALUE: _REAL_VALUE,
        _STRING: _JSON_STRING_NULL,
    }
    with engine.begin() as conn:
        for row_id, cell in shapes.items():
            _insert_job(conn, row_id, cell)
            _insert_journal(conn, row_id, cell)
    return engine


def _cells(engine) -> dict[str, dict[str, tuple[str | None, ...]]]:
    with engine.connect() as conn:
        jobs = conn.execute(text("SELECT id, extra, result FROM background_jobs")).all()
        journal = conn.execute(
            text("SELECT id, trades, allocation, drift FROM portfolio_journal")
        ).all()
    return {
        "jobs": {row.id: tuple(row[1:]) for row in jobs},
        "journal": {row.id: tuple(row[1:]) for row in journal},
    }


def test_revision_chains_from_the_current_head() -> None:
    """The revision extends the single Alembic head."""
    mod = _load()
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_turns_legacy_json_null_into_sql_null(engine) -> None:
    """Historical literal ``null`` rows become SQL NULL in every nullable column."""
    _run(engine, _load().upgrade)

    cells = _cells(engine)
    assert cells["jobs"][_LITERAL] == (None, None)
    assert cells["jobs"][_SQL_NULL] == (None, None)
    assert cells["journal"][_LITERAL] == (None, None, None)
    assert cells["journal"][_SQL_NULL] == (None, None, None)


def test_upgrade_keeps_real_values_and_json_strings(engine) -> None:
    """Populated cells, including the JSON string ``"null"``, are left alone."""
    _run(engine, _load().upgrade)

    cells = _cells(engine)
    assert cells["jobs"][_VALUE] == (_REAL_VALUE,) * 2
    assert cells["jobs"][_STRING] == (_JSON_STRING_NULL,) * 2
    assert cells["journal"][_VALUE] == (_REAL_VALUE,) * 3
    assert cells["journal"][_STRING] == (_JSON_STRING_NULL,) * 3


def test_upgrade_is_idempotent(engine) -> None:
    """Running the backfill twice leaves the data as after the first pass."""
    mod = _load()
    _run(engine, mod.upgrade)
    first = _cells(engine)

    _run(engine, mod.upgrade)

    assert _cells(engine) == first


def test_downgrade_restores_the_legacy_literal_for_empty_values(engine) -> None:
    """Rolling back stores the legacy literal for every empty cell, real values stay."""
    mod = _load()
    _run(engine, mod.upgrade)

    _run(engine, mod.downgrade)

    cells = _cells(engine)
    assert cells["jobs"][_LITERAL] == ("null", "null")
    assert cells["jobs"][_SQL_NULL] == ("null", "null")
    assert cells["journal"][_SQL_NULL] == ("null", "null", "null")
    assert cells["jobs"][_VALUE] == (_REAL_VALUE,) * 2
    assert cells["journal"][_STRING] == (_JSON_STRING_NULL,) * 3
