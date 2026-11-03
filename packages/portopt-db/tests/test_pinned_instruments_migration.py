"""Tests for the ``add_pinned_instruments`` Alembic revision.

Loads the revision module by path and replays ``upgrade()`` / ``downgrade()``
against an in-memory SQLite engine via a raw ``MigrationContext`` +
``Operations`` (no Postgres needed, so it runs in CI). Asserts the table, its
unique constraint and its single source index appear on upgrade and are gone on
downgrade, and that the revision chains from the previous head.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "e3a5c7d9b1f4_add_pinned_instruments.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "e3a5c7d9b1f4"
EXPECTED_DOWN_REVISION = "d2f4b6a8c0e1"

TABLE = "pinned_instruments"
EXPECTED_CONSTRAINT = "uq_pinned_instruments_ticker_source"
EXPECTED_INDEX = "ix_pinned_instruments_source"


def _load_migration() -> ModuleType:
    """Import the revision module directly from its file path."""
    spec = importlib.util.spec_from_file_location(
        "pinned_instruments_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sqlite_engine():
    """A fresh in-memory SQLite engine for one migration replay."""
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _run(engine, fn_name: str) -> None:
    """Run the revision's ``upgrade`` or ``downgrade`` on ``engine``."""
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    mod = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(mod, fn_name)()


def test_revision_ids_chain_from_previous_head() -> None:
    """The revision id is as expected and chains from the previous head."""
    mod = _load_migration()
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


@pytest.fixture
def upgraded_engine():
    """A SQLite engine with the revision's ``upgrade()`` applied."""
    engine = _sqlite_engine()
    _run(engine, "upgrade")
    return engine


def test_upgrade_creates_the_table_columns(upgraded_engine) -> None:
    """Upgrade creates ``pinned_instruments`` with its columns."""
    insp = inspect(upgraded_engine)
    assert TABLE in set(insp.get_table_names())
    cols = {c["name"]: c for c in insp.get_columns(TABLE)}
    assert set(cols) == {"id", "ticker", "source", "created_at", "updated_at"}
    assert cols["ticker"]["nullable"] is False
    assert cols["source"]["nullable"] is False


def test_upgrade_creates_the_unique_constraint(upgraded_engine) -> None:
    """Upgrade creates the ticker-source unique constraint."""
    names = {
        uc["name"] for uc in inspect(upgraded_engine).get_unique_constraints(TABLE)
    }
    assert EXPECTED_CONSTRAINT in names


def test_upgrade_creates_only_the_source_index(upgraded_engine) -> None:
    """Upgrade creates the source index exactly once and no implicit duplicate."""
    indexes = [i["name"] for i in inspect(upgraded_engine).get_indexes(TABLE)]
    assert indexes == [EXPECTED_INDEX]


def test_downgrade_removes_the_table_and_index() -> None:
    """Downgrade drops the table and index created by upgrade."""
    engine = _sqlite_engine()
    _run(engine, "upgrade")
    assert TABLE in set(inspect(engine).get_table_names())

    _run(engine, "downgrade")
    assert TABLE not in set(inspect(engine).get_table_names())
