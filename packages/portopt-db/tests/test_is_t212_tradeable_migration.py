"""Tests for the ``add_is_t212_tradeable_to_instruments`` Alembic revision.

Loads the revision module by path and replays ``upgrade()`` / ``downgrade()``
against an in-memory SQLite ``instruments`` table via a raw ``MigrationContext``
+ ``Operations`` (no Postgres needed, so it runs in CI). Existing rows must come
out of the upgrade as "not evaluated" (NULL), not as "not tradeable".
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "f4b6d8a0c2e3_add_is_t212_tradeable_to_instruments.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "f4b6d8a0c2e3"
EXPECTED_DOWN_REVISION = "e3a5c7d9b1f4"

TABLE = "instruments"
COLUMN = "is_t212_tradeable"


def _load_migration() -> ModuleType:
    """Import the revision module directly from its file path."""
    spec = importlib.util.spec_from_file_location(
        "is_t212_tradeable_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine_with_instruments() -> Engine:
    """A SQLite engine holding the pre-upgrade table with one existing row."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE instruments (id TEXT PRIMARY KEY, ticker TEXT NOT NULL)")
        )
        conn.execute(text("INSERT INTO instruments (id, ticker) VALUES ('a', 'AAPL')"))
    return engine


def _run(engine: Engine, fn_name: str) -> None:
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
def upgraded_engine() -> Engine:
    """A SQLite engine with the revision's ``upgrade()`` applied."""
    engine = _engine_with_instruments()
    _run(engine, "upgrade")
    return engine


def test_upgrade_adds_a_nullable_boolean_column(upgraded_engine: Engine) -> None:
    """Upgrade adds ``is_t212_tradeable`` as a nullable boolean."""
    cols = {c["name"]: c for c in inspect(upgraded_engine).get_columns(TABLE)}
    assert COLUMN in cols
    assert cols[COLUMN]["nullable"] is True
    assert "BOOL" in str(cols[COLUMN]["type"]).upper()


def test_existing_rows_come_out_not_evaluated(upgraded_engine: Engine) -> None:
    """A deployment without Trading 212 keeps NULL, never a false 'not tradeable'."""
    with upgraded_engine.connect() as conn:
        value = conn.execute(text(f"SELECT {COLUMN} FROM instruments")).scalar_one()
    assert value is None


def test_downgrade_removes_the_column() -> None:
    """Downgrade drops the column created by upgrade."""
    engine = _engine_with_instruments()
    _run(engine, "upgrade")
    _run(engine, "downgrade")

    cols = {c["name"] for c in inspect(engine).get_columns(TABLE)}
    assert COLUMN not in cols
