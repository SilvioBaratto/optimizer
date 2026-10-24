"""Tests for the ``add_benchmark_ticker_to_nav_history`` Alembic revision.

Mirrors ``test_nav_history_migration.py``: replays the ``nav_history`` creation
revision and then this one against in-memory SQLite via a raw ``MigrationContext``.
Asserts the nullable ``benchmark_ticker`` column appears on upgrade (existing rows
stay NULL, i.e. "identity unknown"), disappears on downgrade, and that the revision
chains from the ``nav_history`` revision.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
BASE_FILENAME = "d8f0b2c4e6a1_add_nav_history.py"
MIGRATION_FILENAME = "c8e0a2d4f6b1_add_benchmark_ticker_to_nav_history.py"

EXPECTED_REVISION = "c8e0a2d4f6b1"
EXPECTED_DOWN_REVISION = "d8f0b2c4e6a1"

TABLE = "nav_history"
COLUMN = "benchmark_ticker"


def _load(filename: str, name: str) -> ModuleType:
    """Import a revision module directly from its file path."""
    path = str((MIGRATION_DIR / filename).resolve())
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(engine, fn) -> None:
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _engine_at_base():
    """SQLite engine with the ``nav_history`` table (and one legacy row) in place."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _run(engine, _load(BASE_FILENAME, "nav_history_base").upgrade)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO nav_history (id, portfolio_id, date, nav, benchmark) "
                "VALUES ('a', 'p', '2026-10-01', 100, 450)"
            )
        )
    return engine


def test_revision_id_and_down_revision() -> None:
    """The revision chains directly from the ``nav_history`` creation revision."""
    mod = _load(MIGRATION_FILENAME, "nav_history_ticker")
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_adds_a_nullable_column_and_keeps_existing_rows() -> None:
    """Upgrade adds ``benchmark_ticker``; the legacy row keeps a NULL identity."""
    engine = _engine_at_base()

    _run(engine, _load(MIGRATION_FILENAME, "nav_history_ticker").upgrade)

    columns = {c["name"]: c for c in inspect(engine).get_columns(TABLE)}
    assert COLUMN in columns
    assert columns[COLUMN]["nullable"] is True
    with engine.connect() as conn:
        value = conn.execute(text(f"SELECT {COLUMN} FROM {TABLE}")).scalar_one()
    assert value is None


def test_downgrade_drops_the_column() -> None:
    """Downgrade removes the column and leaves the table in place."""
    engine = _engine_at_base()
    mod = _load(MIGRATION_FILENAME, "nav_history_ticker")
    _run(engine, mod.upgrade)

    _run(engine, mod.downgrade)

    assert COLUMN not in {c["name"] for c in inspect(engine).get_columns(TABLE)}
    assert TABLE in set(inspect(engine).get_table_names())
