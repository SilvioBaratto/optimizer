"""T2.2 — the ``add_agent_audit_tables`` Alembic revision.

Mirrors ``test_new_tables_migration.py``: loads the revision module by path and
replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite engine via a
raw ``MigrationContext`` + ``Operations`` (no Postgres needed, so it runs in CI).
Asserts the two audit tables + their indexes appear on upgrade and are gone on
downgrade, and that the revision chains from the current head.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "e9f0a1b2c3d4_add_agent_audit_tables.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "e9f0a1b2c3d4"
EXPECTED_DOWN_REVISION = "d8e9f0a1b2c3"

AUDIT_TABLES = {"agent_runs", "agent_decisions"}
EXPECTED_INDEXES = {
    "ix_agent_runs_portfolio_id",
    "ix_agent_runs_status",
    "ix_agent_runs_asof",
    "ix_agent_decisions_run_id",
}


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "agent_audit_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sqlite_engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_module_is_importable() -> None:
    assert _load_migration() is not None


def test_revision_id_is_correct() -> None:
    assert _load_migration().revision == EXPECTED_REVISION


def test_down_revision_chains_from_head() -> None:
    assert _load_migration().down_revision == EXPECTED_DOWN_REVISION


@pytest.fixture
def upgraded_engine():
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
    return engine


def test_upgrade_creates_both_tables(upgraded_engine) -> None:
    tables = set(inspect(upgraded_engine).get_table_names())
    assert tables >= AUDIT_TABLES


def test_upgrade_creates_expected_indexes(upgraded_engine) -> None:
    names: set[str] = set()
    inspector = inspect(upgraded_engine)
    for table in AUDIT_TABLES:
        names |= {idx["name"] for idx in inspector.get_indexes(table)}
    assert names >= EXPECTED_INDEXES


def test_downgrade_removes_both_tables() -> None:
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
    assert set(inspect(engine).get_table_names()) >= AUDIT_TABLES

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.downgrade()
    assert set(inspect(engine).get_table_names()).isdisjoint(AUDIT_TABLES)
