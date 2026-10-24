"""``agent_decisions.created_at`` index: model declaration and Alembic revision.

The Audit timeline reads ``ORDER BY created_at DESC LIMIT n`` across every run, so
without an index PostgreSQL scans and sorts the whole decision table on each poll.
Mirrors ``test_nav_history_migration.py``: the revision is replayed by path against
in-memory SQLite (after the revision that creates the table), so it runs in CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

from portopt_db.models import AgentDecision

MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
CREATE_TABLES_FILENAME = "e9f0a1b2c3d4_add_agent_audit_tables.py"
MIGRATION_FILENAME = "e4a6c8d0f2b1_add_agent_decisions_created_at_index.py"

EXPECTED_REVISION = "e4a6c8d0f2b1"
EXPECTED_DOWN_REVISION = "d8f0b2c4e6a1"

INDEX = "ix_agent_decisions_created_at"
TABLE = "agent_decisions"


def _load(filename: str) -> ModuleType:
    path = str((MIGRATION_DIR / filename).resolve())
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(engine, fn) -> None:
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _index_names(engine) -> set[str]:
    return {idx["name"] for idx in inspect(engine).get_indexes(TABLE)}


@pytest.fixture
def engine_with_audit_tables():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _run(engine, _load(CREATE_TABLES_FILENAME).upgrade)
    return engine


def test_model_declares_the_created_at_index() -> None:
    names = {idx.name for idx in AgentDecision.__table__.indexes}
    assert INDEX in names


def test_revision_chains_from_the_current_head() -> None:
    mod = _load(MIGRATION_FILENAME)
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_creates_the_index(engine_with_audit_tables) -> None:
    assert INDEX not in _index_names(engine_with_audit_tables)

    _run(engine_with_audit_tables, _load(MIGRATION_FILENAME).upgrade)

    assert INDEX in _index_names(engine_with_audit_tables)


def test_downgrade_drops_the_index(engine_with_audit_tables) -> None:
    mod = _load(MIGRATION_FILENAME)
    _run(engine_with_audit_tables, mod.upgrade)

    _run(engine_with_audit_tables, mod.downgrade)

    assert INDEX not in _index_names(engine_with_audit_tables)
