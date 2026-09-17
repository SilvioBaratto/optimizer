"""T3.6 — the ``add_paper_orders`` Alembic revision.

Mirrors ``test_agent_audit_migration.py``: loads the revision module by path and
replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite engine via a
raw ``MigrationContext`` + ``Operations`` (no Postgres needed, so it runs in CI).
Asserts the ``paper_orders`` table, its indexes and the idempotency UNIQUE
constraint appear on upgrade and are gone on downgrade, and that the revision
chains from the agent-audit head.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "f0a1b2c3d4e5_add_paper_orders.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "f0a1b2c3d4e5"
EXPECTED_DOWN_REVISION = "e9f0a1b2c3d4"

ORDER_TABLES = {"paper_orders"}
EXPECTED_INDEXES = {
    "ix_paper_orders_portfolio_id",
    "ix_paper_orders_asof",
}
EXPECTED_UNIQUE = "uq_paper_order_key"


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "paper_orders_migration", MIGRATION_PATH
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


def test_down_revision_chains_from_agent_audit_head() -> None:
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


def test_upgrade_creates_paper_orders_table(upgraded_engine) -> None:
    tables = set(inspect(upgraded_engine).get_table_names())
    assert tables >= ORDER_TABLES


def test_upgrade_creates_expected_indexes(upgraded_engine) -> None:
    names = {idx["name"] for idx in inspect(upgraded_engine).get_indexes("paper_orders")}
    assert names >= EXPECTED_INDEXES


def test_upgrade_creates_idempotency_unique_constraint(upgraded_engine) -> None:
    constraints = {
        uc["name"]
        for uc in inspect(upgraded_engine).get_unique_constraints("paper_orders")
    }
    assert EXPECTED_UNIQUE in constraints


def test_downgrade_removes_paper_orders_table() -> None:
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
    assert set(inspect(engine).get_table_names()) >= ORDER_TABLES

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.downgrade()
    assert set(inspect(engine).get_table_names()).isdisjoint(ORDER_TABLES)
