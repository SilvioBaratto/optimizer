"""T2 — the ``add_positions_mandates_run_thread_id`` Alembic revision.

Mirrors ``test_agent_audit_migration.py`` / ``test_paper_orders_migration.py``:
loads the revision module by path and replays ``upgrade()`` / ``downgrade()``
against an in-memory SQLite engine via a raw ``MigrationContext`` +
``Operations`` (no Postgres needed, so it runs in CI). Unlike its siblings this
revision both *creates* tables (``positions``, ``portfolio_mandates``) and *adds
a column* (``agent_runs.thread_id``), so the fixture first stands up the minimal
prerequisite tables from earlier migrations — ``paper_orders`` (the
``positions.paper_order_id`` FK target) and ``agent_runs`` (the table the column
is added to) — before replaying the revision. Asserts the two new tables + their
indexes + unique constraints appear and ``agent_runs`` gains ``thread_id`` on
upgrade, that all of it is gone on downgrade, and that the revision chains from
the current (mifid-profiles) head.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "b3c4d5e6f7a8_add_positions_mandates_run_thread_id.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "b3c4d5e6f7a8"
EXPECTED_DOWN_REVISION = "a2b3c4d5e6f7"

NEW_TABLES = {"positions", "portfolio_mandates"}
EXPECTED_INDEXES = {
    "positions": {"ix_positions_portfolio_id"},
    "portfolio_mandates": {"ix_portfolio_mandates_base_currency"},
}
EXPECTED_UNIQUE = {
    "positions": "uq_position_portfolio_ticker",
    "portfolio_mandates": "uq_portfolio_mandate_portfolio_id",
}


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "positions_mandates_migration", MIGRATION_PATH
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


def _create_prerequisites(op) -> None:
    """Stand up the minimal prior-migration tables this revision touches.

    ``positions.paper_order_id`` FKs ``paper_orders.id`` and the revision adds a
    column to ``agent_runs``; both tables come from earlier migrations, so the
    fresh single-revision replay engine must carry them before ``upgrade()``.
    """
    op.create_table(
        "paper_orders", sa.Column("id", UUID(as_uuid=True), primary_key=True)
    )
    op.create_table("agent_runs", sa.Column("id", UUID(as_uuid=True), primary_key=True))


def test_module_is_importable() -> None:
    assert _load_migration() is not None


def test_revision_id_is_correct() -> None:
    assert _load_migration().revision == EXPECTED_REVISION


def test_down_revision_chains_from_mifid_head() -> None:
    assert _load_migration().down_revision == EXPECTED_DOWN_REVISION


@pytest.fixture
def upgraded_engine():
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx) as op:
            _create_prerequisites(op)
            mod.upgrade()
    return engine


def test_upgrade_creates_both_new_tables(upgraded_engine) -> None:
    tables = set(inspect(upgraded_engine).get_table_names())
    assert tables >= NEW_TABLES


def test_upgrade_creates_expected_indexes(upgraded_engine) -> None:
    inspector = inspect(upgraded_engine)
    for table, expected in EXPECTED_INDEXES.items():
        names = {idx["name"] for idx in inspector.get_indexes(table)}
        assert names >= expected


def test_upgrade_creates_expected_unique_constraints(upgraded_engine) -> None:
    inspector = inspect(upgraded_engine)
    for table, expected in EXPECTED_UNIQUE.items():
        names = {uc["name"] for uc in inspector.get_unique_constraints(table)}
        assert expected in names


def test_upgrade_adds_thread_id_to_agent_runs(upgraded_engine) -> None:
    columns = {c["name"] for c in inspect(upgraded_engine).get_columns("agent_runs")}
    assert "thread_id" in columns


def test_downgrade_removes_new_tables_and_thread_id() -> None:
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx) as op:
            _create_prerequisites(op)
            mod.upgrade()
    assert set(inspect(engine).get_table_names()) >= NEW_TABLES
    assert "thread_id" in {c["name"] for c in inspect(engine).get_columns("agent_runs")}

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.downgrade()

    remaining = set(inspect(engine).get_table_names())
    assert remaining.isdisjoint(NEW_TABLES)
    assert "thread_id" not in {
        c["name"] for c in inspect(engine).get_columns("agent_runs")
    }
