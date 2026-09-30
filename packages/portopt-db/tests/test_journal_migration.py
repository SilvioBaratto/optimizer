"""Tests for the ``add_market_and_portfolio_journal`` Alembic revision.

Loads the revision module by path via importlib and drives ``upgrade()`` /
``downgrade()`` through a raw ``MigrationContext`` + ``Operations`` on an
in-memory SQLite engine — no live Postgres required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "d5e6f7a8b9c0_add_market_and_portfolio_journal.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "d5e6f7a8b9c0"
EXPECTED_DOWN_REVISION = "c4d5e6f7a8b9"

JOURNAL_TABLES = {"market_journal", "portfolio_journal"}

EXPECTED_MARKET_INDEXES = {"ix_market_journal_as_of"}
EXPECTED_PORTFOLIO_INDEXES = {
    "ix_portfolio_journal_portfolio_id",
    "ix_portfolio_journal_as_of",
}
EXPECTED_MARKET_UNIQUE = "uq_market_journal_as_of"
EXPECTED_PORTFOLIO_UNIQUE = "uq_portfolio_journal_key"


def _load_migration() -> ModuleType:
    """Load the migration module from its file path."""
    spec = importlib.util.spec_from_file_location("journal_migration", MIGRATION_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sqlite_engine():
    """Return a fresh in-memory SQLite engine for isolated migration tests."""
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_module_is_importable() -> None:
    """The migration module must load without errors."""
    assert _load_migration() is not None


def test_revision_id_is_correct() -> None:
    """revision must equal the expected string constant."""
    assert _load_migration().revision == EXPECTED_REVISION


def test_down_revision_chains_from_eps_tables_head() -> None:
    """down_revision must point at the previous single head."""
    assert _load_migration().down_revision == EXPECTED_DOWN_REVISION


@pytest.fixture
def upgraded_engine():
    """Return an engine after upgrade() has been applied."""
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
    """upgrade() must create market_journal and portfolio_journal."""
    tables = set(inspect(upgraded_engine).get_table_names())
    assert tables >= JOURNAL_TABLES


def test_upgrade_creates_market_journal_index(upgraded_engine) -> None:
    """upgrade() must create ix_market_journal_as_of."""
    names = {
        idx["name"] for idx in inspect(upgraded_engine).get_indexes("market_journal")
    }
    assert names >= EXPECTED_MARKET_INDEXES


def test_upgrade_creates_portfolio_journal_indexes(upgraded_engine) -> None:
    """upgrade() must create both portfolio_journal indexes."""
    names = {
        idx["name"] for idx in inspect(upgraded_engine).get_indexes("portfolio_journal")
    }
    assert names >= EXPECTED_PORTFOLIO_INDEXES


def test_upgrade_creates_market_journal_unique_constraint(upgraded_engine) -> None:
    """upgrade() must create uq_market_journal_as_of."""
    constraints = {
        uc["name"]
        for uc in inspect(upgraded_engine).get_unique_constraints("market_journal")
    }
    assert EXPECTED_MARKET_UNIQUE in constraints


def test_upgrade_creates_portfolio_journal_unique_constraint(upgraded_engine) -> None:
    """upgrade() must create uq_portfolio_journal_key."""
    constraints = {
        uc["name"]
        for uc in inspect(upgraded_engine).get_unique_constraints("portfolio_journal")
    }
    assert EXPECTED_PORTFOLIO_UNIQUE in constraints


def test_downgrade_removes_both_tables() -> None:
    """downgrade() must leave neither market_journal nor portfolio_journal."""
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
    assert set(inspect(engine).get_table_names()) >= JOURNAL_TABLES

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.downgrade()
    assert set(inspect(engine).get_table_names()).isdisjoint(JOURNAL_TABLES)
