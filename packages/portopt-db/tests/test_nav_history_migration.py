"""Tests for the ``add_nav_history`` Alembic revision.

Mirrors ``test_fx_rates_migration.py``: loads the revision module by path and
replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite engine via a
raw ``MigrationContext`` + ``Operations`` (no Postgres needed, so it runs in CI).
Asserts the table + its unique constraint appear on upgrade and are gone on
downgrade, and that the revision chains from the fx_rates revision.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "d8f0b2c4e6a1_add_nav_history.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "d8f0b2c4e6a1"
EXPECTED_DOWN_REVISION = "c7e9a1b3d5f0"

TABLE = "nav_history"
EXPECTED_CONSTRAINT = "uq_nav_history_portfolio_date"


def _load_migration() -> ModuleType:
    """Import the revision module directly from its file path."""
    spec = importlib.util.spec_from_file_location(
        "nav_history_migration", MIGRATION_PATH
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


def test_module_is_importable() -> None:
    """The revision file loads as a module."""
    assert _load_migration() is not None


def test_revision_id_is_correct() -> None:
    """The revision id matches the expected identifier."""
    assert _load_migration().revision == EXPECTED_REVISION


def test_down_revision_chains_from_fx_rates() -> None:
    """The revision chains from the fx_rates revision (T1)."""
    assert _load_migration().down_revision == EXPECTED_DOWN_REVISION


@pytest.fixture
def upgraded_engine():
    """A SQLite engine with the revision's ``upgrade()`` applied."""
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
    return engine


def test_upgrade_creates_the_table(upgraded_engine) -> None:
    """Upgrade creates the ``nav_history`` table."""
    assert TABLE in set(inspect(upgraded_engine).get_table_names())


def test_upgrade_creates_the_unique_constraint(upgraded_engine) -> None:
    """Upgrade creates the portfolio-date unique constraint."""
    names = {
        uc["name"] for uc in inspect(upgraded_engine).get_unique_constraints(TABLE)
    }
    assert EXPECTED_CONSTRAINT in names


def test_downgrade_removes_the_table() -> None:
    """Downgrade drops the table created by upgrade."""
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
    assert TABLE in set(inspect(engine).get_table_names())

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.downgrade()
    assert TABLE not in set(inspect(engine).get_table_names())
