"""Tests for the ``add_fx_rates`` Alembic revision.

Mirrors ``test_fund_autonomy_state_migration.py``: loads the revision module by
path and replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite
engine via a raw ``MigrationContext`` + ``Operations`` (no Postgres needed, so it
runs in CI). Asserts the table + its unique constraint appear on upgrade and are
gone on downgrade, and that the revision chains from the current head.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "c7e9a1b3d5f0_add_fx_rates.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "c7e9a1b3d5f0"
EXPECTED_DOWN_REVISION = "e7d6c5b4a3f2"

TABLE = "fx_rates"
EXPECTED_CONSTRAINT = "uq_fx_rates_pair_date"


def _load_migration() -> ModuleType:
    """Import the revision module directly from its file path."""
    spec = importlib.util.spec_from_file_location("fx_rates_migration", MIGRATION_PATH)
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


def test_down_revision_chains_from_head() -> None:
    """The revision chains from the current Alembic head."""
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
    """Upgrade creates the ``fx_rates`` table."""
    assert TABLE in set(inspect(upgraded_engine).get_table_names())


def test_upgrade_creates_the_unique_constraint(upgraded_engine) -> None:
    """Upgrade creates the pair-date unique constraint."""
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
