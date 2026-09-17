"""T4 — the ``add_mifid_profiles`` Alembic revision.

Mirrors ``test_paper_orders_migration.py``: loads the revision module by path
and replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite engine
via a raw ``MigrationContext`` + ``Operations`` (no Postgres needed, so it runs
in CI). Asserts the ``mifid_profiles`` table, its index and the append-only
``UNIQUE(portfolio_id, version)`` constraint appear on upgrade and are gone on
downgrade, and that the revision chains from the paper-orders head.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "a2b3c4d5e6f7_add_mifid_profiles.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "a2b3c4d5e6f7"
EXPECTED_DOWN_REVISION = "f0a1b2c3d4e5"

PROFILE_TABLES = {"mifid_profiles"}
EXPECTED_INDEXES = {"ix_mifid_profiles_portfolio_id"}
EXPECTED_UNIQUE = "uq_mifid_profile_portfolio_version"


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mifid_profiles_migration", MIGRATION_PATH
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


def test_down_revision_chains_from_paper_orders_head() -> None:
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


def test_upgrade_creates_mifid_profiles_table(upgraded_engine) -> None:
    tables = set(inspect(upgraded_engine).get_table_names())
    assert tables >= PROFILE_TABLES


def test_upgrade_creates_expected_indexes(upgraded_engine) -> None:
    names = {
        idx["name"] for idx in inspect(upgraded_engine).get_indexes("mifid_profiles")
    }
    assert names >= EXPECTED_INDEXES


def test_upgrade_creates_versioning_unique_constraint(upgraded_engine) -> None:
    constraints = {
        uc["name"]
        for uc in inspect(upgraded_engine).get_unique_constraints("mifid_profiles")
    }
    assert EXPECTED_UNIQUE in constraints


def test_downgrade_removes_mifid_profiles_table() -> None:
    engine = _sqlite_engine()
    mod = _load_migration()

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
    assert set(inspect(engine).get_table_names()) >= PROFILE_TABLES

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.downgrade()
    assert set(inspect(engine).get_table_names()).isdisjoint(PROFILE_TABLES)
