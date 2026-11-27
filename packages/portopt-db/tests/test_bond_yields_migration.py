"""Tests for the ``create_bond_yields`` Alembic revision.

Replays ``upgrade()`` / ``downgrade()`` against in-memory SQLite through a raw
``MigrationContext`` + ``Operations`` (no Postgres needed). The upgrade must
build the table as ``_BOND_YIELDS`` declares it, and must be a no-op on databases
where ingestion's ``create_all`` already created it, so existing deployments stay
valid. The model that once mapped the table is gone, so ``_BOND_YIELDS`` stands in
for the table ``create_all`` built.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    Uuid,
    create_engine,
    func,
    inspect,
)
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "f5a7c9e1b3d2_create_bond_yields.py"
MIGRATION_PATH = (
    Path(__file__).resolve().parents[1] / "alembic" / "versions" / MIGRATION_FILENAME
)

EXPECTED_REVISION = "f5a7c9e1b3d2"
EXPECTED_DOWN_REVISION = "d8f0b2c4e6a1"

TABLE = "bond_yields"
UNIQUE_NAME = "uq_bond_yield_country_maturity"
INDEX_NAME = "ix_bond_yields_country"

_BOND_YIELDS = Table(
    TABLE,
    MetaData(),
    Column("id", Uuid, primary_key=True),
    Column("country", String(100), nullable=False),
    Column("maturity", String(10), nullable=False),
    Column("yield_value", Float),
    Column("day_change", Float),
    Column("month_change", Float),
    Column("year_change", Float),
    Column("reference_date", Date),
    Column(
        "created_at", DateTime(timezone=True), nullable=False, server_default=func.now()
    ),
    Column(
        "updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()
    ),
    UniqueConstraint("country", "maturity", name=UNIQUE_NAME),
    Index(INDEX_NAME, "country"),
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "create_bond_yields_migration", MIGRATION_PATH
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


def _run(engine, fn_name: str) -> None:
    mod = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(mod, fn_name)()


@pytest.fixture
def upgraded_engine():
    """A SQLite engine with the revision's ``upgrade()`` applied."""
    engine = _sqlite_engine()
    _run(engine, "upgrade")
    return engine


def test_revision_chains_from_the_nav_history_head() -> None:
    """The revision id and parent match the expected chain position."""
    mod = _load_migration()

    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_creates_the_table(upgraded_engine) -> None:
    """Upgrade creates the ``bond_yields`` table."""
    assert TABLE in set(inspect(upgraded_engine).get_table_names())


def test_upgrade_creates_the_expected_columns(upgraded_engine) -> None:
    """Upgrade yields the column names and nullability ``_BOND_YIELDS`` declares."""
    db_columns = {c["name"]: c for c in inspect(upgraded_engine).get_columns(TABLE)}

    assert set(db_columns) == {c.name for c in _BOND_YIELDS.columns}
    for column in _BOND_YIELDS.columns:
        assert db_columns[column.name]["nullable"] == column.nullable, column.name


def test_upgrade_creates_the_upsert_conflict_target(upgraded_engine) -> None:
    """The (country, maturity) unique constraint backs the repository upsert."""
    uniques = {
        uc["name"]: uc["column_names"]
        for uc in inspect(upgraded_engine).get_unique_constraints(TABLE)
    }

    assert uniques[UNIQUE_NAME] == ["country", "maturity"]


def test_upgrade_creates_the_country_index(upgraded_engine) -> None:
    """Upgrade creates the country lookup index."""
    indexes = {
        ix["name"]: ix["column_names"]
        for ix in inspect(upgraded_engine).get_indexes(TABLE)
    }

    assert indexes[INDEX_NAME] == ["country"]


def test_upgrade_is_a_noop_when_create_all_already_built_the_table() -> None:
    """A database built by ``create_all`` upgrades without a duplicate-table error."""
    engine = _sqlite_engine()
    _BOND_YIELDS.create(bind=engine)

    _run(engine, "upgrade")

    assert TABLE in set(inspect(engine).get_table_names())


def test_downgrade_removes_the_table(upgraded_engine) -> None:
    """Downgrade drops the table created by upgrade."""
    _run(upgraded_engine, "downgrade")

    assert TABLE not in set(inspect(upgraded_engine).get_table_names())


def test_downgrade_keeps_a_table_that_holds_rows_from_before_the_revision() -> None:
    """The upgrade adopts a ``create_all`` table, so rows in it are not ours to drop."""
    engine = _sqlite_engine()
    _BOND_YIELDS.create(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            _BOND_YIELDS.insert().values(
                id=uuid.uuid4(), country="Italy", maturity="10Y", yield_value=3.5
            )
        )
    _run(engine, "upgrade")

    _run(engine, "downgrade")

    assert TABLE in set(inspect(engine).get_table_names())
    with engine.connect() as conn:
        assert conn.execute(_BOND_YIELDS.select()).first() is not None
