"""Tests for the ``add_etf_preferred_convertible_pct`` Alembic revision.

Replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite engine via a
raw ``MigrationContext`` + ``Operations`` (no Postgres needed, so it runs in CI).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "e9a1c3d5f7b2_add_etf_preferred_convertible_pct.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "e9a1c3d5f7b2"
EXPECTED_DOWN_REVISION = "d8f0b2c4e6a1"

TABLE = "etf_asset_classes"
NEW_COLUMNS = {"preferred_pct", "convertible_pct"}


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "etf_asset_class_positions_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine_with_pre_migration_table():
    """SQLite engine holding ``etf_asset_classes`` as it stood before the revision."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    meta = sa.MetaData()
    sa.Table(
        TABLE,
        meta,
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("stock_pct", sa.Numeric(10, 6)),
        sa.Column("bond_pct", sa.Numeric(10, 6)),
        sa.Column("cash_pct", sa.Numeric(10, 6)),
        sa.Column("other_pct", sa.Numeric(10, 6)),
    )
    meta.create_all(engine)
    return engine


def _run(engine, fn_name: str) -> None:
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    mod = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(mod, fn_name)()


def test_revision_chains_from_nav_history() -> None:
    mod = _load_migration()
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_adds_nullable_numeric_columns() -> None:
    engine = _engine_with_pre_migration_table()
    _run(engine, "upgrade")

    cols = {c["name"]: c for c in inspect(engine).get_columns(TABLE)}
    assert set(cols) >= NEW_COLUMNS
    for name in NEW_COLUMNS:
        assert cols[name]["nullable"] is True


def test_downgrade_drops_the_columns() -> None:
    engine = _engine_with_pre_migration_table()
    _run(engine, "upgrade")
    _run(engine, "downgrade")

    names = {c["name"] for c in inspect(engine).get_columns(TABLE)}
    assert not (NEW_COLUMNS & names)


def test_model_declares_the_migrated_columns() -> None:
    from portopt_db.models.market_data.etf_metadata import ETFAssetClass

    assert set(ETFAssetClass.__table__.columns.keys()) >= NEW_COLUMNS


@pytest.mark.parametrize("name", sorted(NEW_COLUMNS))
def test_model_columns_match_migration_type(name: str) -> None:
    from portopt_db.models.market_data.etf_metadata import ETFAssetClass

    col = ETFAssetClass.__table__.columns[name]
    assert col.nullable is True
    assert (col.type.precision, col.type.scale) == (10, 6)
