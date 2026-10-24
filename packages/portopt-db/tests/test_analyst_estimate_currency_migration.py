"""Tests for the ``add_currency_to_analyst_estimate_tables`` Alembic revision.

Replays the revisions that create the four forward-estimate tables and then this
one against in-memory SQLite through a raw ``MigrationContext``. Asserts the
nullable ``currency`` column appears on each table (stored rows stay NULL, i.e.
"currency unknown"), disappears on downgrade, and that the ORM models declare it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

from portopt_db.models.market_data.yfinance_data import (
    EarningsEstimate,
    EpsRevisions,
    EpsTrend,
    RevenueEstimate,
)

MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
BASE_FILENAMES = (
    "e1e2e3e4e5e6_add_analyst_estimate_tables.py",
    "c4d5e6f7a8b9_add_eps_trend_revisions_tables.py",
)
MIGRATION_FILENAME = "a7c9e1b3d5f2_add_currency_to_analyst_estimate_tables.py"

EXPECTED_REVISION = "a7c9e1b3d5f2"
EXPECTED_DOWN_REVISION = "70393a606218"

TABLES = ("earnings_estimate", "revenue_estimate", "eps_trend", "eps_revisions")
COLUMN = "currency"


def _load(filename: str, name: str) -> ModuleType:
    """Import a revision module directly from its file path."""
    path = str((MIGRATION_DIR / filename).resolve())
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(engine, fn) -> None:
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _engine_at_base():
    """SQLite engine with the four estimate tables (and one legacy row) in place."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for i, filename in enumerate(BASE_FILENAMES):
        _run(engine, _load(filename, f"estimate_base_{i}").upgrade)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO earnings_estimate (id, instrument_id, period, avg) "
                "VALUES ('a', 'i', '0q', 1.5)"
            )
        )
    return engine


def test_revision_id_and_down_revision() -> None:
    mod = _load(MIGRATION_FILENAME, "estimate_currency")
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_adds_a_nullable_currency_column_to_every_estimate_table() -> None:
    engine = _engine_at_base()

    _run(engine, _load(MIGRATION_FILENAME, "estimate_currency").upgrade)

    for table in TABLES:
        columns = {c["name"]: c for c in inspect(engine).get_columns(table)}
        assert COLUMN in columns, table
        assert columns[COLUMN]["nullable"] is True, table


def test_upgrade_keeps_existing_rows_with_unknown_currency() -> None:
    engine = _engine_at_base()

    _run(engine, _load(MIGRATION_FILENAME, "estimate_currency").upgrade)

    with engine.connect() as conn:
        row = conn.execute(text("SELECT avg, currency FROM earnings_estimate")).one()
    assert row[0] == 1.5
    assert row[1] is None


def test_downgrade_drops_the_column_and_keeps_the_tables() -> None:
    engine = _engine_at_base()
    mod = _load(MIGRATION_FILENAME, "estimate_currency")
    _run(engine, mod.upgrade)

    _run(engine, mod.downgrade)

    for table in TABLES:
        assert COLUMN not in {c["name"] for c in inspect(engine).get_columns(table)}
    assert set(TABLES) <= set(inspect(engine).get_table_names())


@pytest.mark.parametrize(
    "model", [EarningsEstimate, RevenueEstimate, EpsTrend, EpsRevisions]
)
def test_models_declare_a_nullable_currency_column(model) -> None:
    column = model.__table__.c[COLUMN]
    assert column.nullable is True
    assert column.type.length == 10
