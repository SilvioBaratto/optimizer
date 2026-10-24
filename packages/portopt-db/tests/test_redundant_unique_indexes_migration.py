"""Guards and replay tests for the redundant-index cleanup revision.

A ``UNIQUE`` constraint is enforced through its own b-tree index, so a plain
index on the same columns in the same order doubles write cost and storage for
no query benefit. Two layers are checked here:

* the ORM metadata (what ``create_all`` builds and ``alembic check`` compares
  against) must not declare such a pair, and
* the revision must drop the plain index from databases built by the older
  chain, replayed against in-memory SQLite through a raw ``MigrationContext`` +
  ``Operations`` (no Postgres needed, so it runs in CI).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import (
    Column,
    Index,
    PrimaryKeyConstraint,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.pool import StaticPool

from portopt_db.models import Base

MIGRATION_FILENAME = "c5d9a1e7b3f8_drop_indexes_duplicating_unique_keys.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "c5d9a1e7b3f8"
EXPECTED_DOWN_REVISION = "e3a5c7b9d1f4"

# (table, redundant plain index, its columns, the unique constraint that covers it)
REDUNDANT = [
    (
        "trading_economics_observations",
        "ix_te_obs_country_key_date",
        ["country", "indicator_key", "date"],
        "uq_te_obs_country_key_date",
    ),
    (
        "bond_yield_observations",
        "ix_bond_obs_country_maturity_date",
        ["country", "maturity", "date"],
        "uq_bond_obs_country_mat_date",
    ),
    (
        "economic_indicators",
        "ix_economic_indicators_country",
        ["country"],
        "uq_economic_indicator_country",
    ),
    (
        "analyst_price_targets",
        "ix_analyst_price_targets_instrument_id",
        ["instrument_id"],
        "uq_analyst_pt_instrument",
    ),
    (
        "market_journal",
        "ix_market_journal_as_of",
        ["as_of"],
        "uq_market_journal_as_of",
    ),
    (
        "etf_metadata",
        "ix_etf_metadata_instrument_id",
        ["instrument_id"],
        "uq_etf_metadata_instrument",
    ),
    (
        "major_holders",
        "ix_major_holders_instrument_id",
        ["instrument_id"],
        "uq_major_holders_instrument",
    ),
    (
        "insider_purchases",
        "ix_insider_purchases_instrument_id",
        ["instrument_id"],
        "uq_insider_purchases_instrument",
    ),
    (
        "ticker_profile_extras",
        "ix_ticker_profile_extras_instrument_id",
        ["instrument_id"],
        "uq_ticker_profile_extras_instrument",
    ),
]
_IDS = [row[0] for row in REDUNDANT]


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "drop_indexes_duplicating_unique_keys_migration", MIGRATION_PATH
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
        with Operations.context(MigrationContext.configure(conn)):
            getattr(mod, fn_name)()


def _index_names(engine, table: str) -> set[str]:
    return {ix["name"] for ix in inspect(engine).get_indexes(table)}


def _legacy_engine():
    """SQLite holding the nine tables the way the pre-cleanup chain left them.

    The ORM no longer declares the redundant indexes, so they are added by hand
    to reproduce a database built by ``alembic upgrade`` from the old revisions.
    """
    engine = _sqlite_engine()
    for table, index, columns, _ in REDUNDANT:
        Base.metadata.tables[table].create(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "{index}" '
                    f'ON "{table}" ({", ".join(columns)})'
                )
            )
    return engine


def _unique_column_sets(table) -> set[tuple[str, ...]]:
    """Ordered column tuples a table already enforces as unique."""
    unique: set[tuple[str, ...]] = set()
    for constraint in table.constraints:
        if isinstance(constraint, (UniqueConstraint, PrimaryKeyConstraint)):
            unique.add(tuple(c.name for c in constraint.columns))
    for index in table.indexes:
        if index.unique and _is_full_column_index(index):
            unique.add(tuple(c.name for c in index.columns))
    return unique


def _is_full_column_index(index: Index) -> bool:
    """An index over plain columns that covers every row (no predicate).

    A partial unique index only constrains the rows its predicate selects, so it
    cannot stand in for a full index on the same columns.
    """
    return index.dialect_options["postgresql"]["where"] is None and all(
        isinstance(e, Column) for e in index.expressions
    )


def test_orm_metadata_declares_no_index_duplicating_a_unique_key() -> None:
    """No model pairs a plain ``Index`` with a unique key on the same columns."""
    offenders = sorted(
        f"{table.name}.{index.name}"
        for table in Base.metadata.tables.values()
        for index in table.indexes
        if not index.unique
        and _is_full_column_index(index)
        and tuple(c.name for c in index.columns) in _unique_column_sets(table)
    )

    assert offenders == []


def test_revision_chains_from_the_previous_head() -> None:
    """The revision id and parent match the expected chain position."""
    mod = _load_migration()

    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


@pytest.mark.parametrize(("table", "index", "columns", "unique"), REDUNDANT, ids=_IDS)
def test_upgrade_drops_the_plain_index_and_keeps_the_unique_key(
    table: str, index: str, columns: list[str], unique: str
) -> None:
    """The plain index goes; the unique constraint that covers it stays."""
    engine = _legacy_engine()
    assert index in _index_names(engine, table)

    _run(engine, "upgrade")

    assert index not in _index_names(engine, table)
    constraints = {
        uc["name"]: uc["column_names"]
        for uc in inspect(engine).get_unique_constraints(table)
    }
    assert constraints[unique] == columns


def test_upgrade_is_idempotent_on_databases_without_the_plain_indexes() -> None:
    """A database built from current metadata has nothing to drop."""
    engine = _sqlite_engine()
    for table, *_ in REDUNDANT:
        Base.metadata.tables[table].create(engine)

    _run(engine, "upgrade")

    for table, _, columns, unique in REDUNDANT:
        constraints = {
            uc["name"]: uc["column_names"]
            for uc in inspect(engine).get_unique_constraints(table)
        }
        assert constraints[unique] == columns


def test_upgrade_keeps_a_plain_index_when_no_unique_key_covers_it() -> None:
    """Without a covering unique key the plain index is the only lookup path."""
    engine = _sqlite_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE market_journal (id TEXT PRIMARY KEY, as_of DATE NOT NULL)"
            )
        )
        conn.execute(
            text("CREATE INDEX ix_market_journal_as_of ON market_journal (as_of)")
        )

    _run(engine, "upgrade")

    assert "ix_market_journal_as_of" in _index_names(engine, "market_journal")


def test_upgrade_skips_tables_the_database_does_not_have() -> None:
    """A partially built database must not make the revision raise."""
    _run(_sqlite_engine(), "upgrade")


def test_downgrade_recreates_every_dropped_index() -> None:
    """Downgrade restores the pre-revision index set."""
    engine = _legacy_engine()
    _run(engine, "upgrade")

    _run(engine, "downgrade")

    for table, index, columns, _ in REDUNDANT:
        restored = {
            ix["name"]: ix["column_names"] for ix in inspect(engine).get_indexes(table)
        }
        assert restored[index] == columns
