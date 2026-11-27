"""Revision 2266e24c509f, which drops the Trading 212 columns of the universe tables.

The revision is replayed by path against in-memory SQLite, so it runs in CI without
PostgreSQL. The tables are built from the current models and then given back the three
columns the models no longer declare, which reproduces the schema of the previous head.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Boolean, Engine, Integer, String, create_engine, inspect, text
from sqlalchemy.pool import StaticPool

from portopt_db.models.universe.universe import Exchange, Instrument

MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_FILENAME = "2026_11_27_1711-2266e24c509f_drop_trading_212_columns.py"

EXPECTED_REVISION = "2266e24c509f"
EXPECTED_DOWN_REVISION = "b8d0f2a4c6e1"

# (table, column, type the downgrade must restore it with)
_DROPPED = (
    ("exchanges", "t212_id", Integer),
    ("instruments", "t212_ticker", String),
    ("instruments", "is_t212_tradeable", Boolean),
)

# Width of instruments.t212_ticker at the previous head, which the downgrade restores.
_T212_TICKER_LENGTH = 100

_EXCHANGE_ID = "e" * 32
_INSTRUMENT_ID = "a" * 32
_EXCHANGE_NAME = "XNAS"
_INSTRUMENT_TICKER = "AAPL"
_INSTRUMENT_ISIN = "US0378331005"


def _load() -> ModuleType:
    path = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())
    spec = importlib.util.spec_from_file_location(Path(MIGRATION_FILENAME).stem, path)
    assert spec and spec.loader
    revision_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision_module)
    return revision_module


def _run(engine: Engine, migration_step: Callable[[], None]) -> None:
    with engine.begin() as conn:
        migration_context = MigrationContext.configure(conn)
        with Operations.context(migration_context):
            migration_step()


def _columns(engine: Engine, table: str) -> dict[str, dict]:
    return {info["name"]: info for info in inspect(engine).get_columns(table)}


@pytest.fixture
def engine() -> Engine:
    """SQLite shaped like the previous head, with one populated row per table."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Exchange.__table__.create(engine)
    Instrument.__table__.create(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE exchanges ADD COLUMN t212_id INTEGER"))
        conn.execute(
            text(
                "ALTER TABLE instruments ADD COLUMN t212_ticker "
                f"VARCHAR({_T212_TICKER_LENGTH})"
            )
        )
        conn.execute(
            text("ALTER TABLE instruments ADD COLUMN is_t212_tradeable BOOLEAN")
        )
        conn.execute(
            text("INSERT INTO exchanges (id, name, t212_id) VALUES (:id, :name, 7)"),
            {"id": _EXCHANGE_ID, "name": _EXCHANGE_NAME},
        )
        conn.execute(
            text(
                "INSERT INTO instruments "
                "(id, exchange_id, ticker, short_name, isin, yfinance_ticker, "
                "t212_ticker, is_t212_tradeable) "
                "VALUES (:id, :exchange_id, :ticker, 'Apple', :isin, :ticker, "
                "'AAPL_US_EQ', 1)"
            ),
            {
                "id": _INSTRUMENT_ID,
                "exchange_id": _EXCHANGE_ID,
                "ticker": _INSTRUMENT_TICKER,
                "isin": _INSTRUMENT_ISIN,
            },
        )
    return engine


@pytest.mark.criterion("REQ-DB-007")
def test_revision_follows_b8d0f2a4c6e1() -> None:
    """Validates revision 2266e24c509f: it extends the previous Alembic head."""
    revision_module = _load()
    assert revision_module.revision == EXPECTED_REVISION
    assert revision_module.down_revision == EXPECTED_DOWN_REVISION


@pytest.mark.criterion("REQ-DB-007")
def test_upgrade_drops_exchange_t212_id_and_both_instrument_columns(
    engine: Engine,
) -> None:
    """Validates the upgrade() of revision 2266e24c509f: the three columns are gone."""
    _run(engine, _load().upgrade)

    for table, column, _ in _DROPPED:
        assert column not in _columns(engine, table), f"{table}.{column} survived"


@pytest.mark.criterion("REQ-DB-007")
def test_upgrade_keeps_other_columns_and_rows(engine: Engine) -> None:
    """Validates the upgrade() of revision 2266e24c509f: other columns and rows stay."""
    expected = {
        table: set(_columns(engine, table))
        - {column for dropped_table, column, _ in _DROPPED if dropped_table == table}
        for table in ("exchanges", "instruments")
    }

    _run(engine, _load().upgrade)

    for table, names in expected.items():
        assert set(_columns(engine, table)) == names
    with engine.connect() as conn:
        exchange = conn.execute(text("SELECT id, name FROM exchanges")).one()
        instrument = conn.execute(
            text(
                "SELECT id, exchange_id, ticker, isin, yfinance_ticker FROM instruments"
            )
        ).one()
    assert tuple(exchange) == (_EXCHANGE_ID, _EXCHANGE_NAME)
    assert tuple(instrument) == (
        _INSTRUMENT_ID,
        _EXCHANGE_ID,
        _INSTRUMENT_TICKER,
        _INSTRUMENT_ISIN,
        _INSTRUMENT_TICKER,
    )


@pytest.mark.criterion("REQ-DB-007")
def test_downgrade_restores_three_nullable_columns(engine: Engine) -> None:
    """Validates the downgrade() of revision 2266e24c509f: nullable columns, no data."""
    revision_module = _load()
    _run(engine, revision_module.upgrade)

    _run(engine, revision_module.downgrade)

    for table, column, expected_type in _DROPPED:
        restored = _columns(engine, table)[column]
        assert restored["nullable"] is True
        assert isinstance(restored["type"], expected_type)
    ticker_type = _columns(engine, "instruments")["t212_ticker"]["type"]
    assert ticker_type.length == _T212_TICKER_LENGTH
    with engine.connect() as conn:
        exchange_value = conn.execute(
            text("SELECT t212_id FROM exchanges")
        ).scalar_one()
        instrument_values = tuple(
            conn.execute(
                text("SELECT t212_ticker, is_t212_tradeable FROM instruments")
            ).one()
        )
        survivors = conn.execute(text("SELECT count(*) FROM instruments")).scalar_one()
    assert exchange_value is None
    assert instrument_values == (None, None)
    assert survivors == 1
