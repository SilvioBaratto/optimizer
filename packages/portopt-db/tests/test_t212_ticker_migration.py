"""The ``t212_ticker`` revision backfills ``ticker`` from ``yfinance_ticker``.

``uq_instrument_ticker_exchange (ticker, exchange_id)`` has existed since the
instruments normalisation, and ``yfinance_ticker`` carries no uniqueness of its
own: two Trading 212 lines on one exchange can map to the same Yahoo symbol.
A blind ``UPDATE ... SET ticker = yfinance_ticker`` then violates the
constraint and leaves the database stuck below head. These tests run the real
revision against SQLite tables that mirror the pre-upgrade schema.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.pool import StaticPool

_MIGRATION_NAME = "e6f7a8b9c0d1_add_t212_ticker_to_instruments"
_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / f"{_MIGRATION_NAME}.py"
)

_LSE = "exchange-lse"
_NYSE = "exchange-nyse"


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_MIGRATION_NAME, _MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine_with_instruments(
    rows: list[tuple[str, str, str | None, str, str]],
) -> Engine:
    """Create the pre-upgrade ``instruments`` table and insert ``rows``.

    Each row is ``(id, ticker, yfinance_ticker, exchange_id, created_at)``.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE instruments (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    yfinance_ticker TEXT,
                    exchange_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    CONSTRAINT uq_instrument_ticker_exchange
                        UNIQUE (ticker, exchange_id)
                )
                """
            )
        )
        for row_id, ticker, yf_ticker, exchange_id, created_at in rows:
            conn.execute(
                text(
                    "INSERT INTO instruments "
                    "(id, ticker, yfinance_ticker, exchange_id, created_at) "
                    "VALUES (:id, :ticker, :yf, :exchange_id, :created_at)"
                ),
                {
                    "id": row_id,
                    "ticker": ticker,
                    "yf": yf_ticker,
                    "exchange_id": exchange_id,
                    "created_at": created_at,
                },
            )
    return engine


def _upgrade(engine: Engine) -> None:
    module = _load_migration()
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            module.upgrade()


def _tickers(engine: Engine) -> dict[str, str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, ticker FROM instruments")).tuples().all()
    return dict(rows)


def test_when_two_instruments_on_one_exchange_share_a_yahoo_symbol_then_the_upgrade_completes():
    """The older line takes the Yahoo ticker; the duplicate keeps its own."""
    engine = _engine_with_instruments(
        [
            ("a", "VODl_EQ", "VOD.L", _LSE, "2026-01-01"),
            ("b", "VOD_LSE_EQ", "VOD.L", _LSE, "2026-02-01"),
        ]
    )

    _upgrade(engine)

    assert _tickers(engine) == {"a": "VOD.L", "b": "VOD_LSE_EQ"}


def test_when_the_older_duplicate_has_the_larger_id_then_it_still_wins():
    """Age decides the winner; ids are random UUIDs and carry no ordering."""
    engine = _engine_with_instruments(
        [
            ("z-older", "VODl_EQ", "VOD.L", _LSE, "2026-01-01"),
            ("a-newer", "VOD_LSE_EQ", "VOD.L", _LSE, "2026-02-01"),
        ]
    )

    _upgrade(engine)

    assert _tickers(engine) == {"z-older": "VOD.L", "a-newer": "VOD_LSE_EQ"}


def test_when_duplicates_share_a_created_at_then_the_smaller_id_wins():
    """The id only breaks a tie, and does so independently of insert order."""
    engine = _engine_with_instruments(
        [
            ("b", "VOD_LSE_EQ", "VOD.L", _LSE, "2026-01-01"),
            ("a", "VODl_EQ", "VOD.L", _LSE, "2026-01-01"),
            ("c", "VOD_XYZ_EQ", "VOD.L", _LSE, "2026-01-01"),
        ]
    )

    _upgrade(engine)

    assert _tickers(engine) == {
        "a": "VOD.L",
        "b": "VOD_LSE_EQ",
        "c": "VOD_XYZ_EQ",
    }


def test_when_the_same_yahoo_symbol_sits_on_different_exchanges_then_both_are_backfilled():
    """The unique constraint is per exchange, so each exchange keeps its row."""
    engine = _engine_with_instruments(
        [
            ("a", "VODl_EQ", "VOD.L", _LSE, "2026-01-01"),
            ("b", "VOD_US_EQ", "VOD.L", _NYSE, "2026-01-02"),
        ]
    )

    _upgrade(engine)

    assert _tickers(engine) == {"a": "VOD.L", "b": "VOD.L"}


def test_when_a_yahoo_symbol_is_already_another_rows_ticker_then_that_row_is_left_alone():
    """Backfilling would collide with the row that already holds the symbol."""
    engine = _engine_with_instruments(
        [
            ("holder", "BP.L", None, _LSE, "2026-01-01"),
            ("mapped", "BP_EQ", "BP.L", _LSE, "2026-01-02"),
        ]
    )

    _upgrade(engine)

    assert _tickers(engine) == {"holder": "BP.L", "mapped": "BP_EQ"}


def test_when_a_yahoo_symbol_is_held_by_a_row_that_is_itself_remapped_then_the_holder_still_moves():
    """A holder with its own Yahoo mapping is backfilled; the claimant waits."""
    engine = _engine_with_instruments(
        [
            ("claimant", "A_EQ", "B", _LSE, "2026-01-01"),
            ("holder", "B", "C", _LSE, "2026-01-02"),
        ]
    )

    _upgrade(engine)

    assert _tickers(engine) == {"claimant": "A_EQ", "holder": "C"}


def test_when_yahoo_symbol_is_missing_or_empty_then_the_ticker_is_untouched():
    """Rows without a usable Yahoo mapping keep their previous ticker."""
    engine = _engine_with_instruments(
        [
            ("none", "NONE_EQ", None, _LSE, "2026-01-01"),
            ("blank", "BLANK_EQ", "", _LSE, "2026-01-02"),
            ("mapped", "AAPL_US_EQ", "AAPL", _NYSE, "2026-01-03"),
        ]
    )

    _upgrade(engine)

    assert _tickers(engine) == {
        "none": "NONE_EQ",
        "blank": "BLANK_EQ",
        "mapped": "AAPL",
    }


def test_when_the_revision_is_upgraded_then_the_t212_ticker_column_is_added():
    """The column the backfill is meant to complement still gets created."""
    engine = _engine_with_instruments([])

    _upgrade(engine)

    columns = {c["name"] for c in inspect(engine).get_columns("instruments")}
    assert "t212_ticker" in columns
