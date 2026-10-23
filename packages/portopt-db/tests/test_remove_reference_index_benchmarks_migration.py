"""Tests for the ``remove_reference_index_benchmarks`` Alembic revision.

Replays ``upgrade()`` against in-memory SQLite through a raw ``MigrationContext``
+ ``Operations`` (no Postgres needed, so it runs in CI). The seed rows are built
from the seed revisions' own catalogue so the test breaks if the two drift apart.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"

SPY_SEED = ("SPY", "SPDR S&P 500 ETF Trust", "US78462F1030")


def _load(filename: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, str((MIGRATION_DIR / filename).resolve())
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def engine():
    """In-memory SQLite with minimal ``exchanges`` / ``instruments`` tables."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE exchanges (id TEXT PRIMARY KEY, name TEXT)"))
        conn.execute(
            text(
                "CREATE TABLE instruments (id TEXT PRIMARY KEY, ticker TEXT NOT NULL, "
                "name TEXT, isin TEXT, instrument_type TEXT, exchange_id TEXT)"
            )
        )
        conn.execute(text("INSERT INTO exchanges VALUES ('nyse', 'NYSE')"))
        conn.execute(text("INSERT INTO exchanges VALUES ('nasdaq', 'NASDAQ')"))
        conn.execute(text("INSERT INTO exchanges VALUES ('xetra', 'XETRA')"))
    return eng


def _insert(engine, ticker, name, isin, instrument_type, exchange_id) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO instruments "
                "(id, ticker, name, isin, instrument_type, exchange_id) "
                "VALUES (:id, :t, :n, :i, :ty, :e)"
            ),
            {
                "id": f"{ticker}-{exchange_id}-{name}",
                "t": ticker,
                "n": name,
                "i": isin,
                "ty": instrument_type,
                "e": exchange_id,
            },
        )


def _seed_like_the_seed_revisions(engine) -> list[str]:
    """Insert the rows x4y5z6a7b8c9 / a7b8c9d0e1f2 produce on a populated DB."""
    expanded = _load("a7b8c9d0e1f2_seed_reference_indices_expanded.py", "expanded")
    _insert(engine, SPY_SEED[0], SPY_SEED[1], SPY_SEED[2], "ETF", "nyse")
    seeded = [SPY_SEED[0]]
    for ticker, _short, long_name, isin, exchange, _yf in expanded.INDICES:
        _insert(engine, ticker, long_name, isin, "ETF", exchange.lower())
        seeded.append(ticker)
    return seeded


def _run_upgrade(engine) -> None:
    mod = _load("b9c8d7e6f5a4_remove_reference_index_benchmarks.py", "remove_bench")
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            mod.upgrade()


def _tickers(engine) -> list[str]:
    with engine.connect() as conn:
        return sorted(
            r[0] for r in conn.execute(text("SELECT ticker FROM instruments")).all()
        )


def test_upgrade_removes_the_named_rows_the_seed_revisions_insert(engine) -> None:
    """Seed revisions set a name, so a NULL-name predicate would skip them."""
    _seed_like_the_seed_revisions(engine)

    _run_upgrade(engine)

    assert _tickers(engine) == []


def test_upgrade_removes_name_less_rows_from_the_retired_runtime_seeder(
    engine,
) -> None:
    """The old runtime seeder left no name; URTH and VBINX only came from it."""
    for ticker in ("SPY", "QQQ", "URTH", "VBINX"):
        _insert(engine, ticker, None, None, "ETF", "nyse")

    _run_upgrade(engine)

    assert _tickers(engine) == []


def test_upgrade_keeps_real_instruments_that_share_a_benchmark_ticker(
    engine,
) -> None:
    """A named listing with a different ISIN is a real universe member."""
    _seed_like_the_seed_revisions(engine)
    _insert(engine, "SPY", "Some Other SPY Listing", "XX0000000001", "ETF", "xetra")
    _insert(engine, "GLD", "Different Gold Product", "XX0000000002", "ETF", "xetra")
    _insert(engine, "AAPL", "Apple Inc", "US0378331005", "STOCK", "nasdaq")

    _run_upgrade(engine)

    assert _tickers(engine) == ["AAPL", "GLD", "SPY"]
