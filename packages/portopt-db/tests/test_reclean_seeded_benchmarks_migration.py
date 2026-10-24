"""Tests for the ``reclean_seeded_benchmarks`` Alembic revision.

``b9c8d7e6f5a4`` was widened after some installs had already applied it, so those
databases still hold the ETFs the seed revisions inserted. This revision removes
them. ``upgrade()`` is replayed against in-memory SQLite through a raw
``MigrationContext`` + ``Operations`` (no Postgres needed, so it runs in CI); the
seed rows come from the seed revisions' own catalogue so the test breaks if the
catalogues drift apart.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

PKG_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = PKG_ROOT / "alembic" / "versions"
MIGRATION_FILENAME = "a6f1c8d3e2b7_reclean_seeded_benchmarks.py"

EXPECTED_REVISION = "a6f1c8d3e2b7"
EXPECTED_DOWN_REVISION = "c5d9a1e7b3f8"

SPY_SEED = ("SPY", "SPDR S&P 500 ETF Trust", "US78462F1030")


def _load(filename: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, str((MIGRATION_DIR / filename).resolve())
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _migration() -> ModuleType:
    return _load(MIGRATION_FILENAME, "reclean_seeded_benchmarks")


@pytest.fixture
def engine():
    """In-memory SQLite with a minimal ``instruments`` table."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE instruments (id TEXT PRIMARY KEY, ticker TEXT NOT NULL, "
                "name TEXT, isin TEXT, instrument_type TEXT, exchange_id TEXT)"
            )
        )
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
    """Insert the rows x4y5z6a7b8c9 / a7b8c9d0e1f2 leave on a populated DB."""
    expanded = _load("a7b8c9d0e1f2_seed_reference_indices_expanded.py", "expanded")
    _insert(engine, SPY_SEED[0], SPY_SEED[1], SPY_SEED[2], "ETF", "nyse")
    seeded = [SPY_SEED[0]]
    for ticker, _short, long_name, isin, exchange, _yf in expanded.INDICES:
        _insert(engine, ticker, long_name, isin, "ETF", exchange.lower())
        seeded.append(ticker)
    return seeded


def _run(engine, direction: str = "upgrade") -> None:
    mod = _migration()
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            getattr(mod, direction)()


def _tickers(engine) -> list[str]:
    with engine.connect() as conn:
        return sorted(
            r[0] for r in conn.execute(text("SELECT ticker FROM instruments")).all()
        )


def test_revision_chains_directly_after_the_unique_index_cleanup() -> None:
    revision = ScriptDirectory(str(PKG_ROOT / "alembic")).get_revision(
        EXPECTED_REVISION
    )

    assert revision is not None
    assert revision.down_revision == EXPECTED_DOWN_REVISION


def test_catalogue_is_exactly_what_the_seed_revisions_insert() -> None:
    expanded = _load("a7b8c9d0e1f2_seed_reference_indices_expanded.py", "expanded")
    seeded = {(SPY_SEED[0], SPY_SEED[2])} | {
        (ticker, isin) for ticker, _s, _n, isin, _e, _y in expanded.INDICES
    }
    original = _load("b9c8d7e6f5a4_remove_reference_index_benchmarks.py", "original")

    assert set(_migration()._MIGRATION_SEEDED) == seeded
    assert set(original._MIGRATION_SEEDED) == seeded


def test_upgrade_removes_seed_rows_left_by_installs_that_ran_the_old_cleanup(
    engine,
) -> None:
    """The first version of b9c8d7e6f5a4 kept every row that had a name."""
    _seed_like_the_seed_revisions(engine)

    _run(engine)

    assert _tickers(engine) == []


def test_upgrade_keeps_real_instruments_that_share_a_benchmark_ticker(
    engine,
) -> None:
    """A listing with a different ISIN is a real universe member."""
    _seed_like_the_seed_revisions(engine)
    _insert(engine, "SPY", "Some Other SPY Listing", "XX0000000001", "ETF", "xetra")
    _insert(engine, "AAPL", "Apple Inc", "US0378331005", "STOCK", "nasdaq")

    _run(engine)

    assert _tickers(engine) == ["AAPL", "SPY"]


def test_upgrade_keeps_universe_built_etfs_that_carry_no_isin(engine) -> None:
    """The Yahoo-screen universe leaves isin NULL, so SPY there is a live member."""
    _seed_like_the_seed_revisions(engine)
    _insert(engine, "SPY", "SPDR S&P 500 ETF Trust", None, "ETF", "bats")
    _insert(engine, "QQQ", "Invesco QQQ Trust", None, "ETF", "bats")

    _run(engine)

    assert _tickers(engine) == ["QQQ", "SPY"]


def test_upgrade_keeps_a_non_etf_that_reuses_a_seeded_ticker_and_isin(
    engine,
) -> None:
    _insert(engine, SPY_SEED[0], SPY_SEED[1], SPY_SEED[2], "STOCK", "nyse")

    _run(engine)

    assert _tickers(engine) == ["SPY"]


def test_upgrade_is_a_no_op_on_a_clean_database_and_when_replayed(engine) -> None:
    _insert(engine, "AAPL", "Apple Inc", "US0378331005", "STOCK", "nasdaq")
    _seed_like_the_seed_revisions(engine)

    _run(engine)
    _run(engine)

    assert _tickers(engine) == ["AAPL"]


def test_downgrade_refuses_because_the_deleted_rows_cannot_be_restored(
    engine,
) -> None:
    _insert(engine, "AAPL", "Apple Inc", "US0378331005", "STOCK", "nasdaq")

    with pytest.raises(NotImplementedError, match="one-way"):
        _run(engine, "downgrade")

    assert _tickers(engine) == ["AAPL"]
