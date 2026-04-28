"""Tests for migration a7b8c9d0e1f2: seed_reference_indices_expanded (issue #430).

Covers:
- Migration module imports, revision + down_revision chain.
- ``INDICES`` constant lists the 9 new benchmarks (SPY is owned by the
  predecessor migration ``x4y5z6a7b8c9``).
- Every INSERT statement carries the per-ticker WHERE EXISTS guard and the
  named-constraint ``ON CONFLICT ON CONSTRAINT uq_instrument_ticker_exchange
  DO NOTHING`` clause.
- QQQ targets NASDAQ; the other 8 target NYSE.
- Functional SQLite smoke: inserts succeed when the target exchange exists,
  are a no-op when it does not, and are idempotent under re-runs.
- Downgrade removes all 9 rows (and leaves SPY untouched).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "a7b8c9d0e1f2_seed_reference_indices_expanded.py"
MIGRATION_DIR = Path(__file__).parent / ".." / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "a7b8c9d0e1f2"
EXPECTED_DOWN_REVISION = "z6a7b8c9d0e1"

EXPECTED_TICKERS = (
    "QQQ", "IWM", "EFA", "EEM", "AGG", "VGK", "VWO", "TLT", "GLD",
)
NASDAQ_TICKERS = {"QQQ"}
NYSE_TICKERS = {"IWM", "EFA", "EEM", "AGG", "VGK", "VWO", "TLT", "GLD"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_migration() -> types.ModuleType:
    module_name = MIGRATION_FILENAME[:-3]
    spec = importlib.util.spec_from_file_location(module_name, MIGRATION_PATH)
    assert spec is not None, f"Cannot find migration at {MIGRATION_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _capture_upgrade_statements() -> list[str]:
    """Return every SQL string passed to ``op.execute`` during upgrade()."""
    mod = _load_migration()
    mock_op = MagicMock()
    mod.upgrade.__globals__["op"] = mock_op
    mod.upgrade()
    return [call.args[0] for call in mock_op.execute.call_args_list]


def _capture_downgrade_sql() -> str:
    mod = _load_migration()
    mock_op = MagicMock()
    mod.downgrade.__globals__["op"] = mock_op
    mod.downgrade()
    assert mock_op.execute.called
    return mock_op.execute.call_args[0][0]


def _sqlite_engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _create_tables(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE exchanges (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """))
        conn.execute(text("""
            CREATE TABLE instruments (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                short_name TEXT NOT NULL,
                name TEXT,
                isin TEXT,
                instrument_type TEXT,
                currency_code TEXT,
                yfinance_ticker TEXT,
                exchange_id TEXT NOT NULL REFERENCES exchanges(id),
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                CONSTRAINT uq_instrument_ticker_exchange UNIQUE (ticker, exchange_id)
            )
        """))


def _sqlite_insert_sql(
    ticker: str,
    long_name: str,
    isin: str,
    exchange_name: str,
) -> str:
    """SQLite-compatible version of the migration's INSERT (for smoke testing only)."""
    return f"""
        INSERT OR IGNORE INTO instruments (
            id,
            ticker,
            short_name,
            name,
            isin,
            instrument_type,
            currency_code,
            yfinance_ticker,
            exchange_id
        )
        SELECT
            'uuid-{ticker.lower()}',
            '{ticker}',
            '{ticker}',
            '{long_name}',
            '{isin}',
            'ETF',
            'USD',
            '{ticker}',
            (SELECT id FROM exchanges WHERE name = '{exchange_name}')
        WHERE EXISTS (SELECT 1 FROM exchanges WHERE name = '{exchange_name}')
    """


def _count(engine, ticker: str | None = None) -> int:
    with engine.connect() as conn:
        if ticker is None:
            return conn.execute(
                text("SELECT COUNT(*) FROM instruments WHERE instrument_type = 'ETF'")
            ).scalar()
        return conn.execute(
            text("SELECT COUNT(*) FROM instruments WHERE ticker = :t"),
            {"t": ticker},
        ).scalar()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMigrationMetadata:
    def test_module_imports(self) -> None:
        assert _load_migration() is not None

    def test_revision_id(self) -> None:
        assert _load_migration().revision == EXPECTED_REVISION

    def test_down_revision_chains_from_z6a7b8c9d0e1(self) -> None:
        assert _load_migration().down_revision == EXPECTED_DOWN_REVISION

    def test_branch_labels_is_none(self) -> None:
        assert _load_migration().branch_labels is None

    def test_depends_on_is_none(self) -> None:
        assert _load_migration().depends_on is None


# ---------------------------------------------------------------------------
# INDICES catalogue
# ---------------------------------------------------------------------------


class TestIndicesCatalogue:
    """The static INDICES tuple must cover the nine new benchmarks."""

    def test_catalogue_has_nine_entries(self) -> None:
        mod = _load_migration()
        assert len(mod.INDICES) == 9

    def test_catalogue_tickers_match_spec(self) -> None:
        mod = _load_migration()
        catalogue_tickers = tuple(row[0] for row in mod.INDICES)
        assert catalogue_tickers == EXPECTED_TICKERS

    def test_catalogue_does_not_include_spy(self) -> None:
        """SPY is owned by the predecessor migration x4y5z6a7b8c9, not this one."""
        mod = _load_migration()
        assert all(row[0] != "SPY" for row in mod.INDICES)


# ---------------------------------------------------------------------------
# Upgrade SQL content
# ---------------------------------------------------------------------------


class TestUpgradeSQLContent:
    """Every INSERT must carry the guard + the named ON CONFLICT clause."""

    def test_emits_one_insert_per_ticker(self) -> None:
        statements = _capture_upgrade_statements()
        assert len(statements) == len(EXPECTED_TICKERS)

    def test_every_ticker_appears_in_some_insert(self) -> None:
        sql_all = "\n".join(_capture_upgrade_statements())
        for ticker in EXPECTED_TICKERS:
            assert f"'{ticker}'" in sql_all, f"missing insert for {ticker}"

    def test_every_insert_has_where_exists_guard(self) -> None:
        for sql in _capture_upgrade_statements():
            assert "WHERE EXISTS (SELECT 1 FROM exchanges WHERE name =" in sql

    def test_every_insert_has_named_on_conflict_clause(self) -> None:
        for sql in _capture_upgrade_statements():
            assert "ON CONFLICT ON CONSTRAINT uq_instrument_ticker_exchange" in sql
            assert "DO NOTHING" in sql

    def test_qqq_targets_nasdaq(self) -> None:
        qqq_sql = _single_statement_for("QQQ")
        assert "WHERE name = 'NASDAQ'" in qqq_sql

    def test_non_qqq_tickers_target_nyse(self) -> None:
        for ticker in NYSE_TICKERS:
            sql = _single_statement_for(ticker)
            assert "WHERE name = 'NYSE'" in sql, f"{ticker} should target NYSE"

    def test_every_insert_sets_instrument_type_etf(self) -> None:
        for sql in _capture_upgrade_statements():
            assert "'ETF'" in sql

    def test_every_insert_sets_usd_currency(self) -> None:
        for sql in _capture_upgrade_statements():
            assert "'USD'" in sql


def _single_statement_for(ticker: str) -> str:
    """Return the single upgrade statement that inserts *ticker*."""
    for sql in _capture_upgrade_statements():
        if f"'{ticker}'" in sql:
            return sql
    raise AssertionError(f"No upgrade statement found for {ticker}")


# ---------------------------------------------------------------------------
# Functional SQLite smoke — WHERE EXISTS + idempotency
# ---------------------------------------------------------------------------


class TestWhereExistsGuardBehavior:
    """The guard blocks inserts on fresh DBs and allows them once exchanges exist."""

    def test_no_rows_inserted_when_neither_exchange_exists(self) -> None:
        engine = _sqlite_engine()
        _create_tables(engine)
        mod = _load_migration()

        with engine.begin() as conn:
            for row in mod.INDICES:
                conn.execute(text(_sqlite_insert_sql(row[0], row[2], row[3], row[4])))

        assert _count(engine) == 0

    def test_only_nyse_tickers_inserted_when_only_nyse_exists(self) -> None:
        engine = _sqlite_engine()
        _create_tables(engine)
        mod = _load_migration()

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')"
            ))
            for row in mod.INDICES:
                conn.execute(text(_sqlite_insert_sql(row[0], row[2], row[3], row[4])))

        for ticker in NYSE_TICKERS:
            assert _count(engine, ticker) == 1, f"{ticker} expected once"
        assert _count(engine, "QQQ") == 0, "QQQ must be skipped without NASDAQ"

    def test_all_rows_inserted_when_both_exchanges_exist(self) -> None:
        engine = _sqlite_engine()
        _create_tables(engine)
        mod = _load_migration()

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')"
            ))
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nasdaq-uuid', 'NASDAQ')"
            ))
            for row in mod.INDICES:
                conn.execute(text(_sqlite_insert_sql(row[0], row[2], row[3], row[4])))

        for ticker in EXPECTED_TICKERS:
            assert _count(engine, ticker) == 1

    def test_idempotent_when_run_twice(self) -> None:
        engine = _sqlite_engine()
        _create_tables(engine)
        mod = _load_migration()

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')"
            ))
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nasdaq-uuid', 'NASDAQ')"
            ))
            for _ in range(2):
                for row in mod.INDICES:
                    conn.execute(text(_sqlite_insert_sql(row[0], row[2], row[3], row[4])))

        for ticker in EXPECTED_TICKERS:
            assert _count(engine, ticker) == 1


# ---------------------------------------------------------------------------
# Downgrade
# ---------------------------------------------------------------------------


class TestDowngradeSQLContent:
    def test_downgrade_is_delete_statement(self) -> None:
        sql = _capture_downgrade_sql()
        assert sql.strip().upper().startswith("DELETE")

    def test_downgrade_targets_all_nine_tickers(self) -> None:
        sql = _capture_downgrade_sql()
        for ticker in EXPECTED_TICKERS:
            assert f"'{ticker}'" in sql

    def test_downgrade_does_not_target_spy(self) -> None:
        """SPY is owned by the predecessor migration; this downgrade must not touch it."""
        sql = _capture_downgrade_sql()
        assert "'SPY'" not in sql

    def test_downgrade_filters_by_etf_type(self) -> None:
        sql = _capture_downgrade_sql()
        assert "instrument_type = 'ETF'" in sql

    def test_downgrade_removes_nine_rows_and_leaves_spy(self) -> None:
        engine = _sqlite_engine()
        _create_tables(engine)
        mod = _load_migration()

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')"
            ))
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nasdaq-uuid', 'NASDAQ')"
            ))
            # Seed SPY (owned by predecessor migration) + the 9 expanded rows.
            conn.execute(text("""
                INSERT INTO instruments (id, ticker, short_name, name, instrument_type, currency_code, yfinance_ticker, exchange_id)
                VALUES ('uuid-spy', 'SPY', 'SPY', 'SPDR S&P 500 ETF Trust', 'ETF', 'USD', 'SPY', 'nyse-uuid')
            """))
            for row in mod.INDICES:
                conn.execute(text(_sqlite_insert_sql(row[0], row[2], row[3], row[4])))

            before_total = conn.execute(
                text("SELECT COUNT(*) FROM instruments WHERE instrument_type = 'ETF'")
            ).scalar()
            assert before_total == 1 + len(EXPECTED_TICKERS), (
                f"expected 10 ETFs before downgrade, got {before_total}"
            )

            downgrade_sql = _capture_downgrade_sql()
            conn.execute(text(downgrade_sql))

            spy_inside = conn.execute(
                text("SELECT COUNT(*) FROM instruments WHERE ticker = 'SPY'")
            ).scalar()
            assert spy_inside == 1, (
                f"SPY must be preserved inside begin block (got {spy_inside}); "
                f"downgrade SQL was:\n{downgrade_sql}"
            )
            for ticker in EXPECTED_TICKERS:
                count = conn.execute(
                    text(f"SELECT COUNT(*) FROM instruments WHERE ticker = '{ticker}'")
                ).scalar()
                assert count == 0, f"{ticker} must be removed (got {count})"
        for ticker in EXPECTED_TICKERS:
            assert _count(engine, ticker) == 0
