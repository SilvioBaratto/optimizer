"""Tests for migration x4y5z6a7b8c9: seed_reference_indices.

Covers:
- Migration module imports and revision metadata
- down_revision chains correctly from w3x4y5z6a7b8
- upgrade() SQL contains WHERE EXISTS guard for NYSE
- WHERE EXISTS guard prevents NOT NULL violation on fresh DB (no-op)
- WHERE EXISTS guard allows INSERT when NYSE exists
- Migration is idempotent (ON CONFLICT DO NOTHING)
- downgrade() targets SPY row specifically
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "x4y5z6a7b8c9_seed_reference_indices.py"
MIGRATION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "alembic", "versions"
)
MIGRATION_PATH = os.path.abspath(os.path.join(MIGRATION_DIR, MIGRATION_FILENAME))

EXPECTED_REVISION = "x4y5z6a7b8c9"
EXPECTED_DOWN_REVISION = "w3x4y5z6a7b8"

NYSE_EXISTS_GUARD = "WHERE EXISTS (SELECT 1 FROM exchanges WHERE name = 'NYSE')"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_migration() -> types.ModuleType:
    """Load the migration module from its file path."""
    module_name = MIGRATION_FILENAME[:-3]
    spec = importlib.util.spec_from_file_location(module_name, MIGRATION_PATH)
    assert spec is not None, f"Cannot find migration at {MIGRATION_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sqlite_engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _create_exchanges_and_instruments_tables(engine) -> None:
    """Create minimal exchanges + instruments tables compatible with SQLite."""
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


def _spy_count(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM instruments WHERE ticker = 'SPY'")
        ).scalar()


# ---------------------------------------------------------------------------
# Migration metadata
# ---------------------------------------------------------------------------


class TestMigrationMetadata:
    """Migration module must declare correct revision identifiers."""

    def test_module_is_importable(self) -> None:
        mod = _load_migration()
        assert mod is not None

    def test_revision_id_is_correct(self) -> None:
        mod = _load_migration()
        assert mod.revision == EXPECTED_REVISION

    def test_down_revision_chains_from_w3x4y5z6a7b8(self) -> None:
        mod = _load_migration()
        assert mod.down_revision == EXPECTED_DOWN_REVISION

    def test_branch_labels_is_none(self) -> None:
        mod = _load_migration()
        assert mod.branch_labels is None

    def test_depends_on_is_none(self) -> None:
        mod = _load_migration()
        assert mod.depends_on is None


# ---------------------------------------------------------------------------
# Upgrade SQL content: WHERE EXISTS guard must be present
# ---------------------------------------------------------------------------


class TestUpgradeSQLContent:
    """upgrade() must emit SQL with the NYSE existence guard."""

    def _capture_upgrade_sql(self) -> str:
        """Call upgrade() with a mock op and return the SQL string passed to execute()."""
        mod = _load_migration()
        mock_op = MagicMock()
        mod.upgrade.__globals__["op"] = mock_op
        mod.upgrade()
        assert mock_op.execute.called, "upgrade() must call op.execute()"
        # op.execute receives a single SQL string argument
        sql_arg = mock_op.execute.call_args[0][0]
        return sql_arg

    def test_upgrade_sql_contains_where_exists_guard(self) -> None:
        sql = self._capture_upgrade_sql()
        assert NYSE_EXISTS_GUARD in sql, (
            f"upgrade() SQL must contain WHERE EXISTS guard for NYSE.\n"
            f"Expected: {NYSE_EXISTS_GUARD!r}\nActual SQL:\n{sql}"
        )

    def test_upgrade_sql_inserts_spy_ticker(self) -> None:
        sql = self._capture_upgrade_sql()
        assert "'SPY'" in sql

    def test_upgrade_sql_inserts_spy_as_etf(self) -> None:
        sql = self._capture_upgrade_sql()
        assert "'ETF'" in sql

    def test_upgrade_sql_targets_nyse_exchange(self) -> None:
        sql = self._capture_upgrade_sql()
        assert "WHERE name = 'NYSE'" in sql

    def test_upgrade_sql_has_on_conflict_do_nothing(self) -> None:
        sql = self._capture_upgrade_sql()
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    def test_upgrade_sql_has_uq_instrument_ticker_exchange_constraint(self) -> None:
        sql = self._capture_upgrade_sql()
        assert "uq_instrument_ticker_exchange" in sql


# ---------------------------------------------------------------------------
# WHERE EXISTS guard behavior: functional SQLite tests
#
# These tests verify the guard logic using SQLite-compatible SQL.
# The WHERE EXISTS clause is standard SQL and behaves identically in SQLite
# and PostgreSQL — only gen_random_uuid() and ON CONFLICT ON CONSTRAINT
# syntax differ, which are not the subject of this test.
# ---------------------------------------------------------------------------


class TestWhereExistsGuardBehavior:
    """The WHERE EXISTS guard must prevent insertion when NYSE is missing."""

    _SQLITE_INSERT = """
        INSERT OR IGNORE INTO instruments (
            id,
            ticker,
            short_name,
            name,
            isin,
            instrument_type,
            currency_code,
            yfinance_ticker,
            exchange_id,
            created_at,
            updated_at
        )
        SELECT
            'spy-fixed-uuid',
            'SPY',
            'SPY',
            'SPDR S&P 500 ETF Trust',
            'US78462F1030',
            'ETF',
            'USD',
            'SPY',
            (SELECT id FROM exchanges WHERE name = 'NYSE'),
            datetime('now'),
            datetime('now')
        WHERE EXISTS (SELECT 1 FROM exchanges WHERE name = 'NYSE')
    """

    _SQLITE_DELETE = """
        DELETE FROM instruments
        WHERE ticker = 'SPY'
          AND yfinance_ticker = 'SPY'
          AND instrument_type = 'ETF'
    """

    def test_no_spy_inserted_when_nyse_missing(self) -> None:
        """Fresh DB without NYSE: INSERT is a no-op due to WHERE EXISTS guard."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(text(self._SQLITE_INSERT))

        assert _spy_count(engine) == 0, (
            "Expected 0 SPY rows when NYSE exchange is absent"
        )

    def test_spy_inserted_when_nyse_exists(self) -> None:
        """DB with NYSE: INSERT succeeds and SPY row is present."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')"
            ))
            conn.execute(text(self._SQLITE_INSERT))

        assert _spy_count(engine) == 1, (
            "Expected 1 SPY row when NYSE exchange exists"
        )

    def test_spy_row_has_correct_ticker(self) -> None:
        """Inserted SPY row must have correct ticker and instrument_type."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')"
            ))
            conn.execute(text(self._SQLITE_INSERT))

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT ticker, instrument_type, isin FROM instruments WHERE ticker = 'SPY'")
            ).fetchone()

        assert row is not None
        assert row[0] == "SPY"
        assert row[1] == "ETF"
        assert row[2] == "US78462F1030"

    def test_upgrade_is_idempotent_when_nyse_exists(self) -> None:
        """Running upgrade twice with NYSE present must not fail or duplicate SPY."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')"
            ))
            conn.execute(text(self._SQLITE_INSERT))
            # Second run — must be a no-op due to ON CONFLICT / INSERT OR IGNORE
            conn.execute(text(self._SQLITE_INSERT))

        assert _spy_count(engine) == 1, (
            "Second upgrade run must not duplicate the SPY row"
        )

    def test_downgrade_removes_spy_row(self) -> None:
        """downgrade() must delete the SPY row inserted during upgrade."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')"
            ))
            conn.execute(text(self._SQLITE_INSERT))

        assert _spy_count(engine) == 1

        with engine.begin() as conn:
            conn.execute(text(self._SQLITE_DELETE))

        assert _spy_count(engine) == 0, (
            "Expected SPY row removed after downgrade"
        )

    def test_downgrade_is_safe_when_spy_absent(self) -> None:
        """downgrade() must not error when SPY row is already missing."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            # No SPY row — DELETE should be a no-op
            conn.execute(text(self._SQLITE_DELETE))

        assert _spy_count(engine) == 0


# ---------------------------------------------------------------------------
# Downgrade SQL content
# ---------------------------------------------------------------------------


class TestDowngradeSQLContent:
    """downgrade() must emit SQL that targets the SPY row precisely."""

    def _capture_downgrade_sql(self) -> str:
        mod = _load_migration()
        mock_op = MagicMock()
        mod.downgrade.__globals__["op"] = mock_op
        mod.downgrade()
        assert mock_op.execute.called, "downgrade() must call op.execute()"
        return mock_op.execute.call_args[0][0]

    def test_downgrade_sql_targets_spy_ticker(self) -> None:
        sql = self._capture_downgrade_sql()
        assert "ticker = 'SPY'" in sql

    def test_downgrade_sql_targets_etf_type(self) -> None:
        sql = self._capture_downgrade_sql()
        assert "instrument_type = 'ETF'" in sql

    def test_downgrade_sql_targets_yfinance_spy(self) -> None:
        sql = self._capture_downgrade_sql()
        assert "yfinance_ticker = 'SPY'" in sql

    def test_downgrade_sql_is_delete(self) -> None:
        sql = self._capture_downgrade_sql()
        assert sql.strip().upper().startswith("DELETE")
