from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "x4y5z6a7b8c9_seed_reference_indices.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "x4y5z6a7b8c9"
EXPECTED_DOWN_REVISION = "w3x4y5z6a7b8"

NYSE_EXISTS_GUARD = "WHERE EXISTS (SELECT 1 FROM exchanges WHERE name = 'NYSE')"


def _load_migration() -> types.ModuleType:
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
        conn.execute(
            text("""
            CREATE TABLE exchanges (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        )
        conn.execute(
            text("""
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
        """)
        )


def _spy_count(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM instruments WHERE ticker = 'SPY'")
        ).scalar()


class TestMigrationMetadata:
    """Verify revision chain — a broken chain silently skips the migration."""

    def test_module_is_importable(self) -> None:
        """Catches a missing file or path misconfiguration before any other assertion."""
        mod = _load_migration()
        assert mod is not None

    def test_revision_id_is_correct(self) -> None:
        """Fails fast if the file was renamed without updating the constant."""
        mod = _load_migration()
        assert mod.revision == EXPECTED_REVISION

    def test_down_revision_chains_from_w3x4y5z6a7b8(self) -> None:
        """A wrong down_revision silently breaks rollback to the previous head."""
        mod = _load_migration()
        assert mod.down_revision == EXPECTED_DOWN_REVISION

    def test_branch_labels_is_none(self) -> None:
        """Non-None branch_labels would register a new Alembic branch unintentionally."""
        mod = _load_migration()
        assert mod.branch_labels is None

    def test_depends_on_is_none(self) -> None:
        """Non-None depends_on would add a cross-branch dependency unintentionally."""
        mod = _load_migration()
        assert mod.depends_on is None


class TestUpgradeSQLContent:
    """Guard that the WHERE EXISTS clause is textually present in the emitted SQL."""

    def _capture_upgrade_sql(self) -> str:
        mod = _load_migration()
        mock_op = MagicMock()
        mod.upgrade.__globals__["op"] = mock_op
        mod.upgrade()
        assert mock_op.execute.called, "upgrade() must call op.execute()"
        sql_arg = mock_op.execute.call_args[0][0]
        return sql_arg

    def test_upgrade_sql_contains_where_exists_guard(self) -> None:
        """Missing guard would cause a NOT NULL violation on exchange_id when NYSE is absent."""
        sql = self._capture_upgrade_sql()
        assert NYSE_EXISTS_GUARD in sql, (
            f"upgrade() SQL must contain WHERE EXISTS guard for NYSE.\n"
            f"Expected: {NYSE_EXISTS_GUARD!r}\nActual SQL:\n{sql}"
        )

    def test_upgrade_sql_inserts_spy_ticker(self) -> None:
        """Confirms SPY is the seeded instrument, not a placeholder or typo."""
        sql = self._capture_upgrade_sql()
        assert "'SPY'" in sql

    def test_upgrade_sql_inserts_spy_as_etf(self) -> None:
        """instrument_type drives fund-bridge universe filtering — wrong type silently misclassifies."""
        sql = self._capture_upgrade_sql()
        assert "'ETF'" in sql

    def test_upgrade_sql_targets_nyse_exchange(self) -> None:
        """SPY must be on NYSE; a wrong exchange FK would yield a wrong exchange_id at query time."""
        sql = self._capture_upgrade_sql()
        assert "WHERE name = 'NYSE'" in sql

    def test_upgrade_sql_has_on_conflict_do_nothing(self) -> None:
        """Required for idempotency when upgrade() is run on a DB that already has the row."""
        sql = self._capture_upgrade_sql()
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    def test_upgrade_sql_has_uq_instrument_ticker_exchange_constraint(self) -> None:
        """ON CONFLICT must name the specific constraint, not rely on a primary key conflict."""
        sql = self._capture_upgrade_sql()
        assert "uq_instrument_ticker_exchange" in sql


# WHERE EXISTS is standard SQL — identical behavior in SQLite and PostgreSQL.
# Only gen_random_uuid() and ON CONFLICT ON CONSTRAINT syntax differ (not tested here).
class TestWhereExistsGuardBehavior:
    """Functional SQLite tests for the WHERE EXISTS guard logic."""

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
        """Without NYSE the subselect returns NULL for exchange_id, causing a NOT NULL violation without the guard."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(text(self._SQLITE_INSERT))

        assert _spy_count(engine) == 0, (
            "Expected 0 SPY rows when NYSE exchange is absent"
        )

    def test_spy_inserted_when_nyse_exists(self) -> None:
        """Happy path: guard passes and exchange_id subselect resolves to the inserted NYSE row."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')")
            )
            conn.execute(text(self._SQLITE_INSERT))

        assert _spy_count(engine) == 1, "Expected 1 SPY row when NYSE exchange exists"

    def test_spy_row_has_correct_ticker(self) -> None:
        """Validates the SELECT payload, not just the row count."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')")
            )
            conn.execute(text(self._SQLITE_INSERT))

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT ticker, instrument_type, isin FROM instruments WHERE ticker = 'SPY'"
                )
            ).fetchone()

        assert row is not None
        assert row[0] == "SPY"
        assert row[1] == "ETF"
        assert row[2] == "US78462F1030"

    def test_upgrade_is_idempotent_when_nyse_exists(self) -> None:
        """Running upgrade twice must not raise or duplicate — required for safe re-runs."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')")
            )
            conn.execute(text(self._SQLITE_INSERT))
            # Second run — must be a no-op due to ON CONFLICT / INSERT OR IGNORE
            conn.execute(text(self._SQLITE_INSERT))

        assert _spy_count(engine) == 1, (
            "Second upgrade run must not duplicate the SPY row"
        )

    def test_downgrade_removes_spy_row(self) -> None:
        """Confirms the DELETE predicate matches what upgrade() inserted."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO exchanges (id, name) VALUES ('nyse-uuid', 'NYSE')")
            )
            conn.execute(text(self._SQLITE_INSERT))

        assert _spy_count(engine) == 1

        with engine.begin() as conn:
            conn.execute(text(self._SQLITE_DELETE))

        assert _spy_count(engine) == 0, "Expected SPY row removed after downgrade"

    def test_downgrade_is_safe_when_spy_absent(self) -> None:
        """DELETE on a missing row must not raise — safe for partial-upgrade rollback."""
        engine = _sqlite_engine()
        _create_exchanges_and_instruments_tables(engine)

        with engine.begin() as conn:
            # No SPY row — DELETE should be a no-op
            conn.execute(text(self._SQLITE_DELETE))

        assert _spy_count(engine) == 0


class TestDowngradeSQLContent:
    """Verify downgrade() targets SPY precisely to avoid deleting unrelated rows."""

    def _capture_downgrade_sql(self) -> str:
        mod = _load_migration()
        mock_op = MagicMock()
        mod.downgrade.__globals__["op"] = mock_op
        mod.downgrade()
        assert mock_op.execute.called, "downgrade() must call op.execute()"
        return mock_op.execute.call_args[0][0]

    def test_downgrade_sql_targets_spy_ticker(self) -> None:
        """A WHERE clause without ticker would delete every ETF on NYSE, not just SPY."""
        sql = self._capture_downgrade_sql()
        assert "ticker = 'SPY'" in sql

    def test_downgrade_sql_targets_etf_type(self) -> None:
        """instrument_type predicate narrows the delete to avoid matching a future equity with the same ticker."""
        sql = self._capture_downgrade_sql()
        assert "instrument_type = 'ETF'" in sql

    def test_downgrade_sql_targets_yfinance_spy(self) -> None:
        """yfinance_ticker predicate is a third guard matching exactly the value upgrade() wrote."""
        sql = self._capture_downgrade_sql()
        assert "yfinance_ticker = 'SPY'" in sql

    def test_downgrade_sql_is_delete(self) -> None:
        """Ensures the statement type is DELETE, not a soft-delete UPDATE or TRUNCATE."""
        sql = self._capture_downgrade_sql()
        assert sql.strip().upper().startswith("DELETE")
