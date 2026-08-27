"""Branch-coverage tests for app/repositories/_shared/database_admin_repository.py.

Split strategy
--------------
* Real ``db_session`` (SQLite SAVEPOINT) — for SQLite-safe paths only:
  - check_health() success
  - truncate_table() ValueError for non-allowlisted table
  - truncate_tables() unknown-table branch (no TRUNCATE executed)
  - _missing_table_row() helper directly

* MagicMock session — for every path that executes Postgres-only SQL:
  ``information_schema`` queries, ``pg_total_relation_size``,
  ``pg_size_pretty``, ``TRUNCATE … CASCADE``.
  These are documented inline; they cannot run on SQLite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from portopt_db.repositories.database_admin import (
    DatabaseAdminRepository,
    _missing_table_row,
)

# A table name present in the _ALLOWED_TABLES frozenset.
_ALLOWED = "instruments"
_UNKNOWN = "not_a_real_table_xyz"


# ===========================================================================
# _missing_table_row helper (pure function — no session needed)
# ===========================================================================


class TestMissingTableRow:
    def test_returns_expected_shape(self) -> None:
        row = _missing_table_row("some_table")
        assert row["name"] == "some_table"
        assert row["schema"] == "public"
        assert row["exists"] is False
        assert row["row_count"] is None
        assert row["size_bytes"] is None
        assert row["size_pretty"] == "—"


# ===========================================================================
# check_health — real db_session (SQLite SELECT 1 is valid everywhere)
# ===========================================================================


class TestCheckHealthReal:
    def test_when_db_up_then_returns_true_with_non_negative_latency(
        self, db_session: Session
    ) -> None:
        repo = DatabaseAdminRepository(db_session)
        healthy, latency = repo.check_health()
        assert healthy is True
        assert latency >= 0


# ===========================================================================
# check_health — failure path (MagicMock — session.execute raises)
# Documented: no Postgres SQL is executed; the mock short-circuits the call.
# ===========================================================================


class TestCheckHealthFailure:
    def test_when_execute_raises_then_returns_false(self) -> None:
        # PG-only path is not reached; mock raises before any dialect SQL runs.
        mock_session = MagicMock(spec=Session)
        mock_session.execute.side_effect = Exception("connection refused")
        repo = DatabaseAdminRepository(mock_session)

        healthy, latency = repo.check_health()

        assert healthy is False
        assert latency >= 0


# ===========================================================================
# truncate_table — allowlist guard (real db_session; no TRUNCATE executed)
# ===========================================================================


class TestTruncateTableAllowlistGuard:
    def test_when_table_not_allowlisted_then_raises_value_error(
        self, db_session: Session
    ) -> None:
        repo = DatabaseAdminRepository(db_session)
        with pytest.raises(ValueError, match="not a managed application table"):
            repo.truncate_table(_UNKNOWN)

    def test_error_message_contains_table_name(self, db_session: Session) -> None:
        repo = DatabaseAdminRepository(db_session)
        with pytest.raises(ValueError, match=_UNKNOWN):
            repo.truncate_table(_UNKNOWN)


# ===========================================================================
# truncate_tables — unknown-table branch (real db_session; no TRUNCATE run)
# ===========================================================================


class TestTruncateTablesUnknownBranch:
    def test_when_table_unknown_then_error_recorded_nothing_cleared(
        self, db_session: Session
    ) -> None:
        repo = DatabaseAdminRepository(db_session)
        cleared, errors = repo.truncate_tables([_UNKNOWN])
        assert cleared == []
        assert len(errors) == 1
        assert _UNKNOWN in errors[0]
        assert "not a managed table" in errors[0]

    def test_when_mix_of_unknown_then_only_unknown_recorded(
        self, db_session: Session
    ) -> None:
        repo = DatabaseAdminRepository(db_session)
        cleared, errors = repo.truncate_tables([_UNKNOWN, "also_unknown_abc"])
        assert cleared == []
        assert len(errors) == 2


# ===========================================================================
# get_table_info — MagicMock session (Postgres information_schema + pg_* fns)
# DOCUMENTED: _table_exists uses information_schema.tables (Postgres-only).
#             _row_count uses SELECT COUNT(*) FROM "<table>" (dialect-agnostic
#             but we stay in mock territory because _table_exists is not).
#             _size_bytes/_size_pretty use pg_total_relation_size (PG-only).
# ===========================================================================


class TestGetTableInfoMock:
    def _make_mock_session_for_existing_table(self) -> MagicMock:
        """Return a mock whose execute().scalar() sequence drives the
        _table_exists→True → _row_count → _size_bytes → _size_pretty path."""
        mock_session = MagicMock(spec=Session)
        # Sequence of scalar() return values per execute() call:
        # 1st call → _table_exists → True
        # 2nd call → _row_count   → 42
        # 3rd call → _size_bytes  → 8192
        # 4th call → _size_pretty → "8 kB"
        mock_session.execute.return_value.scalar.side_effect = [
            True,
            42,
            8192,
            "8 kB",
        ]
        return mock_session

    def test_when_table_exists_then_row_has_correct_keys(self) -> None:
        # Postgres-only SQL — mock session required.
        mock_session = self._make_mock_session_for_existing_table()
        repo = DatabaseAdminRepository(mock_session)

        rows = repo.get_table_info([_ALLOWED])

        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == _ALLOWED
        assert row["schema"] == "public"
        assert row["exists"] is True
        assert row["row_count"] == 42
        assert row["size_bytes"] == 8192
        assert row["size_pretty"] == "8 kB"

    def test_when_table_missing_then_missing_row_returned(self) -> None:
        # Postgres-only SQL — mock session required.
        # _table_exists returns False → _missing_table_row branch.
        mock_session = MagicMock(spec=Session)
        mock_session.execute.return_value.scalar.return_value = False
        repo = DatabaseAdminRepository(mock_session)

        rows = repo.get_table_info([_ALLOWED])

        assert len(rows) == 1
        row = rows[0]
        assert row["exists"] is False
        assert row["row_count"] is None
        assert row["size_pretty"] == "—"

    def test_multiple_tables_calls_table_row_for_each(self) -> None:
        # Two tables: first exists, second does not.
        mock_session = MagicMock(spec=Session)
        mock_session.execute.return_value.scalar.side_effect = [
            True,  # _table_exists for instruments
            5,  # _row_count
            4096,  # _size_bytes
            "4 kB",  # _size_pretty
            False,  # _table_exists for ticker_profiles (missing)
        ]
        repo = DatabaseAdminRepository(mock_session)

        rows = repo.get_table_info(["instruments", "ticker_profiles"])

        assert len(rows) == 2
        assert rows[0]["exists"] is True
        assert rows[1]["exists"] is False


# ===========================================================================
# truncate_table — success path (MagicMock)
# DOCUMENTED: TRUNCATE TABLE … CASCADE is Postgres-only DDL.
# ===========================================================================


class TestTruncateTableSuccessMock:
    def test_when_allowed_table_then_executes_and_commits(self) -> None:
        # Postgres-only DDL — mock session required.
        mock_session = MagicMock(spec=Session)
        repo = DatabaseAdminRepository(mock_session)

        repo.truncate_table(_ALLOWED)

        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()


# ===========================================================================
# truncate_tables — success and exception paths (MagicMock)
# DOCUMENTED: TRUNCATE TABLE … CASCADE is Postgres-only DDL.
# ===========================================================================


class TestTruncateTablesMock:
    def test_when_all_allowed_and_execute_succeeds_then_cleared_returned(
        self,
    ) -> None:
        # Postgres-only DDL — mock session required.
        mock_session = MagicMock(spec=Session)
        repo = DatabaseAdminRepository(mock_session)

        cleared, errors = repo.truncate_tables([_ALLOWED])

        assert cleared == [_ALLOWED]
        assert errors == []
        mock_session.commit.assert_called_once()

    def test_when_execute_raises_then_rollback_called_and_error_recorded(
        self,
    ) -> None:
        # Postgres-only DDL — mock session required.
        mock_session = MagicMock(spec=Session)
        mock_session.execute.side_effect = Exception("deadlock detected")
        repo = DatabaseAdminRepository(mock_session)

        cleared, errors = repo.truncate_tables([_ALLOWED])

        assert cleared == []
        assert len(errors) == 1
        assert _ALLOWED in errors[0]
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()

    def test_when_one_succeeds_one_fails_then_only_successful_committed(
        self,
    ) -> None:
        # First table executes fine; second raises.
        mock_session = MagicMock(spec=Session)
        mock_session.execute.side_effect = [None, Exception("error")]
        repo = DatabaseAdminRepository(mock_session)

        cleared, errors = repo.truncate_tables(["instruments", "price_history"])

        assert "instruments" in cleared
        assert len(errors) == 1
        mock_session.commit.assert_called_once()

    def test_when_empty_list_then_nothing_cleared_no_commit(self) -> None:
        mock_session = MagicMock(spec=Session)
        repo = DatabaseAdminRepository(mock_session)

        cleared, errors = repo.truncate_tables([])

        assert cleared == []
        assert errors == []
        mock_session.commit.assert_not_called()
