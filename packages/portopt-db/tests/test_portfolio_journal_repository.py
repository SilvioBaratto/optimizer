"""Tests for PortfolioJournal model and PortfolioJournalRepository.

Covers model round-trips with letter-bearing UUIDs (SQLite UUID affinity guard),
metadata registration, composite UniqueConstraint enforcement, upsert idempotency,
updated_at advancement with created_at preserved, and list_for_portfolio filtering.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from portopt_db.models.orders.portfolio_journal import PortfolioJournal
from portopt_db.repositories.orders.portfolio_journal_repository import (
    PortfolioJournalRepository,
)

# Letter-bearing UUIDs: prevents SQLite REAL-coercion of all-decimal UUID strings,
# which would crash the UUID result processor under statement caching.
_PID_A = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
_PID_B = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
_RUN_ID = uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")

_AS_OF_1 = dt.date(2026, 9, 10)
_AS_OF_2 = dt.date(2026, 9, 17)
_AS_OF_3 = dt.date(2026, 9, 24)

_TRADES = {"AAPL": {"weight": 0.1, "fill": 221.5}}
_ALLOC = {"AAPL": 0.1, "MSFT": 0.2}
_DRIFT = {"AAPL": 0.02, "MSFT": -0.01}
_NARRATIVE = "Rebalanced: added AAPL, trimmed MSFT."


class TestPortfolioJournalMetadata:
    """portfolio_journal must be visible on Base.metadata after the package import."""

    def test_table_registered_on_metadata(self, test_engine) -> None:
        """Verify portfolio_journal is created by Base.metadata.create_all."""
        tables = set(inspect(test_engine).get_table_names())
        assert "portfolio_journal" in tables

    def test_expected_indexes_exist(self, test_engine) -> None:
        """Verify both portfolio_id and as_of indexes are created."""
        names = {
            idx["name"] for idx in inspect(test_engine).get_indexes("portfolio_journal")
        }
        assert "ix_portfolio_journal_portfolio_id" in names
        assert "ix_portfolio_journal_as_of" in names

    def test_unique_constraint_exists(self, test_engine) -> None:
        """Verify uq_portfolio_journal_key composite constraint is created."""
        constraints = {
            uc["name"]
            for uc in inspect(test_engine).get_unique_constraints("portfolio_journal")
        }
        assert "uq_portfolio_journal_key" in constraints


class TestPortfolioJournalRoundTrip:
    """ORM model stores and retrieves all columns without truncation or coercion."""

    def test_json_and_uuid_columns_survive_flush(self, db_session) -> None:
        """Verify portfolio_id, run_id, and JSON columns round-trip correctly."""
        row = PortfolioJournal(
            portfolio_id=_PID_A,
            as_of=_AS_OF_1,
            run_id=_RUN_ID,
            trades=_TRADES,
            allocation=_ALLOC,
            drift=_DRIFT,
            narrative=_NARRATIVE,
        )
        db_session.add(row)
        db_session.flush()
        db_session.expire(row)

        fetched = db_session.execute(
            select(PortfolioJournal).where(PortfolioJournal.portfolio_id == _PID_A)
        ).scalar_one()
        assert fetched.portfolio_id == _PID_A
        assert fetched.run_id == _RUN_ID
        assert fetched.trades == _TRADES
        assert fetched.allocation == _ALLOC
        assert fetched.drift == _DRIFT
        assert fetched.narrative == _NARRATIVE

    def test_run_id_accepts_none(self, db_session) -> None:
        """run_id is nullable; None must persist and round-trip as None."""
        row = PortfolioJournal(
            portfolio_id=_PID_B,
            as_of=_AS_OF_1,
            run_id=None,
            trades=None,
            allocation=None,
            drift=None,
            narrative="",
        )
        db_session.add(row)
        db_session.flush()
        db_session.expire(row)

        fetched = db_session.execute(
            select(PortfolioJournal).where(PortfolioJournal.portfolio_id == _PID_B)
        ).scalar_one()
        assert fetched.run_id is None

    def test_timestamps_populated_after_flush(self, db_session) -> None:
        """Verify created_at and updated_at are non-None after the first flush."""
        pid = uuid.UUID("c3d4e5f6-a7b8-9012-cdef-123456789012")
        row = PortfolioJournal(
            portfolio_id=pid,
            as_of=dt.date(2026, 8, 1),
            run_id=None,
            trades=None,
            allocation=None,
            drift=None,
            narrative="",
        )
        db_session.add(row)
        db_session.flush()
        assert row.created_at is not None
        assert row.updated_at is not None


class TestPortfolioJournalUniqueConstraint:
    """Duplicate (portfolio_id, as_of) must raise; distinct combinations must succeed."""

    def test_duplicate_composite_key_raises_integrity_error(self, db_session) -> None:
        """Two rows sharing (portfolio_id, as_of) must violate uq_portfolio_journal_key."""
        for _ in range(2):
            db_session.add(
                PortfolioJournal(
                    portfolio_id=_PID_A,
                    as_of=dt.date(2026, 6, 1),
                    run_id=None,
                    trades=None,
                    allocation=None,
                    drift=None,
                    narrative="",
                )
            )
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_same_portfolio_different_as_of_is_allowed(self, db_session) -> None:
        """Same portfolio_id with distinct as_of values must not raise."""
        for d in (_AS_OF_1, _AS_OF_2):
            db_session.add(
                PortfolioJournal(
                    portfolio_id=_PID_A,
                    as_of=d,
                    run_id=None,
                    trades=None,
                    allocation=None,
                    drift=None,
                    narrative="",
                )
            )
        db_session.flush()  # must not raise

    def test_same_as_of_different_portfolio_id_is_allowed(self, db_session) -> None:
        """Different portfolio_ids on the same as_of must not raise."""
        for pid in (_PID_A, _PID_B):
            db_session.add(
                PortfolioJournal(
                    portfolio_id=pid,
                    as_of=dt.date(2026, 4, 1),
                    run_id=None,
                    trades=None,
                    allocation=None,
                    drift=None,
                    narrative="",
                )
            )
        db_session.flush()  # must not raise


class TestPortfolioJournalRepositoryUpsert:
    """PortfolioJournalRepository.upsert idempotency and conflict behaviour."""

    def test_upsert_writes_one_row(self, db_session) -> None:
        """A single upsert call must produce exactly one row."""
        repo = PortfolioJournalRepository(db_session)
        n = repo.upsert(
            _PID_A,
            _AS_OF_1,
            run_id=_RUN_ID,
            trades=_TRADES,
            allocation=_ALLOC,
            drift=_DRIFT,
            narrative=_NARRATIVE,
        )
        db_session.flush()
        assert n == 1
        rows = (
            db_session.execute(
                select(PortfolioJournal).where(PortfolioJournal.portfolio_id == _PID_A)
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1

    def test_upsert_twice_same_key_converges_to_one_row(self, db_session) -> None:
        """Two upserts for the same key must converge to one row with the latest data."""
        repo = PortfolioJournalRepository(db_session)
        repo.upsert(
            _PID_A,
            _AS_OF_2,
            run_id=None,
            trades=None,
            allocation=None,
            drift=None,
            narrative="v1",
        )
        db_session.flush()
        repo.upsert(
            _PID_A,
            _AS_OF_2,
            run_id=_RUN_ID,
            trades=_TRADES,
            allocation=_ALLOC,
            drift=_DRIFT,
            narrative="v2",
        )
        db_session.flush()

        rows = (
            db_session.execute(
                select(PortfolioJournal).where(
                    PortfolioJournal.portfolio_id == _PID_A,
                    PortfolioJournal.as_of == _AS_OF_2,
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        # Every column in _UPDATE_COLUMNS must reflect the second upsert: a check on the
        # narrative alone would miss a JSON column dropped from the update set.
        assert rows[0].narrative == "v2"
        assert rows[0].run_id == _RUN_ID
        assert rows[0].trades == _TRADES
        assert rows[0].allocation == _ALLOC
        assert rows[0].drift == _DRIFT

    def test_distinct_keys_produce_separate_rows(self, db_session) -> None:
        """Upserts with distinct (portfolio_id, as_of) pairs must each produce a row."""
        repo = PortfolioJournalRepository(db_session)
        for pid, d in ((_PID_A, _AS_OF_1), (_PID_B, _AS_OF_1), (_PID_A, _AS_OF_2)):
            repo.upsert(
                pid,
                d,
                run_id=None,
                trades=None,
                allocation=None,
                drift=None,
                narrative="",
            )
        db_session.flush()

        total = db_session.execute(select(PortfolioJournal)).scalars().all()
        assert len(total) == 3

    def test_updated_at_advances_created_at_preserved(self, db_session) -> None:
        """On conflict, updated_at must advance while created_at stays frozen."""
        _PAST = dt.datetime(2000, 1, 1, 0, 0, 0)
        repo = PortfolioJournalRepository(db_session)
        repo.upsert(
            _PID_A,
            _AS_OF_3,
            run_id=None,
            trades=None,
            allocation=None,
            drift=None,
            narrative="first",
        )
        db_session.flush()

        row = db_session.execute(
            select(PortfolioJournal).where(
                PortfolioJournal.portfolio_id == _PID_A,
                PortfolioJournal.as_of == _AS_OF_3,
            )
        ).scalar_one()
        created_at = row.created_at

        # Force updated_at back to a known sentinel before the second upsert.
        db_session.execute(
            text(
                "UPDATE portfolio_journal SET updated_at = :ts"
                " WHERE portfolio_id = :pid AND as_of = :d"
            ),
            {"ts": _PAST, "pid": str(_PID_A), "d": _AS_OF_3},
        )
        db_session.flush()
        db_session.expire(row)

        repo.upsert(
            _PID_A,
            _AS_OF_3,
            run_id=_RUN_ID,
            trades=_TRADES,
            allocation=_ALLOC,
            drift=_DRIFT,
            narrative="second",
        )
        db_session.flush()
        db_session.expire(row)

        row = db_session.execute(
            select(PortfolioJournal).where(
                PortfolioJournal.portfolio_id == _PID_A,
                PortfolioJournal.as_of == _AS_OF_3,
            )
        ).scalar_one()
        assert row.narrative == "second"
        assert row.created_at == created_at, "created_at must not change on conflict"
        assert row.updated_at > _PAST, "updated_at must advance on conflict"


class TestPortfolioJournalRepositoryReads:
    """list_for_portfolio ordering, since filtering, and limit."""

    def _seed(self, db_session, portfolio_id: uuid.UUID, dates: list[dt.date]) -> None:
        """Insert one overlay row per date for the given portfolio."""
        repo = PortfolioJournalRepository(db_session)
        for d in dates:
            repo.upsert(
                portfolio_id,
                d,
                run_id=None,
                trades=None,
                allocation=None,
                drift=None,
                narrative=str(d),
            )
        db_session.flush()

    def test_list_for_portfolio_newest_first(self, db_session) -> None:
        """list_for_portfolio must return rows ordered by as_of descending."""
        self._seed(db_session, _PID_A, [_AS_OF_1, _AS_OF_2, _AS_OF_3])
        repo = PortfolioJournalRepository(db_session)
        rows = repo.list_for_portfolio(_PID_A)
        dates = [r.as_of for r in rows]
        assert dates == sorted(dates, reverse=True)

    def test_list_for_portfolio_since_filter(self, db_session) -> None:
        """since parameter must exclude rows with as_of before the cutoff."""
        self._seed(db_session, _PID_A, [_AS_OF_1, _AS_OF_2, _AS_OF_3])
        repo = PortfolioJournalRepository(db_session)
        rows = repo.list_for_portfolio(_PID_A, since=_AS_OF_2)
        assert all(r.as_of >= _AS_OF_2 for r in rows)
        assert len(rows) == 2

    def test_list_for_portfolio_limit(self, db_session) -> None:
        """limit parameter must cap the number of rows returned."""
        self._seed(db_session, _PID_A, [_AS_OF_1, _AS_OF_2, _AS_OF_3])
        repo = PortfolioJournalRepository(db_session)
        rows = repo.list_for_portfolio(_PID_A, limit=2)
        assert len(rows) == 2

    def test_list_for_portfolio_scoped_to_one_portfolio(self, db_session) -> None:
        """list_for_portfolio must not return rows belonging to other portfolios."""
        self._seed(db_session, _PID_A, [_AS_OF_1])
        self._seed(db_session, _PID_B, [_AS_OF_2])
        repo = PortfolioJournalRepository(db_session)
        rows = repo.list_for_portfolio(_PID_A)
        assert all(r.portfolio_id == _PID_A for r in rows)
        assert len(rows) == 1

    def test_list_for_portfolio_empty_when_none_exist(self, db_session) -> None:
        """list_for_portfolio must return an empty sequence for an unknown portfolio."""
        unknown = uuid.UUID("d4e5f6a7-b8c9-0123-def0-123456789abc")
        repo = PortfolioJournalRepository(db_session)
        rows = repo.list_for_portfolio(unknown)
        assert list(rows) == []
