"""Tests for BrokerAccountSnapshot upsert deduplication (issue #327).

Verifies that upsert_account_snapshot deduplicates on (portfolio_id,
snapshot_date) so repeated broker syncs within the same calendar day update
the existing row rather than appending a new one.

Uses SQLite in-memory via the shared db_session fixture.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.portfolio import BrokerAccountSnapshot, Portfolio
from app.repositories.portfolio_repository import PortfolioRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CASH_V1 = {
    "total": 10_000.0,
    "free": 2_000.0,
    "invested": 8_000.0,
    "blocked": None,
    "result": 150.0,
    "currency": "GBP",
}

_CASH_V2 = {
    "total": 10_500.0,
    "free": 1_800.0,
    "invested": 8_700.0,
    "blocked": 50.0,
    "result": 200.0,
    "currency": "GBP",
}


def _make_portfolio(session: Session, name: str) -> Portfolio:
    p = Portfolio(name=name)
    session.add(p)
    session.flush()
    return p


def _count_snapshots(session: Session, portfolio_id: uuid.UUID) -> int:
    stmt = (
        select(func.count())
        .select_from(BrokerAccountSnapshot)
        .where(BrokerAccountSnapshot.portfolio_id == portfolio_id)
    )
    return session.execute(stmt).scalar_one()


# ---------------------------------------------------------------------------
# Tests — same day deduplication
# ---------------------------------------------------------------------------


class TestUpsertSameDay:
    def test_second_call_updates_not_inserts(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_snap_same_day_a")
        repo = PortfolioRepository(db_session)

        t1 = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 24, 14, 30, 0, tzinfo=timezone.utc)

        snap1 = repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t1)
        db_session.flush()
        assert _count_snapshots(db_session, portfolio.id) == 1
        row_id = snap1.id

        snap2 = repo.upsert_account_snapshot(portfolio.id, _CASH_V2, t2)
        db_session.flush()

        assert _count_snapshots(db_session, portfolio.id) == 1
        assert snap2.id == row_id

    def test_second_call_updates_value_columns(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_snap_same_day_b")
        repo = PortfolioRepository(db_session)

        t1 = datetime(2026, 3, 24, 8, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 24, 15, 0, 0, tzinfo=timezone.utc)

        repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t1)
        db_session.flush()
        repo.upsert_account_snapshot(portfolio.id, _CASH_V2, t2)
        db_session.flush()

        row = db_session.execute(
            select(BrokerAccountSnapshot).where(
                BrokerAccountSnapshot.portfolio_id == portfolio.id
            )
        ).scalar_one()

        assert row.total == _CASH_V2["total"]
        assert row.free == _CASH_V2["free"]
        assert row.invested == _CASH_V2["invested"]
        assert row.blocked == _CASH_V2["blocked"]
        assert row.result == _CASH_V2["result"]
        # SQLite strips tzinfo on read; compare naive datetimes
        assert row.synced_at.replace(tzinfo=None) == t2.replace(tzinfo=None)

    def test_snapshot_date_reflects_call_date(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_snap_same_day_c")
        repo = PortfolioRepository(db_session)

        t = datetime(2026, 3, 24, 10, 0, 0, tzinfo=timezone.utc)
        snap = repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t)
        db_session.flush()

        assert snap.snapshot_date == date(2026, 3, 24)


# ---------------------------------------------------------------------------
# Tests — different days create separate rows
# ---------------------------------------------------------------------------


class TestUpsertDifferentDays:
    def test_different_dates_create_two_rows(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_snap_diff_day_a")
        repo = PortfolioRepository(db_session)

        t_day1 = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
        t_day2 = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)

        repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t_day1)
        db_session.flush()
        repo.upsert_account_snapshot(portfolio.id, _CASH_V2, t_day2)
        db_session.flush()

        assert _count_snapshots(db_session, portfolio.id) == 2

    def test_three_syncs_two_days_two_rows(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_snap_diff_day_b")
        repo = PortfolioRepository(db_session)

        t_day1_am = datetime(2026, 3, 23, 8, 0, 0, tzinfo=timezone.utc)
        t_day1_pm = datetime(2026, 3, 23, 16, 0, 0, tzinfo=timezone.utc)
        t_day2 = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)

        repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t_day1_am)
        db_session.flush()
        repo.upsert_account_snapshot(portfolio.id, _CASH_V2, t_day1_pm)
        db_session.flush()
        repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t_day2)
        db_session.flush()

        assert _count_snapshots(db_session, portfolio.id) == 2

    def test_different_portfolios_do_not_collide(self, db_session: Session) -> None:
        p1 = _make_portfolio(db_session, "test_snap_diff_portfolio_a")
        p2 = _make_portfolio(db_session, "test_snap_diff_portfolio_b")
        repo = PortfolioRepository(db_session)

        t = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)

        repo.upsert_account_snapshot(p1.id, _CASH_V1, t)
        db_session.flush()
        repo.upsert_account_snapshot(p2.id, _CASH_V2, t)
        db_session.flush()

        assert _count_snapshots(db_session, p1.id) == 1
        assert _count_snapshots(db_session, p2.id) == 1


# ---------------------------------------------------------------------------
# Tests — get_latest_account_snapshot
# ---------------------------------------------------------------------------


class TestGetLatestAccountSnapshot:
    def test_returns_none_for_portfolio_with_no_snapshots(
        self, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session, "test_snap_latest_empty")
        repo = PortfolioRepository(db_session)
        assert repo.get_latest_account_snapshot(portfolio.id) is None

    def test_returns_single_snapshot(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_snap_latest_single")
        repo = PortfolioRepository(db_session)

        t = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)
        snap = repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t)
        db_session.flush()

        result = repo.get_latest_account_snapshot(portfolio.id)
        assert result is not None
        assert result.id == snap.id

    def test_returns_most_recent_day(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_snap_latest_order")
        repo = PortfolioRepository(db_session)

        t_old = datetime(2026, 3, 22, 9, 0, 0, tzinfo=timezone.utc)
        t_mid = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
        t_new = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)

        repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t_old)
        db_session.flush()
        repo.upsert_account_snapshot(portfolio.id, _CASH_V1, t_mid)
        db_session.flush()
        snap_latest = repo.upsert_account_snapshot(portfolio.id, _CASH_V2, t_new)
        db_session.flush()

        result = repo.get_latest_account_snapshot(portfolio.id)
        assert result is not None
        assert result.id == snap_latest.id
        assert result.total == _CASH_V2["total"]

    def test_isolates_by_portfolio(self, db_session: Session) -> None:
        p1 = _make_portfolio(db_session, "test_snap_latest_iso_a")
        p2 = _make_portfolio(db_session, "test_snap_latest_iso_b")
        repo = PortfolioRepository(db_session)

        t_old = datetime(2026, 3, 23, 9, 0, 0, tzinfo=timezone.utc)
        t_new = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)

        repo.upsert_account_snapshot(p1.id, _CASH_V1, t_new)
        db_session.flush()
        snap_p2 = repo.upsert_account_snapshot(p2.id, _CASH_V2, t_old)
        db_session.flush()

        result = repo.get_latest_account_snapshot(p2.id)
        assert result is not None
        assert result.id == snap_p2.id
        assert result.total == _CASH_V2["total"]
