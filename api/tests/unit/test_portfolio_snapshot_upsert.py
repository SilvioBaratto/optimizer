"""Tests for PortfolioSnapshot upsert on (portfolio_id, snapshot_date, snapshot_type).

A second call to ``create_snapshot`` for the same triple must not raise
``IntegrityError``: it must replace the child EAV rows in place and yield
exactly one row in ``portfolio_snapshots``.
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.portfolio import (
    Portfolio,
    PortfolioSnapshot,
    SnapshotOptimizerParam,
    SnapshotSectorMapping,
    SnapshotSummaryEntry,
    SnapshotWeight,
)
from app.repositories.portfolio_repository import PortfolioRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE = date(2026, 4, 1)


def _make_portfolio(session: Session, name: str) -> Portfolio:
    p = Portfolio(name=name)
    session.add(p)
    session.flush()
    return p


def _count(session: Session, model: type, snapshot_id: uuid.UUID) -> int:
    stmt = (
        select(func.count()).select_from(model).where(model.snapshot_id == snapshot_id)
    )
    return session.execute(stmt).scalar_one()


def _count_snapshots(
    session: Session,
    portfolio_id: uuid.UUID,
    snapshot_type: str | None = None,
) -> int:
    stmt = (
        select(func.count())
        .select_from(PortfolioSnapshot)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
    )
    if snapshot_type is not None:
        stmt = stmt.where(PortfolioSnapshot.snapshot_type == snapshot_type)
    return session.execute(stmt).scalar_one()


# ---------------------------------------------------------------------------
# Tests — same triple replaces in place
# ---------------------------------------------------------------------------


class TestCreateSnapshotUpsert:
    def test_when_called_twice_with_same_triple_one_row_remains(
        self, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session, "test_upsert_one_row")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
        )
        db_session.flush()
        repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"MSFT": 1.0},
        )
        db_session.flush()

        assert _count_snapshots(db_session, portfolio.id, "research_run") == 1

    def test_when_called_twice_weights_are_replaced(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_upsert_weights")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 0.6, "MSFT": 0.4},
        )
        db_session.flush()
        snap = repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"GOOG": 0.7, "META": 0.3},
        )
        db_session.flush()

        assert _count(db_session, SnapshotWeight, snap.id) == 2
        weights = {
            w.ticker: w.weight
            for w in db_session.execute(
                select(SnapshotWeight).where(SnapshotWeight.snapshot_id == snap.id)
            ).scalars()
        }
        assert weights == {"GOOG": 0.7, "META": 0.3}

    def test_when_called_twice_sector_mapping_is_replaced(
        self, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session, "test_upsert_sectors")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
            sector_mapping={"AAPL": "Tech"},
        )
        db_session.flush()
        snap = repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"JPM": 1.0},
            sector_mapping={"JPM": "Financials"},
        )
        db_session.flush()

        assert _count(db_session, SnapshotSectorMapping, snap.id) == 1
        row = db_session.execute(
            select(SnapshotSectorMapping).where(
                SnapshotSectorMapping.snapshot_id == snap.id
            )
        ).scalar_one()
        assert row.ticker == "JPM"
        assert row.sector == "Financials"

    def test_when_called_twice_summary_entries_are_replaced(
        self, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session, "test_upsert_summary")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
            summary={"sharpe": 1.0, "vol": 0.2},
        )
        db_session.flush()
        snap = repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
            summary={"sharpe": 1.5},
        )
        db_session.flush()

        assert _count(db_session, SnapshotSummaryEntry, snap.id) == 1
        row = db_session.execute(
            select(SnapshotSummaryEntry).where(
                SnapshotSummaryEntry.snapshot_id == snap.id
            )
        ).scalar_one()
        assert row.key == "sharpe"

    def test_when_called_twice_optimizer_params_are_replaced(
        self, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session, "test_upsert_optimizer")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
            optimizer_config={"max_weights": 0.10, "l2_coef": 0.05},
        )
        db_session.flush()
        snap = repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
            optimizer_config={"max_weights": 0.05},
        )
        db_session.flush()

        assert _count(db_session, SnapshotOptimizerParam, snap.id) == 1

    def test_when_called_twice_turnover_is_updated(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_upsert_turnover")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
            turnover=0.1,
        )
        db_session.flush()
        snap = repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
            turnover=0.25,
        )
        db_session.flush()

        assert snap.turnover == 0.25

    def test_when_called_twice_holding_count_reflects_latest_weights(
        self, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session, "test_upsert_holdings")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
        )
        db_session.flush()
        snap = repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
        )
        db_session.flush()

        assert snap.holding_count == 3

    def test_when_called_twice_snapshot_id_is_stable(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_upsert_id_stable")
        repo = PortfolioRepository(db_session)

        first = repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"AAPL": 1.0},
        )
        db_session.flush()
        first_id = first.id
        second = repo.create_snapshot(
            portfolio.id,
            _DATE,
            "research_run",
            {"MSFT": 1.0},
        )
        db_session.flush()

        assert second.id == first_id


# ---------------------------------------------------------------------------
# Tests — different triple inserts a separate row
# ---------------------------------------------------------------------------


class TestCreateSnapshotDistinct:
    def test_when_snapshot_type_differs_two_rows_exist(
        self, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session, "test_distinct_type")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(portfolio.id, _DATE, "research_run", {"AAPL": 1.0})
        db_session.flush()
        repo.create_snapshot(portfolio.id, _DATE, "manual", {"AAPL": 1.0})
        db_session.flush()

        assert _count_snapshots(db_session, portfolio.id) == 2

    def test_when_date_differs_two_rows_exist(self, db_session: Session) -> None:
        portfolio = _make_portfolio(db_session, "test_distinct_date")
        repo = PortfolioRepository(db_session)

        repo.create_snapshot(portfolio.id, _DATE, "research_run", {"AAPL": 1.0})
        db_session.flush()
        repo.create_snapshot(
            portfolio.id,
            date(2026, 4, 2),
            "research_run",
            {"AAPL": 1.0},
        )
        db_session.flush()

        assert _count_snapshots(db_session, portfolio.id, "research_run") == 2
