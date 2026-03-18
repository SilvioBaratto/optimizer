"""Repository for portfolio state persistence operations."""

import uuid
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.portfolio import (
    ActivityEvent,
    BrokerAccountSnapshot,
    BrokerPosition,
    Portfolio,
    PortfolioSnapshot,
    RegimeState,
)
from app.repositories.base import RepositoryBase


class PortfolioRepository(RepositoryBase):
    """Repository for portfolio CRUD, snapshots, positions, events, and regime states."""

    def __init__(self, session: Session):
        super().__init__(session)

    # ------------------------------------------------------------------
    # Portfolio CRUD
    # ------------------------------------------------------------------

    def get_by_name(self, name: str) -> Portfolio | None:
        stmt = select(Portfolio).where(Portfolio.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_or_create(self, name: str, **defaults: Any) -> Portfolio:
        existing = self.get_by_name(name)
        if existing:
            return existing
        portfolio = Portfolio(name=name, **defaults)
        self.session.add(portfolio)
        self.session.flush()
        return portfolio

    def get_all_active(self) -> Sequence[Portfolio]:
        stmt = (
            select(Portfolio)
            .where(Portfolio.is_active.is_(True))
            .order_by(Portfolio.name)
        )
        return self.session.execute(stmt).scalars().all()

    def get_by_id(self, portfolio_id: uuid.UUID) -> Portfolio | None:
        stmt = select(Portfolio).where(Portfolio.id == portfolio_id)
        return self.session.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def create_snapshot(
        self,
        portfolio_id: uuid.UUID,
        snapshot_date: date,
        snapshot_type: str,
        weights: dict[str, float],
        *,
        sector_mapping: dict[str, str] | None = None,
        summary: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        turnover: float | None = None,
    ) -> PortfolioSnapshot:
        snap = PortfolioSnapshot(
            portfolio_id=portfolio_id,
            snapshot_date=snapshot_date,
            snapshot_type=snapshot_type,
            weights=weights,
            sector_mapping=sector_mapping,
            summary=summary,
            optimizer_config=optimizer_config,
            turnover=turnover,
            holding_count=len(weights),
        )
        self.session.add(snap)
        self.session.flush()
        return snap

    def get_latest_snapshot(
        self, portfolio_id: uuid.UUID,
    ) -> PortfolioSnapshot | None:
        stmt = (
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.portfolio_id == portfolio_id)
            .order_by(PortfolioSnapshot.snapshot_date.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_snapshots(
        self,
        portfolio_id: uuid.UUID,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 100,
    ) -> Sequence[PortfolioSnapshot]:
        stmt = (
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        )
        if start_date:
            stmt = stmt.where(PortfolioSnapshot.snapshot_date >= start_date)
        if end_date:
            stmt = stmt.where(PortfolioSnapshot.snapshot_date <= end_date)
        stmt = stmt.order_by(PortfolioSnapshot.snapshot_date.desc()).limit(limit)
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Broker positions
    # ------------------------------------------------------------------

    def upsert_positions(
        self,
        portfolio_id: uuid.UUID,
        positions: list[dict[str, Any]],
        synced_at: datetime,
    ) -> int:
        rows = [
            {
                "id": uuid.uuid4(),
                "portfolio_id": portfolio_id,
                "synced_at": synced_at,
                **pos,
            }
            for pos in positions
        ]
        return self._upsert(
            BrokerPosition,
            rows,
            constraint_name="uq_broker_position_portfolio_ticker",
            update_columns=[
                "yfinance_ticker", "name", "quantity", "average_price",
                "current_price", "ppl", "fx_ppl", "initial_fill_date",
                "synced_at", "updated_at",
            ],
        )

    def delete_stale_positions(
        self, portfolio_id: uuid.UUID, current_tickers: set[str],
    ) -> int:
        if not current_tickers:
            return 0
        stmt = (
            delete(BrokerPosition)
            .where(BrokerPosition.portfolio_id == portfolio_id)
            .where(BrokerPosition.ticker.notin_(current_tickers))
        )
        result = self.session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    def get_positions(
        self, portfolio_id: uuid.UUID,
    ) -> Sequence[BrokerPosition]:
        stmt = (
            select(BrokerPosition)
            .where(BrokerPosition.portfolio_id == portfolio_id)
            .order_by(BrokerPosition.ticker)
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Broker account snapshots
    # ------------------------------------------------------------------

    def upsert_account_snapshot(
        self,
        portfolio_id: uuid.UUID,
        cash_data: dict[str, Any],
        synced_at: datetime,
    ) -> BrokerAccountSnapshot:
        snap = BrokerAccountSnapshot(
            portfolio_id=portfolio_id,
            total=cash_data["total"],
            free=cash_data["free"],
            invested=cash_data["invested"],
            blocked=cash_data.get("blocked"),
            result=cash_data.get("result"),
            currency=cash_data.get("currency", "EUR"),
            synced_at=synced_at,
        )
        self.session.add(snap)
        self.session.flush()
        return snap

    def get_latest_account_snapshot(
        self, portfolio_id: uuid.UUID,
    ) -> BrokerAccountSnapshot | None:
        stmt = (
            select(BrokerAccountSnapshot)
            .where(BrokerAccountSnapshot.portfolio_id == portfolio_id)
            .order_by(BrokerAccountSnapshot.synced_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Activity events
    # ------------------------------------------------------------------

    def add_event(
        self,
        event_type: str,
        title: str,
        *,
        portfolio_id: uuid.UUID | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActivityEvent:
        event = ActivityEvent(
            portfolio_id=portfolio_id,
            event_type=event_type,
            title=title,
            description=description,
            metadata_=metadata,
        )
        self.session.add(event)
        self.session.flush()
        return event

    def get_events(
        self,
        *,
        portfolio_id: uuid.UUID | None = None,
        event_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Sequence[ActivityEvent]:
        stmt = select(ActivityEvent)
        if portfolio_id is not None:
            stmt = stmt.where(ActivityEvent.portfolio_id == portfolio_id)
        if event_type is not None:
            stmt = stmt.where(ActivityEvent.event_type == event_type)
        stmt = (
            stmt.order_by(ActivityEvent.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return self.session.execute(stmt).scalars().all()

    def count_events(
        self,
        *,
        portfolio_id: uuid.UUID | None = None,
        event_type: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(ActivityEvent)
        if portfolio_id is not None:
            stmt = stmt.where(ActivityEvent.portfolio_id == portfolio_id)
        if event_type is not None:
            stmt = stmt.where(ActivityEvent.event_type == event_type)
        return self.session.execute(stmt).scalar_one()

    # ------------------------------------------------------------------
    # Regime states
    # ------------------------------------------------------------------

    def upsert_regime_state(
        self,
        state_date: date,
        regime: str,
        probabilities: list[dict[str, Any]],
        *,
        model_type: str = "hmm",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        rows = [
            {
                "id": uuid.uuid4(),
                "state_date": state_date,
                "regime": regime,
                "probabilities": probabilities,
                "model_type": model_type,
                "metadata": metadata,
            },
        ]
        return self._upsert(
            RegimeState,
            rows,
            constraint_name="uq_regime_state_date_model",
            update_columns=["regime", "probabilities", "metadata", "updated_at"],
        )

    def get_latest_regime(
        self, model_type: str = "hmm",
    ) -> RegimeState | None:
        stmt = (
            select(RegimeState)
            .where(RegimeState.model_type == model_type)
            .order_by(RegimeState.state_date.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()
