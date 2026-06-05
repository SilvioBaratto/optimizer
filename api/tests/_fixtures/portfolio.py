"""DB seed builder for the portfolio domain.

Populates the parent ``Portfolio`` + a ``PortfolioSnapshot`` whose
``weights``/``sector_mapping`` are computed from child rows, so the
builder seeds the ``SnapshotWeight`` / ``SnapshotSectorMapping`` tables
directly.  Also seeds one ``BrokerPosition`` and one
``BrokerAccountSnapshot``.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import NamedTuple

from sqlalchemy.orm import Session

from app.models.portfolio.portfolio import (
    BrokerAccountSnapshot,
    BrokerPosition,
    Portfolio,
    PortfolioSnapshot,
    SnapshotSectorMapping,
    SnapshotWeight,
)
from tests._fixtures._helpers import add_and_flush

_DEFAULT_WEIGHTS = {"AAPL": 0.6, "MSFT": 0.4}
_DEFAULT_SECTORS = {"AAPL": "Technology", "MSFT": "Technology"}
_SEED_DATE = date(2024, 1, 1)


class PortfolioSeed(NamedTuple):
    portfolio: Portfolio
    snapshot: PortfolioSnapshot
    position: BrokerPosition
    account: BrokerAccountSnapshot


def _add_weight_rows(
    session: Session, snapshot: PortfolioSnapshot, weights: dict[str, float]
) -> None:
    for ticker, weight in weights.items():
        sector = _DEFAULT_SECTORS.get(ticker, "Unknown")
        session.add(
            SnapshotWeight(snapshot_id=snapshot.id, ticker=ticker, weight=weight)
        )
        session.add(
            SnapshotSectorMapping(snapshot_id=snapshot.id, ticker=ticker, sector=sector)
        )
    session.flush()


def _seed_snapshot(
    session: Session, portfolio: Portfolio, weights: dict[str, float]
) -> PortfolioSnapshot:
    snapshot = add_and_flush(
        session,
        PortfolioSnapshot(
            portfolio_id=portfolio.id,
            snapshot_date=_SEED_DATE,
            snapshot_type="optimization",
            holding_count=len(weights),
        ),
    )
    _add_weight_rows(session, snapshot, weights)
    return snapshot


def _make_portfolio(name: str) -> Portfolio:
    # Explicit values for server_default columns so the returned ORM object
    # is immediately usable without a session.refresh (no commit fires here).
    return Portfolio(
        name=name,
        currency="EUR",
        benchmark_ticker="SPY",
        is_active=True,
    )


def _make_position(portfolio: Portfolio) -> BrokerPosition:
    return BrokerPosition(
        portfolio_id=portfolio.id,
        ticker="AAPL_US_EQ",
        quantity=10.0,
        average_price=150.0,
        synced_at=datetime.now(timezone.utc),
    )


def _make_account(portfolio: Portfolio) -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        portfolio_id=portfolio.id,
        snapshot_date=_SEED_DATE,
        total=10_000.0,
        free=2_000.0,
        invested=8_000.0,
        currency="EUR",
        synced_at=datetime.now(timezone.utc),
    )


def seed_portfolio(
    session: Session,
    *,
    name: str = "test-portfolio",
    weights: dict[str, float] | None = None,
) -> PortfolioSeed:
    weights = dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)
    portfolio = add_and_flush(session, _make_portfolio(name))
    snapshot = _seed_snapshot(session, portfolio, weights)
    position = add_and_flush(session, _make_position(portfolio))
    account = add_and_flush(session, _make_account(portfolio))
    return PortfolioSeed(portfolio, snapshot, position, account)
