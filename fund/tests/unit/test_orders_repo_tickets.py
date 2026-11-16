"""``OrderRepository`` reads of filled tickets and the ``created_at`` re-stamp.

``filled_tickets`` returns :class:`FilledTicket` values ranked oldest to newest by
fill bar, decision bar, ``created_at`` and id; ``latest_filled_ticket`` returns the
last of that ranking, or ``None``. The re-stamp writes through the caller's session
without committing.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, date, datetime

import pytest
from portopt_db.models.orders.paper_order import PaperOrder

from fund.audit.orders_repository import FilledTicket, OrderRepository

_PID = uuid.UUID("33333333-3333-3333-3333-333333333333")
_OTHER = uuid.UUID("44444444-4444-4444-4444-444444444444")


def _order(
    session,
    *,
    fill_date: date,
    created_at: datetime,
    asof: date | None = None,
    portfolio_id: uuid.UUID = _PID,
    notional: float = 1000.0,
) -> PaperOrder:
    order = PaperOrder(
        portfolio_id=portfolio_id,
        asof=asof or fill_date,
        weights_hash=f"h-{fill_date}-{created_at.isoformat()}",
        weights={"AAA": 1.0},
        fill_date=fill_date,
        lines=[{"ticker": "AAA", "notional": notional, "fill_date": str(fill_date)}],
        notional=notional,
        total_commission=1.0,
        total_slippage_cost=0.5,
        created_at=created_at,
    )
    session.add(order)
    session.flush()
    return order


def _seed(session) -> list[PaperOrder]:
    """Four tickets of ``_PID`` in rank order, plus one of another portfolio."""
    _order(
        session,
        fill_date=date(2026, 1, 5),
        created_at=datetime(2026, 1, 9, tzinfo=UTC),
        portfolio_id=_OTHER,
    )
    return [
        _order(
            session,
            fill_date=date(2026, 1, 2),
            created_at=datetime(2026, 1, 9, tzinfo=UTC),
        ),
        _order(
            session,
            fill_date=date(2026, 1, 3),
            asof=date(2026, 1, 1),
            created_at=datetime(2026, 1, 8, tzinfo=UTC),
        ),
        _order(
            session,
            fill_date=date(2026, 1, 3),
            asof=date(2026, 1, 2),
            created_at=datetime(2026, 1, 3, tzinfo=UTC),
        ),
        _order(
            session,
            fill_date=date(2026, 1, 3),
            asof=date(2026, 1, 2),
            created_at=datetime(2026, 1, 4, tzinfo=UTC),
            notional=2500.0,
        ),
    ]


def test_filled_tickets_rank_oldest_first_by_fill_asof_and_created_at(db_session):
    """The portfolio's tickets come back in rank order; the last carries its stored
    fill date, notional, ``created_at`` and lines."""
    orders = _seed(db_session)

    tickets = OrderRepository(db_session).filled_tickets(_PID, filled_through=None)

    assert [ticket.id for ticket in tickets] == [order.id for order in orders]
    newest = tickets[-1]
    assert isinstance(newest, FilledTicket)
    assert (newest.fill_date, newest.notional) == (date(2026, 1, 3), 2500.0)
    assert newest.created_at == orders[-1].created_at
    assert newest.lines == tuple(orders[-1].lines)


def test_filled_tickets_stop_at_the_inclusive_fill_bound(db_session):
    """``filled_through`` keeps tickets filled on or before it."""
    orders = _seed(db_session)
    repo = OrderRepository(db_session)

    assert [
        t.id for t in repo.filled_tickets(_PID, filled_through=date(2026, 1, 2))
    ] == [orders[0].id]
    assert repo.filled_tickets(_PID, filled_through=date(2026, 1, 1)) == []


def test_latest_filled_ticket_is_the_last_of_the_ranked_tickets(db_session):
    """The newest ticket is the one the ascending read ends with."""
    orders = _seed(db_session)
    repo = OrderRepository(db_session)

    latest = repo.latest_filled_ticket(_PID, filled_through=None)
    bounded = repo.latest_filled_ticket(_PID, filled_through=date(2026, 1, 2))

    assert latest is not None
    assert latest.id == orders[-1].id
    assert bounded is not None
    assert bounded.id == orders[0].id
    assert repo.latest_filled_ticket(_PID, filled_through=date(2025, 12, 31)) is None
    assert repo.latest_filled_ticket(uuid.uuid4(), filled_through=None) is None


def test_a_filled_ticket_is_frozen(db_session):
    """Reassigning a ``FilledTicket`` field raises ``FrozenInstanceError``."""
    _seed(db_session)
    ticket = OrderRepository(db_session).filled_tickets(_PID, filled_through=None)[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        ticket.notional = 0.0  # type: ignore[misc]


def test_restamp_created_at_makes_an_older_ticket_the_newest(committing_session):
    """The stamp moves the ticket to the end of the rank, and a rollback undoes it."""
    orders = _seed(committing_session)
    restamped, newest = orders[2].id, orders[-1].id
    committing_session.commit()
    repo = OrderRepository(committing_session)

    repo.restamp_created_at(restamped)
    committing_session.expire_all()
    latest = repo.latest_filled_ticket(_PID, filled_through=None)
    committing_session.rollback()
    restored = repo.latest_filled_ticket(_PID, filled_through=None)

    assert latest is not None
    assert latest.id == restamped
    assert restored is not None
    assert restored.id == newest
