"""T4 — ``PositionRepository`` (current-snapshot holdings) + ``OrderRepository`` reads.

Resolved contracts (SPEC §8, 2026-09-21):

* **O1 = Option B** — ``set_holdings(portfolio_id, holdings: list[dict], *, asof,
  paper_order_id=None) -> None`` **replaces** the whole snapshot (delete-then-insert),
  each ``holdings`` row dict carrying ``ticker``/``weight`` + optional ``shares``/
  ``notional``; ``get_holdings(portfolio_id) -> list[Position]`` (full rows — no
  ``list_positions``, no ``lines`` channel; ``asof`` keyword-only).
* **O2 = Option A** — ``list_for_portfolio(portfolio_id)`` (newest-first) +
  ``latest_for_portfolio(portfolio_id)`` (most recent / ``None`` when empty); no
  ``run_id`` filter (``paper_orders`` has no ``run_id`` column).

Both repos sit on ``RepositoryBase``, take an injected sync session, and never
``commit`` (the caller owns the transaction, D1).
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from portopt_db.models.orders.paper_order import PaperOrder

from fund.audit import OrderRepository, PositionRepository

_PID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_OTHER = uuid.UUID("22222222-2222-2222-2222-222222222222")
_ASOF = date(2026, 1, 30)


def _order(
    portfolio_id: uuid.UUID,
    *,
    asof: date,
    created_at: datetime,
    weights_hash: str = "h",
) -> PaperOrder:
    """A minimal filled ticket; ``created_at`` set explicitly so ordering is
    deterministic under SQLite's second-resolution ``func.now()``."""
    return PaperOrder(
        portfolio_id=portfolio_id,
        asof=asof,
        weights_hash=weights_hash,
        weights={"AAA": 1.0},
        fill_date=asof,
        lines=[{"ticker": "AAA", "weight": 1.0}],
        notional=1000.0,
        total_commission=1.0,
        total_slippage_cost=0.5,
        created_at=created_at,
    )


# --- exports ---------------------------------------------------------------


def test_position_repository_exported_from_audit():
    from fund.audit import PositionRepository as Exported

    assert Exported is PositionRepository


# --- PositionRepository.set_holdings / get_holdings (O1 = Option B) ---------


def test_set_holdings_persists_snapshot_rows(db_session):
    repo = PositionRepository(db_session)

    repo.set_holdings(
        _PID,
        [
            {"ticker": "AAA", "weight": 0.6, "shares": 10.0, "notional": 600.0},
            {"ticker": "BBB", "weight": 0.4, "shares": 5.0, "notional": 400.0},
        ],
        asof=_ASOF,
    )

    rows = repo.get_holdings(_PID)
    assert [r.ticker for r in rows] == ["AAA", "BBB"]
    aaa = {r.ticker: r for r in rows}["AAA"]
    assert aaa.weight == 0.6
    assert aaa.shares == 10.0
    assert aaa.notional == 600.0
    assert aaa.asof == _ASOF


def test_set_holdings_replaces_previous_snapshot(db_session):
    repo = PositionRepository(db_session)

    repo.set_holdings(_PID, [{"ticker": "AAA", "weight": 1.0}], asof=_ASOF)
    repo.set_holdings(
        _PID,
        [{"ticker": "BBB", "weight": 0.7}, {"ticker": "CCC", "weight": 0.3}],
        asof=date(2026, 2, 27),
    )

    rows = repo.get_holdings(_PID)
    # AAA is gone — a snapshot replace, not an accumulation.
    assert [r.ticker for r in rows] == ["BBB", "CCC"]


def test_set_holdings_optional_shares_notional_default_none(db_session):
    repo = PositionRepository(db_session)

    repo.set_holdings(_PID, [{"ticker": "AAA", "weight": 1.0}], asof=_ASOF)

    (row,) = repo.get_holdings(_PID)
    assert row.shares is None
    assert row.notional is None
    assert row.paper_order_id is None


def test_set_holdings_records_paper_order_provenance(db_session):
    order = _order(_PID, asof=_ASOF, created_at=datetime(2026, 1, 31, tzinfo=UTC))
    db_session.add(order)
    db_session.flush()

    repo = PositionRepository(db_session)
    repo.set_holdings(
        _PID,
        [{"ticker": "AAA", "weight": 1.0}],
        asof=_ASOF,
        paper_order_id=order.id,
    )

    (row,) = repo.get_holdings(_PID)
    assert row.paper_order_id == order.id


def test_get_holdings_scoped_to_portfolio(db_session):
    repo = PositionRepository(db_session)

    repo.set_holdings(_PID, [{"ticker": "AAA", "weight": 1.0}], asof=_ASOF)
    repo.set_holdings(_OTHER, [{"ticker": "ZZZ", "weight": 1.0}], asof=_ASOF)

    assert [r.ticker for r in repo.get_holdings(_PID)] == ["AAA"]
    assert [r.ticker for r in repo.get_holdings(_OTHER)] == ["ZZZ"]


def test_get_holdings_empty_returns_empty_list(db_session):
    repo = PositionRepository(db_session)
    assert repo.get_holdings(_PID) == []


def test_set_holdings_does_not_commit(db_session, monkeypatch):
    repo = PositionRepository(db_session)
    called = {"commit": False}
    monkeypatch.setattr(
        db_session, "commit", lambda: called.__setitem__("commit", True)
    )

    repo.set_holdings(_PID, [{"ticker": "AAA", "weight": 1.0}], asof=_ASOF)

    assert called["commit"] is False


# --- OrderRepository.list_for_portfolio / latest_for_portfolio (O2 = A) -----


def test_list_for_portfolio_newest_first(db_session):
    repo = OrderRepository(db_session)
    o1 = _order(
        _PID,
        asof=date(2026, 1, 30),
        created_at=datetime(2026, 1, 31, tzinfo=UTC),
        weights_hash="h1",
    )
    o2 = _order(
        _PID,
        asof=date(2026, 2, 27),
        created_at=datetime(2026, 2, 28, tzinfo=UTC),
        weights_hash="h2",
    )
    o3 = _order(
        _PID,
        asof=date(2026, 3, 30),
        created_at=datetime(2026, 3, 31, tzinfo=UTC),
        weights_hash="h3",
    )
    db_session.add_all([o1, o2, o3])
    db_session.flush()

    listed = repo.list_for_portfolio(_PID)
    assert [o.id for o in listed] == [o3.id, o2.id, o1.id]


def test_list_for_portfolio_scoped_and_empty(db_session):
    repo = OrderRepository(db_session)
    db_session.add(
        _order(_OTHER, asof=_ASOF, created_at=datetime(2026, 1, 31, tzinfo=UTC))
    )
    db_session.flush()

    assert repo.list_for_portfolio(_PID) == []


def test_latest_for_portfolio_returns_most_recent(db_session):
    repo = OrderRepository(db_session)
    old = _order(
        _PID,
        asof=date(2026, 1, 30),
        created_at=datetime(2026, 1, 31, tzinfo=UTC),
        weights_hash="h1",
    )
    new = _order(
        _PID,
        asof=date(2026, 2, 27),
        created_at=datetime(2026, 2, 28, tzinfo=UTC),
        weights_hash="h2",
    )
    db_session.add_all([old, new])
    db_session.flush()

    latest = repo.latest_for_portfolio(_PID)
    assert latest is not None
    assert latest.id == new.id


def test_latest_for_portfolio_none_when_empty(db_session):
    repo = OrderRepository(db_session)
    assert repo.latest_for_portfolio(_PID) is None
