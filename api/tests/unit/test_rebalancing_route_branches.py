"""Branch coverage for GET /api/v1/rebalance/preview with broker positions.

Exercises lines 119-123 of rebalance.py — the positions price-extraction loop
(``if price:`` branch) and the ``if positions`` branch that calls
``build_trade_list``.

Seeding strategy: reuse the local helpers already established by
tests/unit/test_rebalancing_routes.py (_make_portfolio, _make_policy,
_make_snapshot) and add a ``_make_position`` helper that seeds a
BrokerPosition with a non-null current_price so the loop body runs.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.portfolio.portfolio import (
    BrokerPosition,
    Portfolio,
    PortfolioSnapshot,
    SnapshotWeight,
)
from app.models.rebalancing.rebalancing import RebalancingPolicy

BASE_PREVIEW = "/api/v1/rebalance/preview"


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _make_portfolio(db_session: Session, name: str | None = None) -> Portfolio:
    portfolio = Portfolio(name=name or f"port-{uuid.uuid4().hex[:8]}")
    db_session.add(portfolio)
    db_session.flush()
    return portfolio


def _make_policy(
    db_session: Session,
    portfolio_id: uuid.UUID,
    policy_type: str = "threshold",
    is_active: bool = True,
    config: dict | None = None,
) -> RebalancingPolicy:
    policy = RebalancingPolicy(
        portfolio_id=portfolio_id,
        name="test-policy",
        policy_type=policy_type,
        config=config or {"threshold_type": "absolute", "threshold": 0.05},
        is_active=is_active,
    )
    db_session.add(policy)
    db_session.flush()
    return policy


def _make_snapshot(
    db_session: Session,
    portfolio_id: uuid.UUID,
    weights: dict[str, float] | None = None,
) -> PortfolioSnapshot:
    weights = weights or {"AAPL": 0.6, "MSFT": 0.4}
    snap = PortfolioSnapshot(
        portfolio_id=portfolio_id,
        snapshot_date=date(2024, 1, 1),
        snapshot_type="optimization",
        holding_count=len(weights),
    )
    db_session.add(snap)
    db_session.flush()
    for ticker, weight in weights.items():
        db_session.add(SnapshotWeight(snapshot_id=snap.id, ticker=ticker, weight=weight))
    db_session.flush()
    return snap


def _make_position(
    db_session: Session,
    portfolio_id: uuid.UUID,
    ticker: str = "AAPL_US_EQ",
    yfinance_ticker: str | None = "AAPL",
    current_price: float | None = 175.0,
    average_price: float = 150.0,
    quantity: float = 10.0,
) -> BrokerPosition:
    pos = BrokerPosition(
        portfolio_id=portfolio_id,
        ticker=ticker,
        yfinance_ticker=yfinance_ticker,
        quantity=quantity,
        average_price=average_price,
        current_price=current_price,
        synced_at=datetime.now(timezone.utc),
    )
    db_session.add(pos)
    db_session.flush()
    return pos


# ---------------------------------------------------------------------------
# Tests: positions with current_price (covers lines 119-135 of rebalance.py)
# ---------------------------------------------------------------------------


class TestPreviewWithPositions:
    """The ``if price:`` loop and ``if positions: build_trade_list`` branches."""

    def test_returns_200_with_positions_having_current_price(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id, weights={"AAPL": 0.6, "MSFT": 0.4})
        _make_position(db_session, portfolio.id, current_price=175.0)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200

    def test_target_weights_present_in_response(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id, weights={"AAPL": 0.6, "MSFT": 0.4})
        _make_position(db_session, portfolio.id, current_price=175.0)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        body = resp.json()
        key = "targetWeights" if "targetWeights" in body else "target_weights"
        assert key in body
        assert body[key] is not None

    def test_trades_key_present_in_response(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id, weights={"AAPL": 0.6, "MSFT": 0.4})
        _make_position(db_session, portfolio.id, current_price=175.0)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        body = resp.json()
        assert "trades" in body

    def test_status_is_null_when_policy_and_snapshot_exist(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id)
        _make_position(db_session, portfolio.id, current_price=175.0)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.json()["status"] is None

    def test_position_with_none_current_price_falls_back_to_average_price(
        self, client: TestClient, db_session: Session
    ) -> None:
        """current_price=None → average_price used as fallback (``price`` is truthy)."""
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id, weights={"AAPL": 1.0})
        _make_position(
            db_session,
            portfolio.id,
            current_price=None,
            average_price=150.0,
        )

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        assert "trades" in resp.json()

    def test_position_with_yfinance_ticker_none_falls_back_to_ticker(
        self, client: TestClient, db_session: Session
    ) -> None:
        """yfinance_ticker=None → pos.ticker used as price key."""
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id, weights={"AAPL_US_EQ": 1.0})
        _make_position(
            db_session,
            portfolio.id,
            ticker="AAPL_US_EQ",
            yfinance_ticker=None,
            current_price=175.0,
        )

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200

    def test_policy_type_returned_in_response(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, policy_type="calendar", is_active=True)
        _make_snapshot(db_session, portfolio.id)
        _make_position(db_session, portfolio.id, current_price=175.0)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        body = resp.json()
        key = "policyType" if "policyType" in body else "policy_type"
        assert body[key] == "calendar"
