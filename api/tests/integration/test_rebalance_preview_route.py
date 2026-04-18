"""Integration tests for GET /api/v1/rebalance/preview/{name} (issue #425).

Exercises the full route → repository flow against the test SQLite database.
The seeded scenario mirrors production: a ``trading212``-style portfolio that
exists with a snapshot but has no active rebalancing policy yet, which
previously returned 404 and broke the Rebalancing page Trade Preview tab.

Issue #425 semantics:
- portfolio missing           → 404
- portfolio exists, no policy → 200 with ``status="no_active_policy"``
- portfolio exists, policy but no snapshot → 200 with ``status="no_snapshots"``
- portfolio + policy + snapshot → 200 with ``status=null``
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import date

import pytest
from app.models.portfolio import Portfolio, PortfolioSnapshot, SnapshotWeight
from app.models.rebalancing import RebalancingPolicy
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

BASE = "/api/v1/rebalance/preview"

_NAME_EXISTS_NO_POLICY = "integration-preview-no-policy"
_NAME_EXISTS_POLICY_NO_SNAPSHOT = "integration-preview-policy-no-snapshot"
_NAME_FULL = "integration-preview-full"


@pytest.fixture
def seeded_preview_portfolios(
    db_session: Session,
) -> Generator[dict[str, uuid.UUID], None, None]:
    """Seed three portfolios covering every empty-state path defined by #425."""
    p_no_policy = _seed_portfolio(db_session, _NAME_EXISTS_NO_POLICY)
    # portfolio has a snapshot but no active policy — #425 'no_active_policy'
    _seed_snapshot(db_session, p_no_policy.id, {"AAPL": 0.6, "MSFT": 0.4})

    p_policy_no_snap = _seed_portfolio(db_session, _NAME_EXISTS_POLICY_NO_SNAPSHOT)
    _seed_policy(db_session, p_policy_no_snap.id, policy_type="threshold")

    p_full = _seed_portfolio(db_session, _NAME_FULL)
    _seed_policy(db_session, p_full.id, policy_type="calendar")
    _seed_snapshot(db_session, p_full.id, {"AAPL": 0.5, "MSFT": 0.5})

    db_session.flush()
    yield {
        "no_policy": p_no_policy.id,
        "policy_no_snap": p_policy_no_snap.id,
        "full": p_full.id,
    }


def _seed_portfolio(session: Session, name: str) -> Portfolio:
    portfolio = Portfolio(name=name, description="integration test")
    session.add(portfolio)
    session.flush()
    return portfolio


def _seed_policy(
    session: Session,
    portfolio_id: uuid.UUID,
    policy_type: str = "threshold",
) -> RebalancingPolicy:
    policy = RebalancingPolicy(
        portfolio_id=portfolio_id,
        name="integration-policy",
        policy_type=policy_type,
        config={"threshold_type": "absolute", "threshold": 0.05},
        is_active=True,
    )
    session.add(policy)
    session.flush()
    return policy


def _seed_snapshot(
    session: Session,
    portfolio_id: uuid.UUID,
    weights: dict[str, float],
) -> PortfolioSnapshot:
    snapshot = PortfolioSnapshot(
        portfolio_id=portfolio_id,
        snapshot_date=date.today(),
        snapshot_type="optimization",
        holding_count=len(weights),
    )
    session.add(snapshot)
    session.flush()
    session.add_all(
        [
            SnapshotWeight(snapshot_id=snapshot.id, ticker=t, weight=w)
            for t, w in weights.items()
        ]
    )
    session.flush()
    return snapshot


def test_preview_missing_portfolio_returns_404(
    client: TestClient, seeded_preview_portfolios: dict[str, uuid.UUID]
) -> None:
    resp = client.get(f"{BASE}/does-not-exist")
    assert resp.status_code == 404
    assert "does-not-exist" in resp.json()["error"]["message"]


def test_preview_portfolio_without_active_policy_returns_200_empty(
    client: TestClient, seeded_preview_portfolios: dict[str, uuid.UUID]
) -> None:
    """Regression for #425: this previously returned 404 for ``trading212``."""
    resp = client.get(f"{BASE}/{_NAME_EXISTS_NO_POLICY}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "no_active_policy"
    assert body["trades"] == []


def test_preview_portfolio_with_policy_but_no_snapshot_returns_200_empty(
    client: TestClient, seeded_preview_portfolios: dict[str, uuid.UUID]
) -> None:
    resp = client.get(f"{BASE}/{_NAME_EXISTS_POLICY_NO_SNAPSHOT}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "no_snapshots"
    assert body["trades"] == []


def test_preview_happy_path_returns_200_with_null_status(
    client: TestClient, seeded_preview_portfolios: dict[str, uuid.UUID]
) -> None:
    resp = client.get(f"{BASE}/{_NAME_FULL}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] is None
    assert body["trades"] == []  # no broker positions → empty trades
