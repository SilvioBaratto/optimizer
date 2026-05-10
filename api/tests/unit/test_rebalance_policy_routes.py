"""Unit tests for rebalancing policy CRUD and activate endpoints.

TDD Red → Green → Refactor.

Routes under test:
  GET  /api/v1/portfolio/{name}/rebalance-policy
  POST /api/v1/portfolio/{name}/rebalance-policy
  POST /api/v1/portfolio/{name}/activate-policy/{policy_id}

Acceptance criteria covered:
- list policies returns all policies for the named portfolio
- create policy succeeds for calendar / threshold / hybrid types
- invalid policy_type returns 422
- 404 for unknown portfolio (all three routes)
- 404 for unknown policy (activate)
- 409 for duplicate (portfolio_id, name) pair
- 404 when policy_id does not belong to the target portfolio
- activate atomically switches the active policy
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.portfolio.portfolio import Portfolio
from app.models.rebalancing.rebalancing import RebalancingPolicy

BASE = "/api/v1/portfolio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(db_session: Session, name: str | None = None) -> Portfolio:
    portfolio = Portfolio(name=name or f"port-{uuid.uuid4().hex[:8]}")
    db_session.add(portfolio)
    db_session.flush()
    return portfolio


def _make_policy(
    db_session: Session,
    portfolio_id: uuid.UUID,
    *,
    name: str = "default",
    policy_type: str = "calendar",
    is_active: bool = False,
    config: dict | None = None,
) -> RebalancingPolicy:
    policy = RebalancingPolicy(
        portfolio_id=portfolio_id,
        name=name,
        policy_type=policy_type,
        config=config or {"interval_days": 63},
        is_active=is_active,
    )
    db_session.add(policy)
    db_session.flush()
    return policy


# ---------------------------------------------------------------------------
# GET /{name}/rebalance-policy
# ---------------------------------------------------------------------------


class TestListPolicies:
    def test_returns_200_with_empty_list_when_no_policies(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)

        resp = client.get(f"{BASE}/{portfolio.name}/rebalance-policy")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_returns_all_policies_for_portfolio(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, name="monthly")
        _make_policy(db_session, portfolio.id, name="quarterly")

        resp = client.get(f"{BASE}/{portfolio.name}/rebalance-policy")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["items"]) == 2

    def test_does_not_return_policies_from_other_portfolios(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        other = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, name="mine")
        _make_policy(db_session, other.id, name="theirs")

        resp = client.get(f"{BASE}/{portfolio.name}/rebalance-policy")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1

    def test_returns_404_for_unknown_portfolio(self, client: TestClient) -> None:
        resp = client.get(f"{BASE}/no-such-portfolio/rebalance-policy")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /{name}/rebalance-policy
# ---------------------------------------------------------------------------


class TestCreatePolicy:
    _CALENDAR: dict[str, Any] = {
        "name": "quarterly-rebalance",
        "policy_type": "calendar",
        "config": {"interval_days": 63},
    }
    _THRESHOLD: dict[str, Any] = {
        "name": "threshold-5pct",
        "policy_type": "threshold",
        "config": {"threshold_type": "absolute", "threshold": 0.05},
    }
    _HYBRID: dict[str, Any] = {
        "name": "hybrid-policy",
        "policy_type": "hybrid",
        "config": {
            "calendar": {"frequency": "quarterly"},
            "threshold": {"threshold_type": "absolute", "threshold": 0.10},
        },
    }

    def test_creates_calendar_policy(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)

        resp = client.post(
            f"{BASE}/{portfolio.name}/rebalance-policy", json=self._CALENDAR
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "quarterly-rebalance"
        assert body["policyType"] == "calendar"

    def test_creates_threshold_policy(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)

        resp = client.post(
            f"{BASE}/{portfolio.name}/rebalance-policy", json=self._THRESHOLD
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["policyType"] == "threshold"

    def test_creates_hybrid_policy(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)

        resp = client.post(
            f"{BASE}/{portfolio.name}/rebalance-policy", json=self._HYBRID
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["policyType"] == "hybrid"

    def test_invalid_policy_type_returns_422(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)

        resp = client.post(
            f"{BASE}/{portfolio.name}/rebalance-policy",
            json={"name": "bad", "policy_type": "invalid_type", "config": {"k": 1}},
        )

        assert resp.status_code == 422

    def test_returns_404_for_unknown_portfolio(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE}/ghost-portfolio/rebalance-policy", json=self._CALENDAR
        )

        assert resp.status_code == 404

    def test_returns_409_for_duplicate_policy_name(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, name="quarterly-rebalance")

        resp = client.post(
            f"{BASE}/{portfolio.name}/rebalance-policy", json=self._CALENDAR
        )

        assert resp.status_code == 409

    def test_response_has_portfolio_id(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)

        resp = client.post(
            f"{BASE}/{portfolio.name}/rebalance-policy", json=self._CALENDAR
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["portfolioId"] == str(portfolio.id)

    def test_new_policy_is_inactive_by_default(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)

        resp = client.post(
            f"{BASE}/{portfolio.name}/rebalance-policy", json=self._CALENDAR
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["isActive"] is False


# ---------------------------------------------------------------------------
# POST /{name}/activate-policy/{policy_id}
# ---------------------------------------------------------------------------


class TestActivatePolicy:
    def test_activates_policy_and_returns_204(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        policy = _make_policy(db_session, portfolio.id, name="target", is_active=False)

        resp = client.post(f"{BASE}/{portfolio.name}/activate-policy/{policy.id}")

        assert resp.status_code == 204

    def test_activate_deactivates_other_policies(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        first = _make_policy(db_session, portfolio.id, name="first", is_active=True)
        second = _make_policy(db_session, portfolio.id, name="second", is_active=False)

        resp = client.post(f"{BASE}/{portfolio.name}/activate-policy/{second.id}")

        assert resp.status_code == 204
        db_session.expire_all()
        assert db_session.get(RebalancingPolicy, first.id).is_active is False
        assert db_session.get(RebalancingPolicy, second.id).is_active is True

    def test_returns_404_for_unknown_portfolio(self, client: TestClient) -> None:
        resp = client.post(f"{BASE}/ghost-portfolio/activate-policy/{uuid.uuid4()}")

        assert resp.status_code == 404

    def test_returns_404_for_unknown_policy_id(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)

        resp = client.post(f"{BASE}/{portfolio.name}/activate-policy/{uuid.uuid4()}")

        assert resp.status_code == 404

    def test_returns_404_when_policy_belongs_to_different_portfolio(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio_a = _make_portfolio(db_session)
        portfolio_b = _make_portfolio(db_session)
        policy_b = _make_policy(db_session, portfolio_b.id, name="policy-b")

        # Try to activate portfolio_b's policy via portfolio_a's route
        resp = client.post(f"{BASE}/{portfolio_a.name}/activate-policy/{policy_b.id}")

        assert resp.status_code == 404
