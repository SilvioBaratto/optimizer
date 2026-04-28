"""Unit tests for POST /api/v1/rebalance/decide and GET /api/v1/rebalance/preview.

TDD Red → Green → Refactor.

Acceptance criteria covered:
- decide: calendar / threshold / hybrid policies
- decide: returns should_rebalance, turnover, estimated_cost, trade_weights
- decide: hybrid rejects missing last_review_date with 422
- decide: unknown policy_type rejects with 422
- preview: 404 when portfolio not found
- preview: 404 when no active policy
- preview: 404 when no snapshots
- preview: empty trade list when no broker positions
- preview: returns trade list with weight_delta and side

All library calls are mocked to keep tests fast and isolated.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.portfolio import Portfolio, PortfolioSnapshot, SnapshotWeight
from app.models.rebalancing import RebalancingPolicy

BASE_DECIDE = "/api/v1/rebalance/decide"
BASE_PREVIEW = "/api/v1/rebalance/preview"


# ---------------------------------------------------------------------------
# Fixtures
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
    snapshot_date: date | None = None,
) -> PortfolioSnapshot:
    snap = PortfolioSnapshot(
        portfolio_id=portfolio_id,
        snapshot_date=snapshot_date or date(2024, 1, 1),
        snapshot_type="optimization",
        holding_count=len(weights or {}),
    )
    db_session.add(snap)
    db_session.flush()
    for ticker, weight in (weights or {"AAPL": 0.6, "MSFT": 0.4}).items():
        db_session.add(SnapshotWeight(
            snapshot_id=snap.id, ticker=ticker, weight=weight,
        ))
    db_session.flush()
    return snap


# ---------------------------------------------------------------------------
# POST /api/v1/rebalance/decide — threshold policy
# ---------------------------------------------------------------------------


class TestDecideThreshold:
    """Threshold policy decide endpoint."""

    _REQUEST: dict[str, Any] = {
        "current_weights": {"AAPL": 0.65, "MSFT": 0.35},
        "target_weights": {"AAPL": 0.60, "MSFT": 0.40},
        "policy_type": "threshold",
        "policy_config": {"threshold_type": "absolute", "threshold": 0.05},
    }

    def test_returns_200_with_expected_fields(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance",
            return_value=False,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST)

        assert resp.status_code == 200
        body = resp.json()
        assert "shouldRebalance" in body or "should_rebalance" in body

    def test_should_rebalance_true_when_drift_exceeds_threshold(
        self, client: TestClient
    ) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance",
            return_value=True,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST)

        assert resp.status_code == 200
        body = resp.json()
        # Accept both camelCase and snake_case from serialization
        key = "shouldRebalance" if "shouldRebalance" in body else "should_rebalance"
        assert body[key] is True

    def test_response_contains_turnover(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance",
            return_value=False,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST)

        assert resp.status_code == 200
        body = resp.json()
        assert "turnover" in body

    def test_response_contains_estimated_cost(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance",
            return_value=False,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST)

        assert resp.status_code == 200
        body = resp.json()
        key = "estimatedCost" if "estimatedCost" in body else "estimated_cost"
        assert key in body

    def test_response_contains_trade_weights(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance",
            return_value=False,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST)

        assert resp.status_code == 200
        body = resp.json()
        key = "tradeWeights" if "tradeWeights" in body else "trade_weights"
        assert key in body

    def test_trade_weights_computed_correctly(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance",
            return_value=False,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST)

        assert resp.status_code == 200
        body = resp.json()
        trade_w = body.get("tradeWeights") or body.get("trade_weights")
        # AAPL: 0.60 - 0.65 = -0.05, MSFT: 0.40 - 0.35 = +0.05
        assert abs(trade_w["AAPL"] - (-0.05)) < 1e-6
        assert abs(trade_w["MSFT"] - 0.05) < 1e-6


# ---------------------------------------------------------------------------
# POST /api/v1/rebalance/decide — calendar policy
# ---------------------------------------------------------------------------


class TestDecideCalendar:
    """Calendar policy decide endpoint."""

    _REQUEST: dict[str, Any] = {
        "current_weights": {"AAPL": 0.6, "MSFT": 0.4},
        "target_weights": {"AAPL": 0.6, "MSFT": 0.4},
        "policy_type": "calendar",
        "policy_config": {"frequency": "quarterly"},
        "current_date": "2024-04-15",
        "last_review_date": "2024-01-01",
    }

    def test_returns_200(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service._calendar_should_rebalance",
            return_value=True,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST)

        assert resp.status_code == 200

    def test_calendar_rebalance_after_enough_days(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service._calendar_should_rebalance",
            return_value=True,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST)

        assert resp.status_code == 200
        body = resp.json()
        key = "shouldRebalance" if "shouldRebalance" in body else "should_rebalance"
        assert body[key] is True


# ---------------------------------------------------------------------------
# POST /api/v1/rebalance/decide — hybrid policy
# ---------------------------------------------------------------------------


class TestDecideHybrid:
    """Hybrid policy decide endpoint."""

    _REQUEST_OK: dict[str, Any] = {
        "current_weights": {"AAPL": 0.6, "MSFT": 0.4},
        "target_weights": {"AAPL": 0.6, "MSFT": 0.4},
        "policy_type": "hybrid",
        "policy_config": {
            "calendar": {"frequency": "quarterly"},
            "threshold": {"threshold_type": "absolute", "threshold": 0.10},
        },
        "current_date": "2024-04-15",
        "last_review_date": "2024-01-01",
    }

    _REQUEST_NO_LAST_REVIEW: dict[str, Any] = {
        "current_weights": {"AAPL": 0.6, "MSFT": 0.4},
        "target_weights": {"AAPL": 0.6, "MSFT": 0.4},
        "policy_type": "hybrid",
        "policy_config": {
            "calendar": {"frequency": "quarterly"},
            "threshold": {"threshold_type": "absolute", "threshold": 0.10},
        },
    }

    def test_hybrid_returns_200_when_last_review_date_provided(
        self, client: TestClient
    ) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance_hybrid",
            return_value=False,
        ):
            resp = client.post(BASE_DECIDE, json=self._REQUEST_OK)

        assert resp.status_code == 200

    def test_hybrid_rejects_missing_last_review_date_with_422(
        self, client: TestClient
    ) -> None:
        resp = client.post(BASE_DECIDE, json=self._REQUEST_NO_LAST_REVIEW)

        assert resp.status_code == 422

    def test_hybrid_uses_should_rebalance_hybrid(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance_hybrid",
            return_value=True,
        ) as mock_fn:
            resp = client.post(BASE_DECIDE, json=self._REQUEST_OK)

        assert resp.status_code == 200
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/rebalance/decide — invalid policy_type
# ---------------------------------------------------------------------------


class TestDecideUnknownPolicy:
    def test_unknown_policy_type_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            BASE_DECIDE,
            json={
                "current_weights": {"AAPL": 0.6},
                "target_weights": {"AAPL": 1.0},
                "policy_type": "nonexistent",
                "policy_config": {},
            },
        )

        assert resp.status_code == 422

    def test_custom_transaction_costs_accepted(self, client: TestClient) -> None:
        with patch(
            "app.services.rebalancing_service.should_rebalance",
            return_value=False,
        ):
            resp = client.post(
                BASE_DECIDE,
                json={
                    "current_weights": {"AAPL": 0.6, "MSFT": 0.4},
                    "target_weights": {"AAPL": 0.6, "MSFT": 0.4},
                    "policy_type": "threshold",
                    "policy_config": {"threshold_type": "absolute", "threshold": 0.05},
                    "transaction_costs": 0.002,
                },
            )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/v1/rebalance/preview/{portfolio_name} — 404 cases
# ---------------------------------------------------------------------------


class TestPreviewSemantics:
    """Regression for #425: preview returns 200 with empty state when the
    portfolio exists but the optional resources (policy, snapshots) don't.

    404 is preserved strictly for the 'portfolio itself does not exist' case.
    """

    def test_portfolio_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.get(f"{BASE_PREVIEW}/nonexistent-portfolio")

        assert resp.status_code == 404

    def test_no_active_policy_returns_200_with_no_active_policy_status(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=False)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "no_active_policy"
        assert body["trades"] == []

    def test_no_snapshots_returns_200_with_no_snapshots_status(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        # No snapshot created

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "no_snapshots"
        assert body["trades"] == []

    def test_no_snapshots_response_still_carries_policy_type(
        self, client: TestClient, db_session: Session
    ) -> None:
        """When a policy exists but no snapshot, policy_type should still be populated."""
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, policy_type="threshold", is_active=True)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        body = resp.json()
        key = "policyType" if "policyType" in body else "policy_type"
        assert body[key] == "threshold"

    def test_happy_path_status_is_null(
        self, client: TestClient, db_session: Session
    ) -> None:
        """When policy + snapshot both exist, status must be null (no empty state)."""
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        assert resp.json()["status"] is None


# ---------------------------------------------------------------------------
# GET /api/v1/rebalance/preview/{portfolio_name} — happy path
# ---------------------------------------------------------------------------


class TestPreviewHappy:
    def test_empty_trade_list_when_no_broker_positions(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        body = resp.json()
        key = "trades" if "trades" in body else "tradeList"
        assert body[key] == []

    def test_response_has_portfolio_name(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(db_session, portfolio.id)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        body = resp.json()
        key = "portfolioName" if "portfolioName" in body else "portfolio_name"
        assert body[key] == portfolio.name

    def test_response_has_policy_type(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(
            db_session,
            portfolio.id,
            policy_type="threshold",
            is_active=True,
        )
        _make_snapshot(db_session, portfolio.id)

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        body = resp.json()
        key = "policyType" if "policyType" in body else "policy_type"
        assert body[key] == "threshold"

    def test_response_has_target_weights(
        self, client: TestClient, db_session: Session
    ) -> None:
        portfolio = _make_portfolio(db_session)
        _make_policy(db_session, portfolio.id, is_active=True)
        _make_snapshot(
            db_session, portfolio.id, weights={"AAPL": 0.6, "MSFT": 0.4}
        )

        resp = client.get(f"{BASE_PREVIEW}/{portfolio.name}")

        assert resp.status_code == 200
        body = resp.json()
        key = "targetWeights" if "targetWeights" in body else "target_weights"
        assert abs(body[key]["AAPL"] - 0.6) < 1e-6
