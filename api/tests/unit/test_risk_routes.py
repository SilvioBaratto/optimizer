"""Unit tests for risk limits CRUD endpoints (issue #370).

Covers:
  - GET  /api/v1/risk/{portfolio_name}/limits
  - POST /api/v1/risk/{portfolio_name}/limits
  - PUT  /api/v1/risk/{portfolio_name}/limits/{limit_id}
  - DELETE /api/v1/risk/{portfolio_name}/limits/{limit_id}

All external collaborators (PortfolioRepository, RiskRepository) are mocked.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

_PORTFOLIO_REPO = "app.api.v1.risk.PortfolioRepository"
_RISK_REPO = "app.api.v1.risk.RiskRepository"

BASE = "/api/v1/risk"
PORTFOLIO_NAME = "test-portfolio"
LIMITS_URL = f"{BASE}/{PORTFOLIO_NAME}/limits"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(name: str = PORTFOLIO_NAME) -> MagicMock:
    p = MagicMock()
    p.id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    p.name = name
    return p


def _make_limit(
    *,
    metric: str = "volatility",
    limit_type: str = "upper",
    threshold: float = 0.20,
    current_value: float | None = 0.15,
    is_breached: bool = False,
) -> SimpleNamespace:
    """Create a plain-attribute object that Pydantic from_attributes can read."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        portfolio_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        metric=metric,
        limit_type=limit_type,
        threshold=threshold,
        current_value=current_value,
        is_breached=is_breached,
        last_checked_at=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


# ===========================================================================
# GET /risk/{portfolio_name}/limits
# ===========================================================================


class TestListLimits:
    """Tests for GET /risk/{portfolio_name}/limits."""

    def test_returns_200_with_limits(self, client: TestClient) -> None:
        limit = _make_limit()
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.get_limits_by_portfolio.return_value = [limit]

            resp = client.get(LIMITS_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1
        assert data["breachCount"] == 0

    def test_returns_empty_list_for_portfolio_with_no_limits(
        self, client: TestClient
    ) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.get_limits_by_portfolio.return_value = []

            resp = client.get(LIMITS_URL)

        assert resp.status_code == 200
        assert resp.json()["items"] == []
        assert resp.json()["breachCount"] == 0

    def test_returns_404_for_unknown_portfolio(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_name.return_value = None

            resp = client.get(f"{BASE}/unknown-portfolio/limits")

        assert resp.status_code == 404

    def test_breach_count_reflects_breached_limits(self, client: TestClient) -> None:
        limits = [
            _make_limit(metric="vol", is_breached=True),
            _make_limit(metric="drawdown", is_breached=False),
        ]
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.get_limits_by_portfolio.return_value = limits

            resp = client.get(LIMITS_URL)

        assert resp.status_code == 200
        assert resp.json()["breachCount"] == 1

    def test_refresh_true_re_evaluates_breach_flags(
        self, client: TestClient
    ) -> None:
        """upper limit: current_value > threshold → breached."""
        limit = _make_limit(
            limit_type="upper", threshold=0.20, current_value=0.25, is_breached=False
        )
        updated_limit = _make_limit(
            metric=limit.metric,
            limit_type="upper",
            threshold=0.20,
            current_value=0.25,
            is_breached=True,
        )
        updated_limit.id = limit.id  # type: ignore[attr-defined]
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.get_limits_by_portfolio.return_value = [limit]
            MockRiskRepo.return_value.update_limit_value.return_value = updated_limit

            resp = client.get(LIMITS_URL, params={"refresh": "true"})

        assert resp.status_code == 200
        MockRiskRepo.return_value.update_limit_value.assert_called_once_with(
            limit.id, 0.25, True
        )
        assert resp.json()["breachCount"] == 1

    def test_refresh_true_lower_limit_breach(self, client: TestClient) -> None:
        """lower limit: current_value < threshold → breached."""
        limit = _make_limit(
            limit_type="lower", threshold=0.05, current_value=0.03, is_breached=False
        )
        updated_limit = _make_limit(
            metric=limit.metric,
            limit_type="lower",
            threshold=0.05,
            current_value=0.03,
            is_breached=True,
        )
        updated_limit.id = limit.id  # type: ignore[attr-defined]
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.get_limits_by_portfolio.return_value = [limit]
            MockRiskRepo.return_value.update_limit_value.return_value = updated_limit

            resp = client.get(LIMITS_URL, params={"refresh": "true"})

        assert resp.status_code == 200
        MockRiskRepo.return_value.update_limit_value.assert_called_once_with(
            limit.id, 0.03, True
        )

    def test_refresh_true_skips_limits_without_current_value(
        self, client: TestClient
    ) -> None:
        limit = _make_limit(current_value=None, is_breached=False)
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.get_limits_by_portfolio.return_value = [limit]

            resp = client.get(LIMITS_URL, params={"refresh": "true"})

        assert resp.status_code == 200
        MockRiskRepo.return_value.update_limit_value.assert_not_called()


# ===========================================================================
# POST /risk/{portfolio_name}/limits
# ===========================================================================


class TestCreateLimit:
    """Tests for POST /risk/{portfolio_name}/limits."""

    def test_creates_limit_returns_201(self, client: TestClient) -> None:
        created = _make_limit(metric="max_drawdown", limit_type="upper", threshold=0.15)
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.create_limit.return_value = created

            resp = client.post(
                LIMITS_URL,
                json={"metric": "max_drawdown", "limit_type": "upper", "threshold": 0.15},
            )

        assert resp.status_code == 201
        assert resp.json()["metric"] == "max_drawdown"

    def test_returns_404_for_unknown_portfolio(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_name.return_value = None

            resp = client.post(
                f"{BASE}/ghost/limits",
                json={"metric": "vol", "limit_type": "upper", "threshold": 0.2},
            )

        assert resp.status_code == 404

    def test_returns_409_for_duplicate_limit(self, client: TestClient) -> None:
        from sqlalchemy.exc import IntegrityError

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.create_limit.side_effect = IntegrityError(
                "UNIQUE constraint", None, None
            )

            resp = client.post(
                LIMITS_URL,
                json={"metric": "volatility", "limit_type": "upper", "threshold": 0.2},
            )

        assert resp.status_code == 409

    def test_returns_422_for_invalid_limit_type(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()

            resp = client.post(
                LIMITS_URL,
                json={"metric": "vol", "limit_type": "invalid", "threshold": 0.2},
            )

        assert resp.status_code == 422

    def test_returns_422_for_zero_threshold(self, client: TestClient) -> None:
        resp = client.post(
            LIMITS_URL,
            json={"metric": "vol", "limit_type": "upper", "threshold": 0.0},
        )
        assert resp.status_code == 422

    def test_returns_422_for_empty_metric(self, client: TestClient) -> None:
        resp = client.post(
            LIMITS_URL,
            json={"metric": "", "limit_type": "upper", "threshold": 0.2},
        )
        assert resp.status_code == 422

    def test_accepts_lower_as_valid_limit_type(self, client: TestClient) -> None:
        created = _make_limit(metric="sharpe", limit_type="lower", threshold=0.5)
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.create_limit.return_value = created

            resp = client.post(
                LIMITS_URL,
                json={"metric": "sharpe", "limit_type": "lower", "threshold": 0.5},
            )

        assert resp.status_code == 201


# ===========================================================================
# PUT /risk/{portfolio_name}/limits/{limit_id}
# ===========================================================================


class TestUpdateLimit:
    """Tests for PUT /risk/{portfolio_name}/limits/{limit_id}."""

    def test_updates_limit_returns_200(self, client: TestClient) -> None:
        limit_id = uuid.uuid4()
        updated = _make_limit(current_value=0.30, is_breached=True)
        updated.id = limit_id  # type: ignore[attr-defined]
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.update_limit_value.return_value = updated

            resp = client.put(
                f"{LIMITS_URL}/{limit_id}",
                json={"current_value": 0.30, "is_breached": True},
            )

        assert resp.status_code == 200
        assert resp.json()["isBreached"] is True

    def test_returns_404_for_unknown_limit_id(self, client: TestClient) -> None:
        limit_id = uuid.uuid4()
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.update_limit_value.side_effect = ValueError(
                "not found"
            )

            resp = client.put(
                f"{LIMITS_URL}/{limit_id}",
                json={"current_value": 0.1, "is_breached": False},
            )

        assert resp.status_code == 404

    def test_returns_404_for_unknown_portfolio(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_name.return_value = None

            resp = client.put(
                f"{BASE}/ghost/limits/{uuid.uuid4()}",
                json={"current_value": 0.1, "is_breached": False},
            )

        assert resp.status_code == 404


# ===========================================================================
# DELETE /risk/{portfolio_name}/limits/{limit_id}
# ===========================================================================


class TestDeleteLimit:
    """Tests for DELETE /risk/{portfolio_name}/limits/{limit_id}."""

    def test_deletes_limit_returns_204(self, client: TestClient) -> None:
        limit_id = uuid.uuid4()
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.delete_limit.return_value = True

            resp = client.delete(f"{LIMITS_URL}/{limit_id}")

        assert resp.status_code == 204

    def test_returns_404_for_unknown_limit_id(self, client: TestClient) -> None:
        limit_id = uuid.uuid4()
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_RISK_REPO) as MockRiskRepo,
        ):
            MockPortRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRiskRepo.return_value.delete_limit.return_value = False

            resp = client.delete(f"{LIMITS_URL}/{limit_id}")

        assert resp.status_code == 404

    def test_returns_404_for_unknown_portfolio(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_name.return_value = None

            resp = client.delete(f"{BASE}/ghost/limits/{uuid.uuid4()}")

        assert resp.status_code == 404
