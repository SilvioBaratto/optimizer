"""Unit tests for the portfolio activity feed endpoint.

Mocks PortfolioRepository at the router's import path so no real database
is touched.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

_PORTFOLIO_REPO = "app.api.v1.dashboard.PortfolioRepository"

BASE_URL = "/api/v1/portfolio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(name: str = "test") -> MagicMock:
    p = MagicMock()
    p.id = uuid.uuid4()
    p.name = name
    return p


def _make_event(
    event_type: str = "rebalance",
    title: str = "Quarterly rebalance executed",
    description: str | None = "12 trades, 4.2% turnover",
) -> MagicMock:
    e = MagicMock()
    e.id = uuid.uuid4()
    e.event_type = event_type
    e.title = title
    e.description = description
    e.created_at = datetime(2026, 2, 24, 16, 0, 0, tzinfo=timezone.utc)
    return e


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetActivity:
    def test_success_returns_items(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        events = [
            _make_event("rebalance", "Quarterly rebalance"),
            _make_event("trade", "Bought AAPL"),
        ]

        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_events.return_value = events
            repo.count_events.return_value = 45

            resp = client.get(f"{BASE_URL}/myport/activity")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["total"] == 45

    def test_empty_feed_returns_200(self, client: TestClient):
        portfolio = _make_portfolio("myport")

        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_events.return_value = []
            repo.count_events.return_value = 0

            resp = client.get(f"{BASE_URL}/myport/activity")

        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["total"] == 0

    def test_portfolio_not_found(self, client: TestClient):
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None

            resp = client.get(f"{BASE_URL}/missing/activity")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "not found" in msg.lower()

    def test_pagination_params_passed(self, client: TestClient):
        portfolio = _make_portfolio()

        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_events.return_value = []
            repo.count_events.return_value = 0

            client.get(f"{BASE_URL}/test/activity?limit=5&offset=10")

            repo.get_events.assert_called_once_with(
                portfolio_id=portfolio.id,
                event_type=None,
                limit=5,
                offset=10,
            )

    def test_type_filter_passed(self, client: TestClient):
        portfolio = _make_portfolio()

        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_events.return_value = []
            repo.count_events.return_value = 0

            client.get(f"{BASE_URL}/test/activity?type=rebalance")

            repo.get_events.assert_called_once_with(
                portfolio_id=portfolio.id,
                event_type="rebalance",
                limit=20,
                offset=0,
            )
            repo.count_events.assert_called_once_with(
                portfolio_id=portfolio.id,
                event_type="rebalance",
            )

    def test_activity_item_fields(self, client: TestClient):
        portfolio = _make_portfolio()
        events = [_make_event()]

        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_events.return_value = events
            repo.count_events.return_value = 1

            resp = client.get(f"{BASE_URL}/test/activity")

        item = resp.json()["items"][0]
        assert "id" in item
        assert item["type"] == "rebalance"
        assert item["title"] == "Quarterly rebalance executed"
        assert item["description"] == "12 trades, 4.2% turnover"
        assert "timestamp" in item

    def test_camel_case_serialization(self, client: TestClient):
        """All response keys should be camelCase (items/total are single-word)."""
        portfolio = _make_portfolio()
        events = [_make_event()]

        with patch(_PORTFOLIO_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_events.return_value = events
            repo.count_events.return_value = 1

            resp = client.get(f"{BASE_URL}/test/activity")

        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "total" in body
        # snake_case should not appear
        assert "event_type" not in str(body)

    def test_limit_validation_too_large(self, client: TestClient):
        with patch(_PORTFOLIO_REPO):
            resp = client.get(f"{BASE_URL}/test/activity?limit=201")

        assert resp.status_code == 422

    def test_limit_validation_zero(self, client: TestClient):
        with patch(_PORTFOLIO_REPO):
            resp = client.get(f"{BASE_URL}/test/activity?limit=0")

        assert resp.status_code == 422
