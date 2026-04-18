"""Unit tests for the /database router.

Mocks ``DatabaseAdminRepository`` at the router's import path so no real
PostgreSQL is touched (functions like ``pg_total_relation_size`` are
unavailable on the SQLite test backend).
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

_REPO = "app.api.v1.database.DatabaseAdminRepository"

BASE_URL = "/api/v1/database"


# ---------------------------------------------------------------------------
# GET /database/tables — contract test (issue #435)
# ---------------------------------------------------------------------------


_FAKE_TABLES = [
    {
        "name": "price_history",
        "schema": "public",
        "exists": True,
        "row_count": 12345,
        "size_bytes": 1_048_576,
        "size_pretty": "1024 kB",
    },
    {
        "name": "ticker_news",
        "schema": "public",
        "exists": True,
        "row_count": 42,
        "size_bytes": 8192,
        "size_pretty": "8192 bytes",
    },
]


class TestGetDatabaseTables:
    def test_returns_table_info_with_full_schema(self, client: TestClient):
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_table_info.return_value = _FAKE_TABLES

            resp = client.get(f"{BASE_URL}/tables")

        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 2

    def test_response_contains_name_schema_row_count_and_size_keys(
        self, client: TestClient,
    ):
        """Issue #435 contract: every row must carry the four columns the
        Settings → Data Management table renders."""
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_table_info.return_value = _FAKE_TABLES

            resp = client.get(f"{BASE_URL}/tables")

        assert resp.status_code == 200
        for row in resp.json():
            assert "name" in row
            assert "schema" in row
            assert "row_count" in row
            assert "size_bytes" in row
            assert "size_pretty" in row

    def test_response_does_not_use_legacy_table_name_key(
        self, client: TestClient,
    ):
        """Regression: the legacy ``table_name`` key was renamed to ``name``
        to match the frontend ``TableInfo`` model (issue #435)."""
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_table_info.return_value = _FAKE_TABLES

            resp = client.get(f"{BASE_URL}/tables")

        assert resp.status_code == 200
        for row in resp.json():
            assert "table_name" not in row


# ---------------------------------------------------------------------------
# GET /database/health — contract test (issue #436)
# ---------------------------------------------------------------------------


class TestGetDatabaseHealth:
    def test_returns_healthy_true_when_repo_check_succeeds(
        self, client: TestClient,
    ):
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.check_health.return_value = (True, 1.23)

            resp = client.get(f"{BASE_URL}/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["healthy"] is True
        assert body["latency_ms"] == 1.23

    def test_returns_healthy_false_when_repo_check_fails(
        self, client: TestClient,
    ):
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.check_health.return_value = (False, 9999.0)

            resp = client.get(f"{BASE_URL}/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["healthy"] is False

    def test_response_keys_match_frontend_HealthCheck_contract(
        self, client: TestClient,
    ):
        """Issue #436: freezes the {healthy, latency_ms, database_url} shape
        that the frontend ``HealthCheck`` interface depends on."""
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.check_health.return_value = (True, 0.5)

            resp = client.get(f"{BASE_URL}/health")

        assert resp.status_code == 200
        body = resp.json()
        assert sorted(body.keys()) == ["database_url", "healthy", "latency_ms"]
        assert isinstance(body["healthy"], bool)
        assert isinstance(body["latency_ms"], (int, float))
        assert isinstance(body["database_url"], str)

    def test_database_url_is_password_masked(self, client: TestClient):
        """The masked URL must never expose a password (issue #436 guard)."""
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.check_health.return_value = (True, 0.5)

            resp = client.get(f"{BASE_URL}/health")

        assert resp.status_code == 200
        url = resp.json()["database_url"]
        # masked URL contains *** in place of the password
        assert "***" in url


# ---------------------------------------------------------------------------
# DatabaseAdminRepository.get_table_info — drives the repo extension
# ---------------------------------------------------------------------------


class TestGetTableInfoRepo:
    """White-box repo tests with a stubbed Session.

    SQLite cannot run ``pg_total_relation_size`` / ``pg_size_pretty``, so we
    mock ``session.execute`` to return fixed scalars in the order the repo
    invokes them. The test asserts the dict shape — not the SQL itself.
    """

    def _make_session(self, scalars: list):
        """Return a MagicMock ``Session`` whose ``execute`` yields *scalars*."""
        from unittest.mock import MagicMock

        session = MagicMock()
        results = []
        for value in scalars:
            res = MagicMock()
            res.scalar.return_value = value
            results.append(res)
        session.execute.side_effect = results
        return session

    def test_returns_name_schema_size_bytes_size_pretty(self):
        from app.repositories.database_admin_repository import (
            DatabaseAdminRepository,
        )

        # Per-table call sequence: exists?, count(*), pg_total_relation_size,
        # pg_size_pretty(...). Single table, so 4 scalar reads in a row.
        session = self._make_session(
            scalars=[True, 100, 16384, "16 kB"],
        )
        repo = DatabaseAdminRepository(session)

        rows = repo.get_table_info(["price_history"])

        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "price_history"
        assert row["schema"] == "public"
        assert row["row_count"] == 100
        assert row["size_bytes"] == 16384
        assert row["size_pretty"] == "16 kB"

    def test_missing_table_returns_zero_size_and_dash(self):
        from app.repositories.database_admin_repository import (
            DatabaseAdminRepository,
        )

        # exists? returns False → repo skips count + size lookups
        session = self._make_session(scalars=[False])
        repo = DatabaseAdminRepository(session)

        rows = repo.get_table_info(["ghost_table"])

        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "ghost_table"
        assert row["exists"] is False
        assert row["row_count"] is None
        assert row["size_bytes"] is None
        assert row["size_pretty"] == "—"
