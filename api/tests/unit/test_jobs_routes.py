"""Integration tests for GET /api/v1/jobs and GET /api/v1/jobs/{job_id}.

Uses the real SQLite test DB via the ``client`` and ``db_session`` fixtures.
Seeds rows via BackgroundJobRepository directly — no service mocking.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.repositories.jobs.background_job_repository import BackgroundJobRepository

BASE_URL = "/api/v1/jobs"


# ---------------------------------------------------------------------------
# Seeding helper
# ---------------------------------------------------------------------------


def _seed(db_session: Session, job_type: str, *, status: str = "pending") -> uuid.UUID:
    """Create a job row and optionally advance its status."""
    repo = BackgroundJobRepository(db_session)
    jid = repo.claim_or_create(job_type)
    assert jid is not None
    if status != "pending":
        kwargs: dict[str, object] = {"status": status}
        if status in ("completed", "failed"):
            kwargs["finished_at"] = datetime.now(timezone.utc)
        repo.update(jid, **kwargs)
    db_session.flush()
    return jid


# ---------------------------------------------------------------------------
# List jobs — GET /api/v1/jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    def test_when_no_filter_then_200_and_list_returned(
        self, client: TestClient, db_session: Session
    ) -> None:
        _seed(db_session, "route_list_no_filter_a")
        _seed(db_session, "route_list_no_filter_b")

        resp = client.get(BASE_URL)

        assert resp.status_code == 200
        body = resp.json()
        assert "jobs" in body
        assert isinstance(body["jobs"], list)
        assert body["total"] >= 2

    def test_when_domain_filter_then_only_matching_jobs_returned(
        self, client: TestClient, db_session: Session
    ) -> None:
        target_type = "route_domain_filter_target"
        other_type = "route_domain_filter_other"
        _seed(db_session, target_type)
        _seed(db_session, other_type)

        resp = client.get(BASE_URL, params={"domain": target_type})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        for job in body["jobs"]:
            assert job["domain"] == target_type

    def test_when_status_filter_then_only_matching_status_returned(
        self, client: TestClient, db_session: Session
    ) -> None:
        _seed(db_session, "route_status_filter_running", status="running")
        _seed(db_session, "route_status_filter_pending")

        resp = client.get(BASE_URL, params={"status": "running"})

        assert resp.status_code == 200
        body = resp.json()
        for job in body["jobs"]:
            assert job["status"] == "running"

    def test_when_paginated_then_limit_and_offset_respected(
        self, client: TestClient, db_session: Session
    ) -> None:
        # Each job_type is unique because claim_or_create blocks duplicate active jobs.
        # Use a shared prefix and distinct suffixes so we can filter by prefix is not
        # available — instead seed 3 different types and count total >= 3.
        _seed(db_session, "route_pagination_type_a")
        _seed(db_session, "route_pagination_type_b")
        _seed(db_session, "route_pagination_type_c")

        # Fetch with limit=1; total must be >= 3 (may include rows from other tests
        # that share the same in-memory DB within the savepoint).
        resp = client.get(BASE_URL, params={"limit": 1, "offset": 0})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["jobs"]) == 1
        assert body["total"] >= 3
        assert body["limit"] == 1
        assert body["offset"] == 0


# ---------------------------------------------------------------------------
# Get single job — GET /api/v1/jobs/{job_id}
# ---------------------------------------------------------------------------


class TestGetJob:
    def test_when_job_exists_then_200_and_correct_id_returned(
        self, client: TestClient, db_session: Session
    ) -> None:
        jid = _seed(db_session, "route_get_single")

        resp = client.get(f"{BASE_URL}/{jid}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(jid)
        assert body["domain"] == "route_get_single"

    def test_when_job_not_found_then_404_returned(
        self, client: TestClient
    ) -> None:
        random_id = uuid.uuid4()

        resp = client.get(f"{BASE_URL}/{random_id}")

        assert resp.status_code == 404

    def test_when_id_is_not_uuid_then_422_returned(
        self, client: TestClient
    ) -> None:
        resp = client.get(f"{BASE_URL}/not-a-uuid")

        assert resp.status_code == 422
