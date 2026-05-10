"""Tests for POST /api/v1/jobs/{job_id}/cancel (issues #560, #561)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.repositories.background_job_repository import BackgroundJobRepository

BASE_URL = "/api/v1/jobs"


def _seed(
    db_session: Session, job_type: str, status: str | None = None,
) -> UUID:
    repo = BackgroundJobRepository(db_session)
    jid = repo.claim_or_create(job_type)
    assert jid is not None
    if status is not None and status != "pending":
        kwargs: dict[str, object] = {"status": status}
        if status in ("completed", "failed", "cancelled"):
            kwargs["finished_at"] = datetime.now(timezone.utc)
        repo.update(jid, **kwargs)
    db_session.flush()
    db_session.commit()
    return jid


class TestCancelJob:
    """End-to-end coverage of POST /api/v1/jobs/{job_id}/cancel."""

    def test_invalid_uuid_returns_422(self, client: TestClient) -> None:
        resp = client.post(f"{BASE_URL}/not-a-uuid/cancel")
        assert resp.status_code == 422

    def test_unknown_job_returns_404(self, client: TestClient) -> None:
        resp = client.post(f"{BASE_URL}/{uuid4()}/cancel")
        assert resp.status_code == 404

    def test_completed_job_returns_409(
        self, client: TestClient, db_session: Session,
    ) -> None:
        jid = _seed(db_session, "test_561_completed", status="completed")
        resp = client.post(f"{BASE_URL}/{jid}/cancel")
        assert resp.status_code == 409
        assert "already terminal" in resp.json()["error"]["message"]

    def test_failed_job_returns_409(
        self, client: TestClient, db_session: Session,
    ) -> None:
        jid = _seed(db_session, "test_561_failed", status="failed")
        resp = client.post(f"{BASE_URL}/{jid}/cancel")
        assert resp.status_code == 409
        assert "already terminal" in resp.json()["error"]["message"]

    def test_cancelled_job_returns_409(
        self, client: TestClient, db_session: Session,
    ) -> None:
        jid = _seed(db_session, "test_561_cancelled", status="cancelled")
        resp = client.post(f"{BASE_URL}/{jid}/cancel")
        assert resp.status_code == 409
        assert "already terminal" in resp.json()["error"]["message"]

    def test_pending_job_cancels_successfully(
        self, client: TestClient, db_session: Session,
    ) -> None:
        jid = _seed(db_session, "test_561_pending")
        resp = client.post(f"{BASE_URL}/{jid}/cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["error"] is not None
        assert "operator-cancelled" in body["error"]

    def test_running_job_cancels_successfully(
        self, client: TestClient, db_session: Session,
    ) -> None:
        jid = _seed(db_session, "test_561_running", status="running")
        resp = client.post(f"{BASE_URL}/{jid}/cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["error"] is not None
        assert "operator-cancelled" in body["error"]

    def test_cancel_is_idempotent_after_first_cancel(
        self, client: TestClient, db_session: Session,
    ) -> None:
        jid = _seed(db_session, "test_561_idempotent")
        first = client.post(f"{BASE_URL}/{jid}/cancel")
        assert first.status_code == 200
        second = client.post(f"{BASE_URL}/{jid}/cancel")
        assert second.status_code == 409
        assert "already terminal" in second.json()["error"]["message"]
