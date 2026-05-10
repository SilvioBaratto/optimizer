"""Tests for GET /api/v1/jobs/{job_id}/stream SSE endpoint (issue #374).

Covers:
  - SSE stream receives ``event: progress`` events with correct JSON payload
  - Terminal status sends ``event: done`` and closes stream
  - 404 for unknown job_id
  - Already-complete job sends single ``event: done`` immediately
  - Invalid ``interval`` values (0, -1, 100) return 422
  - All events use named SSE format (event: + data: lines)
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

BASE_URL = "/api/v1/jobs"
_MOCK_JOB_ID = "00000000-0000-0000-0000-000000000099"

# Patch target: the repository's get() method as called inside the stream generator
_REPO_GET = "app.api.v1.jobs.jobs.BackgroundJobRepository"


def _make_job_row(
    status: str = "running",
    current: int = 10,
    total: int = 100,
    job_type: str = "test_domain",
) -> MagicMock:
    """Build a minimal mock BackgroundJob ORM row."""
    row = MagicMock()
    row.id = uuid.UUID(_MOCK_JOB_ID)
    row.job_type = job_type
    row.status = status
    row.current = current
    row.total = total
    row.error = None
    row.errors = []
    row.started_at = None
    row.finished_at = None
    return row


def _parse_sse_events(text: str) -> list[dict[str, str]]:
    """Parse raw SSE text into list of dicts with 'event' and 'data' keys."""
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:") :].strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


class TestStreamInvalidInterval:
    """Invalid ?interval= values must return 422 before the generator runs."""

    def test_interval_zero_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}/stream?interval=0")
        assert resp.status_code == 422

    def test_interval_negative_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}/stream?interval=-1")
        assert resp.status_code == 422

    def test_interval_above_max_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}/stream?interval=100")
        assert resp.status_code == 422

    def test_interval_at_max_is_accepted(self, client: TestClient) -> None:
        """interval=10 is the boundary maximum — must not return 422."""
        completed_row = _make_job_row(status="completed", current=100)
        with (
            patch(_REPO_GET) as MockRepo,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            instance = MockRepo.return_value
            # pre-flight check + generator immediate done
            instance.get.return_value = completed_row
            resp = client.get(
                f"{BASE_URL}/{_MOCK_JOB_ID}/stream?interval=10",
                headers={"Accept": "text/event-stream"},
            )
        assert resp.status_code == 200

    def test_interval_at_min_is_accepted(self, client: TestClient) -> None:
        """interval=1 is the boundary minimum — must not return 422."""
        completed_row = _make_job_row(status="completed", current=100)
        with patch(_REPO_GET) as MockRepo:
            instance = MockRepo.return_value
            instance.get.return_value = completed_row
            resp = client.get(
                f"{BASE_URL}/{_MOCK_JOB_ID}/stream?interval=1",
                headers={"Accept": "text/event-stream"},
            )
        assert resp.status_code == 200


class TestStreamNotFound:
    """Unknown job_id returns 404 (not SSE stream)."""

    def test_unknown_job_returns_404(self, client: TestClient) -> None:
        unknown_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with patch(_REPO_GET) as MockRepo:
            instance = MockRepo.return_value
            instance.get.return_value = None
            resp = client.get(f"{BASE_URL}/{unknown_id}/stream")
        assert resp.status_code == 404


class TestStreamAlreadyComplete:
    """A job already in terminal status sends a single event: done and closes."""

    def test_completed_job_sends_single_done_event(self, client: TestClient) -> None:
        completed_row = _make_job_row(status="completed", current=50, total=50)
        with patch(_REPO_GET) as MockRepo:
            instance = MockRepo.return_value
            instance.get.return_value = completed_row
            resp = client.get(
                f"{BASE_URL}/{_MOCK_JOB_ID}/stream",
                headers={"Accept": "text/event-stream"},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse_events(resp.text)
        assert len(events) == 1
        assert events[0]["event"] == "done"
        payload = json.loads(events[0]["data"])
        assert payload["status"] == "completed"

    def test_failed_job_sends_single_done_event(self, client: TestClient) -> None:
        failed_row = _make_job_row(status="failed")
        with patch(_REPO_GET) as MockRepo:
            instance = MockRepo.return_value
            instance.get.return_value = failed_row
            resp = client.get(
                f"{BASE_URL}/{_MOCK_JOB_ID}/stream",
                headers={"Accept": "text/event-stream"},
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert len(events) == 1
        assert events[0]["event"] == "done"


class TestStreamProgressEvents:
    """Running jobs push event: progress then event: done on completion."""

    def test_running_then_complete_sends_progress_then_done(
        self, client: TestClient
    ) -> None:
        running_row = _make_job_row(status="running", current=10, total=100)
        completed_row = _make_job_row(status="completed", current=100, total=100)
        # side_effect order: [pre-flight check, generator poll 1, generator poll 2]
        with (
            patch(_REPO_GET) as MockRepo,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            instance = MockRepo.return_value
            instance.get.side_effect = [running_row, running_row, completed_row]
            resp = client.get(
                f"{BASE_URL}/{_MOCK_JOB_ID}/stream",
                headers={"Accept": "text/event-stream"},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse_events(resp.text)
        assert len(events) == 2

        progress_event = events[0]
        assert progress_event["event"] == "progress"
        payload = json.loads(progress_event["data"])
        assert payload["status"] == "running"
        assert payload["current"] == 10
        assert payload["total"] == 100
        assert payload["domain"] == "test_domain"

        done_event = events[1]
        assert done_event["event"] == "done"
        done_payload = json.loads(done_event["data"])
        assert done_payload["status"] == "completed"

    def test_progress_payload_includes_required_fields(
        self, client: TestClient
    ) -> None:
        running_row = _make_job_row(status="running", current=5, total=20)
        completed_row = _make_job_row(status="completed", current=20, total=20)
        # side_effect order: [pre-flight check, generator poll 1, generator poll 2]
        with (
            patch(_REPO_GET) as MockRepo,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            instance = MockRepo.return_value
            instance.get.side_effect = [running_row, running_row, completed_row]
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}/stream")

        events = _parse_sse_events(resp.text)
        progress_events = [e for e in events if e.get("event") == "progress"]
        assert len(progress_events) >= 1

        payload = json.loads(progress_events[0]["data"])
        for field in ("current", "total", "status", "domain"):
            assert field in payload, f"Missing field: {field}"

    def test_content_type_is_text_event_stream(self, client: TestClient) -> None:
        completed_row = _make_job_row(status="completed")
        with patch(_REPO_GET) as MockRepo:
            instance = MockRepo.return_value
            instance.get.return_value = completed_row
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}/stream")

        assert "text/event-stream" in resp.headers["content-type"]

    def test_each_poll_uses_fresh_db_session(self, client: TestClient) -> None:
        """The repository must be instantiated on each poll (fresh session per iteration).

        The pre-flight check + generator iterations each create a new repo instance,
        confirming that a fresh DB session is used on every DB access.
        """
        running_row = _make_job_row(status="running")
        completed_row = _make_job_row(status="completed")
        # side_effect: [pre-flight, generator poll 1 (running), generator poll 2 (done)]
        with (
            patch(_REPO_GET) as MockRepo,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            instance = MockRepo.return_value
            instance.get.side_effect = [running_row, running_row, completed_row]
            client.get(f"{BASE_URL}/{_MOCK_JOB_ID}/stream")

        # Pre-flight + at least 2 generator iterations = at least 3 instantiations
        assert MockRepo.call_count >= 3
