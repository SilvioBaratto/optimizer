"""Unit tests for macro news summary API endpoints (issue #214)."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.models.macro_regime import MacroNewsSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = "/api/v1/macro-data"
_SUMMARIZE_POST = f"{_BASE}/news/summarize"
_SUMMARIES_LIST = f"{_BASE}/news/summaries"


def _make_summary_row(
    country: str = "USA",
    summary_date: date | None = None,
    summary: str = "Markets rose on strong jobs data.",
    sentiment: str = "BULLISH",
    sentiment_score: float = 0.75,
    article_count: int = 10,
) -> MacroNewsSummary:
    """Build a MacroNewsSummary ORM instance for testing."""
    row = MacroNewsSummary()
    row.id = uuid.uuid4()
    row.country = country
    row.summary_date = summary_date or date(2026, 3, 15)
    row.summary = summary
    row.sentiment = sentiment
    row.sentiment_score = sentiment_score
    row.article_count = article_count
    row.news_summary = "raw article text"
    row.created_at = datetime.now(timezone.utc)
    row.updated_at = datetime.now(timezone.utc)
    return row


# ---------------------------------------------------------------------------
# POST /news/summarize — start background job
# ---------------------------------------------------------------------------


class TestStartNewsSummarize:
    def test_returns_202_with_job_id(self, client: TestClient) -> None:
        job_id = str(uuid.uuid4())
        with patch(
            "app.api.v1.macro_regime._summarize_job_service.create_job",
            return_value=job_id,
        ), patch(
            "app.api.v1.macro_regime._summarize_job_service.start_background",
        ):
            resp = client.post(_SUMMARIZE_POST)

        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_returns_409_when_already_running(self, client: TestClient) -> None:
        from app.services.background_job import JobAlreadyRunningError

        with patch(
            "app.api.v1.macro_regime._summarize_job_service.create_job",
            side_effect=JobAlreadyRunningError("existing-id"),
        ):
            resp = client.post(_SUMMARIZE_POST)

        assert resp.status_code == 409
        body = resp.json()
        message = body.get("detail", "") or body.get("error", {}).get("message", "")
        assert "already in progress" in message

    def test_accepts_force_refresh_param(self, client: TestClient) -> None:
        job_id = str(uuid.uuid4())
        with patch(
            "app.api.v1.macro_regime._summarize_job_service.create_job",
            return_value=job_id,
        ), patch(
            "app.api.v1.macro_regime._summarize_job_service.start_background",
        ):
            resp = client.post(_SUMMARIZE_POST, json={"force_refresh": True})

        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# GET /news/summarize/{job_id} — poll status
# ---------------------------------------------------------------------------


class TestGetNewsSummarizeStatus:
    def test_returns_404_for_unknown_job(self, client: TestClient) -> None:
        resp = client.get(f"{_SUMMARIZE_POST}/unknown-id")
        assert resp.status_code == 404

    def test_returns_progress_for_known_job(self, client: TestClient) -> None:
        job_id = str(uuid.uuid4())
        # Create a job and then poll it
        with patch(
            "app.api.v1.macro_regime._summarize_job_service.create_job",
            return_value=job_id,
        ), patch(
            "app.api.v1.macro_regime._summarize_job_service.start_background",
        ):
            post_resp = client.post(_SUMMARIZE_POST)
        assert post_resp.status_code == 202

        with patch(
            "app.api.v1.macro_regime._summarize_job_service.get_job",
            return_value={
                "job_id": job_id,
                "status": "running",
                "current": 1,
                "total": 5,
                "current_country": "USA",
                "errors": [],
                "result": None,
                "error": None,
            },
        ):
            resp = client.get(f"{_SUMMARIZE_POST}/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("pending", "running", "completed", "failed")
        assert "current_country" in data


# ---------------------------------------------------------------------------
# GET /news/summaries — list all
# ---------------------------------------------------------------------------


class TestGetAllNewsSummaries:
    def test_returns_empty_list_when_no_data(self, client: TestClient) -> None:
        resp = client.get(_SUMMARIES_LIST)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_persisted_summaries(
        self, client: TestClient, db_session
    ) -> None:
        row = _make_summary_row(country="USA")
        db_session.add(row)
        db_session.flush()

        resp = client.get(_SUMMARIES_LIST)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["country"] == "USA"
        assert data[0]["sentiment"] == "BULLISH"
        # news_summary (raw input) should NOT be in response
        assert "news_summary" not in data[0]

    def test_filters_by_summary_date(
        self, client: TestClient, db_session
    ) -> None:
        row1 = _make_summary_row(country="Japan", summary_date=date(2026, 2, 10))
        row2 = _make_summary_row(country="Canada", summary_date=date(2026, 2, 11))
        db_session.add_all([row1, row2])
        db_session.flush()

        resp = client.get(_SUMMARIES_LIST, params={"summary_date": "2026-02-10"})
        assert resp.status_code == 200
        data = resp.json()
        countries = [d["country"] for d in data]
        assert "Japan" in countries
        assert "Canada" not in countries


# ---------------------------------------------------------------------------
# GET /news/summaries/{country} — single country
# ---------------------------------------------------------------------------


class TestGetCountryNewsSummary:
    def test_returns_404_for_unknown_country(self, client: TestClient) -> None:
        resp = client.get(f"{_SUMMARIES_LIST}/Atlantis")
        assert resp.status_code == 404

    def test_returns_summary_for_known_country(
        self, client: TestClient, db_session
    ) -> None:
        row = _make_summary_row(country="Germany")
        db_session.add(row)
        db_session.flush()

        resp = client.get(f"{_SUMMARIES_LIST}/Germany")
        assert resp.status_code == 200
        data = resp.json()
        assert data["country"] == "Germany"
        assert data["summary"] == "Markets rose on strong jobs data."
        assert data["sentiment"] in ("BULLISH", "BEARISH", "NEUTRAL", "MIXED")
        assert "id" in data
        assert "created_at" in data

    def test_filters_by_summary_date(
        self, client: TestClient, db_session
    ) -> None:
        row = _make_summary_row(country="France", summary_date=date(2026, 3, 15))
        db_session.add(row)
        db_session.flush()

        resp = client.get(
            f"{_SUMMARIES_LIST}/France",
            params={"summary_date": "2026-03-15"},
        )
        assert resp.status_code == 200
        assert resp.json()["country"] == "France"

    def test_returns_404_when_date_mismatch(
        self, client: TestClient, db_session
    ) -> None:
        row = _make_summary_row(country="UK", summary_date=date(2026, 3, 15))
        db_session.add(row)
        db_session.flush()

        resp = client.get(
            f"{_SUMMARIES_LIST}/UK",
            params={"summary_date": "2026-01-01"},
        )
        assert resp.status_code == 404
