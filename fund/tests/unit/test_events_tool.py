"""``get_recent_events``: the economist's window onto the global daily digest.

Drives the leaf tool against seeded ``market_journal`` rows on the in-memory
``db_session``, asserting the ``{ok, data}`` envelope, the window filter, the
ascending order, and that a bad ``asof`` degrades to ``{ok: false}`` rather than
raising across the tool boundary.
"""

from __future__ import annotations

import datetime as dt

from portopt_db.repositories.market_data.market_journal_repository import (
    MarketJournalRepository,
)

from fund.tools.events import get_recent_events

_ASOF = dt.date(2024, 1, 15)


def _seed_journal(
    db_session, as_of: dt.date, *, themes=None, narrative="digest"
) -> None:
    """Seed one global digest row for ``as_of`` with optional theme counts."""
    MarketJournalRepository(db_session).upsert_journal(
        as_of,
        macro_deltas={},
        market_moves={},
        news_themes={"themes": themes or {}},
        narrative=narrative,
        source_counts={},
    )


class TestGetRecentEvents:
    """The tool returns the in-window digests as a compact, bounded summary."""

    def test_returns_digests_within_window(self, db_session) -> None:
        """Only digests in ``[asof - window, asof]`` are returned, oldest first."""
        _seed_journal(db_session, dt.date(2024, 1, 10), narrative="d10")
        _seed_journal(
            db_session, dt.date(2024, 1, 14), narrative="d14", themes={"rates": 2}
        )
        _seed_journal(db_session, dt.date(2023, 12, 1), narrative="old")

        result = get_recent_events(db_session, _ASOF, window=14)

        assert result["ok"] is True
        data = result["data"]
        assert data["asof"] == "2024-01-15"
        assert data["window"] == 14
        assert data["count"] == 2
        assert [e["narrative"] for e in data["events"]] == ["d10", "d14"]
        assert data["events"][1]["themes"] == {"rates": 2}

    def test_empty_window_is_ok_with_no_events(self, db_session) -> None:
        """A window with no digests returns a well-formed empty summary."""
        result = get_recent_events(db_session, _ASOF, window=7)

        assert result["ok"] is True
        assert result["data"]["count"] == 0
        assert result["data"]["events"] == []

    def test_bad_asof_returns_error_envelope(self, db_session) -> None:
        """A malformed ``asof`` is caught by the envelope, never raised."""
        result = get_recent_events(db_session, "not-a-date")

        assert result["ok"] is False
        assert "error" in result

    def test_accepts_iso_string_asof(self, db_session) -> None:
        """An ISO ``YYYY-MM-DD`` string is accepted like a ``date``."""
        _seed_journal(db_session, dt.date(2024, 1, 14), narrative="d14")

        result = get_recent_events(db_session, "2024-01-15", window=14)

        assert result["ok"] is True
        assert result["data"]["count"] == 1
