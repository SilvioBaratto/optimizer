"""Branch-coverage tests for MacroRegimeRepository.

Goal: close as many of its branches as possible using the real ``db_session``
SQLite fixture (SAVEPOINT, no second session, no live DB).

IMPORTANT — Postgres-only methods
----------------------------------
All ``_upsert``/``pg_insert … ON CONFLICT`` paths use
``sqlalchemy.dialects.postgresql.insert`` which is **not** executable on
SQLite.  The following public methods are therefore *skipped* for their main
execution path; only their SQLite-safe early-return guard branches are driven:

  - ``upsert_fred_observations``           (pg ON CONFLICT)
  - ``upsert_macro_news``                  (pg ON CONFLICT)

All SELECT / plain-INSERT / DELETE methods run fine on SQLite and are fully
exercised below.
"""

from __future__ import annotations

import datetime
from typing import Any

import pytest
from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import (
    FredObservation,
    MacroNews,
    MacroNewsTheme,
)
from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D = datetime.date
_DT = datetime.datetime


def _flush(session: Session, *objs: object) -> None:
    """Add objects and flush within the SAVEPOINT transaction."""
    for obj in objs:
        session.add(obj)
    session.flush()


def _fred(series: str, date: _D, value: float | None = 0.5) -> FredObservation:
    return FredObservation(series_id=series, date=date, value=value)


def _news(news_id: str, pub: _DT | None = None, ticker: str | None = None) -> MacroNews:
    return MacroNews(
        news_id=news_id,
        title=f"Title {news_id}",
        publish_time=pub,
        source_ticker=ticker,
    )


# ===========================================================================
# Postgres-only early-return guards (empty-input branch = False branch)
# ===========================================================================


class TestUpsertEmptyInputGuards:
    """Each ``upsert_*`` returns 0 immediately when called with an empty payload.

    This drives the ``if not observations / if not rows`` guard branch (the
    'True' side) without executing the pg ON CONFLICT stmt.
    """

    def test_upsert_fred_observations_empty_list_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_fred_observations("T10Y2Y", []) == 0

    def test_upsert_fred_observations_skips_null_values(
        self, db_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[dict[str, Any]] = []

        def _capture(self: Any, model: type, rows: list[dict[str, Any]], **_: Any):
            captured.extend(rows)
            return len(rows)

        monkeypatch.setattr(MacroRegimeRepository, "_upsert", _capture)
        repo = MacroRegimeRepository(db_session)

        processed = repo.upsert_fred_observations(
            "DGS10",
            [
                {"date": _D(2026, 1, 1), "value": 4.1},
                {"date": _D(2026, 1, 2), "value": None},
            ],
        )

        assert processed == 1
        assert [r["date"] for r in captured] == [_D(2026, 1, 1)]

    def test_upsert_fred_observations_all_null_values_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        observations = [{"date": _D(2026, 1, 2), "value": None}]
        assert repo.upsert_fred_observations("DGS10", observations) == 0

    def test_upsert_macro_news_empty_list_returns_zero(
        self, db_session: Session
    ) -> None:
        # upsert_macro_news hits a PG upsert internally — but when rows=[] the
        # early return in _upsert fires BEFORE pg_insert is reached.
        # Postgres-only ON CONFLICT — not executable on SQLite (main path)
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_macro_news([]) == 0


# ===========================================================================
# get_fred_observations
# ===========================================================================


class TestGetFredObservations:
    _D1 = _D(2026, 1, 1)
    _D2 = _D(2026, 2, 1)
    _D3 = _D(2026, 3, 1)
    _D4 = _D(2026, 4, 1)

    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_fred_observations()) == []

    def test_no_filters_returns_all(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("T10Y2Y", self._D1),
            _fred("T10Y2Y", self._D2),
            _fred("DGS10", self._D1),
        )
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_fred_observations()) == 3

    def test_series_id_filter(self, db_session: Session) -> None:
        _flush(db_session, _fred("T10Y2Y", self._D1), _fred("DGS10", self._D1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(series_id="DGS10")
        assert all(r.series_id == "DGS10" for r in results)
        assert len(results) == 1

    def test_series_id_filter_no_match(self, db_session: Session) -> None:
        _flush(db_session, _fred("T10Y2Y", self._D1))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_fred_observations(series_id="UNKNOWN")) == []

    def test_start_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("S", self._D1),
            _fred("S", self._D2),
            _fred("S", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(start_date=self._D2)
        assert all(r.date >= self._D2 for r in results)
        assert len(results) == 2

    def test_end_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("S", self._D1),
            _fred("S", self._D2),
            _fred("S", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(end_date=self._D2)
        assert all(r.date <= self._D2 for r in results)
        assert len(results) == 2

    def test_limit_returns_most_recent_in_asc_order(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("S", self._D1, 1.0),
            _fred("S", self._D2, 2.0),
            _fred("S", self._D3, 3.0),
            _fred("S", self._D4, 4.0),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(series_id="S", limit=2)
        assert len(results) == 2
        assert results[0].date < results[1].date
        assert results[0].date >= self._D3  # D3 and D4 are the two most recent

    def test_limit_without_series_id_applies_per_series(
        self, db_session: Session
    ) -> None:
        _flush(
            db_session,
            *(_fred("A", _D(2026, 1, day), float(day)) for day in range(1, 6)),
            *(_fred("B", _D(2026, 1, day), float(day)) for day in range(1, 6)),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(limit=2)
        assert [(r.series_id, r.date.day) for r in results] == [
            ("A", 4),
            ("A", 5),
            ("B", 4),
            ("B", 5),
        ]

    def test_limit_without_series_id_respects_date_filters(
        self, db_session: Session
    ) -> None:
        _flush(
            db_session,
            *(_fred("A", _D(2026, 1, day)) for day in range(1, 6)),
            *(_fred("B", _D(2026, 1, day)) for day in range(1, 6)),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(end_date=_D(2026, 1, 3), limit=1)
        assert [(r.series_id, r.date.day) for r in results] == [("A", 3), ("B", 3)]

    def test_limit_none_uses_ascending_order(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("S", self._D3),
            _fred("S", self._D1),
            _fred("S", self._D2),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(series_id="S", limit=None)
        dates = [r.date for r in results]
        assert dates == sorted(dates)

    def test_limit_with_series_and_date_filters(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("A", self._D1),
            _fred("A", self._D2),
            _fred("A", self._D3),
            _fred("B", self._D1),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(
            series_id="A", start_date=self._D2, limit=10
        )
        assert all(r.series_id == "A" for r in results)
        assert all(r.date >= self._D2 for r in results)

    def test_all_filters_no_limit(self, db_session: Session) -> None:
        _flush(db_session, _fred("X", self._D1), _fred("X", self._D2))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(
            series_id="X", start_date=self._D1, end_date=self._D1
        )
        assert len(results) == 1


class TestGetFredObservationsExcludeNull:
    _D1 = _D(2026, 1, 1)
    _D2 = _D(2026, 2, 1)
    _D3 = _D(2026, 3, 1)

    def test_exclude_null_drops_placeholder_rows_before_limit(
        self, db_session: Session
    ) -> None:
        _flush(
            db_session,
            _fred("S", self._D1, 1.0),
            _fred("S", self._D2, 2.0),
            _fred("S", self._D3, None),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(series_id="S", limit=2, exclude_null=True)
        assert [r.date for r in results] == [self._D1, self._D2]

    def test_default_keeps_placeholder_rows(self, db_session: Session) -> None:
        _flush(db_session, _fred("S", self._D1, 1.0), _fred("S", self._D2, None))
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_fred_observations(series_id="S")) == 2


# ===========================================================================
# get_fred_latest_date
# ===========================================================================


class TestGetFredLatestDate:
    _D1 = _D(2026, 1, 1)
    _D2 = _D(2026, 2, 1)

    def test_when_no_rows_returns_none(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.get_fred_latest_date("NONEXISTENT") is None

    def test_returns_maximum_date_for_series(self, db_session: Session) -> None:
        _flush(db_session, _fred("S", self._D1), _fred("S", self._D2))
        repo = MacroRegimeRepository(db_session)
        assert repo.get_fred_latest_date("S") == self._D2

    def test_isolates_by_series_id(self, db_session: Session) -> None:
        _flush(db_session, _fred("A", self._D1), _fred("B", self._D2))
        repo = MacroRegimeRepository(db_session)
        assert repo.get_fred_latest_date("A") == self._D1
        assert repo.get_fred_latest_date("B") == self._D2

    def test_ignores_null_placeholder_rows(self, db_session: Session) -> None:
        _flush(db_session, _fred("S", self._D1, 1.0), _fred("S", self._D2, None))
        repo = MacroRegimeRepository(db_session)
        assert repo.get_fred_latest_date("S") == self._D1

    def test_returns_none_when_only_placeholder_rows_exist(
        self, db_session: Session
    ) -> None:
        _flush(db_session, _fred("S", self._D1, None))
        repo = MacroRegimeRepository(db_session)
        assert repo.get_fred_latest_date("S") is None


# ===========================================================================
# get_macro_news
# ===========================================================================


class TestGetMacroNews:
    _PUB1 = _DT(2026, 1, 1, 12, 0, 0)
    _PUB2 = _DT(2026, 2, 1, 12, 0, 0)
    _PUB3 = _DT(2026, 3, 1, 12, 0, 0)

    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_macro_news()) == []

    def test_no_filters_returns_up_to_limit(self, db_session: Session) -> None:
        for i in range(5):
            _flush(db_session, _news(f"id{i}", self._PUB1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(limit=3)
        assert len(results) == 3

    def test_orders_by_publish_time_desc(self, db_session: Session) -> None:
        _flush(
            db_session,
            _news("n1", self._PUB1),
            _news("n2", self._PUB2),
            _news("n3", self._PUB3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(limit=50)
        dates = [r.publish_time for r in results]
        assert dates == sorted(dates, reverse=True)

    def test_theme_filter_returns_matching(self, db_session: Session) -> None:
        article = _news("themed_001", self._PUB1)
        db_session.add(article)
        db_session.flush()
        db_session.add(MacroNewsTheme(news_id=article.id, theme="inflation"))
        db_session.flush()

        unrelated = _news("other_002", self._PUB2)
        db_session.add(unrelated)
        db_session.flush()

        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(theme="inflation")
        ids = [r.news_id for r in results]
        assert "themed_001" in ids
        assert "other_002" not in ids

    def test_theme_filter_no_match_returns_empty(self, db_session: Session) -> None:
        _flush(db_session, _news("n1", self._PUB1))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_macro_news(theme="nonexistent")) == []

    def test_start_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _news("early", self._PUB1),
            _news("late", self._PUB3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(start_date=self._PUB2, limit=50)
        ids = {r.news_id for r in results}
        assert "late" in ids
        assert "early" not in ids

    def test_end_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _news("early", self._PUB1),
            _news("late", self._PUB3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(end_date=self._PUB2, limit=50)
        ids = {r.news_id for r in results}
        assert "early" in ids
        assert "late" not in ids

    def test_start_and_end_date_combined(self, db_session: Session) -> None:
        _flush(
            db_session,
            _news("jan", self._PUB1),
            _news("feb", self._PUB2),
            _news("mar", self._PUB3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(
            start_date=self._PUB2, end_date=self._PUB2, limit=50
        )
        assert len(results) == 1
        assert results[0].news_id == "feb"

    def test_no_theme_filter_skips_join(self, db_session: Session) -> None:
        _flush(db_session, _news("plain", self._PUB1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(limit=10)
        assert any(r.news_id == "plain" for r in results)


# ===========================================================================
# delete_old_macro_news
# ===========================================================================


class TestDeleteOldMacroNews:
    _PUB1 = _DT(2025, 6, 1, 0, 0, 0)
    _PUB2 = _DT(2026, 1, 1, 0, 0, 0)
    _PUB3 = _DT(2026, 6, 1, 0, 0, 0)

    def test_when_empty_returns_zero(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.delete_old_macro_news(_DT(2026, 1, 1)) == 0

    def test_deletes_rows_before_cutoff(self, db_session: Session) -> None:
        _flush(db_session, _news("old", self._PUB1), _news("new", self._PUB3))
        repo = MacroRegimeRepository(db_session)
        count = repo.delete_old_macro_news(self._PUB2)
        db_session.flush()
        assert count == 1
        remaining = repo.get_macro_news(limit=100)
        ids = {r.news_id for r in remaining}
        assert "old" not in ids
        assert "new" in ids

    def test_preserves_rows_on_cutoff_boundary(self, db_session: Session) -> None:
        _flush(db_session, _news("boundary", self._PUB2))
        repo = MacroRegimeRepository(db_session)
        count = repo.delete_old_macro_news(self._PUB2)
        assert count == 0

    def test_deletes_multiple_rows(self, db_session: Session) -> None:
        for i in range(4):
            _flush(db_session, _news(f"old{i}", self._PUB1))
        _flush(db_session, _news("keep", self._PUB3))
        repo = MacroRegimeRepository(db_session)
        assert repo.delete_old_macro_news(self._PUB2) == 4
