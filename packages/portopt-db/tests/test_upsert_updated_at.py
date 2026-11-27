"""Regression tests for _upsert() updated_at stamping.

Verifies that on ON CONFLICT DO UPDATE, updated_at is stamped with the
server's current time rather than the stale excluded.updated_at value.

Strategy: force updated_at to a fixed past sentinel value via raw SQL before
each conflict upsert, then assert updated_at > sentinel after the upsert.
This avoids reliance on sub-second timing precision in SQLite.
"""

import datetime
import uuid

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import FredObservation, MacroNews
from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository

_PAST = datetime.datetime(2000, 1, 1, 0, 0, 0)
_NEWS_ID = "UPDATED_AT_NEWS"
_FRED_SERIES = "UPDATED_AT_FRED"


def _news_row(title: str) -> dict:
    return {
        "id": uuid.uuid4(),
        "news_id": _NEWS_ID,
        "title": title,
        "full_content": "body",
    }


def _fred_observations(obs_date: datetime.date, value: float) -> list[dict]:
    return [{"date": obs_date, "value": value}]


class TestUpsertUpdatedAtStamping:
    """_upsert() must advance updated_at on conflict and preserve created_at."""

    def test_updated_at_advances_after_conflict_fallback_path(
        self, db_session: Session
    ) -> None:
        """upsert_macro_news uses update_columns=None (fallback dict-scan) for an
        article that carries a body. On conflict, updated_at must advance."""
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news([_news_row("first headline")])
        db_session.flush()

        row = db_session.execute(
            select(MacroNews).where(MacroNews.news_id == _NEWS_ID)
        ).scalar_one()
        created_at = row.created_at

        db_session.execute(
            text("UPDATE macro_news SET updated_at = :ts WHERE news_id = :n"),
            {"ts": _PAST, "n": _NEWS_ID},
        )
        db_session.flush()
        db_session.expire(row)

        repo.upsert_macro_news([_news_row("revised headline")])
        db_session.flush()
        db_session.expire(row)

        row = db_session.execute(
            select(MacroNews).where(MacroNews.news_id == _NEWS_ID)
        ).scalar_one()
        assert row.title == "revised headline"
        assert row.created_at == created_at, "created_at must never change on conflict"
        assert row.updated_at > _PAST, "updated_at must advance on conflict"

    def test_narrow_update_columns_omitting_updated_at_still_advances_it(
        self, db_session: Session
    ) -> None:
        """Callers with an explicit update_columns list that lacks 'updated_at'
        (e.g. upsert_fred_observations) must still get updated_at refreshed when a
        revised value overwrites the row; Core ON CONFLICT never fires onupdate."""
        repo = MacroRegimeRepository(db_session)
        obs_date = datetime.date(2026, 3, 15)
        repo.upsert_fred_observations(_FRED_SERIES, _fred_observations(obs_date, 2.0))
        db_session.flush()

        row = db_session.execute(
            select(FredObservation).where(
                FredObservation.series_id == _FRED_SERIES,
                FredObservation.date == obs_date,
            )
        ).scalar_one()
        created_at = row.created_at

        db_session.execute(
            text(
                "UPDATE fred_observations SET updated_at = :ts"
                " WHERE series_id = :s AND date = :d"
            ),
            {"ts": _PAST, "s": _FRED_SERIES, "d": obs_date},
        )
        db_session.flush()

        repo.upsert_fred_observations(_FRED_SERIES, _fred_observations(obs_date, 3.0))
        db_session.flush()
        db_session.expire(row)

        row = db_session.execute(
            select(FredObservation).where(
                FredObservation.series_id == _FRED_SERIES,
                FredObservation.date == obs_date,
            )
        ).scalar_one()
        assert row.value == 3.0
        assert row.created_at == created_at, "created_at must never change on conflict"
        assert row.updated_at > _PAST, (
            "updated_at must advance even when absent from update_columns"
        )
