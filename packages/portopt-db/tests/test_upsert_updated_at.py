"""Regression tests for _upsert() updated_at stamping.

Verifies that on ON CONFLICT DO UPDATE, updated_at is stamped with the
server's current time rather than the stale excluded.updated_at value.

Strategy: force updated_at to a fixed past sentinel value via raw SQL before
each conflict upsert, then assert updated_at > sentinel after the upsert.
This avoids reliance on sub-second timing precision in SQLite.
"""

import datetime
import uuid

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import FredObservation, MacroNews
from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository

_PAST = datetime.datetime(2000, 1, 1, 0, 0, 0)
_NEWS_ID = "UPDATED_AT_NEWS"
_NEWS_BODY = "body"
_FIRST_HEADLINE = "first headline"
_REVISED_HEADLINE = "revised headline"
_FRED_SERIES = "UPDATED_AT_FRED"
_OBSERVATION_DATE = datetime.date(2026, 3, 15)
_FIRST_VALUE = 2.0
_REVISED_VALUE = 3.0


def _news_row(title: str) -> dict:
    return {
        "id": uuid.uuid4(),
        "news_id": _NEWS_ID,
        "title": title,
        "full_content": _NEWS_BODY,
    }


def _fred_observations(value: float) -> list[dict]:
    return [{"date": _OBSERVATION_DATE, "value": value}]


class TestUpsertUpdatedAtStamping:
    """_upsert() must advance updated_at on conflict and preserve created_at."""

    @pytest.mark.criterion("REQ-DB-011")
    def test_updated_at_advances_after_conflict_fallback_path(
        self, db_session: Session
    ) -> None:
        """Validates RepositoryBase._upsert as called by
        MacroRegimeRepository.upsert_macro_news, which passes no update_columns
        for an article that carries a body (fallback dict-scan): on conflict the
        row is updated in place, created_at is kept and updated_at advances."""
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news([_news_row(_FIRST_HEADLINE)])
        db_session.flush()

        row = db_session.execute(
            select(MacroNews).where(MacroNews.news_id == _NEWS_ID)
        ).scalar_one()
        created_at = row.created_at

        db_session.execute(
            update(MacroNews)
            .where(MacroNews.news_id == _NEWS_ID)
            .values(updated_at=_PAST)
        )
        db_session.flush()
        db_session.expire(row)

        repo.upsert_macro_news([_news_row(_REVISED_HEADLINE)])
        db_session.flush()
        db_session.expire(row)

        row = db_session.execute(
            select(MacroNews).where(MacroNews.news_id == _NEWS_ID)
        ).scalar_one()
        assert row.title == _REVISED_HEADLINE
        assert row.created_at == created_at, "created_at must never change on conflict"
        assert row.updated_at > _PAST, "updated_at must advance on conflict"

    @pytest.mark.criterion("REQ-DB-011")
    def test_narrow_update_columns_omitting_updated_at_still_advances_it(
        self, db_session: Session
    ) -> None:
        """Validates RepositoryBase._upsert as called by
        MacroRegimeRepository.upsert_fred_observations, whose explicit
        update_columns list lacks updated_at: a revised value still overwrites the
        row and re-stamps updated_at (Core ON CONFLICT never fires onupdate)."""
        repo = MacroRegimeRepository(db_session)
        repo.upsert_fred_observations(_FRED_SERIES, _fred_observations(_FIRST_VALUE))
        db_session.flush()

        row = db_session.execute(
            select(FredObservation).where(
                FredObservation.series_id == _FRED_SERIES,
                FredObservation.date == _OBSERVATION_DATE,
            )
        ).scalar_one()
        created_at = row.created_at

        db_session.execute(
            update(FredObservation)
            .where(
                FredObservation.series_id == _FRED_SERIES,
                FredObservation.date == _OBSERVATION_DATE,
            )
            .values(updated_at=_PAST)
        )
        db_session.flush()

        repo.upsert_fred_observations(_FRED_SERIES, _fred_observations(_REVISED_VALUE))
        db_session.flush()
        db_session.expire(row)

        row = db_session.execute(
            select(FredObservation).where(
                FredObservation.series_id == _FRED_SERIES,
                FredObservation.date == _OBSERVATION_DATE,
            )
        ).scalar_one()
        assert row.value == _REVISED_VALUE
        assert row.created_at == created_at, "created_at must never change on conflict"
        assert row.updated_at > _PAST, (
            "updated_at must advance even when absent from update_columns"
        )
