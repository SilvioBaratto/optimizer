"""Regression tests for _upsert() updated_at stamping (issue #311).

Verifies that on ON CONFLICT DO UPDATE, updated_at is stamped with the
server's current time rather than the stale excluded.updated_at value.

Strategy: force updated_at to a fixed past sentinel value via raw SQL before
each conflict upsert, then assert updated_at > sentinel after the upsert.
This avoids reliance on sub-second timing precision in SQLite.
"""

import datetime

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from app.models.macro_regime import (
    EconomicIndicator,
    TradingEconomicsObservation,
)
from app.repositories.macro_regime_repository import MacroRegimeRepository

_PAST = datetime.datetime(2000, 1, 1, 0, 0, 0)


class TestUpsertUpdatedAtStamping:
    """_upsert() must advance updated_at on conflict and preserve created_at."""

    def test_updated_at_advances_after_conflict_explicit_update_columns(
        self, db_session: Session
    ) -> None:
        """upsert_macro_news_summary uses an explicit update_columns list.
        On conflict, updated_at must advance beyond the pre-conflict sentinel."""
        repo = MacroRegimeRepository(db_session)
        d = datetime.date(2026, 1, 10)
        data = {
            "summary": "First",
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "article_count": 1,
            "news_summary": "X",
        }
        repo.upsert_macro_news_summary("USA_311_A", d, data)
        db_session.flush()

        row = repo.get_macro_news_summary("USA_311_A", d)
        assert row is not None
        created_at = row.created_at

        # Force updated_at to a known past value so the conflict upsert's func.now()
        # is guaranteed to produce a newer timestamp — avoids SQLite clock precision issues.
        db_session.execute(
            text(
                "UPDATE macro_news_summaries SET updated_at = :ts"
                " WHERE country = :c AND summary_date = :d"
            ),
            {"ts": _PAST, "c": "USA_311_A", "d": d},
        )
        db_session.flush()
        db_session.expire(row)

        repo.upsert_macro_news_summary("USA_311_A", d, {**data, "summary": "Updated"})
        db_session.flush()
        db_session.expire(row)

        row = repo.get_macro_news_summary("USA_311_A", d)
        assert row is not None
        assert row.summary == "Updated"
        assert row.created_at == created_at, "created_at must never change on conflict"
        assert row.updated_at > _PAST, "updated_at must advance on conflict"

    def test_updated_at_advances_after_conflict_fallback_path(
        self, db_session: Session
    ) -> None:
        """upsert_economic_indicator uses update_columns=None (fallback dict-scan).
        On conflict, updated_at must advance."""
        repo = MacroRegimeRepository(db_session)
        data_v1 = {
            "last_inflation": 2.1,
            "inflation_6m": None,
            "inflation_10y_avg": None,
            "gdp_growth_6m": None,
            "earnings_12m": None,
            "eps_expected_12m": None,
            "peg_ratio": None,
            "lt_rate_forecast": None,
            "reference_date": None,
        }
        repo.upsert_economic_indicator("USA_311_B", data_v1)
        db_session.flush()

        row = db_session.execute(
            select(EconomicIndicator).where(EconomicIndicator.country == "USA_311_B")
        ).scalar_one()
        created_at = row.created_at

        db_session.execute(
            text(
                "UPDATE economic_indicators SET updated_at = :ts WHERE country = :c"
            ),
            {"ts": _PAST, "c": "USA_311_B"},
        )
        db_session.flush()
        db_session.expire(row)

        repo.upsert_economic_indicator("USA_311_B", {**data_v1, "last_inflation": 3.5})
        db_session.flush()
        db_session.expire(row)

        row = db_session.execute(
            select(EconomicIndicator).where(EconomicIndicator.country == "USA_311_B")
        ).scalar_one()
        assert row.last_inflation == 3.5
        assert row.created_at == created_at, "created_at must never change on conflict"
        assert row.updated_at > _PAST, "updated_at must advance on conflict"

    def test_created_at_never_overwritten_on_conflict(
        self, db_session: Session
    ) -> None:
        """created_at is excluded from the update_dict and must remain frozen."""
        repo = MacroRegimeRepository(db_session)
        d = datetime.date(2026, 2, 1)
        data = {
            "summary": "A",
            "sentiment": "positive",
            "sentiment_score": 0.5,
            "article_count": 3,
            "news_summary": "Y",
        }
        repo.upsert_macro_news_summary("DE_311", d, data)
        db_session.flush()

        row = repo.get_macro_news_summary("DE_311", d)
        assert row is not None
        original_created_at = row.created_at

        repo.upsert_macro_news_summary("DE_311", d, {**data, "summary": "B"})
        db_session.flush()
        db_session.expire(row)

        row = repo.get_macro_news_summary("DE_311", d)
        assert row is not None
        assert row.created_at == original_created_at

    def test_first_insert_sets_both_timestamps(self, db_session: Session) -> None:
        """On a clean INSERT (no conflict), both created_at and updated_at are set."""
        repo = MacroRegimeRepository(db_session)
        d = datetime.date(2026, 3, 1)
        data = {
            "summary": "New",
            "sentiment": "negative",
            "sentiment_score": -0.3,
            "article_count": 2,
            "news_summary": "Z",
        }
        repo.upsert_macro_news_summary("FR_311", d, data)
        db_session.flush()

        row = repo.get_macro_news_summary("FR_311", d)
        assert row is not None
        assert row.created_at is not None
        assert row.updated_at is not None

    def test_narrow_update_columns_omitting_updated_at_leaves_it_unchanged(
        self, db_session: Session
    ) -> None:
        """Callers that omit 'updated_at' from update_columns (e.g. upsert_te_observations)
        do not get updated_at touched — correct for append-only time-series rows."""
        repo = MacroRegimeRepository(db_session)
        obs_date = datetime.date(2026, 3, 15)
        repo.upsert_te_observations(
            "USA_311_TE", obs_date, {"GDP_GROWTH": {"value": 2.0}}
        )
        db_session.flush()

        row = db_session.execute(
            select(TradingEconomicsObservation).where(
                TradingEconomicsObservation.country == "USA_311_TE",
                TradingEconomicsObservation.date == obs_date,
            )
        ).scalar_one()
        original_updated_at = row.updated_at

        # Force updated_at to a sentinel, then upsert — it must stay at sentinel
        # because 'updated_at' is not in the update_columns for this method.
        db_session.execute(
            text(
                "UPDATE trading_economics_observations SET updated_at = :ts"
                " WHERE country = :c AND date = :d AND indicator_key = :k"
            ),
            {"ts": _PAST, "c": "USA_311_TE", "d": obs_date, "k": "GDP_GROWTH"},
        )
        db_session.flush()

        repo.upsert_te_observations(
            "USA_311_TE", obs_date, {"GDP_GROWTH": {"value": 3.0}}
        )
        db_session.flush()
        db_session.expire(row)

        row = db_session.execute(
            select(TradingEconomicsObservation).where(
                TradingEconomicsObservation.country == "USA_311_TE",
                TradingEconomicsObservation.date == obs_date,
            )
        ).scalar_one()
        assert row.value == 3.0
        assert row.updated_at == _PAST, (
            "updated_at must not change when excluded from update_columns"
        )
