"""Tests for MacroNewsSummary repository methods."""

import datetime

from sqlalchemy.orm import Session

from app.repositories.macro_regime_repository import MacroRegimeRepository


def _make_data(
    summary: str = "Markets stable.",
    sentiment: str = "neutral",
    sentiment_score: float = 0.0,
    article_count: int = 5,
    news_summary: str = "Summary text.",
) -> dict:
    return {
        "summary": summary,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "article_count": article_count,
        "news_summary": news_summary,
    }


class TestUpsertMacroNewsSummary:
    def test_inserts_new_row(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        count = repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data())
        db_session.flush()

        assert count == 1
        row = repo.get_macro_news_summary("USA", datetime.date(2026, 3, 15))
        assert row is not None
        assert row.summary == "Markets stable."
        assert row.article_count == 5

    def test_updates_existing_row(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        date = datetime.date(2026, 3, 15)
        repo.upsert_macro_news_summary("USA", date, _make_data())
        db_session.flush()

        repo.upsert_macro_news_summary(
            "USA", date, _make_data(summary="Updated.", article_count=10)
        )
        db_session.flush()

        row = repo.get_macro_news_summary("USA", date)
        assert row is not None
        assert row.summary == "Updated."
        assert row.article_count == 10

    def test_same_day_upsert_produces_single_row(self, db_session: Session) -> None:
        """Idempotency: two upserts on the same (country, date) produce exactly one row."""
        repo = MacroRegimeRepository(db_session)
        d = datetime.date(2026, 3, 15)
        repo.upsert_macro_news_summary("USA", d, _make_data(summary="First"))
        db_session.flush()
        repo.upsert_macro_news_summary("USA", d, _make_data(summary="Second"))
        db_session.flush()

        all_rows = repo.get_all_news_summaries(summary_date=d)
        usa_rows = [r for r in all_rows if r.country == "USA"]
        assert len(usa_rows) == 1
        assert usa_rows[0].summary == "Second"


class TestGetMacroNewsSummary:
    def test_returns_none_when_no_data(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.get_macro_news_summary("USA") is None

    def test_returns_row_for_exact_date(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 14), _make_data(summary="Day 14"))
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data(summary="Day 15"))
        db_session.flush()

        row = repo.get_macro_news_summary("USA", datetime.date(2026, 3, 14))
        assert row is not None
        assert row.summary == "Day 14"

    def test_returns_latest_when_no_date(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 14), _make_data(summary="Day 14"))
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data(summary="Day 15"))
        db_session.flush()

        row = repo.get_macro_news_summary("USA")
        assert row is not None
        assert row.summary == "Day 15"

    def test_returns_none_for_unknown_country(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data())
        db_session.flush()

        assert repo.get_macro_news_summary("Germany") is None

    def test_country_isolation(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data(summary="US news"))
        repo.upsert_macro_news_summary("UK", datetime.date(2026, 3, 15), _make_data(summary="UK news"))
        db_session.flush()

        row = repo.get_macro_news_summary("UK", datetime.date(2026, 3, 15))
        assert row is not None
        assert row.summary == "UK news"


class TestGetAllNewsSummaries:
    def test_returns_empty_list_when_no_data(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.get_all_news_summaries() == []

    def test_returns_all_rows_when_no_date_filter(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 14), _make_data())
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data())
        repo.upsert_macro_news_summary("UK", datetime.date(2026, 3, 15), _make_data())
        db_session.flush()

        results = repo.get_all_news_summaries()
        assert len(results) == 3

    def test_filters_by_date(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 14), _make_data())
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data())
        repo.upsert_macro_news_summary("UK", datetime.date(2026, 3, 15), _make_data())
        db_session.flush()

        results = repo.get_all_news_summaries(summary_date=datetime.date(2026, 3, 15))
        assert len(results) == 2
        assert all(r.summary_date == datetime.date(2026, 3, 15) for r in results)

    def test_orders_by_country_then_date_desc(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 14), _make_data())
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data())
        repo.upsert_macro_news_summary("UK", datetime.date(2026, 3, 15), _make_data())
        db_session.flush()

        results = repo.get_all_news_summaries()
        countries = [r.country for r in results]
        assert countries == ["UK", "USA", "USA"]
        usa_dates = [r.summary_date for r in results if r.country == "USA"]
        assert usa_dates == [datetime.date(2026, 3, 15), datetime.date(2026, 3, 14)]


class TestDeleteOldNewsSummaries:
    def test_deletes_rows_before_cutoff(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2025, 12, 1), _make_data())
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data())
        db_session.flush()

        deleted = repo.delete_old_news_summaries(datetime.date(2026, 1, 1))
        db_session.flush()

        assert deleted == 1
        assert repo.get_macro_news_summary("USA", datetime.date(2025, 12, 1)) is None
        assert repo.get_macro_news_summary("USA", datetime.date(2026, 3, 15)) is not None

    def test_preserves_rows_on_or_after_cutoff(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        cutoff = datetime.date(2026, 3, 15)
        repo.upsert_macro_news_summary("USA", cutoff, _make_data())
        db_session.flush()

        deleted = repo.delete_old_news_summaries(cutoff)
        db_session.flush()

        assert deleted == 0
        assert repo.get_macro_news_summary("USA", cutoff) is not None

    def test_returns_correct_count(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        for day in range(1, 6):
            repo.upsert_macro_news_summary("USA", datetime.date(2025, 1, day), _make_data())
        db_session.flush()

        deleted = repo.delete_old_news_summaries(datetime.date(2025, 1, 4))
        assert deleted == 3

    def test_empty_table_returns_zero(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.delete_old_news_summaries(datetime.date(2026, 3, 15)) == 0

    def test_all_rows_newer_than_cutoff_returns_zero(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news_summary("USA", datetime.date(2026, 3, 15), _make_data())
        repo.upsert_macro_news_summary("UK", datetime.date(2026, 3, 15), _make_data())
        db_session.flush()

        assert repo.delete_old_news_summaries(datetime.date(2026, 1, 1)) == 0
