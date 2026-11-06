"""Retention of ``ticker_news``: article bodies are pruned in bounded batches."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from portopt_db.models.market_data.yfinance_data import TickerNews
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.yfinance_repository import (
    YFinanceRepository,
)

_NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
_CUTOFF = _NOW - timedelta(days=365)


def _instrument(session: Session, ticker: str = "TSM") -> Instrument:
    exchange = Exchange(name=f"EX-{ticker}")
    session.add(exchange)
    session.flush()
    instrument = Instrument(
        ticker=ticker,
        short_name=ticker,
        exchange=exchange,
        yfinance_ticker=ticker,
    )
    session.add(instrument)
    session.flush()
    return instrument


def _article(
    session: Session,
    instrument: Instrument,
    news_uuid: str,
    *,
    published: datetime | None,
    ingested: datetime | None = None,
) -> None:
    session.add(
        TickerNews(
            instrument_id=instrument.id,
            news_uuid=news_uuid,
            title=news_uuid,
            publish_time=published,
            full_content="body " * 100,
            **({"created_at": ingested} if ingested is not None else {}),
        )
    )
    session.flush()


def _remaining(session: Session) -> set[str]:
    return set(session.execute(select(TickerNews.news_uuid)).scalars())


def _count(session: Session) -> int:
    return session.execute(select(func.count()).select_from(TickerNews)).scalar_one()


class TestPruneTickerNews:
    """``prune_ticker_news`` removes one bounded batch of expired articles."""

    def test_old_articles_go_and_recent_ones_stay(self, db_session: Session) -> None:
        """Expired rows of every instrument are deleted, newer rows survive."""
        inst = _instrument(db_session)
        other = _instrument(db_session, "OTHER")
        _article(db_session, inst, "old", published=_CUTOFF - timedelta(days=1))
        _article(db_session, other, "old-other", published=_CUTOFF - timedelta(days=90))
        _article(db_session, inst, "recent", published=_CUTOFF + timedelta(days=1))
        _article(db_session, inst, "fresh", published=_NOW - timedelta(hours=3))

        removed = YFinanceRepository(db_session).prune_ticker_news(_CUTOFF)

        assert removed == 2
        assert _remaining(db_session) == {"recent", "fresh"}

    def test_an_article_published_exactly_at_the_cutoff_is_kept(
        self, db_session: Session
    ) -> None:
        """The cutoff is exclusive: only strictly older rows are deleted."""
        inst = _instrument(db_session)
        _article(db_session, inst, "boundary", published=_CUTOFF)

        assert YFinanceRepository(db_session).prune_ticker_news(_CUTOFF) == 0
        assert _remaining(db_session) == {"boundary"}

    def test_a_missing_publish_time_falls_back_to_the_ingest_time(
        self, db_session: Session
    ) -> None:
        """An undated article ages by its ``created_at`` instead of living forever."""
        inst = _instrument(db_session)
        _article(
            db_session,
            inst,
            "undated-old",
            published=None,
            ingested=_CUTOFF - timedelta(days=30),
        )
        _article(
            db_session,
            inst,
            "undated-recent",
            published=None,
            ingested=_CUTOFF + timedelta(days=30),
        )

        removed = YFinanceRepository(db_session).prune_ticker_news(_CUTOFF)

        assert removed == 1
        assert _remaining(db_session) == {"undated-recent"}

    def test_the_publish_time_wins_over_the_ingest_time_when_both_exist(
        self, db_session: Session
    ) -> None:
        """A dated article is judged by when it was published, not ingested."""
        inst = _instrument(db_session)
        _article(
            db_session,
            inst,
            "published-recently-ingested-long-ago",
            published=_NOW - timedelta(days=2),
            ingested=_CUTOFF - timedelta(days=400),
        )
        _article(
            db_session,
            inst,
            "published-long-ago",
            published=_CUTOFF - timedelta(days=2),
            ingested=_NOW - timedelta(days=1),
        )

        removed = YFinanceRepository(db_session).prune_ticker_news(_CUTOFF)

        assert removed == 1
        assert _remaining(db_session) == {"published-recently-ingested-long-ago"}

    def test_one_call_deletes_at_most_one_batch(self, db_session: Session) -> None:
        """The caller drives the loop, so each commit covers a bounded delete."""
        inst = _instrument(db_session)
        for i in range(5):
            _article(
                db_session,
                inst,
                f"old-{i}",
                published=_CUTOFF - timedelta(days=i + 1),
            )
        _article(db_session, inst, "recent", published=_NOW)
        repo = YFinanceRepository(db_session)

        assert repo.prune_ticker_news(_CUTOFF, batch_size=2) == 2
        assert _count(db_session) == 4
        assert repo.prune_ticker_news(_CUTOFF, batch_size=2) == 2
        assert repo.prune_ticker_news(_CUTOFF, batch_size=2) == 1
        assert repo.prune_ticker_news(_CUTOFF, batch_size=2) == 0
        assert _remaining(db_session) == {"recent"}

    def test_a_repeat_run_finds_nothing_more_to_delete(
        self, db_session: Session
    ) -> None:
        """Pruning is idempotent."""
        inst = _instrument(db_session)
        _article(db_session, inst, "old", published=_CUTOFF - timedelta(days=5))
        repo = YFinanceRepository(db_session)

        assert repo.prune_ticker_news(_CUTOFF) == 1
        assert repo.prune_ticker_news(_CUTOFF) == 0

    def test_an_empty_table_is_a_no_op(self, db_session: Session) -> None:
        """Nothing stored means nothing deleted."""
        assert YFinanceRepository(db_session).prune_ticker_news(_CUTOFF) == 0

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_a_non_positive_batch_size_is_rejected_and_deletes_nothing(
        self, db_session: Session, batch_size: int
    ) -> None:
        """A bad batch size must never turn into an unbounded delete."""
        inst = _instrument(db_session)
        _article(db_session, inst, "old", published=_CUTOFF - timedelta(days=5))

        with pytest.raises(ValueError, match="batch_size"):
            YFinanceRepository(db_session).prune_ticker_news(
                _CUTOFF, batch_size=batch_size
            )
        assert _remaining(db_session) == {"old"}

    def test_a_plain_date_cuts_at_midnight_utc(self, db_session: Session) -> None:
        """A ``date`` cutoff means the start of that day in UTC."""
        inst = _instrument(db_session)
        _article(
            db_session,
            inst,
            "before",
            published=datetime(2025, 11, 5, 23, 59, tzinfo=timezone.utc),
        )
        _article(
            db_session,
            inst,
            "after",
            published=datetime(2025, 11, 6, 0, 1, tzinfo=timezone.utc),
        )

        removed = YFinanceRepository(db_session).prune_ticker_news(date(2025, 11, 6))

        assert removed == 1
        assert _remaining(db_session) == {"after"}

    def test_a_naive_cutoff_is_read_as_utc_and_an_aware_one_is_converted(
        self, db_session: Session
    ) -> None:
        """Offsets are normalised to UTC before comparing against stored times."""
        inst = _instrument(db_session)
        _article(
            db_session,
            inst,
            "inside",
            published=datetime(2025, 11, 6, 0, 30, tzinfo=timezone.utc),
        )
        repo = YFinanceRepository(db_session)

        # 2025-11-06 02:00 at UTC+3 is 2025-11-05 23:00 UTC, before the article.
        plus_three = timezone(timedelta(hours=3))
        assert repo.prune_ticker_news(datetime(2025, 11, 6, 2, tzinfo=plus_three)) == 0
        assert repo.prune_ticker_news(datetime(2025, 11, 6, 1)) == 1

    def test_the_default_batch_size_is_bounded(self) -> None:
        """The default keeps a single transaction small."""
        default = (
            inspect.signature(YFinanceRepository.prune_ticker_news)
            .parameters["batch_size"]
            .default
        )
        assert isinstance(default, int)
        assert 1 <= default <= 10_000
