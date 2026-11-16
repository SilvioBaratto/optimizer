"""Repository for the news articles stored per instrument."""

import contextlib
from collections.abc import Sequence
from datetime import date, datetime, timezone
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, and_, delete, or_, select

from portopt_db.models.market_data.yfinance_data import TickerNews
from portopt_db.models.universe.universe import Instrument
from portopt_db.repositories.market_data._mapping import (
    _last_wins,
    _safe_str,
    _strip_nul,
)
from portopt_db.repository import RepositoryBase

# Bound parameters per IN list, well under the 32,767 a PostgreSQL statement
# (and SQLite's 32,766) accepts.
_NEWS_ID_CHUNK = 500

# Rows per ``prune_ticker_news`` call. Each one carries a full article body, so
# the batch size is what keeps a single delete transaction small.
_NEWS_PRUNE_BATCH = 1000


class NewsRepository(RepositoryBase):
    """News articles stored per instrument."""

    def upsert_news(self, instrument_id: UUID, articles: list[dict[str, Any]]) -> int:
        """Upsert news articles from yf.Ticker.news list of dicts.

        Handles both old yfinance format (flat dict with ``uuid``, ``title``,
        ``providerPublishTime``) and new format (nested ``content`` dict with
        ``pubDate`` ISO 8601 string).
        """
        ticker_name = self.session.execute(
            select(Instrument.name).where(Instrument.id == instrument_id)
        ).scalar_one_or_none()

        rows = []
        for article in articles:
            # New format nests data under "content"; old format is flat
            content = article.get("content", article)

            # UUID: new format uses top-level "id", old uses "uuid"
            news_uuid = _safe_str(article.get("id") or article.get("uuid"), 200)
            if not news_uuid:
                continue

            title = _safe_str(content.get("title") or article.get("title"))

            # Publisher: new format nests under provider dict
            provider = content.get("provider")
            if isinstance(provider, dict):
                publisher = _safe_str(provider.get("displayName", ""), 500)
            else:
                publisher = _safe_str(article.get("publisher"), 500)

            # Link: new format uses canonicalUrl or previewUrl
            canonical = content.get("canonicalUrl")
            if isinstance(canonical, dict):
                link = _safe_str(canonical.get("url", ""))
            else:
                link = _safe_str(content.get("previewUrl") or article.get("link"))

            # Publish time: new format is ISO 8601, old is epoch int
            publish_time = None
            pub_date_str = content.get("pubDate")
            if pub_date_str and isinstance(pub_date_str, str):
                try:
                    from dateutil import parser as dateutil_parser

                    publish_time = dateutil_parser.isoparse(pub_date_str)
                    # The column is timestamptz: keep UTC rather than a naive
                    # value the database would read in its session timezone.
                    if publish_time.tzinfo is None:
                        publish_time = publish_time.replace(tzinfo=timezone.utc)
                    else:
                        publish_time = publish_time.astimezone(timezone.utc)
                except (ValueError, TypeError):
                    pass
            if publish_time is None:
                pt = article.get("providerPublishTime")
                if pt is not None:
                    with contextlib.suppress(
                        ValueError, TypeError, OSError, OverflowError
                    ):
                        publish_time = datetime.fromtimestamp(int(pt), tz=timezone.utc)

            news_type = _safe_str(
                content.get("contentType") or article.get("type"), 100
            )

            # Full content (scraped upstream by NewsClient if available)
            full_content = article.get("full_content")
            if isinstance(full_content, str):
                full_content = _strip_nul(full_content)

            rows.append(
                {
                    "instrument_id": instrument_id,
                    "news_uuid": news_uuid,
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "publish_time": publish_time,
                    "news_type": news_type,
                    "ticker_name": _safe_str(ticker_name, 500),
                    "full_content": full_content,
                }
            )

        # The same articles reappear in ticker.news for days and each run
        # re-scrapes them. A scrape that fails (timeout, paywall, consent wall)
        # yields no body, and rewriting every column would NULL a body stored by
        # an earlier run, so those rows leave full_content out of the conflict
        # update.
        rows = _last_wins(rows, "news_uuid")
        with_body = [r for r in rows if r["full_content"]]
        without_body = [r for r in rows if not r["full_content"]]
        keep_body_columns = [
            col.name
            for col in TickerNews.__table__.columns
            if col.name not in {"id", "created_at", "full_content"}
        ]

        written = 0
        if with_body:
            written += self._upsert(
                TickerNews,
                with_body,
                constraint_name="uq_ticker_news_instrument_uuid",
            )
        if without_body:
            written += self._upsert(
                TickerNews,
                without_body,
                constraint_name="uq_ticker_news_instrument_uuid",
                update_columns=keep_body_columns,
            )
        return written

    def get_scraped_news_uuids(
        self, instrument_id: UUID, news_uuids: Sequence[str]
    ) -> set[str]:
        """Return which of the given articles already have a stored body.

        Selects the id column only, so the stored article text is never loaded.

        Args:
            instrument_id: Instrument the articles were published for.
            news_uuids: Article ids as stored, i.e. truncated to the column width
                like the upsert truncates them.

        Returns:
            The subset of *news_uuids* whose row holds a non-empty body.
        """
        found: set[str] = set()
        for start in range(0, len(news_uuids), _NEWS_ID_CHUNK):
            chunk = list(news_uuids[start : start + _NEWS_ID_CHUNK])
            found.update(
                self.session.execute(
                    select(TickerNews.news_uuid).where(
                        TickerNews.instrument_id == instrument_id,
                        TickerNews.news_uuid.in_(chunk),
                        TickerNews.full_content.is_not(None),
                        TickerNews.full_content != "",
                    )
                ).scalars()
            )
        return found

    def get_news(self, instrument_id: UUID) -> Sequence[TickerNews]:
        """Return news articles for an instrument, ordered by publish time descending."""
        stmt = (
            select(TickerNews)
            .where(TickerNews.instrument_id == instrument_id)
            .order_by(TickerNews.publish_time.desc().nullslast())
        )
        return self.session.execute(stmt).scalars().all()

    def prune_ticker_news(
        self,
        older_than: datetime | date,
        *,
        batch_size: int = _NEWS_PRUNE_BATCH,
    ) -> int:
        """Delete one batch of articles published before ``older_than``.

        Each row carries the scraped article body, so the table grows by
        megabytes a day without a retention rule. At most ``batch_size`` rows
        go per call and the repository never commits: the caller commits after
        every call and repeats until fewer than ``batch_size`` rows come back,
        which keeps each transaction, and the row locks it holds, small.

        A row is aged by ``publish_time``; an undated one by ``created_at``, so
        it cannot outlive the rule.

        Args:
            older_than: Exclusive cutoff. A naive datetime is read as UTC and a
                plain ``date`` as midnight UTC.
            batch_size: Maximum rows deleted by this call; must be at least 1 so
                a bad value can never turn into an unbounded delete.

        Returns:
            Number of rows deleted by this call; ``0`` once nothing older than
            the cutoff is left.

        Raises:
            ValueError: ``batch_size`` is below 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if isinstance(older_than, datetime):
            cutoff = (
                older_than.replace(tzinfo=timezone.utc)
                if older_than.tzinfo is None
                else older_than.astimezone(timezone.utc)
            )
        else:
            cutoff = datetime(
                older_than.year, older_than.month, older_than.day, tzinfo=timezone.utc
            )
        # Two plain comparisons rather than COALESCE(publish_time, created_at):
        # an expression would not use the index on publish_time.
        expired = select(TickerNews.id).where(
            or_(
                TickerNews.publish_time < cutoff,
                and_(
                    TickerNews.publish_time.is_(None),
                    TickerNews.created_at < cutoff,
                ),
            )
        )
        result = self.session.execute(
            delete(TickerNews).where(TickerNews.id.in_(expired.limit(batch_size)))
        )
        return cast("CursorResult[Any]", result).rowcount
