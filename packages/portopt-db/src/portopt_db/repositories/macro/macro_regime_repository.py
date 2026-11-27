"""Repository for macro regime data access with PostgreSQL upsert support."""

import datetime
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import (
    FredObservation,
    MacroNews,
    MacroNewsTheme,
)
from portopt_db.repository import RepositoryBase

logger = logging.getLogger(__name__)


class MacroRegimeRepository(RepositoryBase):
    """Sync repository for macro regime data. Uses PostgreSQL ON CONFLICT upsert."""

    def __init__(self, session: Session):
        super().__init__(session)

    # ------------------------------------------------------------------
    # FRED Observations
    # ------------------------------------------------------------------

    def upsert_fred_observations(
        self,
        series_id: str,
        observations: list[dict[str, Any]],
    ) -> int:
        """Bulk upsert FRED observations for a single series.

        Args:
            series_id: FRED series identifier (e.g. ``"BAMLH0A0HYM2"``).
            observations: List of ``{"date": datetime.date, "value": float|None}``.

        Returns:
            Number of rows processed. Observations with a ``None`` value
            (FRED's ``"."`` placeholder) are skipped.
        """
        if not observations:
            return 0

        rows: list[dict[str, Any]] = [
            {
                "id": uuid.uuid4(),
                "series_id": series_id,
                "date": obs["date"],
                "value": obs["value"],
            }
            for obs in observations
            if obs.get("value") is not None
        ]

        return self._upsert(
            FredObservation,
            rows,
            constraint_name="uq_fred_observation_series_date",
            update_columns=["value"],
        )

    def get_fred_observations(
        self,
        series_id: str | None = None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        limit: int | None = None,
        exclude_null: bool = False,
    ) -> Sequence[FredObservation]:
        """Return FRED observations with optional filters.

        Args:
            series_id: Filter to this series. ``None`` returns all series.
            start_date: Inclusive lower bound on observation date.
            end_date: Inclusive upper bound on observation date.
            limit: When supplied, the *limit* most-recent rows of each series
                are kept, so ``result[-1]`` of a single series is always its
                latest value and no series is starved when *series_id* is
                ``None``.
            exclude_null: Drop rows whose ``value`` is NULL (FRED ``.``
                placeholders for holidays/missing readings) before *limit* is
                applied, so the window holds the most recent real readings.

        Returns:
            Rows ordered by series_id and date ascending. When *limit* is
            applied the bounded window is still returned in ascending order.
        """
        stmt = select(FredObservation)
        if series_id:
            stmt = stmt.where(FredObservation.series_id == series_id)
        if start_date:
            stmt = stmt.where(FredObservation.date >= start_date)
        if end_date:
            stmt = stmt.where(FredObservation.date <= end_date)
        if exclude_null:
            stmt = stmt.where(FredObservation.value.is_not(None))

        if limit is not None:
            # A plain LIMIT would be taken across series in series_id order and
            # return only the alphabetically first series.
            recency = sa_func.row_number().over(
                partition_by=FredObservation.series_id,
                order_by=FredObservation.date.desc(),
            )
            ranked = stmt.with_only_columns(
                FredObservation.id, recency.label("recency")
            ).subquery()
            stmt = (
                select(FredObservation)
                .join(ranked, FredObservation.id == ranked.c.id)
                .where(ranked.c.recency <= limit)
            )

        stmt = stmt.order_by(FredObservation.series_id, FredObservation.date)
        return self.session.execute(stmt).scalars().all()

    def get_fred_latest_date(self, series_id: str) -> datetime.date | None:
        """Return the most recent observation date that carries a value.

        Rows whose value is ``NULL`` (FRED's ``"."`` placeholder) are ignored so
        an incremental fetch anchored on this date never skips past a day whose
        real value has not been stored yet.

        Args:
            series_id: FRED series identifier to query.

        Returns:
            The latest stored date with a non-null value, or ``None`` if the
            series has no such observation.
        """
        result = self.session.execute(
            select(sa_func.max(FredObservation.date)).where(
                FredObservation.series_id == series_id,
                FredObservation.value.isnot(None),
            )
        ).scalar()
        return result

    # ------------------------------------------------------------------
    # Macro News
    # ------------------------------------------------------------------

    def upsert_macro_news(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert macro news rows. Deduplicates on ``news_id``.

        Extracts ``themes`` from each row dict, upserts the MacroNews rows,
        then manages MacroNewsTheme child rows for each article.

        Parent rows use ``INSERT ... ON CONFLICT DO UPDATE`` on
        ``uq_macro_news_id``. The theme children are replaced (clear-then-re-add)
        rather than upserted on ``uq_macro_news_theme``, deliberately: a re-run
        must converge to *exactly* the current theme set, and a bare per-theme
        upsert would leave stale themes behind when an article's classification
        changes. The clear+re-add runs inside the caller's single transaction
        (atomic — a crash rolls back both halves), and the one-daemon-per-DB
        invariant precludes a concurrent writer, so the sequence is idempotent
        on re-run and convergent on a changed set.

        A row whose ``full_content`` is ``None`` (a snippet-only fetch) never
        overwrites a body already stored for the same ``news_id``; every other
        column is still refreshed. A row carrying a body replaces the stored one.

        A ``news_id`` repeated within *rows* is collapsed to its last row, keeping
        a body an earlier occurrence carried when the last has none. A theme
        repeated within one article is stored once.
        """
        # themes is no longer a DB column; must be extracted before upsert
        themes_by_news_id: dict[str, list[str]] = {}
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_copy = dict(row)
            raw_themes = row_copy.pop("themes", None)
            if raw_themes and isinstance(raw_themes, str):
                themes_by_news_id[row_copy["news_id"]] = list(
                    dict.fromkeys(t.strip() for t in raw_themes.split(",") if t.strip())
                )
            earlier = latest.get(row_copy["news_id"])
            if (
                earlier is not None
                and row_copy.get("full_content") is None
                and earlier.get("full_content") is not None
            ):
                row_copy["full_content"] = earlier["full_content"]
            latest[row_copy["news_id"]] = row_copy
        clean_rows = list(latest.values())

        with_body = [r for r in clean_rows if r.get("full_content") is not None]
        snippet_only = [r for r in clean_rows if r.get("full_content") is None]

        count = self._upsert(
            MacroNews,
            with_body,
            constraint_name="uq_macro_news_id",
        )
        # A separate statement because ON CONFLICT DO UPDATE applies one column
        # set to the whole batch: excluding full_content here keeps a body that
        # an earlier scrape run stored from being nulled by a snippet-only run.
        count += self._upsert(
            MacroNews,
            snippet_only,
            constraint_name="uq_macro_news_id",
            update_columns=[
                col.name
                for col in MacroNews.__table__.columns
                if col.name not in {"id", "created_at", "full_content"}
            ],
        )

        if themes_by_news_id:
            news_ids = list(themes_by_news_id.keys())
            stmt = select(MacroNews).where(MacroNews.news_id.in_(news_ids))
            news_rows = self.session.execute(stmt).scalars().all()
            for news_row in news_rows:
                theme_list = themes_by_news_id.get(news_row.news_id, [])
                if not theme_list:
                    continue
                news_row.theme_entries.clear()
                self.session.flush()
                for theme_str in theme_list:
                    self.session.add(
                        MacroNewsTheme(
                            news_id=news_row.id,
                            theme=theme_str,
                        )
                    )
            self.session.flush()

        return count

    def get_macro_news(
        self,
        theme: str | None = None,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        limit: int = 50,
    ) -> Sequence[MacroNews]:
        """Return stored macro news, most-recent first.

        Args:
            theme: Filter to articles tagged with this theme. ``None`` returns
                all themes.
            start_date: Inclusive lower bound on publish_time.
            end_date: Inclusive upper bound on publish_time.
            limit: Maximum rows to return.

        Returns:
            Up to *limit* rows ordered by publish_time descending.
        """
        stmt = select(MacroNews)
        if theme:
            stmt = stmt.join(MacroNewsTheme).where(MacroNewsTheme.theme == theme)
        if start_date:
            stmt = stmt.where(MacroNews.publish_time >= start_date)
        if end_date:
            stmt = stmt.where(MacroNews.publish_time <= end_date)
        stmt = stmt.order_by(MacroNews.publish_time.desc()).limit(limit)
        return self.session.execute(stmt).scalars().all()

    def delete_old_macro_news(self, before_date: datetime.datetime) -> int:
        """Delete macro news articles published before the given datetime.

        Args:
            before_date: Articles with publish_time strictly before this value
                are deleted.

        Returns:
            Number of rows deleted.
        """
        stmt = select(MacroNews).where(MacroNews.publish_time < before_date)
        rows = self.session.execute(stmt).scalars().all()
        count = len(rows)
        for row in rows:
            self.session.delete(row)
        return count
