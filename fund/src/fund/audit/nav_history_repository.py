"""``NavHistoryRepository`` — fund-side behavior over the ``nav_history`` model.

The ``nav_history`` *model* lives in ``portopt_db`` (shared schema, Alembic-owned);
its *behavior* lives here, mirroring the ``positions`` / ``agent_runs`` /
``paper_orders`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session and owns the transaction boundary (no ``commit`` here).

Module-level imports stay clear of the agent stack (``deepagents``/``langgraph``)
so the daily NAV-mark scheduler step can lazy-import this repo without dragging
the deep-agent framework into a bare ``import fund.scheduler``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from portopt_db.models import NavHistory
from portopt_db.repository import RepositoryBase
from sqlalchemy import func, or_, select


class NavHistoryRepository(RepositoryBase):
    """Upsert/read a portfolio's persisted NAV series on an injected session."""

    def upsert_navs(self, rows: list[dict[str, Any]]) -> int:
        """Upsert NAV rows idempotently, returning the count written.

        Rows are deduplicated on ``(portfolio_id, date)`` (last value wins within
        one payload), so re-marking a date overwrites rather than duplicates.
        Rows missing portfolio id, date, or nav are skipped. The caller owns the
        transaction; this issues the upsert without committing.

        Args:
            rows: Mappings with ``portfolio_id``, ``date``, ``nav`` and optional
                ``benchmark`` (level) and ``benchmark_ticker`` (its identity) keys.

        Returns:
            Number of distinct rows passed to the upsert.
        """
        out: dict[tuple[uuid.UUID, dt.date], dict[str, Any]] = {}
        for r in rows:
            pid = r.get("portfolio_id")
            date = r.get("date")
            nav = r.get("nav")
            if pid is None or date is None or nav is None:
                continue
            out[(pid, date)] = {
                "id": uuid.uuid4(),
                "portfolio_id": pid,
                "date": date,
                "nav": nav,
                "benchmark": r.get("benchmark"),
                "benchmark_ticker": r.get("benchmark_ticker"),
            }
        if not out:
            return 0
        self._upsert(
            NavHistory,
            list(out.values()),
            constraint_name="uq_nav_history_portfolio_date",
            update_columns=["nav", "benchmark", "benchmark_ticker", "updated_at"],
        )
        return len(out)

    def get_navs(
        self,
        portfolio_id: uuid.UUID,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> list[NavHistory]:
        """Return a portfolio's NAV series, ordered ascending by date.

        Args:
            portfolio_id: Portfolio to query.
            start: Inclusive lower date bound; ``None`` leaves it open.
            end: Inclusive upper date bound; ``None`` leaves it open.

        Returns:
            Matching :class:`NavHistory` rows ordered by ``date``; empty when none
            match.
        """
        stmt = select(NavHistory).where(NavHistory.portfolio_id == portfolio_id)
        if start is not None:
            stmt = stmt.where(NavHistory.date >= start)
        if end is not None:
            stmt = stmt.where(NavHistory.date <= end)
        stmt = stmt.order_by(NavHistory.date)
        return list(self.session.execute(stmt).scalars().all())

    def has_foreign_benchmark(
        self, portfolio_id: uuid.UUID, benchmark: str | None
    ) -> bool:
        """Whether any persisted benchmark level is not known to be ``benchmark``'s.

        A level counts as foreign when it is tagged with another ticker, or untagged
        (written before the identity was recorded, so it cannot be trusted). Rows
        without a level cannot mix benchmarks and are ignored. The daily NAV mark uses
        this to decide that an incremental re-mark would leave a stitched series.

        Args:
            portfolio_id: Portfolio to query.
            benchmark: The mandate's current benchmark ticker, or ``None`` when it
                has none (any persisted level is then foreign).

        Returns:
            ``True`` when at least one persisted level needs re-marking.
        """
        conditions = [
            NavHistory.portfolio_id == portfolio_id,
            NavHistory.benchmark.is_not(None),
        ]
        if benchmark is not None:
            # ``!=`` alone would skip NULL tags (NULL != x is NULL), so test them too.
            conditions.append(
                or_(
                    NavHistory.benchmark_ticker.is_(None),
                    NavHistory.benchmark_ticker != benchmark,
                )
            )
        stmt = select(NavHistory.id).where(*conditions).limit(1)
        return self.session.execute(stmt).first() is not None

    def last_nav_date(self, portfolio_id: uuid.UUID) -> dt.date | None:
        """Return the most recent persisted NAV date for a portfolio, or ``None``.

        Lets the daily mark run incrementally — recomputing only from near this
        date instead of rewriting the whole history — without loading the series.

        Args:
            portfolio_id: Portfolio to query.

        Returns:
            The maximum ``date`` persisted for the portfolio, or ``None`` when it
            has never been marked.
        """
        stmt = select(func.max(NavHistory.date)).where(
            NavHistory.portfolio_id == portfolio_id
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def last_nav_before(
        self, portfolio_id: uuid.UUID, before: dt.date
    ) -> NavHistory | None:
        """Return the newest persisted NAV row strictly earlier than ``before``.

        The incremental mark never re-emits bars before its lookback window, so this
        row is the boundary the stored history is trusted up to; comparing it with a
        fresh computation detects history persisted under an older NAV convention.

        Args:
            portfolio_id: Portfolio to query.
            before: Exclusive upper date bound.

        Returns:
            The latest :class:`NavHistory` row with ``date < before``, or ``None``
            when the portfolio has no earlier row.
        """
        stmt = (
            select(NavHistory)
            .where(NavHistory.portfolio_id == portfolio_id, NavHistory.date < before)
            .order_by(NavHistory.date.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalars().first()


__all__ = ["NavHistoryRepository"]
