"""Repository for per-portfolio rebalance overlay rows (``portfolio_journal``).

Uses ``index_elements`` upserts keyed on ``(portfolio_id, as_of)``; the caller
owns the session and controls commit/rollback.  Never opens a session internally.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import select

from portopt_db.models.orders.portfolio_journal import PortfolioJournal
from portopt_db.repository import RepositoryBase

_UPDATE_COLUMNS = [
    "run_id",
    "trades",
    "allocation",
    "drift",
    "narrative",
    "updated_at",
]


class PortfolioJournalRepository(RepositoryBase):
    """Persists and retrieves per-portfolio rebalance overlay rows.

    All writes go through the idempotent ``_upsert`` path keyed on
    ``(portfolio_id, as_of)``.  Reads never issue commits.
    """

    def upsert(
        self,
        portfolio_id: uuid.UUID,
        as_of: date,
        *,
        run_id: uuid.UUID | None,
        trades: dict[str, Any] | None,
        allocation: dict[str, Any] | None,
        drift: dict[str, Any] | None,
        narrative: str,
    ) -> int:
        """Write or refresh the overlay for one portfolio/rebalance-bar pair.

        Idempotent: re-upserting the same ``(portfolio_id, as_of)`` overwrites
        all data columns and advances ``updated_at``.

        Args:
            portfolio_id: Opaque portfolio identifier; no FK enforced.
            as_of: Rebalance-bar date forming the composite key with
                ``portfolio_id``.
            run_id: Soft link to ``agent_runs.id``; pass ``None`` when writing
                outside a fund run.
            trades: Per-ticker fill detail from committed ``paper_orders.lines``.
            allocation: Target weight snapshot for this run.
            drift: Drift vs prior holdings and which bands crossed.
            narrative: Deterministic template string for agent injection.

        Returns:
            Number of rows processed (always 1 when a row is supplied).
        """
        row: dict[str, Any] = {
            "id": uuid.uuid4(),
            "portfolio_id": portfolio_id,
            "as_of": as_of,
            "run_id": run_id,
            "trades": trades,
            "allocation": allocation,
            "drift": drift,
            "narrative": narrative,
        }
        return self._upsert(
            PortfolioJournal,
            [row],
            index_elements=["portfolio_id", "as_of"],
            update_columns=_UPDATE_COLUMNS,
        )

    def list_for_portfolio(
        self,
        portfolio_id: uuid.UUID,
        *,
        since: date | None = None,
        limit: int | None = None,
    ) -> Sequence[PortfolioJournal]:
        """Return overlay rows for a portfolio, newest first.

        Args:
            portfolio_id: Portfolio to filter by.
            since: When provided, only rows with ``as_of >= since`` are returned.
            limit: Maximum number of rows to return; ``None`` returns all.

        Returns:
            Sequence of matching rows ordered by ``as_of`` descending, possibly
            empty.
        """
        stmt = (
            select(PortfolioJournal)
            .where(PortfolioJournal.portfolio_id == portfolio_id)
            .order_by(PortfolioJournal.as_of.desc())
        )
        if since is not None:
            stmt = stmt.where(PortfolioJournal.as_of >= since)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.execute(stmt).scalars().all()
