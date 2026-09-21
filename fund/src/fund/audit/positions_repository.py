"""``PositionRepository`` — fund-side behavior over the ``positions`` model.

The ``positions`` *model* lives in ``portopt_db`` (shared schema, Alembic-owned);
its *behavior* lives here, mirroring the ``agent_runs`` / ``paper_orders`` /
``portfolio_mandates`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session (D1). No ``commit``: the caller owns the transaction boundary.

``positions`` is a **current snapshot**, not a time series: one row per held
``(portfolio_id, ticker)`` after the latest approved fill (history stays derivable
from ``paper_orders``). :meth:`set_holdings` therefore **replaces** the whole
snapshot for a portfolio (delete-then-insert) rather than appending — written only
from ``_finalize_hitl`` on approve.

Holdings I/O shape is **O1 = Option B** (SPEC §8, resolved 2026-09-21):
:meth:`set_holdings` takes ``holdings`` — a list of per-ticker row dicts carrying
``ticker``/``weight`` (+ optional ``shares``/``notional``) — and :meth:`get_holdings`
returns the full ``list[Position]``. Weight-only readers (e.g. ``observe.drift_l1``)
build ``{p.ticker: p.weight for p in get_holdings(...)}``.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from portopt_db.models import Position
from portopt_db.repository import RepositoryBase
from sqlalchemy import delete, select


class PositionRepository(RepositoryBase):
    """Replace/read a portfolio's current-snapshot holdings on an injected session."""

    def set_holdings(
        self,
        portfolio_id: uuid.UUID,
        holdings: list[dict[str, Any]],
        *,
        asof: date,
        paper_order_id: uuid.UUID | None = None,
    ) -> None:
        """Replace the portfolio's current holdings with ``holdings``.

        Delete-then-insert: every existing row for ``portfolio_id`` is removed and
        the ``holdings`` rows inserted fresh, so the snapshot is exactly the new
        set (one row per ticker, ``UNIQUE(portfolio_id, ticker)``). Each ``holdings``
        row is ``{"ticker": str, "weight": float, "shares"?: float, "notional"?:
        float}``; ``asof`` and ``paper_order_id`` stamp every row. No ``commit`` —
        the caller owns the transaction.
        """
        self.session.execute(
            delete(Position)
            .where(Position.portfolio_id == portfolio_id)
            .execution_options(synchronize_session=False)
        )
        for row in holdings:
            self.session.add(
                Position(
                    portfolio_id=portfolio_id,
                    ticker=row["ticker"],
                    weight=row["weight"],
                    shares=row.get("shares"),
                    notional=row.get("notional"),
                    asof=asof,
                    paper_order_id=paper_order_id,
                )
            )
        self.session.flush()

    def get_holdings(self, portfolio_id: uuid.UUID) -> list[Position]:
        """Return the portfolio's current-snapshot rows, ordered by ticker."""
        stmt = (
            select(Position)
            .where(Position.portfolio_id == portfolio_id)
            .order_by(Position.ticker)
        )
        return list(self.session.execute(stmt).scalars().all())


__all__ = ["PositionRepository"]
