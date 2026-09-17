"""``OrderRepository`` — fund-side behavior over the ``paper_orders`` model.

The ``paper_orders`` *model* lives in ``portopt_db`` (shared schema, Alembic-
owned); its *behavior* lives here, mirroring the ``agent_runs`` / ``background_jobs``
model/repo split. Sits on ``portopt_db.repository``'s ``RepositoryBase`` and opens
no session of its own — the caller injects a sync session (D1). No ``commit``: the
caller owns the transaction boundary.

The idempotency key is ``(portfolio_id, asof, weights_hash)`` — :meth:`get_by_key`
finds an existing ticket so ``place_orders`` can be re-run safely under the HITL
``Command(resume=…)`` gotcha (SPEC D3) without double-placing.

Concurrency: the get-then-create is find-or-create, not atomic. That is safe under
the D1 model (one daemon, one sync writer per run — see ``CLAUDE.md`` reaper is
host-scoped). Should two writers ever race the same key, the DB-level
``UNIQUE(portfolio_id, asof, weights_hash)`` is the backstop: the second insert
raises ``IntegrityError``, which ``tool_envelope`` degrades to ``{ok: false}``
rather than double-placing. The caller owns the transaction boundary (D1), so it —
not this repo — rolls back that failed session.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from portopt_db.models import PaperOrder
from portopt_db.repository import RepositoryBase
from sqlalchemy import select


class OrderRepository(RepositoryBase):
    """Find-or-create simulated paper-order tickets on an injected sync session."""

    def get_by_key(
        self,
        *,
        portfolio_id: uuid.UUID,
        asof: date,
        weights_hash: str,
    ) -> PaperOrder | None:
        """Return the ticket for the idempotency key, or ``None``."""
        stmt = select(PaperOrder).where(
            PaperOrder.portfolio_id == portfolio_id,
            PaperOrder.asof == asof,
            PaperOrder.weights_hash == weights_hash,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create(
        self,
        *,
        portfolio_id: uuid.UUID,
        asof: date,
        weights_hash: str,
        weights: dict[str, float],
        fill_date: date,
        lines: list[dict[str, Any]],
        notional: float,
        total_commission: float,
        total_slippage_cost: float,
        status: str = "filled",
    ) -> PaperOrder:
        """Insert a ticket and return the flushed row (id/defaults populated)."""
        order = PaperOrder(
            portfolio_id=portfolio_id,
            asof=asof,
            weights_hash=weights_hash,
            weights=weights,
            fill_date=fill_date,
            lines=lines,
            notional=notional,
            total_commission=total_commission,
            total_slippage_cost=total_slippage_cost,
            status=status,
        )
        self.session.add(order)
        # `flush` populates the client-side UUID default; a `refresh` here would
        # re-read the row, and `load_on_pk_identity` mis-maps this model's shape
        # (pg UUID + JSON + Float) on SQLite. Every attribute is already set.
        self.session.flush()
        return order


__all__ = ["OrderRepository"]
