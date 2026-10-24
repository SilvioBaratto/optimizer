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

import hashlib
import json
import uuid
from datetime import date
from typing import Any

from portopt_db.models import PaperOrder
from portopt_db.repository import RepositoryBase
from sqlalchemy import select


def weights_hash(weights: dict[str, float]) -> str:
    """sha256 of the canonical weights JSON — stable across calls, order-free.

    The weights leg of the ticket idempotency key, shared by ``place_orders`` (to
    find-or-create) and by readers that need the ticket an approved run placed.

    Args:
        weights: Ticker to target weight. Insertion order is irrelevant: keys are
            sorted before hashing, so equal mappings always hash equal.

    Returns:
        The 64-character lowercase hex digest. Values must be JSON-serialisable
        (``float``/``int``); anything else raises ``TypeError`` from ``json.dumps``.
    """
    canonical = json.dumps(weights, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

    def list_for_portfolio(self, portfolio_id: uuid.UUID) -> list[PaperOrder]:
        """Return all tickets for a portfolio, newest first (by ``created_at``).

        The read surface behind the report/transcript views (O2 = Option A, SPEC
        §8). Portfolio-scoped only — ``paper_orders`` carries no ``run_id`` column.
        """
        stmt = (
            select(PaperOrder)
            .where(PaperOrder.portfolio_id == portfolio_id)
            .order_by(PaperOrder.created_at.desc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def latest_on_or_before(
        self, portfolio_id: uuid.UUID, asof: date
    ) -> PaperOrder | None:
        """Return the newest ticket decided at or before ``asof``, or ``None``.

        The book a new ticket rebalances *from*: ordered by decision bar, then fill
        bar, then insertion time so a same-bar re-proposal with different weights
        still resolves deterministically.

        Args:
            portfolio_id: Portfolio to query.
            asof: Inclusive upper bound on the ticket's decision bar.
        """
        stmt = (
            select(PaperOrder)
            .where(PaperOrder.portfolio_id == portfolio_id, PaperOrder.asof <= asof)
            .order_by(
                PaperOrder.asof.desc(),
                PaperOrder.fill_date.desc(),
                PaperOrder.created_at.desc(),
            )
        )
        return self.session.execute(stmt).scalars().first()

    def latest_for_portfolio(self, portfolio_id: uuid.UUID) -> PaperOrder | None:
        """Return the most recent ticket for a portfolio, or ``None`` when it has none.

        Only the run being approved right now can assume its ticket is the newest by
        ``created_at`` (O2); a reader of an arbitrary run resolves its ticket with
        :meth:`get_by_key` and :func:`weights_hash` instead.
        """
        stmt = (
            select(PaperOrder)
            .where(PaperOrder.portfolio_id == portfolio_id)
            .order_by(PaperOrder.created_at.desc())
        )
        return self.session.execute(stmt).scalars().first()


__all__ = ["OrderRepository", "weights_hash"]
