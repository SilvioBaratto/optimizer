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
the D1 model (one daemon per database, one sync writer per run; see the reaper
gotcha in ``CLAUDE.md``). Should two writers ever race the same key, the DB-level
``UNIQUE(portfolio_id, asof, weights_hash)`` is the backstop: the second insert
raises ``IntegrityError``, which ``tool_envelope`` degrades to ``{ok: false}``
rather than double-placing. The caller owns the transaction boundary (D1), so it —
not this repo — rolls back that failed session.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from portopt_db.base import ClockTimestamp
from portopt_db.models import PaperOrder
from portopt_db.repository import RepositoryBase
from sqlalchemy import Select, select, update


@dataclass(frozen=True)
class FilledTicket:
    """One paper ticket's id, fill bar, ``created_at``, notional and lines.

    ``frozen`` blocks attribute assignment, not edits to the dicts in ``lines``:
    they are the loaded row's own line dicts, so editing one edits the loaded row.

    Attributes:
        id: The ticket's primary key.
        fill_date: The ticket's stored fill bar; a line may carry its own.
        created_at: When the ticket was stored, or last re-stamped by
            :meth:`OrderRepository.restamp_created_at`.
        notional: The paper notional the weights were sized against.
        lines: The stored per-ticker fill lines, in stored order, as a tuple of the
            row's line dicts.
    """

    id: uuid.UUID
    fill_date: date
    created_at: datetime
    notional: float
    lines: tuple[dict[str, Any], ...]


def _filled_ticket(order: PaperOrder) -> FilledTicket:
    return FilledTicket(
        id=order.id,
        fill_date=order.fill_date,
        created_at=order.created_at,
        notional=order.notional,
        lines=tuple(order.lines),
    )


def _ticket_rank() -> tuple[Any, ...]:
    """Sort keys that rank paper tickets from oldest to newest, with no ties.

    Tickets for the same decision bar share ``asof`` and ``fill_date`` but may carry
    different weights (a re-run after a reject, a sweep plus a manual run), so those
    two keys alone leave their order to the database. ``created_at`` then ``id``
    make the newest ticket supersede the others deterministically. NAV and the
    holdings table read the ascending order (the holdings take its last ticket as the
    current book) and ``marked_weights`` reads the newest ticket, so under the same
    fill bound all three treat the same ticket as the newest.
    """
    return (
        PaperOrder.fill_date,
        PaperOrder.asof,
        PaperOrder.created_at,
        PaperOrder.id,
    )


def _filled_through(
    portfolio_id: uuid.UUID, through: date | None
) -> Select[tuple[PaperOrder]]:
    stmt = select(PaperOrder).where(PaperOrder.portfolio_id == portfolio_id)
    if through is not None:
        stmt = stmt.where(PaperOrder.fill_date <= through)
    return stmt


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

    def filled_tickets(
        self, portfolio_id: uuid.UUID, *, filled_through: date | None
    ) -> list[FilledTicket]:
        """Return a portfolio's tickets oldest first, the chain NAV is built from.

        Ranked by fill bar, then decision bar, then ``created_at``, then ``id``.

        Args:
            portfolio_id: Portfolio to query.
            filled_through: Inclusive upper bound on ``fill_date``; ``None`` reads
                every ticket.

        Returns:
            The tickets in rank order; empty when none matches.
        """
        stmt = _filled_through(portfolio_id, filled_through).order_by(*_ticket_rank())
        return [_filled_ticket(order) for order in self.session.execute(stmt).scalars()]

    def latest_filled_ticket(
        self, portfolio_id: uuid.UUID, *, filled_through: date | None
    ) -> FilledTicket | None:
        """Return the ticket that ranks last in :meth:`filled_tickets`.

        Args:
            portfolio_id: Portfolio to query.
            filled_through: Inclusive upper bound on ``fill_date``; ``None`` reads
                every ticket.

        Returns:
            The newest ticket, or ``None`` when none matches.
        """
        stmt = _filled_through(portfolio_id, filled_through).order_by(
            *(column.desc() for column in _ticket_rank())
        )
        order = self.session.execute(stmt).scalars().first()
        return None if order is None else _filled_ticket(order)

    def restamp_created_at(self, order_id: uuid.UUID) -> None:
        """Set a ticket's ``created_at`` to the database clock's current time.

        Issued as a bulk ``UPDATE`` with ``synchronize_session=False``, so a
        ``PaperOrder`` already loaded in the session keeps the ``created_at`` it was
        loaded with. It calls neither ``flush`` nor ``commit``: the caller owns the
        transaction.

        Args:
            order_id: The ticket to re-stamp.
        """
        self.session.execute(
            update(PaperOrder)
            .where(PaperOrder.id == order_id)
            .values(created_at=ClockTimestamp())
            .execution_options(synchronize_session=False)
        )


__all__ = ["FilledTicket", "OrderRepository", "weights_hash"]
