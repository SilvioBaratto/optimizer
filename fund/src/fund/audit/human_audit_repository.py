"""``HumanAuditLogRepository`` — fund-side behavior over ``human_audit_log``.

The ``human_audit_log`` *model* lives in ``portopt_db`` (shared schema,
Alembic-owned); its *behavior* lives here, mirroring the ``agent_runs`` /
``portfolio_mandates`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session (D1). No ``commit``: the caller owns the transaction boundary.

This is the append-only home for every human action the shell's unified action
path funnels through; :meth:`append` is the single write and :meth:`list_recent`
the fund-wide read the Audit view merges with ``agent_decisions``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from portopt_db.models import HumanAuditLog
from portopt_db.repository import RepositoryBase
from sqlalchemy import select


class HumanAuditLogRepository(RepositoryBase):
    """Append/read the human-decision audit trail on an injected sync session."""

    def append(
        self,
        *,
        action: str,
        portfolio_id: uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
        actor: str = "human",
        target: str | None = None,
        prev_value: Any | None = None,
        new_value: Any | None = None,
        source: str | None = None,
        override_flag: bool = False,
        suitability_flag: bool = False,
        idempotency_key: str | None = None,
        rationale: str | None = None,
    ) -> HumanAuditLog:
        """Append one human action and return the flushed row.

        When ``idempotency_key`` is supplied and a row already carries it, that
        existing row is returned unchanged and nothing is written — so a retried
        action (e.g. a double-pressed Approve) logs once. Un-keyed actions never
        de-dup. No ``commit``: the caller owns the transaction.

        Args:
            action: The action kind (``approve`` / ``reject`` / ``killswitch`` /
                ``mandate_edit`` / ``guardrail_override``).
            portfolio_id: Portfolio the action targets, or ``None`` for a
                fund-wide action.
            run_id: Run the action resolves, when applicable.
            actor: Who acted; defaults to ``"human"``.
            target: The field, gate, or run the action touched.
            prev_value: Prior value for an edit/override (arbitrary JSON).
            new_value: New value for an edit/override (arbitrary JSON).
            source: Provenance of the prior value (e.g. ``"profile"``).
            override_flag: ⚑ a guardrail edited past its MiFID band.
            suitability_flag: ⚠ outside the suitability band.
            idempotency_key: Dedup key; a repeat returns the first row.
            rationale: Free-text reason recorded with the action.

        Returns:
            The flushed row (its own existing row on an idempotent repeat).
        """
        if idempotency_key is not None:
            existing = self.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing
        row = HumanAuditLog(
            action=action,
            portfolio_id=portfolio_id,
            run_id=run_id,
            actor=actor,
            target=target,
            prev_value=prev_value,
            new_value=new_value,
            source=source,
            override_flag=override_flag,
            suitability_flag=suitability_flag,
            idempotency_key=idempotency_key,
            rationale=rationale,
        )
        self.session.add(row)
        self.session.flush()
        self.session.refresh(row)
        return row

    def get_by_idempotency_key(self, idempotency_key: str) -> HumanAuditLog | None:
        """Return the row carrying ``idempotency_key``, or ``None``."""
        stmt = select(HumanAuditLog).where(
            HumanAuditLog.idempotency_key == idempotency_key
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_recent(
        self,
        *,
        portfolio_id: uuid.UUID | None = None,
        action: str | None = None,
        actor: str | None = None,
        overrides_only: bool = False,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        limit: int | None = 200,
    ) -> list[HumanAuditLog]:
        """Human actions newest first, with composable filters.

        Every filter is optional and ANDs with the rest; passing none returns the
        most recent ``limit`` rows fund-wide. ``overrides_only`` keeps only ⚑
        override rows. ``start`` / ``end`` bound ``created_at`` inclusively.

        Args:
            portfolio_id: Keep only this portfolio's rows.
            action: Keep only this action kind.
            actor: Keep only this actor's rows.
            overrides_only: Keep only rows flagged as overrides.
            start: Lower ``created_at`` bound (inclusive).
            end: Upper ``created_at`` bound (inclusive).
            limit: Maximum rows to return; ``None`` for no cap.

        Returns:
            Matching rows ordered by ``created_at`` descending.
        """
        stmt = select(HumanAuditLog)
        if portfolio_id is not None:
            stmt = stmt.where(HumanAuditLog.portfolio_id == portfolio_id)
        if action is not None:
            stmt = stmt.where(HumanAuditLog.action == action)
        if actor is not None:
            stmt = stmt.where(HumanAuditLog.actor == actor)
        if overrides_only:
            stmt = stmt.where(HumanAuditLog.override_flag.is_(True))
        if start is not None:
            stmt = stmt.where(HumanAuditLog.created_at >= start)
        if end is not None:
            stmt = stmt.where(HumanAuditLog.created_at <= end)
        stmt = stmt.order_by(HumanAuditLog.created_at.desc()).limit(limit)
        return list(self.session.execute(stmt).scalars().all())


__all__ = ["HumanAuditLogRepository"]
