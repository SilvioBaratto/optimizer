"""``AgentRunRepository`` — fund-side behavior over the audit models.

The ``agent_runs`` / ``agent_decisions`` *models* live in ``portopt_db`` (shared
schema, Alembic-owned); their *behavior* lives here, mirroring the
``background_jobs`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session (D1). No ``commit``: the caller owns the transaction boundary.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

from portopt_db.models import AgentDecision, AgentRun
from portopt_db.repository import RepositoryBase
from sqlalchemy import func, select


class AgentRunRepository(RepositoryBase):
    """Create/append/read the allocator audit trail on an injected sync session."""

    def create_run(
        self,
        *,
        portfolio_id: uuid.UUID | None,
        asof: date,
        seed: int | None,
        universe: Sequence[str],
        optimizer_config: dict[str, Any],
        status: str = "pending",
    ) -> AgentRun:
        """Insert a pending run and return the flushed row (id/defaults populated)."""
        run = AgentRun(
            portfolio_id=portfolio_id,
            asof=asof,
            seed=seed,
            universe=list(universe),
            optimizer_config=optimizer_config,
            status=status,
        )
        self.session.add(run)
        self.session.flush()
        self.session.refresh(run)
        return run

    def append_decision(
        self,
        run_id: uuid.UUID,
        *,
        agent: str,
        step: str,
        constraint_set: dict[str, Any] | None = None,
        views: dict[str, Any] | None = None,
        llm_prompt: str | None = None,
        llm_response: str | None = None,
        llm_response_hash: str | None = None,
        hitl_decision: dict[str, Any] | None = None,
    ) -> AgentDecision:
        """Append a decision with the next 0-based ``decision_index`` for the run.

        The index is derived from a ``count()`` of existing decisions. Safe under
        the D1 model (one sync writer per run appends sequentially); a concurrent
        appender would collide on ``uq_agent_decision_run_index``, which is the
        backstop (IntegrityError, not a duplicated index). The caller owns the
        transaction boundary and rolls back a failed session.
        """
        next_index = self.session.execute(
            select(func.count()).where(AgentDecision.run_id == run_id)
        ).scalar_one()
        decision = AgentDecision(
            run_id=run_id,
            decision_index=next_index,
            agent=agent,
            step=step,
            constraint_set=constraint_set,
            views=views,
            llm_prompt=llm_prompt,
            llm_response=llm_response,
            llm_response_hash=llm_response_hash,
            hitl_decision=hitl_decision,
        )
        self.session.add(decision)
        self.session.flush()
        self.session.refresh(decision)
        return decision

    def finalize_run(
        self,
        run_id: uuid.UUID,
        *,
        weights: dict[str, float],
        status: str = "completed",
        finished_at: datetime | None = None,
    ) -> AgentRun | None:
        """Stamp the run's output weights + terminal status; ``None`` if not found."""
        run = self.get_run(run_id)
        if run is None:
            return None
        run.weights = weights
        run.status = status
        run.finished_at = finished_at or datetime.now(UTC)
        self.session.flush()
        self.session.refresh(run)
        return run

    def get_run(self, run_id: uuid.UUID) -> AgentRun | None:
        """Return the run by id, or ``None``."""
        return self.session.get(AgentRun, run_id)

    def list_runs_for_portfolio(self, portfolio_id: uuid.UUID) -> list[AgentRun]:
        """All runs for a portfolio, newest first."""
        stmt = (
            select(AgentRun)
            .where(AgentRun.portfolio_id == portfolio_id)
            .order_by(AgentRun.created_at.desc())
        )
        return list(self.session.execute(stmt).scalars().all())


__all__ = ["AgentRunRepository"]
