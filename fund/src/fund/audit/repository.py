"""``AgentRunRepository`` — fund-side behavior over the audit models.

The ``agent_runs`` / ``agent_decisions`` *models* live in ``portopt_db`` (shared
schema, Alembic-owned); their *behavior* lives here, mirroring the
``background_jobs`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session (D1). No ``commit``: the caller owns the transaction boundary.
"""

from __future__ import annotations

import json
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

    def latest_optimizer_weights(self, run_id: uuid.UUID) -> dict[str, float]:
        """The single authoritative read of a run's optimizer weights.

        Returns the allocator's latest audited ``optimize_portfolio`` weights (the
        skfolio output logged inside the bound tool). Both the paper ticket and the
        finalised ``agent_runs.weights`` derive from THESE, never from what the
        PM/LLM passed — keeping ``optimize_portfolio`` the single source of weights.
        Returns ``{}`` when the allocator never produced a proposal (or the payload
        is missing/unparseable).
        """
        stmt = (
            select(AgentDecision)
            .where(
                AgentDecision.run_id == run_id,
                AgentDecision.agent == "allocator",
                AgentDecision.step == "optimize_portfolio",
            )
            .order_by(AgentDecision.decision_index.desc())
        )
        row = self.session.execute(stmt).scalars().first()
        return self.decode_weights(row)

    @staticmethod
    def decode_weights(decision: AgentDecision | None) -> dict[str, float]:
        """Decode one allocator decision's logged optimizer weights, or ``{}``.

        Parses the ``{"weights": {ticker: float}}`` payload the bound
        ``optimize_portfolio`` tool logs in ``llm_response`` (the skfolio output, never
        PM/LLM-passed). The per-step drill-down reads a specific selected proposal with
        this; :meth:`latest_optimizer_weights` reads the run's newest. Returns ``{}``
        when ``decision`` is ``None``, carries no response, or the payload is
        missing/unparseable.
        """
        if decision is None or not decision.llm_response:
            return {}
        try:
            payload = json.loads(decision.llm_response)
        except (ValueError, TypeError):
            return {}
        weights = payload.get("weights") or {}
        return {str(k): float(v) for k, v in weights.items()}

    def latest_profiler_answers(self, run_id: uuid.UUID) -> str | None:
        """The typed ``MiFIDAnswers`` JSON the profiler logged at ``normalize_answers``.

        The single authoritative read of a profiler run's interpreted answers — the
        cross-process :func:`~fund.agents.profiler.resume_profiler` recovers them
        from here. The MiFID→knob mapping is a *pure* function of these answers, so
        the ``ConstraintSet`` need not be persisted before the HITL gate: it is
        re-derived on resume. Returns ``None`` when the step never ran or its payload
        is missing/empty (a caller treats that as an unresumable run).
        """
        stmt = (
            select(AgentDecision)
            .where(
                AgentDecision.run_id == run_id,
                AgentDecision.agent == "profiler",
                AgentDecision.step == "normalize_answers",
            )
            .order_by(AgentDecision.decision_index.desc())
        )
        row = self.session.execute(stmt).scalars().first()
        if row is None or not row.llm_response:
            return None
        return row.llm_response

    def set_thread_id(self, run_id: uuid.UUID, thread_id: str) -> AgentRun | None:
        """Stamp the run's LangGraph checkpointer thread id; ``None`` if not found.

        Defaults to ``str(run_id)`` at run start so the resume path can rebuild
        the PM agent against the same thread.
        """
        run = self.get_run(run_id)
        if run is None:
            return None
        run.thread_id = thread_id
        self.session.flush()
        self.session.refresh(run)
        return run

    def mark_paused(self, run_id: uuid.UUID) -> AgentRun | None:
        """Flip the run to ``"paused"`` at a HITL gate; ``None`` if not found."""
        run = self.get_run(run_id)
        if run is None:
            return None
        run.status = "paused"
        self.session.flush()
        self.session.refresh(run)
        return run

    def mark_responded(self, run_id: uuid.UUID) -> AgentRun | None:
        """Flip the run to ``"responded"`` after a barge-in re-gate; ``None`` if absent.

        Mirrors :meth:`mark_paused`. A ``respond`` decision threads the adviser's
        directive back to the PM, which may re-gate (call ``place_orders`` again): the
        run then stays alive as ``"responded"`` — distinct from ``"paused"`` so an
        observer tells a plain pause from a barged-in re-gate — and remains in
        :meth:`list_paused_runs`, still awaiting approval.
        """
        run = self.get_run(run_id)
        if run is None:
            return None
        run.status = "responded"
        self.session.flush()
        self.session.refresh(run)
        return run

    def list_paused_runs(self, portfolio_id: uuid.UUID | None = None) -> list[AgentRun]:
        """Runs awaiting a human decision, newest first.

        Returns rows at a live HITL gate — ``status`` in ``("paused", "responded")`` —
        so a barged-in run that re-gated stays visible and approvable alongside a plain
        pause. Optionally scoped to one ``portfolio_id``. This is the pending-HITL query
        the observers and the resume driver read.
        """
        stmt = select(AgentRun).where(AgentRun.status.in_(("paused", "responded")))
        if portfolio_id is not None:
            stmt = stmt.where(AgentRun.portfolio_id == portfolio_id)
        stmt = stmt.order_by(AgentRun.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

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

    def list_portfolio_ids(self) -> list[uuid.UUID]:
        """Distinct non-null portfolio ids that have at least one run.

        ``AgentRun.portfolio_id`` is nullable — a fund-wide/backtest run carries
        no portfolio — so those rows are excluded. Order is unspecified; the
        caller (``observe.list_portfolios``) dedups across sources and sorts.
        """
        stmt = (
            select(AgentRun.portfolio_id)
            .where(AgentRun.portfolio_id.is_not(None))
            .distinct()
        )
        # is_not(None) drops NULLs in SQL; re-narrow the Optional for the checker.
        return [pid for pid in self.session.execute(stmt).scalars() if pid is not None]

    def list_decisions(
        self,
        *,
        portfolio_id: uuid.UUID | None = None,
        step: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[AgentDecision, uuid.UUID | None]]:
        """Agent decisions joined to their run's portfolio, newest first.

        The cross-portfolio read the Audit view unions with ``human_audit_log``: each
        row carries the owning ``portfolio_id`` (from the joined run, nullable for a
        fund-wide run). Every filter is optional and ANDs with the rest.

        Args:
            portfolio_id: Keep only decisions whose run targets this portfolio.
            step: Keep only decisions at this step (the decision "type").
            start: Lower ``created_at`` bound (inclusive).
            end: Upper ``created_at`` bound (inclusive).
            limit: Maximum rows to return.

        Returns:
            ``(decision, portfolio_id)`` pairs ordered by ``created_at`` descending,
            ties broken per run by ``decision_index`` descending.
        """
        stmt = select(AgentDecision, AgentRun.portfolio_id).join(
            AgentRun, AgentDecision.run_id == AgentRun.id
        )
        if portfolio_id is not None:
            stmt = stmt.where(AgentRun.portfolio_id == portfolio_id)
        if step is not None:
            stmt = stmt.where(AgentDecision.step == step)
        if start is not None:
            stmt = stmt.where(AgentDecision.created_at >= start)
        if end is not None:
            stmt = stmt.where(AgentDecision.created_at <= end)
        # Rows stamped inside one transaction before per-statement clocks (and
        # any same-microsecond pair) tie on created_at; keep a run's steps
        # grouped and newest-first rather than leaving the tie to the planner.
        stmt = stmt.order_by(
            AgentDecision.created_at.desc(),
            AgentDecision.run_id,
            AgentDecision.decision_index.desc(),
        ).limit(limit)
        return [(dec, pid) for dec, pid in self.session.execute(stmt).all()]


__all__ = ["AgentRunRepository"]
