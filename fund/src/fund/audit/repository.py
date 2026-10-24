"""``AgentRunRepository`` — fund-side behavior over the audit models.

The ``agent_runs`` / ``agent_decisions`` *models* live in ``portopt_db`` (shared
schema, Alembic-owned); their *behavior* lives here, mirroring the
``background_jobs`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session (D1). No ``commit``: the caller owns the transaction boundary.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from typing import Any, TypeVar

from portopt_db.models import AgentDecision, AgentRun
from portopt_db.repository import RepositoryBase
from sqlalchemy import Select, func, select

from fund.audit.orders_repository import weights_hash

_T = TypeVar("_T")

# Rows fetched per round trip when a caller streams a history and usually stops
# after the first few.
_STREAM_BATCH = 20

# Statuses at a live HITL gate: the only runs an adviser decision may resolve. A
# "pending" row is a run still being driven (or one that crashed before the gate), so
# it is deliberately absent.
RESUMABLE_RUN_STATUSES: tuple[str, ...] = ("paused", "responded")


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
        error: str | None = None,
    ) -> AgentRun | None:
        """Stamp the run's output weights + terminal status; ``None`` if not found.

        ``error`` records why a run ended ``"failed"``; successful finalisations leave
        it unset.
        """
        run = self.get_run(run_id)
        if run is None:
            return None
        run.weights = weights
        run.status = status
        run.finished_at = finished_at or datetime.now(UTC)
        if error is not None:
            run.error = error
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

    @staticmethod
    def weights_digest(weights: dict[str, float]) -> str:
        """sha256 of the canonical weights JSON, order-free.

        The same canonical form as a paper ticket's ``weights_hash`` (shared through
        :func:`fund.audit.orders_repository.weights_hash`), so a risk verdict and the
        ticket it cleared cross-reference.
        """
        return weights_hash(weights)

    def append_risk_verdict(
        self,
        run_id: uuid.UUID,
        *,
        weights: dict[str, float],
        passed: bool,
        violations: Sequence[dict[str, Any]],
        not_enforced: Sequence[str] = (),
        source: str = "risk_check",
    ) -> AgentDecision:
        """Record the risk gate's verdict on ``weights`` as an audited decision.

        Args:
            run_id: The run the verdict belongs to.
            weights: The vector the gate evaluated; only its digest is stored, so the
                verdict is bound to exactly these weights.
            passed: Whether the gate passed them.
            violations: The structured breach records (empty when ``passed``).
            not_enforced: Suitability inputs the gate cannot enforce yet, so a pass
                is not read as full suitability.
            source: What produced the verdict (``"risk_check"`` for the bound tool,
                ``"adviser_edit"`` for an adviser's weight edit).

        Returns:
            The appended ``agent="risk"`` / ``step="risk_check"`` decision.
        """
        payload = json.dumps(
            {
                "passed": passed,
                "violations": list(violations),
                "not_enforced": list(not_enforced),
                "weights_hash": self.weights_digest(weights),
                "source": source,
            },
            sort_keys=True,
            default=str,
        )
        return self.append_decision(
            run_id,
            agent="risk",
            step="risk_check",
            llm_response=payload,
            llm_response_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )

    def risk_gate(self, run_id: uuid.UUID, weights: dict[str, float]) -> dict[str, Any]:
        """The run's newest risk verdict, read against exactly ``weights``.

        Args:
            run_id: The run whose verdicts to read.
            weights: The vector the verdict must have judged to count.

        Returns:
            ``{"status", "violations", "not_enforced", "source"}`` where ``status`` is
            ``"passed"`` (the newest verdict passed these weights), ``"failed"`` (it
            judged these weights and found breaches), ``"stale"`` (it judged a
            different vector, so a superseded proposal) or ``"not_run"`` (no usable
            verdict). Anything but ``"passed"`` is a closed gate.
        """
        stmt = (
            select(AgentDecision)
            .where(
                AgentDecision.run_id == run_id,
                AgentDecision.agent == "risk",
                AgentDecision.step == "risk_check",
            )
            .order_by(AgentDecision.decision_index.desc())
        )
        row = self.session.execute(stmt).scalars().first()
        not_run: dict[str, Any] = {
            "status": "not_run",
            "violations": [],
            "not_enforced": [],
            "source": None,
        }
        if row is None or not row.llm_response:
            return not_run
        try:
            verdict = json.loads(row.llm_response)
        except (ValueError, TypeError):
            return not_run
        if not isinstance(verdict, dict):
            return not_run
        if verdict.get("weights_hash") != self.weights_digest(weights):
            status = "stale"
        elif verdict.get("passed") is True:
            status = "passed"
        else:
            status = "failed"
        return {
            "status": status,
            "violations": list(verdict.get("violations") or []),
            "not_enforced": list(verdict.get("not_enforced") or []),
            "source": verdict.get("source"),
        }

    def risk_passed(self, run_id: uuid.UUID, weights: dict[str, float]) -> bool:
        """Whether the run's newest risk verdict passed exactly ``weights``.

        Fails closed: no verdict, an unparseable one, a failed one, or one computed
        on a different vector (a superseded proposal) all return ``False``.
        """
        return self.risk_gate(run_id, weights)["status"] == "passed"

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
        stmt = select(AgentRun).where(AgentRun.status.in_(RESUMABLE_RUN_STATUSES))
        if portfolio_id is not None:
            stmt = stmt.where(AgentRun.portfolio_id == portfolio_id)
        stmt = stmt.order_by(AgentRun.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def newer_completed_rebalance_run(self, run: AgentRun) -> AgentRun | None:
        """The newest completed rebalance run of ``run``'s portfolio that began later.

        "Newer" is ``started_at`` order, so a round opened later on fresher data wins
        over one that merely finished later. Profiler runs are skipped: completing a
        risk profile sets no holdings. A run with no portfolio has nothing to compare
        against. A tie on ``started_at`` is not "newer".

        Returns:
            The most recently started such run, or ``None`` when ``run`` is current.
        """
        if run.portfolio_id is None:
            return None
        stmt = (
            select(AgentRun)
            .where(
                AgentRun.portfolio_id == run.portfolio_id,
                AgentRun.status == "completed",
                AgentRun.started_at > run.started_at,
            )
            .order_by(AgentRun.started_at.desc())
        )
        for candidate in self.session.execute(stmt).scalars():
            if (candidate.optimizer_config or {}).get("step") != "profiler":
                return candidate
        return None

    def get_run(self, run_id: uuid.UUID) -> AgentRun | None:
        """Return the run by id, or ``None``."""
        return self.session.get(AgentRun, run_id)

    def list_runs_for_portfolio(
        self,
        portfolio_id: uuid.UUID,
        *,
        since: date | None = None,
        limit: int | None = None,
    ) -> list[AgentRun]:
        """Runs for a portfolio, newest first.

        Args:
            portfolio_id: Owning portfolio.
            since: Keep only runs whose ``asof`` is on or after this date.
            limit: Maximum rows to return; ``None`` returns the full history.

        Returns:
            The matching runs, ordered by ``created_at`` descending. Decisions are
            not loaded; read them per run when a caller needs them.
        """
        stmt = select(AgentRun).where(AgentRun.portfolio_id == portfolio_id)
        if since is not None:
            stmt = stmt.where(AgentRun.asof >= since)
        stmt = stmt.order_by(AgentRun.created_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def iter_runs_for_portfolio(
        self, portfolio_id: uuid.UUID, *, status: str | None = None
    ) -> Iterator[AgentRun]:
        """Stream a portfolio's runs newest first so a caller can stop early.

        Args:
            portfolio_id: Owning portfolio.
            status: Keep only runs in this status (e.g. ``"completed"``).

        Yields:
            Runs ordered by ``created_at`` descending, fetched in batches. Closing
            the iterator early releases the cursor without reading the tail.
        """
        stmt = select(AgentRun).where(AgentRun.portfolio_id == portfolio_id)
        if status is not None:
            stmt = stmt.where(AgentRun.status == status)
        yield from self._stream(stmt.order_by(AgentRun.created_at.desc()))

    def iter_optimizer_decisions(
        self,
        portfolio_id: uuid.UUID,
        *,
        run_statuses: Sequence[str] | None = None,
    ) -> Iterator[AgentDecision]:
        """Stream the allocator ``optimize_portfolio`` decisions, newest first.

        Spans every run of the portfolio in one query, so the observers can find the
        latest proposal without a per-run lookup.

        Args:
            portfolio_id: Owning portfolio.
            run_statuses: Keep only decisions of runs in one of these statuses (e.g.
                the live HITL gate pair); ``None`` keeps every run's.

        Yields:
            Decisions ordered newest run first, then ``decision_index`` descending
            within a run, fetched in batches.
        """
        stmt = (
            select(AgentDecision)
            .join(AgentRun, AgentDecision.run_id == AgentRun.id)
            .where(
                AgentRun.portfolio_id == portfolio_id,
                AgentDecision.agent == "allocator",
                AgentDecision.step == "optimize_portfolio",
            )
            .order_by(AgentRun.created_at.desc(), AgentDecision.decision_index.desc())
        )
        if run_statuses is not None:
            stmt = stmt.where(AgentRun.status.in_(run_statuses))
        yield from self._stream(stmt)

    def _stream(self, stmt: Select[tuple[_T]]) -> Iterator[_T]:
        result = self.session.execute(stmt.execution_options(yield_per=_STREAM_BATCH))
        try:
            yield from result.scalars()
        finally:
            result.close()

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
        limit: int | None = 200,
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
            limit: Maximum rows to return; ``None`` for no cap.

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


__all__ = ["RESUMABLE_RUN_STATUSES", "AgentRunRepository"]
