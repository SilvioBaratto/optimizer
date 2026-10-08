"""``fund.actions`` — the single confirm→apply→audit path for every human mutation.

The shell's oversight views (Decisions, Mandate, killswitch, and the Phase-1 drive
flows) never mutate state directly: they funnel every human action through
:func:`apply_action`, which applies the change and appends one immutable
``human_audit_log`` row, so no mutation can bypass the audit trail. The two flags the
Audit view surfaces — ⚑ ``override_flag`` (a guardrail edited past its MiFID band)
and ⚠ ``suitability_flag`` — are stamped here.

This module may drive the agent stack (``resume_run``), but imports it **lazily
inside the action bodies**, mirroring ``fund.observe`` / ``fund.agents.graph``:
``import fund.actions`` itself stays agent-stack-free, and the TUI imports it lazily
inside a ``@work(thread=True)`` worker so ``import fund.tui.app`` keeps its light,
keyless import.

Transaction discipline (D1): :func:`apply_action` opens no session of its own and
never commits — the caller injects a sync session and owns the transaction boundary
(the TUI worker commits once the action lands, exactly like ``resume_run``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fund.config import FundConfig, settings

if TYPE_CHECKING:
    from portopt_db.models import HumanAuditLog
    from sqlalchemy.orm import Session

# Past-tense verb per resume decision, for the status line the shell reports. Its
# keys double as the approve/reject dispatch set in apply_action.
_PAST = {"approve": "approved", "reject": "rejected"}


@dataclass(frozen=True)
class ActionResult:
    """Outcome of an applied action — what the shell reports plus the audit row id.

    Attributes:
        action: The action kind that was applied (e.g. ``"approve"``).
        run_id: The run the action resolved, or ``None`` for a run-less action.
        portfolio_id: The affected portfolio, or ``None`` for a fund-wide action.
        audit_id: Primary key of the ``human_audit_log`` row written.
        message: A one-line summary for the status bar.
        detail: The underlying driver's return value (e.g. ``resume_run``'s result),
            or ``None`` when the action produces none.
    """

    action: str
    run_id: uuid.UUID | None
    portfolio_id: uuid.UUID | None
    audit_id: uuid.UUID
    message: str
    detail: dict[str, Any] | None = None


def apply_action(
    action: str,
    *,
    session: Session,
    run_id: uuid.UUID | str | None = None,
    portfolio_id: uuid.UUID | None = None,
    checkpointer: Any | None = None,
    store: Any | None = None,
    model: Any | None = None,
    actor: str = "human",
    source: str | None = None,
    override_flag: bool = False,
    suitability_flag: bool = False,
    idempotency_key: str | None = None,
    rationale: str | None = None,
    config: FundConfig = settings,
) -> ActionResult:
    """Apply one human action and append its ``human_audit_log`` row.

    The single entry point every shell mutation routes through. This phase handles
    ``approve`` / ``reject``; killswitch, mandate edits, and the drive flows extend
    the dispatch in later tasks. Opens no session and never commits: the caller
    injects ``session`` and owns the transaction (D1).

    ``approve`` / ``reject`` resume the paused run via
    :func:`fund.agents.graph.resume_run` (dispatched by gate), then log the human
    decision with run/portfolio provenance. When ``portfolio_id`` is omitted it is
    recovered from the run so the audit row is always attributed.

    Args:
        action: The action kind; ``"approve"`` / ``"reject"`` in this phase.
        session: Injected sync session; not owned here (no commit).
        run_id: The run to act on; required for approve/reject.
        portfolio_id: The affected portfolio; derived from the run when omitted.
        checkpointer: LangGraph checkpointer for the resume (approve/reject).
        store: LangGraph store for the resume (approve/reject).
        model: Chat model the resumed run drives (approve/reject).
        actor: Who acted; defaults to ``"human"``.
        source: Provenance recorded on the audit row.
        override_flag: ⚑ a guardrail edited past its MiFID band.
        suitability_flag: ⚠ outside the suitability band.
        idempotency_key: Dedup key; a repeated key returns the first audit row.
        rationale: Free-text reason recorded with the action.
        config: Fund settings threaded to the resumer.

    Returns:
        An :class:`ActionResult` describing the outcome and the audit row id.

    Raises:
        ValueError: The action kind is unknown, or a required argument (e.g.
            ``run_id`` for approve/reject) is missing.
        LookupError: ``run_id`` names no run (propagated from ``resume_run``).
    """
    if action in _PAST:
        return _apply_resume(
            action,
            decision=action,
            session=session,
            run_id=run_id,
            portfolio_id=portfolio_id,
            checkpointer=checkpointer,
            store=store,
            model=model,
            actor=actor,
            source=source,
            override_flag=override_flag,
            suitability_flag=suitability_flag,
            idempotency_key=idempotency_key,
            rationale=rationale,
            config=config,
        )
    raise ValueError(f"unknown action {action!r}")


def _apply_resume(
    action: str,
    *,
    decision: str,
    session: Session,
    run_id: uuid.UUID | str | None,
    portfolio_id: uuid.UUID | None,
    checkpointer: Any | None,
    store: Any | None,
    model: Any | None,
    actor: str,
    source: str | None,
    override_flag: bool,
    suitability_flag: bool,
    idempotency_key: str | None,
    rationale: str | None,
    config: FundConfig,
) -> ActionResult:
    """Resume a paused run by its gate, then log the decision.

    ``decision`` is the ``resume_run`` verb (``approve`` / ``reject``); ``action`` is
    what gets logged — they coincide for approve/reject but will diverge for
    killswitch (logged distinctly, resumed as a reject). The agent stack is imported
    here, never at module scope.
    """
    from fund.agents.graph import resume_run

    if run_id is None:
        raise ValueError(f"{action!r} requires a run_id")
    rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
    result = resume_run(
        rid,
        decision,
        session=session,
        checkpointer=checkpointer,
        store=store,
        model=model,
        config=config,
    )
    pid = portfolio_id if portfolio_id is not None else _portfolio_of(session, rid)
    row = _append_audit(
        session,
        action=action,
        run_id=rid,
        portfolio_id=pid,
        actor=actor,
        source=source,
        override_flag=override_flag,
        suitability_flag=suitability_flag,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    return ActionResult(
        action=action,
        run_id=rid,
        portfolio_id=pid,
        audit_id=row.id,
        message=f"run {rid} {_PAST.get(action, action)}",
        detail=result,
    )


def _portfolio_of(session: Session, run_id: uuid.UUID) -> uuid.UUID | None:
    """The run's portfolio id, read lazily through the single-source run repo."""
    from fund.audit.repository import AgentRunRepository

    run = AgentRunRepository(session).get_run(run_id)
    return run.portfolio_id if run is not None else None


def _append_audit(
    session: Session,
    *,
    action: str,
    run_id: uuid.UUID | None,
    portfolio_id: uuid.UUID | None,
    actor: str,
    source: str | None,
    override_flag: bool,
    suitability_flag: bool,
    idempotency_key: str | None,
    rationale: str | None,
) -> HumanAuditLog:
    """Append one ``human_audit_log`` row and return it.

    Imports the repo from its submodule (never the ``fund.audit`` root, which pulls
    the langgraph persistence bootstrap).
    """
    from fund.audit.human_audit_repository import HumanAuditLogRepository

    return HumanAuditLogRepository(session).append(
        action=action,
        run_id=run_id,
        portfolio_id=portfolio_id,
        actor=actor,
        source=source,
        override_flag=override_flag,
        suitability_flag=suitability_flag,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )


__all__ = ["ActionResult", "apply_action"]
