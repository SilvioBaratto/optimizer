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

# Each resume-style action maps to the ``resume_run`` verb it finalises the paused
# run with. approve/reject are self-named; killswitch finalises a paused run as a
# reject but is logged distinctly (its own audit action, no paired undo). The keys
# are the resume-dispatch set in apply_action.
_RESUME_DECISION = {"approve": "approve", "reject": "reject", "killswitch": "reject"}

# Past-tense verb per action, for the status line the shell reports.
_PAST = {"approve": "approved", "reject": "rejected", "killswitch": "killswitched"}


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
    mandate: Any | None = None,
    target: str | None = None,
    prev_value: Any | None = None,
    new_value: Any | None = None,
    actor: str = "human",
    source: str | None = None,
    override_flag: bool = False,
    suitability_flag: bool = False,
    idempotency_key: str | None = None,
    rationale: str | None = None,
    config: FundConfig = settings,
) -> ActionResult:
    """Apply one human action and append its ``human_audit_log`` row.

    The single entry point every shell mutation routes through. This phase handles the
    resume actions (``approve`` / ``reject`` / ``killswitch``) plus ``mandate_edit`` and
    ``guardrail_override``; the drive flows extend the dispatch later. Opens no session
    and never commits: the caller injects ``session`` and owns the transaction (D1).

    ``approve`` / ``reject`` resume the paused run via
    :func:`fund.agents.graph.resume_run` (dispatched by gate); ``killswitch`` finalises
    a paused run through the same reject path but is logged under its own action (no
    paired undo). ``mandate_edit`` upserts the run-policy mandate (persisted);
    ``guardrail_override`` is **audit-only** — it records a guardrail touch and, when
    the caller marks it past the MiFID suitability band, stamps the ⚑/⚠ flags. Each logs
    the human decision with run/portfolio provenance; a resume action recovers
    ``portfolio_id`` from its run when omitted.

    Args:
        action: The action kind; the resume actions, ``"mandate_edit"``, or
            ``"guardrail_override"`` in this phase.
        session: Injected sync session; not owned here (no commit).
        run_id: The run to act on; required for the resume actions.
        portfolio_id: The affected portfolio; derived from the run when omitted.
        checkpointer: LangGraph checkpointer for the resume.
        store: LangGraph store for the resume.
        model: Chat model the resumed run drives.
        mandate: The ``fund.schemas.mandate.PortfolioMandate`` to upsert
            (``mandate_edit`` only).
        target: The field or guardrail the action touched (edit/override).
        prev_value: Prior value recorded on the audit row (edit/override).
        new_value: New value recorded on the audit row (edit/override).
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
        ValueError: The action kind is unknown, or a required argument (``run_id`` for a
            resume action, ``mandate`` for ``mandate_edit``) is missing.
        LookupError: ``run_id`` names no run (propagated from ``resume_run``).
    """
    if action in _RESUME_DECISION:
        return _apply_resume(
            action,
            decision=_RESUME_DECISION[action],
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
    if action == "mandate_edit":
        return _apply_mandate_edit(
            session=session,
            portfolio_id=portfolio_id,
            mandate=mandate,
            actor=actor,
            source=source,
            prev_value=prev_value,
            idempotency_key=idempotency_key,
            rationale=rationale,
        )
    if action == "guardrail_override":
        return _apply_guardrail_override(
            session=session,
            portfolio_id=portfolio_id,
            target=target,
            prev_value=prev_value,
            new_value=new_value,
            actor=actor,
            source=source,
            override_flag=override_flag,
            suitability_flag=suitability_flag,
            idempotency_key=idempotency_key,
            rationale=rationale,
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
    target: str | None = None,
    prev_value: Any | None = None,
    new_value: Any | None = None,
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
        target=target,
        prev_value=prev_value,
        new_value=new_value,
        source=source,
        override_flag=override_flag,
        suitability_flag=suitability_flag,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )


def _apply_mandate_edit(
    *,
    session: Session,
    portfolio_id: uuid.UUID | None,
    mandate: Any | None,
    actor: str,
    source: str | None,
    prev_value: Any | None,
    idempotency_key: str | None,
    rationale: str | None,
) -> ActionResult:
    """Upsert the run-policy mandate (persisted) and log an unflagged edit row.

    ``mandate`` is a ``fund.schemas.mandate.PortfolioMandate`` — the repo (imported
    from its submodule, never the ``fund.audit`` root) owns the DB write via its
    non-committing ``upsert``; the new value logged is the mandate's JSON form.
    """
    if mandate is None:
        raise ValueError("'mandate_edit' requires a mandate")
    from fund.audit.mandate_repository import MandateRepository

    MandateRepository(session).upsert(mandate)
    pid = portfolio_id if portfolio_id is not None else uuid.UUID(mandate.portfolio_id)
    row = _append_audit(
        session,
        action="mandate_edit",
        run_id=None,
        portfolio_id=pid,
        actor=actor,
        target="run_policy",
        prev_value=prev_value,
        new_value=mandate.model_dump(mode="json"),
        source=source,
        override_flag=False,
        suitability_flag=False,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    return ActionResult(
        action="mandate_edit",
        run_id=None,
        portfolio_id=pid,
        audit_id=row.id,
        message=f"mandate updated for {pid}",
    )


def _apply_guardrail_override(
    *,
    session: Session,
    portfolio_id: uuid.UUID | None,
    target: str | None,
    prev_value: Any | None,
    new_value: Any | None,
    actor: str,
    source: str | None,
    override_flag: bool,
    suitability_flag: bool,
    idempotency_key: str | None,
    rationale: str | None,
) -> ActionResult:
    """Log a guardrail touch — **audit-only** this phase (no enforcement, no persist).

    The caller decides whether the proposed value sits past the MiFID suitability band
    (see :func:`is_within_suitability_band`) and passes the ⚑/⚠ flags; this just records
    the prev/new value on an immutable row. Live enforcement is ``03-copilot``.
    """
    row = _append_audit(
        session,
        action="guardrail_override",
        run_id=None,
        portfolio_id=portfolio_id,
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
    flagged = " ⚠ override" if suitability_flag else ""
    return ActionResult(
        action="guardrail_override",
        run_id=None,
        portfolio_id=portfolio_id,
        audit_id=row.id,
        message=f"guardrail {target} logged{flagged}",
    )


# RiskToleranceBand ordered least→most risk-tolerant. The active SuitabilityAssessment's
# band is the ceiling the client is suited for; a guardrail edit to a *riskier* band is
# the override we flag (de-risking below the ceiling is always suitable).
_BAND_RANK = {
    "defensive": 0,
    "conservative": 1,
    "balanced": 2,
    "growth": 3,
    "aggressive": 4,
}


def is_within_suitability_band(proposed_band: str, suitability_band: str) -> bool:
    """Whether a proposed risk band stays within the suitability ceiling.

    Args:
        proposed_band: The edited ``RiskToleranceBand`` value.
        suitability_band: The active assessment's ``RiskToleranceBand`` value (ceiling).

    Returns:
        ``True`` when ``proposed_band`` is no riskier than ``suitability_band``; an
        unknown band is treated conservatively (proposal most-conservative, ceiling
        most-permissive) so a typo never silently flags an override.
    """
    return _BAND_RANK.get(proposed_band, 0) <= _BAND_RANK.get(
        suitability_band, max(_BAND_RANK.values())
    )


__all__ = ["ActionResult", "apply_action", "is_within_suitability_band"]
