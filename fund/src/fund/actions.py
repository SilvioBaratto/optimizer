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

import datetime as dt
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fund.config import FundConfig, settings

if TYPE_CHECKING:
    from portopt_db.models import HumanAuditLog
    from sqlalchemy.orm import Session

# Each resume-style action maps to the ``resume_run`` verb it drives the paused run
# with. approve/reject/respond are self-named; killswitch finalises a paused run as a
# reject but is logged distinctly (its own audit action, no paired undo). ``respond``
# is a ``place_orders``-only barge-in that threads the adviser's ``pm_directive`` back
# to the PM — not a terminal decision: the run re-gates as ``responded`` (still
# awaiting approval) or finalises incomplete. The keys are the resume-dispatch set in
# apply_action.
_RESUME_DECISION = {
    "approve": "approve",
    "reject": "reject",
    "respond": "respond",
    "killswitch": "reject",
}

# Past-tense verb per action, for the status line the shell reports.
_PAST = {
    "approve": "approved",
    "reject": "rejected",
    "respond": "responded",
    "killswitch": "killswitched",
    "edit": "approved with edited weights",
    "freeze": "frozen",
    "rearm": "re-armed",
}


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
    questionnaire: Any | None = None,
    asof: dt.date | None = None,
    base_currency: str | None = None,
    target: str | None = None,
    prev_value: Any | None = None,
    new_value: Any | None = None,
    actor: str = "human",
    source: str | None = None,
    override_flag: bool = False,
    suitability_flag: bool = False,
    idempotency_key: str | None = None,
    rationale: str | None = None,
    pm_directive: str | None = None,
    scope: str | None = None,
    config: FundConfig = settings,
) -> ActionResult:
    """Apply one human action and append its ``human_audit_log`` row.

    The single entry point every shell mutation routes through. This phase handles the
    resume actions (``approve`` / ``reject`` / ``respond`` / ``killswitch``), the policy
    edits (``mandate_edit`` / ``guardrail_override``), and the onboarding drive flows
    (``new_portfolio`` / ``profile`` / ``run``) so a full cycle — create → profile →
    rebalance → approve — funnels through one audited path. Opens no session and never
    commits: the caller injects ``session`` and owns the transaction (D1).

    ``approve`` / ``reject`` resume the paused run via
    :func:`fund.agents.graph.resume_run` (dispatched by gate); ``respond`` threads the
    adviser's ``pm_directive`` back to the PM at the ``place_orders`` gate (the run
    re-gates as ``responded`` or finalises incomplete) and, like every action here, is
    recorded with its own ``human_audit_log`` row; ``killswitch`` finalises a paused run
    through the same reject path but is logged under its own action (no paired undo).
    ``edit`` validates the adviser-supplied ``new_value`` weights against the risk
    invariants, logs them as a fresh allocator decision, and resumes via the approve
    path so the ticket/positions/run all commit the edited allocation (⚑ override).
    ``freeze`` / ``rearm`` flip the autonomy posture at ``scope``: a ``run`` freeze is
    the killswitch (reject-finalise, no autonomy change), while ``portfolio`` / ``fund``
    freeze FROZEN (halting live runs) and ``rearm`` restores ARMED — a **fund** re-arm
    requires a non-empty ``rationale``.
    ``mandate_edit`` upserts the run-policy mandate (persisted);
    ``guardrail_override`` is **audit-only** — it records a guardrail touch and, when
    the caller marks it past the MiFID suitability band, stamps the ⚑/⚠ flags. Each logs
    the human decision with run/portfolio provenance; a resume action recovers
    ``portfolio_id`` from its run when omitted.

    Args:
        action: The action kind; the resume actions, ``"edit"``, ``"mandate_edit"``,
            ``"guardrail_override"``, or a drive flow (``"new_portfolio"`` /
            ``"profile"`` / ``"run"``).
        session: Injected sync session; not owned here (no commit).
        run_id: The run to act on; required for the resume actions.
        portfolio_id: The affected portfolio; derived from the run when omitted.
        checkpointer: LangGraph checkpointer for the resume / drive flows.
        store: LangGraph store for the resume / drive flows.
        model: Chat model the resumed or driven run uses.
        mandate: The ``fund.schemas.mandate.PortfolioMandate`` to upsert
            (``mandate_edit`` / ``new_portfolio``) or run (``run``).
        questionnaire: Free-text/typed MiFID answers the profiler normalises
            (``profile`` only).
        asof: Decision bar for a ``profile`` / ``run`` drive; a ``run`` defaults to
            the newest bar with a stored close after it (``default_decision_bar``).
        base_currency: Reporting currency override for the ``profile`` drive.
        target: The field or guardrail the action touched (edit/override).
        prev_value: Prior value recorded on the audit row (edit/override).
        new_value: New value recorded on the audit row (edit/override).
        actor: Who acted; defaults to ``"human"``.
        source: Provenance recorded on the audit row.
        override_flag: ⚑ a guardrail edited past its MiFID band.
        suitability_flag: ⚠ outside the suitability band.
        idempotency_key: Dedup key; a repeated key returns the first audit row.
        rationale: Free-text reason recorded with the action.
        pm_directive: Adviser free text threaded to the PM for a ``respond`` barge-in;
            required (non-empty) for ``respond``, ignored by the other actions.
        scope: Autonomy scope for ``freeze`` (``run`` / ``portfolio`` / ``fund``) or
            ``rearm`` (``portfolio`` / ``fund``); ignored by the other actions.
        config: Fund settings threaded to the resumer.

    Returns:
        An :class:`ActionResult` describing the outcome and the audit row id.

    Raises:
        ValueError: The action kind is unknown, or a required argument is missing
            (``run_id`` for a resume action; ``pm_directive`` for ``respond``;
            ``mandate`` for ``mandate_edit`` / ``new_portfolio`` / ``run``;
            ``portfolio_id`` for ``profile`` / ``run``).
        LookupError: ``run_id`` names no run (propagated from ``resume_run``).
        RuntimeError: ``run`` on a portfolio with no active ``ConstraintSet``
            (propagated from ``run_fund`` — profile it first).
    """
    if idempotency_key is not None:
        replay = _replayed_action(session, idempotency_key)
        if replay is not None:
            return replay
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
            pm_directive=pm_directive,
            actor=actor,
            source=source,
            override_flag=override_flag,
            suitability_flag=suitability_flag,
            idempotency_key=idempotency_key,
            rationale=rationale,
            config=config,
        )
    if action == "edit":
        return _apply_edit(
            session=session,
            run_id=run_id,
            portfolio_id=portfolio_id,
            checkpointer=checkpointer,
            store=store,
            model=model,
            new_value=new_value,
            actor=actor,
            source=source,
            idempotency_key=idempotency_key,
            rationale=rationale,
            config=config,
        )
    if action == "freeze":
        return _apply_freeze(
            session=session,
            scope=scope,
            run_id=run_id,
            portfolio_id=portfolio_id,
            checkpointer=checkpointer,
            store=store,
            model=model,
            actor=actor,
            source=source,
            idempotency_key=idempotency_key,
            rationale=rationale,
            config=config,
        )
    if action == "rearm":
        return _apply_rearm(
            session=session,
            scope=scope,
            portfolio_id=portfolio_id,
            actor=actor,
            source=source,
            idempotency_key=idempotency_key,
            rationale=rationale,
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
    if action == "new_portfolio":
        return _apply_new_portfolio(
            session=session,
            portfolio_id=portfolio_id,
            mandate=mandate,
            actor=actor,
            source=source,
            idempotency_key=idempotency_key,
            rationale=rationale,
        )
    if action == "profile":
        return _apply_profile(
            session=session,
            portfolio_id=portfolio_id,
            questionnaire=questionnaire,
            model=model,
            checkpointer=checkpointer,
            store=store,
            base_currency=base_currency,
            asof=asof,
            actor=actor,
            source=source,
            idempotency_key=idempotency_key,
            rationale=rationale,
            config=config,
        )
    if action == "run":
        return _apply_run(
            session=session,
            portfolio_id=portfolio_id,
            mandate=mandate,
            model=model,
            checkpointer=checkpointer,
            store=store,
            asof=asof,
            actor=actor,
            source=source,
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
    pm_directive: str | None,
    actor: str,
    source: str | None,
    override_flag: bool,
    suitability_flag: bool,
    idempotency_key: str | None,
    rationale: str | None,
    config: FundConfig,
) -> ActionResult:
    """Resume a paused run by its gate, then log the decision.

    ``decision`` is the ``resume_run`` verb (``approve`` / ``reject`` / ``respond``);
    ``action`` is what gets logged — they coincide for approve/reject/respond but
    diverge for killswitch (logged distinctly, resumed as a reject). ``respond`` threads
    the adviser's ``pm_directive`` (required) back to the PM at the ``place_orders``
    gate. The agent stack is imported here, never at module scope.
    """
    from fund.agents.graph import resume_run

    if run_id is None:
        raise ValueError(f"{action!r} requires a run_id")
    if action == "respond" and not pm_directive:
        raise ValueError("'respond' requires a pm_directive")
    rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
    result = resume_run(
        rid,
        decision,
        session=session,
        checkpointer=checkpointer,
        store=store,
        model=model,
        pm_directive=pm_directive,
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
        message=resume_message(session, rid, action, result),
        detail=result,
    )


def resume_message(
    session: Session, run_id: uuid.UUID, action: str, result: dict[str, Any] | None
) -> str:
    """The status line for a resumed run, honest about what the resume really did.

    ``place_orders`` reports a missing fill bar or unknown ticker as a soft failure, so
    the resume itself succeeds. An approve/edit whose run did not end ``completed``
    (finalised ``incomplete``, or re-gated and left ``paused`` / ``responded``) placed
    no paper ticket; claiming plain success would hide that from the adviser. A PM
    follow-up turn that failed after the gate was consumed (``resume_error``) is
    appended: the decision stands, but the adviser should know the model did not finish.

    Args:
        session: A sync session that sees the resume's writes; not committed here.
        run_id: The resumed run.
        action: The logged action (``approve`` / ``edit`` / ``reject`` / ...).
        result: The ``resume_run`` return value, or ``None``.

    Returns:
        The one-line status message.
    """
    from fund.audit.repository import AgentRunRepository

    message = f"run {run_id} {_PAST.get(action, action)}"
    if action in ("approve", "edit"):
        run = AgentRunRepository(session).get_run(run_id)
        if run is not None and run.status != "completed":
            message += f", but no paper ticket was placed (run is {run.status})"
    error = (result or {}).get("resume_error")
    if error:
        message += f" (PM follow-up failed: {error})"
    return message


def _apply_edit(
    *,
    session: Session,
    run_id: uuid.UUID | str | None,
    portfolio_id: uuid.UUID | None,
    checkpointer: Any | None,
    store: Any | None,
    model: Any | None,
    new_value: Any | None,
    actor: str,
    source: str | None,
    idempotency_key: str | None,
    rationale: str | None,
    config: FundConfig,
) -> ActionResult:
    """Commit adviser-edited weights, then resume the paused run via ``approve``.

    The edit never touches the gated tool's args. It validates ``new_value`` (the
    ``{ticker: weight}`` the adviser typed) against the long-only / fully-invested
    invariants with ``risk_check``, then appends a fresh **allocator /
    optimize_portfolio** decision carrying the edited weights and resumes through the
    existing approve path. ``_place_orders_impl`` re-reads ``latest_optimizer_weights``
    on resume, so that appended row is the single source every downstream record (the
    paper ticket + ``weights_hash``, ``agent_runs.weights``, and the positions
    snapshot) derives from — no graph change, no divergence. The audit row is stamped
    ⚑ ``override_flag`` since the human overrode the optimizer's allocation.

    Raises:
        ValueError: ``run_id`` or ``new_value`` is missing, or the edited weights
            fail the risk invariants (the structured ``violations`` are reported).
    """
    from fund.agents.graph import resume_run
    from fund.audit.repository import AgentRunRepository
    from fund.tools.risk import risk_check

    if run_id is None:
        raise ValueError("'edit' requires a run_id")
    if not new_value:
        raise ValueError("'edit' requires new_value (the edited weights)")
    rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
    edited = {str(k): float(v) for k, v in dict(new_value).items()}

    check = risk_check(edited)
    if not check.get("ok"):
        raise ValueError(f"'edit' rejected: {check.get('error')}")
    report = check["data"]
    if not report["passed"]:
        raise ValueError(f"'edit' violates risk invariants: {report['violations']}")

    AgentRunRepository(session).append_decision(
        rid,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": edited}),
        hitl_decision={"decision": "edit", "human_override": True},
    )
    result = resume_run(
        rid,
        "approve",
        session=session,
        checkpointer=checkpointer,
        store=store,
        model=model,
        config=config,
    )
    pid = portfolio_id if portfolio_id is not None else _portfolio_of(session, rid)
    row = _append_audit(
        session,
        action="edit",
        run_id=rid,
        portfolio_id=pid,
        actor=actor,
        target="weights",
        new_value=edited,
        source=source,
        override_flag=True,
        suitability_flag=False,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    return ActionResult(
        action="edit",
        run_id=rid,
        portfolio_id=pid,
        audit_id=row.id,
        message=resume_message(session, rid, "edit", result),
        detail=result,
    )


_FREEZE_SCOPES = ("run", "portfolio", "fund")
_REARM_SCOPES = ("portfolio", "fund")


def _reject_live_runs(session: Session, portfolio_id: uuid.UUID | None) -> None:
    """Finalise any paused/responded runs in scope as rejected (DB-level halt).

    A portfolio/fund freeze halts in-flight runs: each live gate is marked rejected
    without a graph resume (an emergency stop needs no model/checkpointer). Scoped to
    ``portfolio_id`` when given, else fund-wide. No commit (the caller owns it).
    """
    from fund.audit.repository import AgentRunRepository

    repo = AgentRunRepository(session)
    for run in repo.list_paused_runs(portfolio_id=portfolio_id):
        repo.finalize_run(run.id, weights={}, status="rejected")


def halt_run_if_frozen(
    session: Session, run_id: uuid.UUID, portfolio_id: uuid.UUID | None
) -> bool:
    """Reject a round's just-opened gate when its scope froze while the round ran.

    A freeze halts live runs by rejecting the *committed* paused gates, but a round in
    flight holds its run row uncommitted, so the freeze cannot see it and the round
    would go on to commit an approvable gate on a frozen scope. Call this at the point
    a round reaches its gate, before the caller commits: it re-reads the fund and
    portfolio posture and, when either is frozen, finalises the run ``rejected`` exactly
    as the freeze would have. No commit (the caller owns the transaction).

    Args:
        session: The round's session, still holding its uncommitted run row.
        run_id: The run that just reached its gate.
        portfolio_id: The run's portfolio (fund scope is always checked).

    Returns:
        ``True`` when the run was halted (finalised ``rejected``), else ``False``.
    """
    from fund.audit.autonomy_repository import AutonomyRepository
    from fund.audit.repository import AgentRunRepository

    if AutonomyRepository(session).frozen_scope(portfolio_id) is None:
        return False
    AgentRunRepository(session).finalize_run(run_id, weights={}, status="rejected")
    return True


def _apply_freeze(
    *,
    session: Session,
    scope: str | None,
    run_id: uuid.UUID | str | None,
    portfolio_id: uuid.UUID | None,
    checkpointer: Any | None,
    store: Any | None,
    model: Any | None,
    actor: str,
    source: str | None,
    idempotency_key: str | None,
    rationale: str | None,
    config: FundConfig,
) -> ActionResult:
    """Freeze autonomy at ``scope`` (``run`` / ``portfolio`` / ``fund``) and audit it.

    ``run`` keeps the killswitch semantics — reject-finalise the paused run, touching
    no autonomy posture. ``portfolio`` / ``fund`` flip the autonomy row FROZEN (via
    :class:`AutonomyRepository`, recording ``rationale`` as the frozen reason) and halt
    any live run in scope. Every path appends one ⚑ ``human_audit_log`` row.

    Raises:
        ValueError: ``scope`` is not one of ``run`` / ``portfolio`` / ``fund``, or the
            required ``run_id`` (``run``) / ``portfolio_id`` (``portfolio``) is missing.
    """
    if scope not in _FREEZE_SCOPES:
        raise ValueError(f"'freeze' requires scope in {_FREEZE_SCOPES}, got {scope!r}")

    if scope == "run":
        from fund.agents.graph import resume_run

        if run_id is None:
            raise ValueError("'freeze' scope='run' requires a run_id")
        rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
        result = resume_run(
            rid,
            "reject",
            session=session,
            checkpointer=checkpointer,
            store=store,
            model=model,
            config=config,
        )
        pid = portfolio_id if portfolio_id is not None else _portfolio_of(session, rid)
        row = _append_audit(
            session,
            action="freeze",
            run_id=rid,
            portfolio_id=pid,
            actor=actor,
            target="run",
            source=source,
            override_flag=False,
            suitability_flag=False,
            idempotency_key=idempotency_key,
            rationale=rationale,
        )
        return ActionResult(
            action="freeze",
            run_id=rid,
            portfolio_id=pid,
            audit_id=row.id,
            message=f"run {rid} {_PAST['freeze']}",
            detail=result,
        )

    from fund.audit.autonomy_repository import (
        AutonomyRepository,
        AutonomyScope,
        AutonomyState,
    )

    repo = AutonomyRepository(session)
    if scope == "fund":
        repo.set_state(AutonomyScope.FUND, None, AutonomyState.FROZEN, reason=rationale)
        pid = None
    else:
        if portfolio_id is None:
            raise ValueError("'freeze' scope='portfolio' requires a portfolio_id")
        repo.set_state(
            AutonomyScope.PORTFOLIO,
            portfolio_id,
            AutonomyState.FROZEN,
            reason=rationale,
        )
        pid = portfolio_id
    _reject_live_runs(session, pid)
    row = _append_audit(
        session,
        action="freeze",
        run_id=None,
        portfolio_id=pid,
        actor=actor,
        target=scope,
        new_value={"state": "frozen", "reason": rationale},
        source=source,
        override_flag=False,
        suitability_flag=False,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    return ActionResult(
        action="freeze",
        run_id=None,
        portfolio_id=pid,
        audit_id=row.id,
        message=f"{_scope_subject(scope, pid)} {_PAST['freeze']}",
    )


def _apply_rearm(
    *,
    session: Session,
    scope: str | None,
    portfolio_id: uuid.UUID | None,
    actor: str,
    source: str | None,
    idempotency_key: str | None,
    rationale: str | None,
) -> ActionResult:
    """Re-arm autonomy at ``scope`` (``portfolio`` / ``fund``) and audit it.

    Flips the autonomy row back to ARMED (clearing the frozen provenance) and appends
    one ``human_audit_log`` row. A **fund** re-arm requires a non-empty ``rationale``
    (the graduated ceremony for the fund-wide killswitch); a portfolio re-arm does not.

    Raises:
        ValueError: ``scope`` is not ``portfolio`` / ``fund``; a ``portfolio`` re-arm is
            missing its ``portfolio_id``; or a ``fund`` re-arm has no ``rationale``.
    """
    if scope not in _REARM_SCOPES:
        raise ValueError(f"'rearm' requires scope in {_REARM_SCOPES}, got {scope!r}")

    from fund.audit.autonomy_repository import (
        AutonomyRepository,
        AutonomyScope,
        AutonomyState,
    )

    repo = AutonomyRepository(session)
    if scope == "fund":
        if not rationale:
            raise ValueError("'rearm' scope='fund' requires a non-empty rationale")
        repo.set_state(AutonomyScope.FUND, None, AutonomyState.ARMED)
        pid = None
    else:
        if portfolio_id is None:
            raise ValueError("'rearm' scope='portfolio' requires a portfolio_id")
        repo.set_state(AutonomyScope.PORTFOLIO, portfolio_id, AutonomyState.ARMED)
        pid = portfolio_id
    row = _append_audit(
        session,
        action="rearm",
        run_id=None,
        portfolio_id=pid,
        actor=actor,
        target=scope,
        new_value={"state": "armed"},
        source=source,
        override_flag=False,
        suitability_flag=False,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    return ActionResult(
        action="rearm",
        run_id=None,
        portfolio_id=pid,
        audit_id=row.id,
        message=f"{_scope_subject(scope, pid)} {_PAST['rearm']}",
    )


def _scope_subject(scope: str, portfolio_id: uuid.UUID | None) -> str:
    """The status-line subject of an autonomy change: the portfolio it hit, else scope.

    Naming the portfolio matters because the Decisions queue is fund-wide, so the
    portfolio a freeze lands on need not be the one the header is focused on.
    """
    if scope == "portfolio" and portfolio_id is not None:
        return f"portfolio {portfolio_id}"
    return scope


def _portfolio_of(session: Session, run_id: uuid.UUID) -> uuid.UUID | None:
    """The run's portfolio id, read lazily through the single-source run repo."""
    from fund.audit.repository import AgentRunRepository

    run = AgentRunRepository(session).get_run(run_id)
    return run.portfolio_id if run is not None else None


def _replayed_action(session: Session, idempotency_key: str) -> ActionResult | None:
    """The prior outcome for ``idempotency_key``, or ``None`` if unseen.

    Checked at the top of :func:`apply_action`, before any driver runs, so a repeated
    keyed action (a double-submitted approve, a retried drive) neither re-invokes
    ``resume_run`` / the profiler / the rebalance nor creates an orphan run — it replays
    the first audit row's outcome. The row carries no stored ``detail``, so a replay's
    :attr:`ActionResult.detail` is ``None``.
    """
    from fund.audit.human_audit_repository import HumanAuditLogRepository

    row = HumanAuditLogRepository(session).get_by_idempotency_key(idempotency_key)
    if row is None:
        return None
    return ActionResult(
        action=row.action,
        run_id=row.run_id,
        portfolio_id=row.portfolio_id,
        audit_id=row.id,
        message=f"{row.action} already applied (idempotent replay)",
    )


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


def _apply_new_portfolio(
    *,
    session: Session,
    portfolio_id: uuid.UUID | None,
    mandate: Any | None,
    actor: str,
    source: str | None,
    idempotency_key: str | None,
    rationale: str | None,
) -> ActionResult:
    """Create a portfolio by upserting its initial run-policy mandate + logging it.

    The onboarding entry point: a fresh portfolio exists once it carries a mandate (the
    roster reads ``list_portfolios``, which unions mandate rows). ``mandate`` is a
    ``fund.schemas.mandate.PortfolioMandate``; the repo (imported from its submodule,
    never the ``fund.audit`` root) owns the write via its non-committing ``upsert``.
    """
    if mandate is None:
        raise ValueError("'new_portfolio' requires a mandate")
    from fund.audit.mandate_repository import MandateRepository

    MandateRepository(session).upsert(mandate)
    pid = portfolio_id if portfolio_id is not None else uuid.UUID(mandate.portfolio_id)
    row = _append_audit(
        session,
        action="new_portfolio",
        run_id=None,
        portfolio_id=pid,
        actor=actor,
        target="portfolio",
        new_value=mandate.model_dump(mode="json"),
        source=source,
        override_flag=False,
        suitability_flag=False,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    return ActionResult(
        action="new_portfolio",
        run_id=None,
        portfolio_id=pid,
        audit_id=row.id,
        message=f"portfolio {pid} created",
    )


def _apply_profile(
    *,
    session: Session,
    portfolio_id: uuid.UUID | None,
    questionnaire: Any | None,
    model: Any | None,
    checkpointer: Any | None,
    store: Any | None,
    base_currency: str | None,
    asof: dt.date | None,
    actor: str,
    source: str | None,
    idempotency_key: str | None,
    rationale: str | None,
    config: FundConfig,
) -> ActionResult:
    """Drive the MiFID profiler to its ``save_profile`` gate, then log the run.

    Invocation-only over ``run_profiler`` (no API change): it normalises the
    ``questionnaire`` into typed answers, derives the ``ConstraintSet`` /
    ``SuitabilityAssessment``, and pauses at the always-confirm gate — approving it
    later (through ``apply_action("approve", ...)``) persists the profile and caches
    the active set. The agent stack is imported here, never at module scope.
    """
    if portfolio_id is None:
        raise ValueError("'profile' requires a portfolio_id")
    from fund.agents.profiler import run_profiler

    run = run_profiler(
        model,
        questionnaire,
        portfolio_id=portfolio_id,
        session=session,
        checkpointer=checkpointer,
        store=store,
        base_currency=base_currency,
        asof=asof,
        config=config,
    )
    row = _append_audit(
        session,
        action="profile",
        run_id=run.run_id,
        portfolio_id=portfolio_id,
        actor=actor,
        target="mifid_profile",
        source=source,
        override_flag=False,
        suitability_flag=False,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    return ActionResult(
        action="profile",
        run_id=run.run_id,
        portfolio_id=portfolio_id,
        audit_id=row.id,
        message=f"profiler run {run.run_id} awaiting approval",
    )


def _apply_run(
    *,
    session: Session,
    portfolio_id: uuid.UUID | None,
    mandate: Any | None,
    model: Any | None,
    checkpointer: Any | None,
    store: Any | None,
    asof: dt.date | None,
    actor: str,
    source: str | None,
    idempotency_key: str | None,
    rationale: str | None,
    config: FundConfig,
) -> ActionResult:
    """Drive one paper rebalance to the ``place_orders`` gate, then log the run.

    Invocation-only over ``run_fund`` (no API change): it resolves the portfolio's
    active ``ConstraintSet`` from the Store (a missing one raises ``RuntimeError`` —
    the portfolio must be profiled first), runs the PM pipeline, and pauses at the
    HITL gate for approval. The agent stack is imported here, never at module scope.
    """
    if portfolio_id is None:
        raise ValueError("'run' requires a portfolio_id")
    if mandate is None:
        raise ValueError("'run' requires a mandate")
    from fund import observe
    from fund.agents.graph import run_fund

    run = run_fund(
        model,
        mandate,
        portfolio_id=portfolio_id,
        asof=asof if asof is not None else observe.default_decision_bar(session),
        session=session,
        checkpointer=checkpointer,
        store=store,
        config=config,
    )
    halted = run.status == "paused" and halt_run_if_frozen(
        session, run.run_id, portfolio_id
    )
    row = _append_audit(
        session,
        action="run",
        run_id=run.run_id,
        portfolio_id=portfolio_id,
        actor=actor,
        target="rebalance",
        source=source,
        override_flag=False,
        suitability_flag=False,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    outcome = "awaiting approval" if run.status == "paused" else "no ticket to approve"
    return ActionResult(
        action="run",
        run_id=run.run_id,
        portfolio_id=portfolio_id,
        audit_id=row.id,
        message=(
            f"run {run.run_id} rejected: autonomy was frozen while it ran"
            if halted
            else f"run {run.run_id} {run.status} ({outcome})"
        ),
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
        ``True`` when ``proposed_band`` is no riskier than ``suitability_band``. An
        unrecognised band resolves toward *flagging*, not silence (the compliance-safe
        direction): an unknown proposal is treated as most-risky and an unknown ceiling
        as most-restrictive, so a typo or an unmapped future band surfaces the override
        rather than hiding it.
    """
    return _BAND_RANK.get(proposed_band, max(_BAND_RANK.values())) <= _BAND_RANK.get(
        suitability_band, 0
    )


__all__ = [
    "ActionResult",
    "apply_action",
    "halt_run_if_frozen",
    "is_within_suitability_band",
    "resume_message",
]
