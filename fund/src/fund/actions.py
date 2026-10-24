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
(the TUI worker commits once the action lands, exactly like ``resume_run``). The one
exception is a ``profile`` action that hits a MiFID suitability breach:
``run_profiler`` then rolls the session back and commits only the blocked run's
trail (see its docstring), so the staged work of that action sequence is discarded.
"""

from __future__ import annotations

import contextlib
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

# Actions that settle in the DB alone, so a caller needs no model, store or
# checkpointer to drive them (the human's emergency stop must not depend on the LLM
# provider being up).
MODEL_FREE_ACTIONS = frozenset({"reject", "killswitch"})

# A run in one of these states is finished; resuming or re-rejecting it would rewrite
# a terminal outcome (mirrors the guard in ``resume_fund``).
_FINAL_STATUSES = ("completed", "rejected", "incomplete", "blocked", "failed")

# Resolution of the weights the shell shows and the adviser edits (4 decimals).
_EDIT_RESOLUTION = 1e-4
# Float noise on the headroom comparison, so a residual exactly at capacity is absorbed.
_ROUNDING_EPS = 1e-12

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
        detail: The underlying driver's return value (e.g. ``resume_run``'s result;
            ``{"status": ...}`` for ``run``), a ``{"blocked": True, "reason": ...}``
            marker for a ``profile`` stopped by a suitability breach, or ``None`` when
            the action produces none.
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

    ``approve`` resumes the paused run via :func:`fund.agents.graph.resume_run`
    (dispatched by gate); ``reject`` finalises it rejected in the DB alone, so the
    emergency stop needs no model, store or checkpointer (:data:`MODEL_FREE_ACTIONS`).
    ``respond`` threads the adviser's ``pm_directive`` back to the PM at the
    ``place_orders`` gate (the run re-gates as ``responded`` or finalises incomplete)
    and, like every action here, is recorded with its own ``human_audit_log`` row;
    ``killswitch`` finalises a paused run through the same reject path but is logged
    under its own action (no paired undo).
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
        questionnaire: MiFID answers for the profiler (``profile`` only): free text it
            normalises via the LLM, or a ``MiFIDAnswers`` (instance or JSON) it uses
            verbatim.
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
        sqlalchemy.exc.DBAPIError: A resume-style action whose ``idempotency_key`` is
            held by a concurrent in-flight submit; that submit owns the action.
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
            actor=actor,
            source=source,
            idempotency_key=idempotency_key,
            rationale=rationale,
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
    """Log the decision (claiming its idempotency key), then resume the paused run.

    ``decision`` is the ``resume_run`` verb (``approve`` / ``reject`` / ``respond``);
    ``action`` is what gets logged — they coincide for approve/reject/respond but
    diverge for killswitch (logged distinctly, resumed as a reject). ``respond`` threads
    the adviser's ``pm_directive`` (required) back to the PM at the ``place_orders``
    gate. The agent stack is imported here, never at module scope.
    """
    if run_id is None:
        raise ValueError(f"{action!r} requires a run_id")
    if action == "respond" and not pm_directive:
        raise ValueError("'respond' requires a pm_directive")
    rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
    pid = portfolio_id if portfolio_id is not None else _portfolio_of(session, rid)
    ticketed = _ticketed_weights(session, rid) if decision == "approve" else None
    # Claim the idempotency key before resuming: the checkpointer commits on its own
    # connection, so once the graph has moved nothing rolls it back, and the audit row's
    # unique key is the only thing that can turn a concurrent duplicate submit away.
    row = _append_audit(
        session,
        action=action,
        run_id=rid,
        portfolio_id=pid,
        actor=actor,
        target="weights" if ticketed else None,
        new_value=ticketed,
        source=source,
        override_flag=override_flag,
        suitability_flag=suitability_flag,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    result: dict[str, Any] | None = None
    if decision == "reject":
        _reject_at_gate(session, rid)
    else:
        from fund.agents.graph import resume_run

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
    no paper ticket; claiming plain success would hide that from the adviser. A respond
    the PM answered without re-gating is reported ``incomplete``, not ``responded``.
    A PM follow-up turn that failed after the gate was consumed (``resume_error``) is
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
    elif action == "respond":
        run = AgentRunRepository(session).get_run(run_id)
        if run is not None and run.status == "incomplete":
            # The PM answered without re-gating, so the run was finalised: "responded"
            # alone would send the adviser to wait for a gate that never appears.
            message = (
                f"run {run_id} incomplete (the PM did not re-gate after the response)"
            )
    error = (result or {}).get("resume_error")
    if error:
        message += f" (PM follow-up failed: {error})"
    return message


def _ticketed_weights(session: Session, run_id: uuid.UUID) -> dict[str, Any] | None:
    """The allocation an approve will ticket, in the shape the audit row records.

    The gated ``place_orders`` ignores the PM's own ``weights`` argument and trades the
    allocator's audited optimizer weights, so those (not what the gate echoed) are what
    the adviser's approval commits. ``None`` when the run has no allocator proposal,
    e.g. a profiler gate.
    """
    from fund.audit.orders_repository import weights_hash
    from fund.audit.repository import AgentRunRepository

    weights = AgentRunRepository(session).latest_optimizer_weights(run_id)
    if not weights:
        return None
    return {"weights": weights, "weights_hash": weights_hash(weights)}


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

    The edit never touches the gated tool's args. It resolves ``new_value`` (the
    ``{ticker: weight}`` the adviser typed) onto the tickers the run's allocator
    proposed (case/whitespace-insensitive; anything else is refused), validates it
    against the same gate the risk agent's ``risk_check`` applies (long-only /
    fully-invested bounds, the K&E position cap, the ESG and instrument screen; a book
    that misses the budget only by display rounding is nudged onto it first, see
    :func:`_absorb_rounding`), confirms every line can fill, then appends a fresh
    **allocator / optimize_portfolio** decision carrying the edited weights and resumes
    through the existing approve path.
    ``_place_orders_impl`` re-reads ``latest_optimizer_weights`` on resume, so that
    appended row is the single source every downstream record (the paper ticket +
    ``weights_hash``, ``agent_runs.weights``, and the positions snapshot) derives
    from — no graph change, no divergence. The audit row is stamped ⚑
    ``override_flag`` since the human overrode the optimizer's allocation.

    Raises:
        ValueError: ``run_id`` or ``new_value`` is missing, a ticker is outside the
            run's allocator universe (or listed twice), the edited weights fail the
            risk invariants (the structured ``violations`` are reported), or a line
            has no fill bar.
        LookupError: ``run_id`` names no run.
    """
    from fund.agents.graph import resume_run
    from fund.agents.toolsets import (
        RunContext,
        check_weights_against_mandate,
        mandate_bounds,
    )
    from fund.audit.repository import AgentRunRepository
    from fund.tools.orders import fill_shortfalls

    if run_id is None:
        raise ValueError("'edit' requires a run_id")
    if not new_value:
        raise ValueError("'edit' requires new_value (the edited weights)")
    rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))

    run_repo = AgentRunRepository(session)
    run = run_repo.get_run(rid)
    if run is None:
        raise LookupError(f"no agent_run {rid}")
    if run.portfolio_id is None:
        raise ValueError(f"agent_run {rid} has no portfolio_id")
    # Every allocator row, not just the latest: a prior edit may have dropped a name
    # the adviser now wants back.
    universe = {
        ticker
        for decision in run.decisions
        if decision.agent == "allocator" and decision.step == "optimize_portfolio"
        for ticker in run_repo.decode_weights(decision)
    }
    edited = _canonical_weights(new_value, universe)
    ctx = RunContext(
        session=session,
        asof=run.asof,
        store=store,
        config=config,
        run_id=rid,
        portfolio_id=run.portfolio_id,
    )
    edited = _absorb_rounding(edited, mandate_bounds(ctx))
    check = check_weights_against_mandate(ctx, edited)
    if not check.get("ok"):
        raise ValueError(f"'edit' rejected: {check.get('error')}")
    report = check["data"]
    if not report["passed"]:
        raise ValueError(f"'edit' violates risk invariants: {report['violations']}")
    shortfalls = fill_shortfalls(session, run.asof, edited)
    if shortfalls:
        raise ValueError(f"'edit' cannot be ticketed: {'; '.join(shortfalls)}")

    # Claimed before the resume: the checkpointer commits on its own connection, so a
    # second submit of the same keyed action must be turned away by the unique
    # idempotency key before it can drive the same gate again.
    pid = portfolio_id if portfolio_id is not None else run.portfolio_id
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
    run_repo.append_decision(
        rid,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": edited}),
        hitl_decision={"decision": "edit", "human_override": True},
    )
    # The bound ``place_orders`` refuses weights the risk gate has not passed. The
    # edit just cleared the same mandate check ``risk_check`` applies, so record that
    # as the verdict, labelled as the adviser's.
    run_repo.append_risk_verdict(
        rid, weights=edited, passed=True, violations=[], source="adviser_edit"
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
    return ActionResult(
        action="edit",
        run_id=rid,
        portfolio_id=pid,
        audit_id=row.id,
        message=resume_message(session, rid, "edit", result),
        detail=result,
    )


def _absorb_rounding(
    weights: dict[str, float], bounds: dict[str, float] | None
) -> dict[str, float]:
    """Spread a display-rounding residual over ``weights`` without breaching ``bounds``.

    The shell shows and parses weights at :data:`_EDIT_RESOLUTION`, so a book that is
    economically fully invested (a third each) comes back summing to 0.9999, which the
    risk gate's 1e-6 budget tolerance would bounce. Each name is only known to within
    half a step, so the residual is shared out in proportion to how far each held name
    can move in the needed direction: at most half a step, never past its min/max
    bound, and a name the adviser dropped stays dropped. A residual that headroom
    cannot cover is a real hole in the book and is left for ``risk_check`` to refuse.
    """
    limits = bounds or {}
    residual = limits.get("budget", 1.0) - sum(weights.values())
    half_step = _EDIT_RESOLUTION / 2
    if residual > 0:
        ceiling = limits.get("max_weights", 1.0)
        room = {
            ticker: min(half_step, max(ceiling - weight, 0.0)) if weight > 0 else 0.0
            for ticker, weight in weights.items()
        }
    else:
        floor = limits.get("min_weights", 0.0)
        room = {
            ticker: min(half_step, max(weight - floor, 0.0))
            for ticker, weight in weights.items()
        }
    capacity = sum(room.values())
    # ``not 0 < ...`` also turns a NaN residual (a non-finite weight) away.
    if capacity <= 0 or not 0 < abs(residual) <= capacity + _ROUNDING_EPS:
        return weights
    share = max(-1.0, min(1.0, residual / capacity))
    return {ticker: weight + share * room[ticker] for ticker, weight in weights.items()}


def _canonical_weights(new_value: Any, universe: set[str]) -> dict[str, float]:
    """Map the adviser's typed ``{ticker: weight}`` onto the allocator's own tickers.

    Matching ignores case and surrounding whitespace so ``" aaa "`` lands on ``AAA``
    instead of becoming a phantom position. Tickers outside ``universe`` bypass the
    run's universe screen and are refused, as is one name spelled two ways (the later
    weight would silently win).

    Raises:
        ValueError: A ticker is outside ``universe`` or appears twice.
    """
    canonical = {ticker.strip().upper(): ticker for ticker in universe}
    edited: dict[str, float] = {}
    unknown: list[str] = []
    for raw, weight in dict(new_value).items():
        ticker = canonical.get(str(raw).strip().upper())
        if ticker is None:
            unknown.append(str(raw))
        elif ticker in edited:
            raise ValueError(f"'edit' lists {ticker!r} twice")
        else:
            edited[ticker] = float(weight)
    if unknown:
        raise ValueError(
            f"'edit' rejected: {sorted(unknown)} not in the run's allocator universe "
            f"{sorted(universe)}"
        )
    return edited


_FREEZE_SCOPES = ("run", "portfolio", "fund")
_REARM_SCOPES = ("portfolio", "fund")


def _reject_at_gate(session: Session, run_id: uuid.UUID) -> None:
    """Finalise one paused run as rejected straight in the DB — no graph, no model.

    Reject / killswitch / run-scope freeze are the human's emergency stop, so they must
    work with the provider down or the portfolio's ConstraintSet gone. Resuming the
    graph cannot promise that (it rebuilds the PM agent, re-guards the ConstraintSet and
    takes a post-reject model turn), and a failure there consumes the checkpoint's
    interrupt without recording the reject. The standing interrupt is left alone, as a
    portfolio/fund freeze leaves it: the finalised status keeps ``resume_fund`` from
    ever re-driving it. Writes the same decision row the graph path logs for the gate.
    No commit (the caller owns it).

    Raises:
        LookupError: ``run_id`` names no run.
        ValueError: The run is already finalised.
    """
    from fund.audit.repository import AgentRunRepository

    repo = AgentRunRepository(session)
    run = repo.get_run(run_id)
    if run is None:
        raise LookupError(f"no agent_run {run_id}")
    if run.status in _FINAL_STATUSES:
        raise ValueError(
            f"agent_run {run_id} is {run.status!r}, already finalised and not "
            "rejectable (it completed, failed, or was rejected/frozen/blocked since "
            "the gate opened)"
        )
    if (run.optimizer_config or {}).get("step") == "profiler":
        agent, step = "profiler", "save_profile"
    else:
        agent, step = "orchestrator", "place_orders"
    repo.append_decision(
        run_id, agent=agent, step=step, hitl_decision={"decision": "reject"}
    )
    repo.finalize_run(run_id, weights={}, status="rejected")


def _reject_live_runs(session: Session, portfolio_id: uuid.UUID | None) -> None:
    """Finalise any paused/responded runs in scope as rejected (DB-level halt).

    A portfolio/fund freeze halts in-flight runs: each live gate is marked rejected
    without a graph resume (an emergency stop needs no model/checkpointer). Scoped to
    ``portfolio_id`` when given, else fund-wide. No commit (the caller owns it).
    """
    from fund.audit.repository import AgentRunRepository, RunNotOpenError

    repo = AgentRunRepository(session)
    for run in repo.list_paused_runs(portfolio_id=portfolio_id):
        # A resolver that finalised the run since the listing (an approve in flight)
        # won the race; its outcome must stand instead of being rewritten to rejected.
        with contextlib.suppress(RunNotOpenError):
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
    actor: str,
    source: str | None,
    idempotency_key: str | None,
    rationale: str | None,
) -> ActionResult:
    """Freeze autonomy at ``scope`` (``run`` / ``portfolio`` / ``fund``) and audit it.

    ``run`` keeps the killswitch semantics — reject-finalise the paused run in the DB
    (no model needed), touching no autonomy posture. ``portfolio`` / ``fund`` flip the
    autonomy row FROZEN (via :class:`AutonomyRepository`, recording ``rationale`` as the
    frozen reason) and halt any live run in scope. Every path appends one ⚑
    ``human_audit_log`` row.

    Raises:
        ValueError: ``scope`` is not one of ``run`` / ``portfolio`` / ``fund``, or the
            required ``run_id`` (``run``) / ``portfolio_id`` (``portfolio``) is missing.
    """
    if scope not in _FREEZE_SCOPES:
        raise ValueError(f"'freeze' requires scope in {_FREEZE_SCOPES}, got {scope!r}")

    if scope == "run":
        if run_id is None:
            raise ValueError("'freeze' scope='run' requires a run_id")
        rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
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
        _reject_at_gate(session, rid)
        return ActionResult(
            action="freeze",
            run_id=rid,
            portfolio_id=pid,
            audit_id=row.id,
            message=f"run {rid} {_PAST['freeze']}",
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
    :attr:`ActionResult.detail` is ``None``. This read only sees committed rows; the
    resume-style actions additionally claim the key by writing their audit row before
    any side effect, so a submit racing an in-flight one is stopped by the unique key.
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

    A hard suitability breach (ESG exclusions covering every sector) does not raise:
    the blocked run is audited and returned as an :class:`ActionResult` whose
    ``detail`` is ``{"blocked": True, "reason": ...}``.

    Raises:
        ValueError: ``portfolio_id`` is missing.
        RuntimeError: The profiler ended without pausing at the gate, so there is
            nothing to approve (the run is finalised ``incomplete``).
    """
    if portfolio_id is None:
        raise ValueError("'profile' requires a portfolio_id")
    from fund.agents.profiler import SuitabilityBreachError, run_profiler

    try:
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
    except SuitabilityBreachError as exc:
        # A hard breach is a terminal business outcome, not a failure: returning (not
        # raising) lets the caller commit the blocked run and this audit row, which
        # an exception would roll back with the rest of the transaction.
        row = _append_audit(
            session,
            action="profile",
            run_id=exc.run_id,
            portfolio_id=portfolio_id,
            actor=actor,
            target="mifid_profile",
            new_value={"outcome": "blocked", "reason": str(exc)},
            source=source,
            override_flag=False,
            suitability_flag=True,
            idempotency_key=idempotency_key,
            rationale=rationale,
        )
        return ActionResult(
            action="profile",
            run_id=exc.run_id,
            portfolio_id=portfolio_id,
            audit_id=row.id,
            message=f"profile blocked by a suitability breach: {exc}",
            detail={"blocked": True, "reason": str(exc)},
        )
    if run.interrupt is None:
        raise RuntimeError(
            f"profiler run {run.run_id} ended without reaching the approval gate; "
            "no profile was saved and there is nothing to approve"
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


def append_run_audit(
    session: Session,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID | None,
    actor: str = "human",
    source: str | None = None,
    idempotency_key: str | None = None,
    rationale: str | None = None,
) -> HumanAuditLog:
    """Append the ``human_audit_log`` ``run`` row for a rebalance a human launched.

    The blocking ``run`` action and the live-streamed round (which drives
    ``stream_fund`` itself so it can push each step to the timeline) both record the
    request through this one row shape, so the Audit view shows who triggered a round
    whichever launchpad they used. No commit (the caller owns the transaction).

    Args:
        session: The round's session; the row joins its transaction.
        run_id: The run the round opened.
        portfolio_id: The portfolio the round rebalanced.
        actor: Who launched the round.
        source: Provenance tag recorded with the row.
        idempotency_key: Dedup key; a repeat returns the first row.
        rationale: Free-text reason recorded with the action.

    Returns:
        The flushed audit row.
    """
    return _append_audit(
        session,
        action="run",
        run_id=run_id,
        portfolio_id=portfolio_id,
        actor=actor,
        target="rebalance",
        source=source,
        override_flag=False,
        suitability_flag=False,
        idempotency_key=idempotency_key,
        rationale=rationale,
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
    HITL gate for approval. A pipeline that ends before the gate finalises
    ``incomplete``; the returned ``detail["status"]`` tells the two apart. The agent
    stack is imported here, never at module scope.
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
    row = append_run_audit(
        session,
        run_id=run.run_id,
        portfolio_id=portfolio_id,
        actor=actor,
        source=source,
        idempotency_key=idempotency_key,
        rationale=rationale,
    )
    outcome = "awaiting approval" if run.status == "paused" else "no ticket to approve"
    status = "rejected" if halted else run.status
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
        detail={"status": status},
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
    "MODEL_FREE_ACTIONS",
    "ActionResult",
    "append_run_audit",
    "apply_action",
    "halt_run_if_frozen",
    "is_within_suitability_band",
    "resume_message",
]
