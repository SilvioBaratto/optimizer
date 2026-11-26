"""Settling and resuming a round paused at its HITL gate.

Reads the interrupt a round ended on, finalises the run row for each
ending, and builds the resume command for an approve, reject or respond,
with the guards that refuse a stale, frozen or superseded resume.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from fund.agents.graph_journal import _holdings_from_ticket, _write_portfolio_journal

if TYPE_CHECKING:
    import datetime as dt

    from fund.schemas.mandate import PortfolioMandate


def _coerce_uuid(portfolio_id: uuid.UUID | str) -> uuid.UUID:
    """Normalise ``portfolio_id`` to a ``UUID`` (accepts a canonical string)."""
    if isinstance(portfolio_id, uuid.UUID):
        return portfolio_id
    return uuid.UUID(portfolio_id)


def _extract_interrupt(result: Any) -> dict[str, Any] | None:
    """Pull the HITL interrupt payload from a compiled-graph invoke result."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    value = interrupts[0].value
    if not isinstance(value, dict):
        return None
    return cast("dict[str, Any]", value)


def _finalize_hitl(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
    decision: str,
    order_lines: list[dict[str, Any]] | None = None,
    ticket: Any | None = None,
) -> None:
    """Finalise a HITL decision — the single write path both resume routes share.

    Reached through :func:`_settle_hitl` by :meth:`FundRun.resume` (the live in-process
    handle) **and** :func:`resume_fund` (the cross-process rebuild), so the two cannot
    diverge.
    ``approve`` finalises the run ``completed`` with the optimizer ``weights`` and
    replaces the portfolio's ``positions`` snapshot built from them (O1 = Option B
    per-ticker row dicts; ``order_lines``, when supplied, carries the filled
    shares/notional). ``ticket``, the paper order the approve placed, links every
    snapshot row to it and supplies each row's base notional and the journal's trades.
    Any other ``decision`` writes no order, records the orchestrator's HITL choice,
    and finalises the run ``rejected``.

    The ``agent.invoke(Command(resume=…))`` that actually runs (approve) or denies
    (reject) the gated ``place_orders`` tool is the **caller's** job — inside that
    invoke the executor's execution decision and the paper ticket are written. This
    helper only stamps the terminal audit state + the current-holdings snapshot. No
    ``commit`` (the caller owns the transaction).
    """
    from fund.audit import AgentRunRepository, PositionRepository

    audit = AgentRunRepository(session)
    if decision == "approve":
        audit.finalize_run(run_id, weights=weights, status="completed")
        positions = PositionRepository(session)
        # Capture prior holdings *before* set_holdings overwrites the snapshot, so
        # the overlay's drift is this bar's change rather than a no-op against itself.
        prior_weights = {
            p.ticker: float(p.weight) for p in positions.get_holdings(portfolio_id)
        }
        if order_lines:
            holdings = order_lines
        elif ticket is not None:
            holdings = _holdings_from_ticket(ticket)
        else:
            holdings = [
                {"ticker": ticker, "weight": weight}
                for ticker, weight in weights.items()
            ]
        positions.set_holdings(
            portfolio_id,
            holdings,
            asof=asof,
            paper_order_id=ticket.id if ticket is not None else None,
        )
        journal_lines = order_lines
        if journal_lines is None and ticket is not None:
            journal_lines = ticket.lines
        _write_portfolio_journal(
            session,
            run_id=run_id,
            portfolio_id=portfolio_id,
            asof=asof,
            weights=weights,
            order_lines=journal_lines,
            prior_weights=prior_weights,
        )
    else:
        audit.append_decision(
            run_id,
            agent="orchestrator",
            step="place_orders",
            hitl_decision={"decision": decision},
        )
        audit.finalize_run(run_id, weights={}, status="rejected")


def _placed_ticket(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
) -> Any | None:
    """The ticket this run's approved ``place_orders`` left for ``weights``, if any.

    ``place_orders`` can decline an approved gate (a missing fill bar, the current
    mandate no longer admits the audited weights, or the risk gate had not passed
    them) and still return normally to the graph, so a resume that completed without
    raising proves nothing about the ticket. The ticket row, keyed
    ``(portfolio_id, asof, weights_hash)`` like the tool's own idempotency lookup, is
    the only evidence that the book moved; an idempotent re-run that returned the
    stored ticket counts as placed. The tool is refused unless the risk gate passed
    exactly these weights, so a ticket an earlier run left on the same bar does not
    count when risk never passed this one. No weights means nothing could have been
    placed.
    """
    from fund.audit import AgentRunRepository, OrderRepository
    from fund.audit.orders_repository import weights_hash

    if not weights or not AgentRunRepository(session).risk_passed(run_id, weights):
        return None
    return OrderRepository(session).get_by_key(
        portfolio_id=portfolio_id, asof=asof, weights_hash=weights_hash(weights)
    )


def _ticket_placed(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
) -> bool:
    """Whether :func:`_placed_ticket` finds a ticket for ``weights``."""
    return (
        _placed_ticket(
            session,
            run_id=run_id,
            portfolio_id=portfolio_id,
            asof=asof,
            weights=weights,
        )
        is not None
    )


def _settle_hitl(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
    decision: str,
    result: Any,
    regated_from: str | None = None,
) -> None:
    """Finalise an approve/reject resume, verifying an approve actually ticketed.

    ``place_orders`` reports a missing fill bar, an unknown ticker or a mandate breach
    as an ``{ok: false}`` envelope instead of raising, so resuming the gate succeeds
    even when no ticket was written. An approve with no ticket must not stamp the run
    ``completed`` nor replace the ``positions`` snapshot with weights no order backs:
    it finalises ``incomplete`` (``"place_orders_failed"``). When the resumed PM
    answered by re-gating (``result`` carries a fresh interrupt) the run stays alive as
    ``"responded"`` instead, with the declined approve on the audit trail. Reject and
    a ticketed approve go through :func:`_finalize_hitl`.

    An approve that ``regated_from`` an earlier, non-order gate (see
    :class:`_GateResume`) finalises nothing: it only records the decision and leaves
    the run alive at the fresh gate, whose own approve is the order commit. Finalising
    there would stamp the run ``completed`` with no ticket, replace the holdings with
    whatever weights existed before the allocator ran, and strand the real gate behind
    the finalised-run guard. A reject always closes the run, whatever the PM does
    afterwards, so the freeze-a-run killswitch (a reject) cannot be outlived by a
    re-gating PM. No commit.
    """
    if regated_from is not None and decision == "approve":
        from fund.audit import AgentRunRepository

        AgentRunRepository(session).append_decision(
            run_id,
            agent="orchestrator",
            step=regated_from,
            hitl_decision={"decision": decision, "regated": True},
        )
        return
    ticket = (
        _placed_ticket(
            session,
            run_id=run_id,
            portfolio_id=portfolio_id,
            asof=asof,
            weights=weights,
        )
        if decision == "approve"
        else None
    )
    if decision != "approve" or ticket is not None:
        _finalize_hitl(
            session,
            run_id=run_id,
            portfolio_id=portfolio_id,
            asof=asof,
            weights=weights,
            decision=decision,
            ticket=ticket,
        )
        return

    from fund.audit import AgentRunRepository

    audit = AgentRunRepository(session)
    if _extract_interrupt(result) is None:
        _finalize_incomplete(audit, run_id, "place_orders_failed")
        return
    audit.append_decision(
        run_id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "approve", "reason": "place_orders_failed"},
    )
    audit.mark_responded(run_id)


_FINALISED_STATUSES = ("completed", "rejected", "incomplete", "blocked", "failed")


def _require_resumable(run: Any) -> None:
    """Refuse to resume a run that is not at a live gate.

    An external freeze marks live gates "rejected" at the DB without resolving the
    checkpoint (a completed / incomplete / blocked run, or a stale observer racing a
    concurrent approve, is the same shape): the LangGraph checkpoint is left standing,
    so without this guard a blind ``Command(resume=…)`` would re-drive
    ``place_orders`` and ship live orders against a halted run. A ``pending`` row is no
    gate either: the run is still being driven, or crashed before reaching one, and
    resuming it would re-run the PM headlessly and finalise it as approved with no
    gate and no ticket. Only ``paused`` / ``responded`` runs are resumable. Shared by
    :meth:`FundRun.resume` and :func:`resume_fund`.

    Raises:
        ValueError: ``run`` is finalised, or has not reached a live gate.
    """
    from fund.audit.repository import RESUMABLE_RUN_STATUSES

    if run.status in RESUMABLE_RUN_STATUSES:
        return
    if run.status in _FINALISED_STATUSES:
        raise ValueError(
            f"agent_run {run.id} is {run.status!r}, already finalised and not "
            "resumable (it completed, failed, or was rejected/frozen/blocked since "
            "the gate opened)"
        )
    raise ValueError(
        f"agent_run {run.id} is {run.status!r}, not at a live gate and not resumable "
        "(it is still being driven, or crashed before reaching its gate)"
    )


# Provider stop reasons (lower-cased) that say the output cap, not the model, ended a
# turn, under the key each provider family reports them with.
_OUTPUT_CAP_STOPS = frozenset({"max_tokens", "length"})
_STOP_REASON_KEYS = ("stop_reason", "finish_reason", "done_reason", "stopReason")


def _last_message(result: Any) -> Any:
    """The final message of a compiled-graph result (``None`` when it has none)."""
    messages = result.get("messages") if isinstance(result, dict) else None
    return messages[-1] if messages else None


def _ungated_end_reason(last_message: Any, *, pm_round_cap_reached: bool) -> str:
    """Why the PM's last turn ended the run without reaching the ``place_orders`` gate.

    ``create_agent`` treats any turn without a valid tool call as a final answer, and
    langchain surfaces a truncated or unparsable tool call only as
    ``invalid_tool_calls``. Telling those apart from a blocking risk verdict is the
    only way the adviser can see why a round ended: ``"pm_round_cap"`` when the
    delegation cap tripped, ``"output_truncated"`` when the provider reported an
    output-cap stop, ``"invalid_tool_call"`` when the call was malformed, and
    ``"no_order"`` otherwise (nothing left to commit).
    """
    if pm_round_cap_reached:
        return "pm_round_cap"
    metadata = getattr(last_message, "response_metadata", None) or {}
    if any(
        str(metadata.get(key, "")).lower() in _OUTPUT_CAP_STOPS
        for key in _STOP_REASON_KEYS
    ):
        return "output_truncated"
    if getattr(last_message, "invalid_tool_calls", None):
        return "invalid_tool_call"
    return "no_order"


def _finalize_incomplete(audit: Any, run_id: uuid.UUID, reason: str) -> None:
    """Finalise a run that ended without a ticket — the shared no-ticket end.

    Records the orchestrator's incomplete decision with ``reason`` (``"round_cap"``,
    ``"place_orders_failed"`` or one of :func:`_ungated_end_reason`'s) and finalises
    the run ``incomplete`` with no weights. Shared by the blocking :func:`run_fund`, the
    streaming :func:`stream_fund` and the approve resume so they cannot diverge on the
    no-ticket path. No commit.
    """
    audit.append_decision(
        run_id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "incomplete", "reason": reason},
    )
    audit.finalize_run(run_id, weights={}, status="incomplete")


# The PM prompt and the fund-orchestration skill teach the PM to recognise this
# opening; keep the marker in sync with them.
_RESPOND_FRAME = (
    "ADVISER DIRECTIVE — place_orders was NOT executed and no ticket exists. "
    "Revise the proposal per the directive below, re-run the affected pipeline "
    "stages (the risk gate included) and call place_orders again. "
    "Directive: {directive}"
)


def _resume_command(
    decision: str, pm_directive: str | None, *, actions: int = 1
) -> Any:
    """Build the HITL ``Command(resume=…)`` for a gate decision.

    ``approve`` / ``reject`` carry only the decision type; ``respond`` carries the
    adviser's free-text ``pm_directive`` as the gated tool's result, threading it back
    to the PM so the gate continues instead of terminating. The middleware always
    delivers a respond as a *successful* ``place_orders`` ``ToolMessage``, which a bare
    sentence would make read like a placed order, so the directive is framed as a
    non-execution (see :data:`_RESPOND_FRAME`). The middleware demands exactly one
    decision per gated tool call, so the same decision is repeated ``actions`` times
    (a PM turn that emits parallel gated calls pauses with several action requests).
    Imports ``langgraph`` lazily to keep a bare ``import fund.agents.graph`` light.
    """
    from langgraph.types import Command

    decisions: list[dict[str, Any]] = [
        {
            "type": "respond",
            "message": _RESPOND_FRAME.format(directive=pm_directive or "(none given)"),
        }
        if decision == "respond"
        else {"type": decision}
        for _ in range(max(actions, 1))
    ]
    return Command(resume={"decisions": decisions})


def _pending_action_count(agent: Any, thread_config: dict[str, Any]) -> int:
    """How many gated tool calls the paused PM is awaiting a decision on.

    Read from the checkpointed interrupt rather than a cached payload, because a
    ``respond`` can re-gate with a different number of calls than the first pause.
    ``1`` when no interrupt carries action requests.
    """
    for pending in agent.get_state(thread_config).interrupts:
        value = pending.value
        requests = value.get("action_requests") if isinstance(value, dict) else None
        if requests:
            return len(requests)
    return 1


def _resume_agent(
    agent: Any,
    thread_config: dict[str, Any],
    decision: str,
    pm_directive: str | None,
) -> dict[str, Any]:
    """Resume the paused PM, answering every gated tool call it is waiting on."""
    result: dict[str, Any] = agent.invoke(
        _resume_command(
            decision,
            pm_directive,
            actions=_pending_action_count(agent, thread_config),
        ),
        config=thread_config,
    )
    return result


def _gate_consumed(agent: Any, thread_config: dict[str, Any]) -> bool:
    """Whether the thread already ran its ``place_orders`` call and moved past the gate.

    A gate decision is one-shot: once the checkpoint has advanced past the interrupt, a
    second ``Command(resume=...)`` replays nothing. A failed approve leaves exactly
    that behind when a later step of the same resume (the finalisation, the caller's
    commit) raises after ``place_orders`` ran. The caller then rolls the SQL session
    back and the ticket is gone, but the LangGraph saver commits each superstep on its
    own connection, so the checkpoint keeps the call.
    """
    state = agent.get_state(thread_config)
    if state.interrupts:
        return False
    messages = (state.values or {}).get("messages") or []
    return any(
        getattr(m, "type", None) == "tool"
        and getattr(m, "name", None) == "place_orders"
        for m in messages
    )


def _replace_consumed_ticket(ctx: Any, weights: dict[str, float]) -> None:
    """Place the approved ticket directly for a gate an earlier attempt consumed.

    Resuming a consumed gate cannot reach ``place_orders`` again, so a retried approve
    would otherwise dead-end on a missing ticket. The placement is the one the gate
    would have run (audited weights, mandate capital, risk-gate and mandate checks,
    executor decision) and is idempotent: an existing ticket is left alone.

    Raises:
        RuntimeError: The placement was refused (e.g. still no fill bar after the
            decision date); nothing was written.
    """
    from fund.agents.toolsets import place_audited_orders

    if _ticket_placed(
        ctx.session,
        run_id=ctx.run_id,
        portfolio_id=ctx.portfolio_id,
        asof=ctx.asof,
        weights=weights,
    ):
        return
    outcome = place_audited_orders(ctx)
    if not outcome.get("ok"):
        raise RuntimeError(
            f"approve could not place the paper ticket: {outcome.get('error')}"
        )


def _pending_gate_names(agent: Any, thread_config: dict[str, Any]) -> tuple[str, ...]:
    """Names of the gated tools the thread is currently paused on, in request order."""
    names: list[str] = []
    for interrupt in agent.get_state(thread_config).interrupts:
        value = getattr(interrupt, "value", None)
        requests = value.get("action_requests") if isinstance(value, dict) else None
        names.extend(str(r.get("name")) for r in requests or [] if isinstance(r, dict))
    return tuple(names)


@dataclass(frozen=True)
class _GateResume:
    """What driving one gate decision through the graph produced.

    ``regated_from`` names the gate an approve resolved when that gate was not the
    order commit and the PM, running on, stopped at a fresh one: the run is still
    awaiting an adviser decision. ``None`` for every other outcome.
    """

    result: dict[str, Any]
    regated_from: str | None = None


def _refuse_stale_approval(ctx: Any) -> None:
    """Refuse an approve that would backdate the ticket or roll the holdings back.

    The paper ticket fills at the first close after the run's ``asof`` whatever the
    date of the approval, so a gate left open past the approval window would let the
    adviser pick, with the bars since in view, which fills to keep. And a run for an
    earlier bar than the portfolio's current holdings would replace them with weights
    the later round already superseded. Either run can still be rejected.

    Raises:
        ValueError: The run opened more than ``approval_max_age_days`` ago, or the
            portfolio already holds positions stamped with a later ``asof``.
    """
    import datetime as dt

    from fund.audit import AgentRunRepository, PositionRepository
    from fund.audit.repository import run_age

    run = AgentRunRepository(ctx.session).get_run(ctx.run_id)
    window = dt.timedelta(days=ctx.config.approval_max_age_days)
    if run is not None and (age := run_age(run)) > window:
        raise ValueError(
            f"agent_run {ctx.run_id} is stale: it opened {age.days} days ago, past the "
            f"{ctx.config.approval_max_age_days}-day approval window; reject it and "
            "run a fresh round"
        )
    holdings = PositionRepository(ctx.session).get_holdings(ctx.portfolio_id)
    held = max((p.asof for p in holdings), default=None)
    if held is not None and held > ctx.asof:
        raise ValueError(
            f"agent_run {ctx.run_id} (asof {ctx.asof}) is older than the portfolio's "
            f"current holdings (asof {held}); reject it and run a fresh round"
        )


def _require_fill_bars(ctx: Any, weights: dict[str, float]) -> None:
    """Refuse an approve of the order gate that ``place_orders`` could not fill yet.

    Approving resumes the graph, which runs the gated ``place_orders`` and so consumes
    its interrupt. When that call fails for want of a fill bar the checkpoint is
    already past the gate (the saver commits on its own connection, so no rollback
    undoes it) while the run row stays ``paused``: the Decisions queue, which lists
    only live interrupts, no longer shows it. The ticket fills at the first close after
    ``asof`` and the weekly sweep sets ``asof`` to a Saturday, so an approve that lands
    before the next close is ingested is the ordinary case, not a rare error. Looking
    for the fills first leaves the gate untouched for the retry. A ticket that already
    exists needs none (``place_orders`` returns it as is).

    Raises:
        RuntimeError: A line has no fill bar after ``asof``; nothing was resumed.
    """
    if not weights:
        return
    if _ticket_placed(
        ctx.session,
        run_id=ctx.run_id,
        portfolio_id=ctx.portfolio_id,
        asof=ctx.asof,
        weights=weights,
    ):
        return
    from fund.tools.orders import fill_shortfalls

    shortfalls = fill_shortfalls(
        ctx.session, ctx.asof, weights, base_currency=ctx.base_currency
    )
    if shortfalls:
        raise RuntimeError(
            f"approve would leave no paper ticket ({'; '.join(shortfalls)}); the gate "
            f"stays open, so approve again once the next close after {ctx.asof} is "
            "ingested"
        )


def _finalize_respond(
    session: Any,
    *,
    run_id: uuid.UUID,
    result: Any,
    pm_directive: str | None,
) -> None:
    """Settle a ``respond`` barge-in: keep the run alive iff the PM re-gated.

    Shared by :meth:`FundRun.resume` and :func:`resume_fund` so the in-process and
    rebuild-to-resume paths cannot diverge. The respond is always appended to the audit
    trail (carrying the directive). When the resumed PM re-gates — ``result`` carries a
    fresh interrupt — the run is flipped to ``"responded"`` (alive, still awaiting
    approval) and is **not** finalised; a genuinely terminal result (no re-gate, so no
    ticket passed the gate) finalises the run ``incomplete`` — the same no-gate end as
    :func:`run_fund`. No commit (the caller owns the transaction).
    """
    from fund.audit import AgentRunRepository

    audit = AgentRunRepository(session)
    audit.append_decision(
        run_id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "respond", "message": pm_directive},
    )
    if _extract_interrupt(result) is not None:
        audit.mark_responded(run_id)
    else:
        reason = _ungated_end_reason(
            _last_message(result),
            pm_round_cap_reached=bool(
                isinstance(result, dict) and result.get("pm_round_cap_reached")
            ),
        )
        _finalize_incomplete(audit, run_id, reason)


def _frozen_scope(session: Any, portfolio_id: uuid.UUID) -> str | None:
    """The autonomy scope FROZEN for ``portfolio_id`` (``"fund"`` wins), else ``None``.

    Reads the database rather than the session's cached rows: a run session loads the
    armed rows when it opens, and a freeze committed by the operator's session while
    the round is in flight is otherwise invisible to it. Read-only: it never creates a
    row.
    """
    from fund.audit.autonomy_repository import AutonomyRepository

    scope = AutonomyRepository(session).frozen_scope(portfolio_id)
    return scope.value if scope is not None else None


def _raise_if_profile_superseded(session: Any, run: Any) -> None:
    """Raise ``RuntimeError`` when the portfolio was re-profiled after ``run`` opened.

    A gate parks the optimizer weights computed under the profile that was active when
    the round opened, and ``place_orders`` sizes the ticket from exactly those weights.
    A re-profile appends a new ``mifid_profiles`` version (the old one stays), so a
    gate left waiting would otherwise still be approvable and trade a book the new
    ``ConstraintSet`` never validated. A run opened before the version was recorded
    carries none and is not checked.

    Args:
        session: A sync ``portopt_db`` session; not owned here.
        run: The ``agent_run`` row being resumed; its ``optimizer_config`` holds the
            ``profile_version`` it was opened under.

    Raises:
        RuntimeError: A newer profile version exists; the message tells the adviser
            to reject the run and start a new round.
    """
    from fund.audit import MifidProfileRepository

    built_under = (run.optimizer_config or {}).get("profile_version")
    if built_under is None:
        return
    current = MifidProfileRepository(session).get_active(run.portfolio_id)
    if current is not None and current.version > built_under:
        raise RuntimeError(
            f"portfolio {run.portfolio_id} was re-profiled (v{built_under} -> "
            f"v{current.version}) after run {run.id} opened; reject this run and "
            "start a new round under the current profile"
        )


def _ticket_capital(run: Any, mandate: PortfolioMandate | None) -> float | None:
    """Capital the paper ticket is priced at when a gate resumes.

    The capital recorded when the run opened wins: it is the book the PM was told
    to allocate, so a mandate edited while the gate was pending must not silently
    re-price the ticket. Runs opened before capital was recorded fall back to the
    current mandate, and ``None`` leaves the paper-cost model's default notional.

    Args:
        run: The persisted ``agent_run`` row being resumed.
        mandate: The portfolio's current mandate, or ``None`` if it vanished.

    Returns:
        The notional to size the ticket with, or ``None`` for the default book.
    """
    recorded = (run.optimizer_config or {}).get("capital")
    if recorded is not None:
        return float(recorded)
    return float(mandate.capital) if mandate is not None else None
