"""Phase-7 orchestration graph: the subagents + the PM deep agent.

Assembles the Phase-7A foundations (model / backend / toolsets / prompts / skills)
into one running deep agent. A PM/orchestrator (``create_deep_agent``) delegates
via the built-in ``task`` tool to four stateless subagents in a fixed pipeline —
**economist → allocator → risk (blocking gate) → executor** — and owns the single
gated action: committing the paper trades behind the HITL ``place_orders`` gate.

Two shapes live here (Task 5 builds them; Task 6 drives them end-to-end):

* ``build_<role>_subagent(ctx)`` → a deepagents ``SubAgent`` dict. Subagents do
  **not** inherit tools/skills, so each carries its own — the per-run toolset
  (:func:`fund.agents.toolsets.bind_toolset`, session already bound) and the
  per-role skills. **No subagent sets ``interrupt_on``** (HITL is PM-level only);
  the critical roles (allocator/risk/executor) attach a
  ``ModelFallbackMiddleware`` when a fallback model is supplied.
* ``build_fund_agent(...)`` → the PM ``create_deep_agent`` wiring the orchestrator
  toolset (the top-level gated ``place_orders``), the four subagents, the
  root-relative skills, the ``virtual_mode`` backend, the checkpointer/store, and
  the effective ``interrupt_on`` gate set (config baseline ∪ the mandate's
  ``hitl_gates``, computed by the caller).

**Risk R1 (verified):** deepagents loads skills **through the backend**, which
runs in ``virtual_mode`` and blocks absolute paths outside its root — so every
agent's ``skills`` are the root-relative
:func:`fund.agents.skills.skill_sources` (the per-role dirs the backend stages),
never the absolute ``skill_paths``.

The agent stack (``deepagents`` / ``langchain``) is imported **inside** the
builders, so a bare ``import fund.agents.graph`` drags in no agent runtime and
needs no environment (mirrors :mod:`fund.agents.profiler`).

The subagent factory and caps, the run instruction, the HITL settle and resume
helpers, the failed-round record and the stream-event shaping live in the
``graph_*`` modules beside this one. The run drivers, ``FundRun``,
``build_fund_agent`` and every function that calls a name the tests replace on
this module stay here, so ``monkeypatch.setattr(fund.agents.graph, ...)`` still
reaches the caller; the role builders and their descriptions stay too, as a
skills test reads ``_ALLOCATOR_DESCRIPTION`` from this file's source. Every name
defined here before that split is re-exported.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from fund.agents.backend import build_backend, read_only_permissions
from fund.agents.graph_assembly import (
    _general_purpose_subagent,
    _pm_round_cap_middleware,
    _subagent,
)
from fund.agents.graph_assembly import (
    _subagent_call_cap_middleware as _subagent_call_cap_middleware,
)
from fund.agents.graph_failures import _FAILED_ROUND_ATTR as _FAILED_ROUND_ATTR
from fund.agents.graph_failures import _RUN_ERROR_MAX_CHARS as _RUN_ERROR_MAX_CHARS
from fund.agents.graph_failures import (
    FailedRound,
    _consume_first_round,
    _finalize_failed,
    _hand_back_first_round,
    failed_round_of,
    record_failed_round,
)
from fund.agents.graph_failures import _attach_failed_round as _attach_failed_round
from fund.agents.graph_failures import (
    _drop_checkpoint_thread as _drop_checkpoint_thread,
)
from fund.agents.graph_failures import _own_session as _own_session
from fund.agents.graph_failures import _restore_first_round as _restore_first_round
from fund.agents.graph_hitl import _FINALISED_STATUSES as _FINALISED_STATUSES
from fund.agents.graph_hitl import _OUTPUT_CAP_STOPS as _OUTPUT_CAP_STOPS
from fund.agents.graph_hitl import _RESPOND_FRAME as _RESPOND_FRAME
from fund.agents.graph_hitl import _STOP_REASON_KEYS as _STOP_REASON_KEYS
from fund.agents.graph_hitl import (
    _coerce_uuid,
    _extract_interrupt,
    _finalize_incomplete,
    _finalize_respond,
    _frozen_scope,
    _gate_consumed,
    _GateResume,
    _last_message,
    _pending_gate_names,
    _raise_if_profile_superseded,
    _refuse_stale_approval,
    _replace_consumed_ticket,
    _require_fill_bars,
    _require_resumable,
    _resume_agent,
    _settle_hitl,
    _ticket_capital,
    _ungated_end_reason,
)
from fund.agents.graph_hitl import _finalize_hitl as _finalize_hitl
from fund.agents.graph_hitl import _pending_action_count as _pending_action_count
from fund.agents.graph_hitl import _placed_ticket as _placed_ticket
from fund.agents.graph_hitl import _resume_command as _resume_command
from fund.agents.graph_hitl import _ticket_placed as _ticket_placed
from fund.agents.graph_journal import _drift_vs_prior as _drift_vs_prior
from fund.agents.graph_journal import _holdings_from_ticket as _holdings_from_ticket
from fund.agents.graph_journal import _optimizer_weights
from fund.agents.graph_journal import _portfolio_narrative as _portfolio_narrative
from fund.agents.graph_journal import _trades_from_lines as _trades_from_lines
from fund.agents.graph_journal import (
    _write_portfolio_journal as _write_portfolio_journal,
)
from fund.agents.graph_prompts import _BROKER_BAR_EXPLANATION as _BROKER_BAR_EXPLANATION
from fund.agents.graph_prompts import _BROKER_SCOPE_INTRO as _BROKER_SCOPE_INTRO
from fund.agents.graph_prompts import _DIRECTIVE_MAX_CHARS as _DIRECTIVE_MAX_CHARS
from fund.agents.graph_prompts import _DIRECTIVES_BUDGET as _DIRECTIVES_BUDGET
from fund.agents.graph_prompts import (
    _ELIGIBLE_ALLOWED_BUDGET as _ELIGIBLE_ALLOWED_BUDGET,
)
from fund.agents.graph_prompts import _ELIGIBLE_BARRED_BUDGET as _ELIGIBLE_BARRED_BUDGET
from fund.agents.graph_prompts import (
    _HISTORY_DIGEST_WINDOW_DAYS as _HISTORY_DIGEST_WINDOW_DAYS,
)
from fund.agents.graph_prompts import _HISTORY_SUMMARY_BUDGET as _HISTORY_SUMMARY_BUDGET
from fund.agents.graph_prompts import _HISTORY_SUMMARY_PREFIX as _HISTORY_SUMMARY_PREFIX
from fund.agents.graph_prompts import _UNBOUNDED as _UNBOUNDED
from fund.agents.graph_prompts import _constraint_digest as _constraint_digest
from fund.agents.graph_prompts import _directives_block as _directives_block
from fund.agents.graph_prompts import _eligible_block as _eligible_block
from fund.agents.graph_prompts import _fair_shares as _fair_shares
from fund.agents.graph_prompts import _fit_entries as _fit_entries
from fund.agents.graph_prompts import _fit_prose as _fit_prose
from fund.agents.graph_prompts import _history_summary, _run_instruction
from fund.agents.graph_stream import _UNGATED_END_TEXT as _UNGATED_END_TEXT
from fund.agents.graph_stream import _gate_why as _gate_why
from fund.agents.graph_stream import (
    _message_text,
    _round_end_event,
    _RoundEnd,
    _terminal_event,
    _tool_calls,
)
from fund.agents.prompts import (
    ALLOCATOR_SYSTEM_PROMPT,
    ECONOMIST_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    PM_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
)
from fund.agents.skills import skill_sources
from fund.agents.summarization import summarization_override
from fund.config import DEFAULT_LOOKBACK_DAYS, FundConfig, settings

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterator, Sequence

    # ``fund.agents.toolsets`` reaches ``fund.tools`` (the agent stack) at import;
    # keep it type-only here and import ``bind_toolset`` lazily inside the builders
    # so a bare ``import fund.agents.graph`` stays agent-stack-free (profiler style).
    # The stream DTO below is type-only too; the generator imports it lazily.
    from fund.agents.toolsets import RunContext
    from fund.schemas import ConstraintSet
    from fund.schemas.mandate import PortfolioMandate
    from fund.tui.stream import StreamEvent


_log = logging.getLogger(__name__)


__all__ = [
    "FailedRound",
    "FundRun",
    "build_allocator_subagent",
    "build_economist_subagent",
    "build_executor_subagent",
    "build_fund_agent",
    "build_risk_subagent",
    "failed_round_of",
    "record_failed_round",
    "resume_fund",
    "resume_run",
    "run_fund",
    "stream_fund",
]


# The backtest's rolling training window recorded on every run (the
# ``RunContext.lookback_days`` default — 3y, ~756 trading days). Moments and the
# optimiser fit on the full history up to ``asof``, so the record names only the
# backtest as applying it.
_LOOKBACK_DAYS = DEFAULT_LOOKBACK_DAYS

# One-line descriptions the PM reads (via the built-in ``task`` tool) to decide
# when to delegate — each names the role's fixed pipeline position.
_ECONOMIST_DESCRIPTION = (
    "Pipeline step 1. Reads the macro and price backdrop and hands the allocator a "
    "qualitative regime narrative plus candidate views, kept to the eligible "
    "universe when the instruction lists one. Delegate first."
)
_ALLOCATOR_DESCRIPTION = (
    "Pipeline step 2. Pre-selects the candidate tickers it is handed (all of the "
    "eligible universe when the instruction lists one), estimates moments, and "
    "runs the optimizer to produce candidate weights under the resolved "
    "constraints; the economist's views are advisory context, not an optimizer "
    "input. Delegate after the economist."
)
_RISK_DESCRIPTION = (
    "Pipeline step 3 — the blocking gate. Checks the allocator's proposal against "
    "the MiFID/ESG/validation limits and reports pass or fail; nothing proceeds to "
    "the executor until it passes."
)
_EXECUTOR_DESCRIPTION = (
    "Pipeline step 4. Turns the risk-approved target weights into a paper rebalance "
    "proposal for the PM to commit. Delegate only once risk has passed."
)


def build_economist_subagent(
    ctx: RunContext, *, summarization: Sequence[Any] = ()
) -> dict[str, Any]:
    """The economist subagent (step 1) — narrative regime read, no fallback."""
    return _subagent(
        "economist",
        description=_ECONOMIST_DESCRIPTION,
        system_prompt=ECONOMIST_SYSTEM_PROMPT,
        ctx=ctx,
        summarization=summarization,
    )


def build_allocator_subagent(
    ctx: RunContext,
    *,
    fallback: Any | None = None,
    summarization: Sequence[Any] = (),
) -> dict[str, Any]:
    """The allocator subagent (step 2) — the optimizer path; critical role."""
    return _subagent(
        "allocator",
        description=_ALLOCATOR_DESCRIPTION,
        system_prompt=ALLOCATOR_SYSTEM_PROMPT,
        ctx=ctx,
        fallback=fallback,
        summarization=summarization,
    )


def build_risk_subagent(
    ctx: RunContext,
    *,
    fallback: Any | None = None,
    summarization: Sequence[Any] = (),
) -> dict[str, Any]:
    """The risk subagent (step 3) — the blocking gate; critical role."""
    return _subagent(
        "risk",
        description=_RISK_DESCRIPTION,
        system_prompt=RISK_SYSTEM_PROMPT,
        ctx=ctx,
        fallback=fallback,
        summarization=summarization,
    )


def build_executor_subagent(
    ctx: RunContext,
    *,
    fallback: Any | None = None,
    summarization: Sequence[Any] = (),
) -> dict[str, Any]:
    """The executor subagent (step 4) — proposes the rebalance; critical role.

    Binds no tool from the frozen eight (``place_orders`` is the PM's tool, behind
    the HITL gate): the executor *proposes*, the PM *commits*.
    """
    return _subagent(
        "executor",
        description=_EXECUTOR_DESCRIPTION,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        ctx=ctx,
        fallback=fallback,
        summarization=summarization,
    )


def build_fund_agent(
    model: Any,
    *,
    checkpointer: Any,
    store: Any | None,
    ctx: RunContext,
    config: FundConfig = settings,
    fallback: Any | None = None,
    interrupt_on: dict[str, Any] | None = None,
) -> Any:
    """Assemble the PM ``deepagents`` agent that orchestrates one fund run.

    The PM runs under ``PM_SYSTEM_PROMPT`` with the orchestrator toolset — the
    single top-level ``place_orders`` tool, gated by ``interrupt_on`` (which
    requires the ``checkpointer``) so the commit always pauses for the adviser. The
    gate set is the ``interrupt_on`` override when supplied (the caller folds the
    mandate's ``hitl_gates`` in), else the config default (``place_orders``). It
    delegates through the built-in ``task`` tool to the four subagents in pipeline
    order (economist → allocator → risk → executor); the critical three carry the
    ``fallback`` middleware when one is supplied, and every delegation target (the
    ``general-purpose`` one included) is bounded by the per-delegation model-call
    cap. Skills are the root-relative
    staged sources (Risk R1) resolved through the ``virtual_mode`` ``backend``, which
    every role may read but none may write (:func:`read_only_permissions`).
    ``temperature=0`` / the DeepSeek route ride on ``model`` (built from
    ``FundConfig``), not passed here.
    """
    from deepagents import create_deep_agent

    from fund.agents.toolsets import bind_toolset, build_gate_weights_middleware

    gates = interrupt_on if interrupt_on is not None else config.interrupt_on_map()
    backend = build_backend(config, run_id=ctx.run_id)

    def summarization() -> list[Any]:
        return summarization_override(model, backend, config)

    subagents = [
        build_economist_subagent(ctx, summarization=summarization()),
        build_allocator_subagent(ctx, fallback=fallback, summarization=summarization()),
        build_risk_subagent(ctx, fallback=fallback, summarization=summarization()),
        build_executor_subagent(ctx, fallback=fallback, summarization=summarization()),
        _general_purpose_subagent(config, summarization=summarization()),
    ]
    return create_deep_agent(
        model=model,
        tools=bind_toolset("orchestrator", ctx),
        system_prompt=PM_SYSTEM_PROMPT,
        subagents=cast("list[Any]", subagents),
        skills=skill_sources("orchestrator"),
        backend=backend,
        permissions=read_only_permissions(),
        interrupt_on=cast("dict[str, Any]", gates),
        middleware=[
            _pm_round_cap_middleware(config.max_pm_rounds),
            build_gate_weights_middleware(ctx),
            *summarization(),
        ],
        checkpointer=checkpointer,
        store=store,
    )


# The full set of PM-level gates round-0 forces (every shipped orchestrator gate).
# A brand-new portfolio's first round pauses at all of them whatever the config or
# mandate say, on top of whatever those gate; ``first_round_gated`` then flips so
# later rounds use the config and mandate gates alone.
# before_risk is deferred (needs a synthetic checkpoint tool), so place_orders is
# the only real PM gate today — grow this tuple as gates ship.
_ALL_ORCHESTRATOR_GATES: tuple[str, ...] = ("place_orders",)

# The gated tool whose approval commits the paper ticket; any other gate a mandate adds
# only steers the PM towards it.
_ORDER_GATE = "place_orders"


def _effective_gates(config: FundConfig, mandate: PortfolioMandate) -> dict[str, bool]:
    """Union the config's gated tools with the mandate's ``hitl_gates``.

    The effective ``interrupt_on`` for a run: every tool the config gates plus every
    tool the mandate adds, each mapped to ``True``. A default mandate leaves the gate
    set at the config baseline (``place_orders``); a mandate can only widen it.
    """
    gates = frozenset(config.interrupt_on) | frozenset(mandate.hitl_gates)
    return dict.fromkeys(gates, True)


# ---------------------------------------------------------------------------
# ``run_fund`` + ``FundRun``: the one complete paper-run path.
#
# Mirrors ``run_profiler`` / ``ProfilerRun``: resolve the run's inputs, build the
# PM agent, invoke it so the ``place_orders`` gate pauses for the adviser, and hand
# back a ``FundRun`` whose ``.resume(decision)`` either commits the paper ticket
# (approve) or discards it (reject). The agent stack (deepagents / langgraph) is
# imported lazily so a bare ``import fund.agents.graph`` stays agent-stack-free.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FundRun:
    """Handle for one fund run paused at the ``place_orders`` adviser gate.

    Mirrors :class:`~fund.agents.profiler.ProfilerRun`. Carries the resolved
    ``constraint_set``, the load-bearing optimizer ``weights`` (captured from the
    audit trail before the pause), the ``interrupt`` payload the adviser reviews,
    the run ``status``, and enough state to resume. ``resume("approve")`` commits
    the paper ticket and finalises the run completed, or incomplete when
    ``place_orders`` wrote no ticket (see :func:`_settle_hitl`); ``resume("reject")``
    commits no order, records the HITL choice, and finalises the run rejected.
    """

    constraint_set: ConstraintSet
    run_id: uuid.UUID
    interrupt: dict[str, Any] | None
    weights: dict[str, float]
    status: str
    agent: Any = field(repr=False, compare=False)
    thread_config: dict[str, Any] = field(repr=False, compare=False)
    session: Any = field(repr=False, compare=False)
    ctx: Any = field(repr=False, compare=False)

    def resume(
        self, decision: str, *, pm_directive: str | None = None
    ) -> dict[str, Any]:
        """Resume the paused agent with an adviser ``decision``.

        ``approve`` runs the gated ``place_orders`` tool (the paper ticket is written
        and the executor decision logged inside the closure), then finalises the run
        as completed with the optimizer's weights — unless ``place_orders`` placed no
        ticket (a missing fill bar, the audited weights breach the current mandate, or
        the risk gate had not passed them), in which case nothing is booked and the run
        ends ``incomplete`` (or stays ``responded`` if the PM re-gated). ``reject``
        writes no order, records the HITL choice, and finalises rejected. ``respond``
        threads ``pm_directive`` back to the PM and lets it continue: if it re-gates
        the run stays alive as ``"responded"`` (still awaiting approval), else it
        finalises ``incomplete`` — delegated to :func:`_finalize_respond`.

        Approve/reject finalisation is delegated to the shared :func:`_settle_hitl`,
        and respond to :func:`_finalize_respond`, so this in-process path and the
        cross-process :func:`resume_fund` cannot diverge — both stamp the audit trail
        identically, and both refuse a run that is already finalised. The run's
        ``portfolio_id`` / ``asof`` (which ``FundRun`` does not carry) are read back
        from its persisted row for the approve/reject path, and so are the audited
        weights: the vector captured at the pause may have been superseded by the time
        the adviser decides (a ``respond`` can make the PM re-optimise), and
        finalisation must record what ``place_orders`` placed. They are read *before*
        the resume, because the gated ``place_orders`` is the first thing the resume
        runs and a PM that re-runs the allocator afterwards would move the latest
        optimizer decision off the weights the ticket was placed with.
        """
        from fund.audit import AgentRunRepository

        run = AgentRunRepository(self.session).get_run(self.run_id)
        if run is None:
            raise LookupError(f"no agent_run {self.run_id}")
        _require_resumable(run)
        if run.portfolio_id is None:
            raise ValueError(f"agent_run {self.run_id} has no portfolio_id")
        approved_weights = _optimizer_weights(self.session, self.run_id)
        regated_from: str | None = None
        try:
            resumed = _resume_gate(
                self.agent,
                self.ctx,
                self.thread_config,
                decision=decision,
                pm_directive=pm_directive,
                approved_weights=approved_weights,
            )
            result = resumed.result
            regated_from = resumed.regated_from
        except Exception as exc:
            # Same rule as ``resume_fund``: the checkpointer commits on its own
            # connection, so a failure of the PM's follow-up turn after the gate was
            # consumed would leave the checkpoint past the tool step while the caller
            # rolls the ticket back. A gate still pending means nothing moved and the
            # failure is safely retryable.
            if _pending_gate_names(self.agent, self.thread_config):
                raise
            _log.warning(
                "resume of run %s failed after the gate was consumed",
                self.run_id,
                exc_info=True,
            )
            result = {"resume_error": f"{type(exc).__name__}: {exc}"}
        if decision == "respond":
            _finalize_respond(
                self.session,
                run_id=self.run_id,
                result=result,
                pm_directive=pm_directive,
            )
            return result
        _settle_hitl(
            self.session,
            run_id=self.run_id,
            portfolio_id=run.portfolio_id,
            asof=run.asof,
            weights=approved_weights,
            decision=decision,
            result=result,
            regated_from=regated_from,
        )
        return result


def _resume_gate(
    agent: Any,
    ctx: Any,
    thread_config: dict[str, Any],
    *,
    decision: str,
    pm_directive: str | None,
    approved_weights: dict[str, float],
) -> _GateResume:
    """Resume the paused PM for a gate decision, shared by both resume routes.

    The run is claimed first (:meth:`AgentRunRepository.claim_for_resume`): the status
    guard that precedes this is a plain read, so a concurrent reject, freeze or second
    approve may have finalised the run since, and driving the graph then would place
    orders for a run the adviser already settled. An approve whose gate was already
    consumed by an earlier attempt that did not survive (:func:`_gate_consumed`) has
    its ticket placed directly, since the resume can no longer reach ``place_orders``;
    every other decision is the plain :func:`_resume_agent`.

    An approve is also refused when the run is stale or older than the portfolio's
    holdings (:func:`_refuse_stale_approval`), and an approve of the order gate while
    no fill bar exists leaves the gate untouched (:func:`_require_fill_bars`).

    Raises:
        RuntimeError: The fund or the portfolio is frozen and ``decision`` is not a
            reject, or the order gate's ticket has no fill bar yet.
        ValueError: The run is no longer at a live gate, or an approve is stale.
    """
    from fund.audit.repository import AgentRunRepository

    # The status guard and the freeze check of ``resume_fund`` are plain reads, so a
    # freeze committed since (a round that was mid-flight when the killswitch fired
    # commits its gate afterwards) must be seen here too, and the in-process
    # ``FundRun.resume`` route has no earlier check at all. Reject stays open so the
    # adviser can still close the gate.
    if decision != "reject":
        _raise_if_frozen(ctx.session, ctx.portfolio_id, doing=f"{decision} a run")
    if decision == "approve":
        _refuse_stale_approval(ctx)
    if not AgentRunRepository(ctx.session).claim_for_resume(ctx.run_id):
        raise ValueError(
            f"agent_run {ctx.run_id} is no longer at a live gate (another resolver "
            "finalised it first); nothing was resumed"
        )
    gated: tuple[str, ...] = (
        _pending_gate_names(agent, thread_config) if decision == "approve" else ()
    )
    if _ORDER_GATE in gated:
        _require_fill_bars(ctx, approved_weights)
    if decision == "approve" and _gate_consumed(agent, thread_config):
        _replace_consumed_ticket(ctx, approved_weights)
    result = _resume_agent(agent, thread_config, decision, pm_directive)
    # Approving a gate other than the order commit (a mandate-added tool gate) just
    # lets the PM run on; if it stops at another gate the approve settled nothing.
    if gated and _ORDER_GATE not in gated and _extract_interrupt(result) is not None:
        return _GateResume(result, regated_from=gated[0])
    return _GateResume(result)


def _raise_if_frozen(session: Any, portfolio_id: uuid.UUID, *, doing: str) -> None:
    """Raise ``RuntimeError`` when the fund or the portfolio is FROZEN (re-arm first).

    ``doing`` completes the message ("re-arm before ``doing``").
    """
    scope = _frozen_scope(session, portfolio_id)
    if scope == "fund":
        raise RuntimeError(f"fund autonomy is frozen; re-arm before {doing}")
    if scope == "portfolio":
        raise RuntimeError(f"portfolio {portfolio_id} is frozen; re-arm before {doing}")


def _halt_if_frozen(
    session: Any, audit: Any, run_id: uuid.UUID, portfolio_id: uuid.UUID
) -> bool:
    """Finalise a run ``rejected`` when a freeze landed while it was in flight.

    The freeze verbs only see runs already committed at a gate, so a round still
    executing when the operator freezes would otherwise open its gate afterwards and
    be approvable under a FROZEN posture. Returns ``True`` when the run was halted.
    No commit.
    """
    scope = _frozen_scope(session, portfolio_id)
    if scope is None:
        return False
    audit.append_decision(
        run_id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "halted", "reason": "frozen", "scope": scope},
    )
    audit.finalize_run(run_id, weights={}, status="rejected")
    return True


@dataclass(frozen=True)
class _FundRunSetup:
    """The built agent + opened run + seed turn one fund round is driven from."""

    agent: Any = field(repr=False, compare=False)
    thread_config: dict[str, Any] = field(repr=False, compare=False)
    run: Any = field(repr=False, compare=False)
    constraint_set: ConstraintSet = field(repr=False, compare=False)
    seed_input: dict[str, Any] = field(repr=False, compare=False)
    audit: Any = field(repr=False, compare=False)
    ctx: Any = field(repr=False, compare=False)
    first_round: bool = field(default=False, repr=False, compare=False)


def _prepare_fund_run(
    model: Any,
    mandate: PortfolioMandate,
    *,
    portfolio_id: uuid.UUID | str,
    asof: dt.date,
    session: Any,
    checkpointer: Any,
    store: Any,
    config: FundConfig,
    fallback: Any | None,
    thread_id: str | None,
    seed: int | None,
) -> _FundRunSetup:
    """Resolve the constraint set, open the run, and build the PM agent for one round.

    The setup :func:`run_fund` and :func:`stream_fund` share so neither drifts from the
    other: a FROZEN fund or portfolio raises ``RuntimeError`` (re-arm first — Q3, fund
    scope wins); a missing active ``ConstraintSet`` raises ``RuntimeError`` (profile
    first); a pending ``agent_run`` is opened recording the seed + the effective
    ``temperature`` (``None`` for a model built without one) + the 3y backtest training
    window (D31) + the mandate's ``capital`` (the ticket's gross
    book) + the active MiFID ``profile_version`` (``None`` when unprofiled); its
    checkpointer thread is pinned to ``str(run_id)`` so the rebuild-to-resume path
    recovers it; and the PM agent is built against this run's ``RunContext`` with the
    effective gate set — config baseline ∪ ``mandate.hitl_gates``, which the
    portfolio's **first** round widens with every PM gate
    (``_ALL_ORCHESTRATOR_GATES``). The caller drives the agent (invoke or stream). An
    error once the run is open (agent build, seed) finalises it ``failed`` and hands
    back a first-round gate it spent, then propagates. Every write goes through
    ``session`` (caller owns the transaction).
    """
    import secrets

    from fund.agents.model import effective_temperature
    from fund.agents.toolsets import RunContext
    from fund.audit import (
        AgentRunRepository,
        MifidProfileRepository,
        resolve_constraint_set,
    )
    from fund.schemas import ConstraintSetRef

    pid_uuid = _coerce_uuid(portfolio_id)
    pid_str = str(pid_uuid)

    # FROZEN blocks every new run — scheduled and human-triggered — until an explicit
    # re-arm (Q3). Fund scope wins over portfolio scope. The scheduler pre-checks and
    # skips cleanly; this raise is the backstop that blocks a manual run before any run
    # row is opened. Read-only on purpose: a row created here would sit uncommitted in
    # the round's transaction for its whole run, and a concurrent freeze would then
    # wait on the unique index until the round ends instead of acting at once.
    _raise_if_frozen(session, pid_uuid, doing="running a rebalance")

    if store is None:
        raise RuntimeError(
            f"A store is required to resolve the ConstraintSet of portfolio {pid_str}."
        )
    constraint_set = resolve_constraint_set(
        store,
        ConstraintSetRef(
            portfolio_id=pid_str, store_key=config.constraint_set_store_key
        ),
        session=session,
    )
    if constraint_set is None:
        raise RuntimeError(
            f"No active ConstraintSet for portfolio {pid_str}; run the profiler "
            f"first to establish a risk profile."
        )

    # 31 bits: ``agent_runs.seed`` is a signed 32-bit INTEGER on PostgreSQL, so a full
    # 32-bit draw overflows the INSERT about half the time (SQLite never notices). The
    # seed is an audit label only: the allocator path draws no random numbers (a convex
    # min-variance QP over deterministic estimators), so identical inputs reproduce
    # identical weights without it.
    run_seed = seed if seed is not None else secrets.randbits(31)
    audit = AgentRunRepository(session)
    run = None
    resolved_thread_id: str | None = None
    first_round = False
    # The portfolio's first-round gate is spent before the run row is written, in a
    # session of its own, so no write of the round's transaction precedes it. An error
    # anywhere below hands the gate back, and finalises the run ``failed`` once the row
    # exists, rather than leaving a ``pending`` orphan the roster shows as running.
    try:
        first_round = _consume_first_round(session, pid_uuid)
        active_profile = MifidProfileRepository(session).get_active(pid_uuid)
        run = audit.create_run(
            portfolio_id=pid_uuid,
            asof=asof,
            seed=run_seed,
            # Candidates are picked at runtime; optimize_portfolio records them.
            universe=[],
            optimizer_config={
                "step": "fund",
                "temperature": effective_temperature(config, config.primary_model),
                "backtest_train_size": _LOOKBACK_DAYS,
                # Recorded so the rebuild-to-resume path sizes the ticket at the
                # capital the PM was told, even if the mandate is edited before
                # approval.
                "capital": float(mandate.capital),
                # Recorded so approving the gate can tell the portfolio was
                # re-profiled while the round was running or waiting.
                "profile_version": (
                    active_profile.version if active_profile is not None else None
                ),
            },
        )
        resolved_thread_id = thread_id or str(run.id)
        audit.set_thread_id(run.id, resolved_thread_id)

        # Lazy: the tools package pulls the optimizer stack in, and this module's
        # bare import must stay free of it.
        from fund.tools.eligibility import resolve_mandate_eligibility

        eligible = resolve_mandate_eligibility(
            session, mandate, asof, constraint_set=constraint_set
        )
        ctx = RunContext(
            session=session,
            asof=asof,
            store=store,
            config=config,
            run_id=run.id,
            portfolio_id=pid_uuid,
            base_currency=mandate.base_currency,
            notional=float(mandate.capital),
            eligible=eligible,
        )
        effective_gates = _effective_gates(config, mandate)
        if first_round:
            # A portfolio's first round adds every PM gate for maximum oversight. It
            # widens the effective set rather than replacing it: a gate the mandate
            # adds must not be dropped for round one and only reinstated from round
            # two.
            effective_gates = {
                **effective_gates,
                **dict.fromkeys(_ALL_ORCHESTRATOR_GATES, True),
            }
        agent = build_fund_agent(
            model,
            checkpointer=checkpointer,
            store=store,
            ctx=ctx,
            config=config,
            fallback=fallback,
            interrupt_on=effective_gates,
        )
        thread_config = {
            "configurable": {"thread_id": resolved_thread_id},
            "recursion_limit": config.recursion_limit,
        }
        seed_input = {
            "messages": [
                {
                    "role": "user",
                    "content": _run_instruction(
                        mandate,
                        pid_str,
                        asof,
                        _history_summary(session, pid_uuid, asof),
                        constraint_set=constraint_set,
                        eligible=eligible,
                    ),
                }
            ]
        }
    except Exception as exc:
        if run is None:
            if first_round:
                _hand_back_first_round(session, pid_uuid)
        else:
            _finalize_failed(
                audit,
                run.id,
                exc,
                session=session,
                portfolio_id=pid_uuid,
                first_round=first_round,
                checkpointer=checkpointer,
                thread_id=resolved_thread_id,
            )
        raise
    return _FundRunSetup(
        agent=agent,
        thread_config=thread_config,
        run=run,
        constraint_set=constraint_set,
        seed_input=seed_input,
        audit=audit,
        ctx=ctx,
        first_round=first_round,
    )


def _settle_round(
    session: Any,
    setup: _FundRunSetup,
    portfolio_id: uuid.UUID | str,
    *,
    interrupt: dict[str, Any] | None,
    last_message: Any,
    pm_round_cap_reached: bool,
) -> _RoundEnd:
    """Close a round that ran to its end: the one ending shared by run and stream.

    With no gate, the PM tripped the delegation round cap, its order call was cut off
    or malformed, or a blocking risk verdict left nothing to commit; each is recorded
    as its own reason, distinct from the recursion-limit ``"round_cap"``. A freeze is
    looked at only once a gate was reached. No commit.
    """
    run_id = setup.run.id
    if interrupt is None:
        reason = _ungated_end_reason(
            last_message, pm_round_cap_reached=pm_round_cap_reached
        )
        _finalize_incomplete(setup.audit, run_id, reason)
        return _RoundEnd("incomplete", reason)
    if _halt_if_frozen(session, setup.audit, run_id, _coerce_uuid(portfolio_id)):
        return _RoundEnd("rejected")
    # Flip the row to "paused" so observers (CLI/TUI, the Decisions queue) find
    # awaiting-approval runs and the rebuild-to-resume path knows a gate is live.
    setup.audit.mark_paused(run_id)
    return _RoundEnd("paused")


def _fund_run(
    setup: _FundRunSetup,
    session: Any,
    *,
    status: str,
    weights: dict[str, float],
    interrupt: dict[str, Any] | None = None,
) -> FundRun:
    """The handle :func:`run_fund` returns for a round that ended in ``status``."""
    return FundRun(
        constraint_set=setup.constraint_set,
        run_id=setup.run.id,
        interrupt=interrupt,
        weights=weights,
        status=status,
        agent=setup.agent,
        thread_config=setup.thread_config,
        session=session,
        ctx=setup.ctx,
    )


def run_fund(
    model: Any,
    mandate: PortfolioMandate,
    *,
    portfolio_id: uuid.UUID | str,
    asof: dt.date,
    session: Any,
    checkpointer: Any,
    store: Any,
    config: FundConfig = settings,
    fallback: Any | None = None,
    thread_id: str | None = None,
    seed: int | None = None,
) -> FundRun:
    """Run one paper rebalance to the adviser-confirmation gate.

    Orchestrates: (1) resolve the portfolio's active ``ConstraintSet`` from the
    Store via a Phase-4 ``ConstraintSetRef`` — **missing raises ``RuntimeError``**,
    routing the caller to the profiler (a fund run needs a risk profile). (2) open a
    pending ``agent_run`` recording the seed + ``temperature`` + 3y lookback for
    reproducibility (D31). (3) build the PM agent with per-run toolsets bound to this
    run's ``RunContext`` (``asof`` bounds every price read; no look-ahead). (4) invoke
    with the mandate + directive; the pipeline (economist → allocator → risk →
    executor) runs and the run pauses at the ``place_orders`` HITL gate.

    Returns a :class:`FundRun`. When the pipeline never reaches the gate — a blocking
    risk verdict, the delegation round cap tripping the recursion limit, or a PM order
    call that was cut off or malformed — the run is finalised ``incomplete`` (no
    ticket) with the cause as its reason and surfaced for review instead. A freeze
    committed while the round was running finalises it ``rejected`` rather than
    opening a gate under a FROZEN posture. Any other error (provider outage, tool or
    DB failure) finalises the run ``failed`` with the error recorded, then propagates.

    ``model`` must be a tool-calling chat model. ``store`` is required: the active
    ``ConstraintSet`` is resolved from it. Every write goes through the injected
    ``session`` (caller owns the transaction) and the ``store``.
    """
    from langgraph.errors import GraphRecursionError

    setup = _prepare_fund_run(
        model,
        mandate,
        portfolio_id=portfolio_id,
        asof=asof,
        session=session,
        checkpointer=checkpointer,
        store=store,
        config=config,
        fallback=fallback,
        thread_id=thread_id,
        seed=seed,
    )

    # The place_orders gate pauses the run for the adviser.
    try:
        result = setup.agent.invoke(setup.seed_input, config=setup.thread_config)
    except GraphRecursionError:
        # Round cap: the PM kept delegating past the bound without committing.
        _finalize_incomplete(setup.audit, setup.run.id, "round_cap")
        return _fund_run(setup, session, status="incomplete", weights={})
    except Exception as exc:
        _finalize_failed(
            setup.audit,
            setup.run.id,
            exc,
            session=session,
            portfolio_id=setup.ctx.portfolio_id,
            first_round=setup.first_round,
            checkpointer=checkpointer,
            thread_id=setup.thread_config["configurable"]["thread_id"],
        )
        raise

    interrupt = _extract_interrupt(result)
    weights = _optimizer_weights(session, setup.run.id)
    end = _settle_round(
        session,
        setup,
        portfolio_id,
        interrupt=interrupt,
        last_message=_last_message(result),
        pm_round_cap_reached=bool(result.get("pm_round_cap_reached")),
    )
    return _fund_run(
        setup,
        session,
        status=end.status,
        weights={} if end.status == "rejected" else weights,
        interrupt=interrupt if end.status == "paused" else None,
    )


def stream_fund(
    model: Any,
    mandate: PortfolioMandate,
    *,
    portfolio_id: uuid.UUID | str,
    asof: dt.date,
    session: Any,
    checkpointer: Any,
    store: Any,
    config: FundConfig = settings,
    fallback: Any | None = None,
    thread_id: str | None = None,
    seed: int | None = None,
) -> Iterator[StreamEvent]:
    """Stream one paper rebalance to the adviser gate, yielding ordered step events.

    The live-deliberation counterpart to :func:`run_fund`: it shares
    :func:`_prepare_fund_run`'s setup, then drives the PM agent with
    ``agent.stream(stream_mode="updates")`` and yields one
    :class:`~fund.tui.stream.StreamEvent` per meaningful pipeline step — each subagent
    report (economist → allocator → risk → executor), then the ``place_orders``
    commit — in emission order, filtering the interleaved middleware-hook chunks. The
    stream ends on exactly one terminal event: ``paused`` once the ``place_orders`` HITL
    gate is live (the run is flipped ``paused`` so the Decisions queue finds it), or
    ``incomplete`` on a round-cap / no-order end (the run is finalised ``incomplete``)
    or when a freeze landed mid-round (the run is finalised ``rejected``). Any other
    error finalises the run ``failed`` with the error recorded, then propagates.

    The agent stack and the ``StreamEvent`` DTO are imported lazily, so a bare
    ``import fund.agents.graph`` stays agent-stack-free. ``store`` is required: the
    active ``ConstraintSet`` is resolved from it. Every write goes through ``session``
    (caller owns the transaction). Not thread-safe — one stream per thread.

    Yields:
        :class:`~fund.tui.stream.StreamEvent` in emission order, terminated by a single
        ``paused`` / ``incomplete`` event.
    """
    from langgraph.errors import GraphRecursionError

    from fund.tui.stream import StepStatus, StreamEvent

    setup = _prepare_fund_run(
        model,
        mandate,
        portfolio_id=portfolio_id,
        asof=asof,
        session=session,
        checkpointer=checkpointer,
        store=store,
        config=config,
        fallback=fallback,
        thread_id=thread_id,
        seed=seed,
    )
    run_id = setup.run.id
    audit = setup.audit

    index = 0
    # A PM turn may batch several delegations, and each result streams back as its own
    # tools update, so the subagent is matched to its call by id, not by recency.
    delegated: dict[str, str | None] = {}
    interrupt_value: dict[str, Any] | None = None
    pm_round_cap_reached = False
    last_pm_message: Any = None
    try:
        for chunk in setup.agent.stream(
            setup.seed_input, config=setup.thread_config, stream_mode="updates"
        ):
            if not isinstance(chunk, dict):
                continue
            if "__interrupt__" in chunk:
                interrupt_value = _extract_interrupt(chunk)
                continue
            # Only the PM's own model/tools nodes carry the pipeline; the middleware
            # hooks (Skills/PMRoundCap/HumanInTheLoop/PatchToolCalls) are noise here.
            for node, update in chunk.items():
                if isinstance(update, dict) and update.get("pm_round_cap_reached"):
                    # Mirror run_fund: a PM-round-cap end must finalise as
                    # "pm_round_cap", not "no_order". The updates stream never carries
                    # the final merged state, so latch the flag as the cap's state
                    # update streams past.
                    pm_round_cap_reached = True
                messages = update.get("messages") if isinstance(update, dict) else None
                message = messages[-1] if messages else None
                if node == "model":
                    if message is not None:
                        last_pm_message = message
                    for call in _tool_calls(message):
                        name = call.get("name")
                        if name == "task":
                            delegated[call.get("id") or ""] = (
                                call.get("args") or {}
                            ).get("subagent_type")
                        elif name == "place_orders":
                            yield StreamEvent(
                                kind="step",
                                run_id=run_id,
                                index=index,
                                agent="orchestrator",
                                step="place_orders",
                                status=StepStatus.LIVE,
                                text="committing the risk-approved trades",
                            )
                            index += 1
                elif node == "tools" and getattr(message, "name", None) == "task":
                    call_id = getattr(message, "tool_call_id", None) or ""
                    yield StreamEvent(
                        kind="step",
                        run_id=run_id,
                        index=index,
                        agent=delegated.get(call_id) or "subagent",
                        step="report",
                        status=StepStatus.LIVE,
                        text=_message_text(message),
                    )
                    index += 1
    except GraphRecursionError:
        _finalize_incomplete(audit, run_id, "round_cap")
        yield _terminal_event(
            run_id, index, kind="incomplete", text="round cap reached before a ticket"
        )
        return
    except Exception as exc:
        _finalize_failed(
            audit,
            run_id,
            exc,
            session=session,
            portfolio_id=setup.ctx.portfolio_id,
            first_round=setup.first_round,
            checkpointer=checkpointer,
            thread_id=setup.thread_config["configurable"]["thread_id"],
        )
        raise

    end = _settle_round(
        session,
        setup,
        portfolio_id,
        interrupt=interrupt_value,
        last_message=last_pm_message,
        pm_round_cap_reached=pm_round_cap_reached,
    )
    yield _round_end_event(end, run_id, index, interrupt_value)


def resume_fund(
    run_id: uuid.UUID | str,
    decision: str,
    *,
    session: Any,
    checkpointer: Any,
    store: Any | None,
    model: Any,
    config: FundConfig = settings,
    fallback: Any | None = None,
    pm_directive: str | None = None,
) -> dict[str, Any]:
    """Resume a paused fund run from a fresh process — the rebuild-to-resume path.

    :class:`FundRun` is not serialisable across processes (its ``agent``/``session``
    are live objects), so an observer (the CLI ``approve``/``reject`` commands, the
    TUI HITL queue) resumes a paused ``place_orders`` gate by **rebuilding** the PM
    agent against the *same* ``checkpointer`` + ``thread_id`` and issuing
    ``Command(resume=…)`` itself. Approving executes the interrupted commit inside
    the graph (so a ``model`` is required to resume); the shared :func:`_settle_hitl`
    then stamps the terminal audit state and, on an approve whose ticket was placed, the
    positions snapshot — leaving DB state identical to :meth:`FundRun.resume`. A PM
    follow-up turn that fails after the gate was consumed does not raise: the decision
    stands and the failure is returned as ``{"resume_error": ...}``.

    Loads the ``agent_run`` to recover its ``thread_id`` / ``portfolio_id`` / ``asof``
    (an unknown ``run_id`` raises ``LookupError``; a run that is not at a live gate —
    ``paused`` / ``responded`` — raises ``ValueError``), re-guards the portfolio's
    active ``ConstraintSet`` from the Store (missing raises ``RuntimeError`` — a run
    whose profile vanished cannot be committed), re-reads the mandate so a gate set
    edited since the run opened applies on resume too (a vanished mandate falls back to
    the config default gate), rebuilds the ``RunContext`` + PM agent, resumes, and
    finalises. ``decision`` is ``approve``, ``reject``, or ``respond``.
    Any other value raises ``ValueError``; ``approve`` / ``respond`` raise
    ``RuntimeError`` while the fund or portfolio is FROZEN, or when the portfolio was
    re-profiled after the run opened (``reject`` always passes). A ``respond`` threads
    ``pm_directive`` (the adviser's free text) back to the PM via
    :func:`_finalize_respond`, keeping the run
    alive as ``responded`` if it re-gates; the resume ``thread_config`` keeps the full
    ``recursion_limit`` so a re-gate has headroom. Every write goes through the injected
    ``session`` (caller owns the transaction); the agent stack is imported lazily so a
    bare ``import fund.agents.graph`` stays clean.
    """
    if decision not in ("approve", "reject", "respond"):
        raise ValueError(
            f"decision must be 'approve', 'reject', or 'respond', not {decision!r}"
        )

    from fund.agents.toolsets import RunContext
    from fund.audit import AgentRunRepository, resolve_constraint_set
    from fund.schemas import ConstraintSetRef

    rid = _coerce_uuid(run_id)
    audit = AgentRunRepository(session)
    run = audit.get_run(rid)
    if run is None:
        raise LookupError(f"no agent_run {rid}")
    if run.portfolio_id is None:
        raise ValueError(f"agent_run {rid} has no portfolio_id")
    _require_resumable(run)

    # set_holdings replaces the whole snapshot, so approving a round that started before
    # an already-completed one would silently roll the book back to older weights.
    # Rejecting a stale run stays allowed: that is how an adviser clears it.
    if decision == "approve":
        newer = audit.newer_completed_rebalance_run(run)
        if newer is not None:
            raise ValueError(
                f"agent_run {rid} is stale: newer run {newer.id} for portfolio "
                f"{run.portfolio_id} already completed, and approving would overwrite "
                "its holdings with older weights; reject this run instead"
            )

    portfolio_id = run.portfolio_id
    pid_str = str(portfolio_id)
    resolved_thread_id = run.thread_id or str(rid)

    # A gate that opened before (or raced) a freeze, or under a since-replaced profile,
    # stays approvable at the DB level, so approve/respond re-check both. Reject is the
    # way out and always passes.
    if decision != "reject":
        _raise_if_frozen(session, portfolio_id, doing="resuming a run")
        _raise_if_profile_superseded(session, run)

    # Re-guard the active ConstraintSet — the run's risk profile must still exist.
    constraint_set = None
    if store is not None:
        constraint_set = resolve_constraint_set(
            store,
            ConstraintSetRef(
                portfolio_id=pid_str, store_key=config.constraint_set_store_key
            ),
            session=session,
        )
    if constraint_set is None:
        raise RuntimeError(
            f"No active ConstraintSet for portfolio {pid_str}; run the profiler "
            f"first to establish a risk profile."
        )

    # Re-read the mandate so a gate set edited since the run opened takes effect on
    # resume too (mirrors the ConstraintSet re-guard above); a vanished mandate falls
    # back to the config default gate and the tool's default notional.
    from fund import observe

    mandate = observe.get_mandate(session, portfolio_id)
    # The allowlist follows the current mandate, unlike the capital recorded at run
    # open: an adviser who removed a ticker after the proposal must also stop the order.
    from fund.tools.eligibility import resolve_mandate_eligibility

    ctx = RunContext(
        session=session,
        asof=run.asof,
        store=store,
        config=config,
        run_id=rid,
        portfolio_id=portfolio_id,
        base_currency=mandate.base_currency if mandate is not None else None,
        notional=_ticket_capital(run, mandate),
        eligible=resolve_mandate_eligibility(
            session, mandate, run.asof, constraint_set=constraint_set
        ),
    )
    agent = build_fund_agent(
        model,
        checkpointer=checkpointer,
        store=store,
        ctx=ctx,
        config=config,
        fallback=fallback,
        interrupt_on=_effective_gates(config, mandate) if mandate is not None else None,
    )
    thread_config = {
        "configurable": {"thread_id": resolved_thread_id},
        "recursion_limit": config.recursion_limit,
    }
    # Read before the resume: the gated ``place_orders`` runs first and places these
    # weights, while a PM that re-runs the allocator afterwards moves the latest
    # optimizer decision off the weights the ticket was placed with.
    approved_weights = _optimizer_weights(session, rid)
    regated_from: str | None = None
    try:
        resumed = _resume_gate(
            agent,
            ctx,
            thread_config,
            decision=decision,
            pm_directive=pm_directive,
            approved_weights=approved_weights,
        )
        result = resumed.result
        regated_from = resumed.regated_from
    except Exception as exc:
        # The checkpointer commits on its own connection while the ticket and
        # positions live in the caller's uncommitted session. If the PM's follow-up
        # turn fails after the gate was consumed, propagating would roll the ticket
        # back yet leave the checkpoint past the tool step: the run would stay
        # "paused" with no pending interrupt, gone from the queue and unretryable.
        # So once the gate is consumed the decision stands and is finalised; a gate
        # still pending means nothing moved and the failure is safely retryable.
        if observe._has_pending_interrupt(checkpointer, resolved_thread_id):
            raise
        _log.warning(
            "resume of run %s failed after the gate was consumed", rid, exc_info=True
        )
        result = {"resume_error": f"{type(exc).__name__}: {exc}"}
    if decision == "respond":
        _finalize_respond(session, run_id=rid, result=result, pm_directive=pm_directive)
        return result
    _settle_hitl(
        session,
        run_id=rid,
        portfolio_id=portfolio_id,
        asof=run.asof,
        weights=approved_weights,
        decision=decision,
        result=result,
        regated_from=regated_from,
    )
    return result


def resume_run(
    run_id: uuid.UUID | str,
    decision: str,
    *,
    session: Any,
    checkpointer: Any,
    store: Any | None,
    model: Any,
    config: FundConfig = settings,
    fallback: Any | None = None,
    pm_directive: str | None = None,
) -> dict[str, Any]:
    """Resume a paused run by its gate — the single entry point observers call.

    A run pauses at one of two HITL gates: the profiler's ``save_profile`` (step
    ``"profiler"``) or the rebalance PM's ``place_orders`` (step ``"fund"``). This
    dispatches on the run's recorded ``optimizer_config["step"]`` to
    :func:`~fund.agents.profiler.resume_profiler` or :func:`resume_fund`, so the CLI
    ``approve``/``reject`` and the TUI HITL queue drive either gate through one call
    and cannot diverge on which path a given run takes. An unknown ``run_id`` raises
    ``LookupError``; ``decision`` validation is delegated to the chosen resumer.

    ``pm_directive`` carries the adviser's free text for a ``respond`` barge-in; it is
    threaded only to the ``place_orders`` gate. The profiler gate keeps approve/reject
    (``resume_profiler`` rejects ``respond``), so a respond there raises ``ValueError``.
    """
    from fund.audit import AgentRunRepository

    rid = _coerce_uuid(run_id)
    run = AgentRunRepository(session).get_run(rid)
    if run is None:
        raise LookupError(f"no agent_run {rid}")

    if (run.optimizer_config or {}).get("step") == "profiler":
        from fund.agents.profiler import resume_profiler

        return resume_profiler(
            rid,
            decision,
            session=session,
            checkpointer=checkpointer,
            store=store,
            model=model,
            config=config,
        )
    return resume_fund(
        rid,
        decision,
        session=session,
        checkpointer=checkpointer,
        store=store,
        model=model,
        config=config,
        fallback=fallback,
        pm_directive=pm_directive,
    )
