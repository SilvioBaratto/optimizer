"""``fund.observe`` — the shared, model-free read model (Phase 8, Task 5).

The single read model both frontends use: the CLI ``status``/``report`` commands and
the TUI's four panels are thin wrappers over these functions. It constructs **no**
LLM and imports **no** agent stack at module load — ``import fund.observe`` drags in
no ``deepagents`` / ``langchain`` / ``langgraph`` (asserted by a subprocess guard).
The repositories it reads through live under ``fund.audit`` (whose package
``__init__`` pulls the LangGraph persistence bootstrap), so they are imported lazily
**inside** the functions, mirroring ``fund.agents.graph``'s lazy builders — the bare
import stays clean, the call path still routes through the single-source repos.

Every function takes an injected sync ``Session`` (the caller — CLI/TUI — owns the
transaction; these read paths never ``commit``). Where a run's transcript or a
pending-HITL cross-check is needed, an already-bootstrapped LangGraph ``saver`` is
passed in and typed ``Any`` (no ``langgraph`` import here).

Two sources feed the transcript (SPEC §4a): the verbatim PM/subagent turns live in
the checkpoint (``saver.get_tuple(...).checkpoint["channel_values"]["messages"]`` —
read-only, no compiled graph, no model), and the load-bearing structured facts live
in ``agent_decisions``. :func:`load_run_transcript` interleaves them; economist/risk
narrative exists only in ``messages`` (they log no decision row), so the merge never
assumes every agent has one.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only import: ``fund.schemas.mandate`` is agent-stack-free, but with
    # ``from __future__ import annotations`` the return type is a string, so guarding
    # it under TYPE_CHECKING keeps the module's *runtime* imports minimal while still
    # advertising the SPEC §4a signature ``get_mandate -> PortfolioMandate | None``.
    from fund.schemas.mandate import PortfolioMandate

# LangGraph writes a pending dynamic interrupt to this reserved checkpoint channel
# (``langgraph.constants.INTERRUPT``, now private). Kept as a literal so importing
# this module never pulls in ``langgraph`` — the agent-stack-free invariant (SPEC R2).
_INTERRUPT_CHANNEL = "__interrupt__"


@dataclass(frozen=True)
class RunSummary:
    """One row for the history / status / HITL-queue panels."""

    run_id: uuid.UUID
    portfolio_id: uuid.UUID | None
    asof: dt.date
    status: str
    created_at: dt.datetime
    finished_at: dt.datetime | None
    n_weights: int
    awaiting_hitl: bool


@dataclass(frozen=True)
class TranscriptEntry:
    """A merged narrative/structured line, ordered within a run's transcript."""

    source: str  # "message" | "decision"
    agent: str
    step: str | None
    index: int
    text: str  # message content, or a rendered decision summary
    payload: dict[str, Any] | None  # decision JSON when present, else None


@dataclass(frozen=True)
class PortfolioState:
    """Panel 2: current vs target holdings, with L1 drift + best-effort metrics."""

    portfolio_id: uuid.UUID
    current: dict[str, float]
    target: dict[str, float]
    drift_l1: float
    metrics: dict[str, float]  # best-effort — empty when none persisted (never faked)


def drift_l1(current: dict[str, float], target: dict[str, float]) -> float:
    """``Σ_i |current_i - target_i|`` over the union of tickers (D12).

    A ticker present on only one side counts as weight ``0.0`` on the other.
    """
    tickers = set(current) | set(target)
    return sum(abs(current.get(t, 0.0) - target.get(t, 0.0)) for t in tickers)


def list_portfolio_runs(session: Any, portfolio_id: uuid.UUID) -> list[RunSummary]:
    """All runs for a portfolio, newest first, as :class:`RunSummary` rows.

    ``awaiting_hitl`` here is the plain ``status == "paused"`` flag (the history
    panel); the HITL queue uses :func:`pending_hitl`, which additionally checks the
    checkpoint for a live interrupt.
    """
    from fund.audit.repository import AgentRunRepository

    runs = AgentRunRepository(session).list_runs_for_portfolio(portfolio_id)
    return [_run_summary(run, awaiting_hitl=run.status == "paused") for run in runs]


def pending_hitl(
    session: Any, saver: Any, portfolio_id: uuid.UUID | None = None
) -> list[RunSummary]:
    """Paused runs awaiting a human decision, newest first.

    A row is returned only when it is ``status == "paused"`` **and** its checkpoint
    still carries a pending interrupt (§4f): being marked paused at the gate is
    necessary but not sufficient — the ``saver`` cross-check keeps a stale/finished
    thread out of the Approve-able queue.
    """
    from fund.audit.repository import AgentRunRepository

    paused = AgentRunRepository(session).list_paused_runs(portfolio_id)
    return [
        _run_summary(run, awaiting_hitl=True)
        for run in paused
        if _has_pending_interrupt(saver, run.thread_id)
    ]


def load_run_transcript(session: Any, saver: Any, run: Any) -> list[TranscriptEntry]:
    """Merge the checkpoint's ``messages`` with the run's ``agent_decisions``.

    Each decision is anchored right after the message that triggered its agent — a
    ``task(subagent_type=...)`` delegation or a top-level tool call whose name
    matches the decision's ``agent``/``step``. Decisions with no matching message
    (an agent that logged a row without a PM-thread tool call) trail in
    ``decision_index`` order. Messages from agents that log no decision (economist,
    risk) simply appear on their own.
    """
    entries: list[TranscriptEntry] = []
    messages = _messages_for(saver, getattr(run, "thread_id", None))
    remaining = list(run.decisions)
    for index, message in enumerate(messages):
        entries.append(_message_entry(message, index))
        tokens = _message_tokens(message)
        if not tokens:
            continue
        for decision in [d for d in remaining if _decision_matches(d, tokens)]:
            entries.append(_decision_entry(decision))
            remaining.remove(decision)
    entries.extend(_decision_entry(decision) for decision in remaining)
    return entries


def portfolio_state(session: Any, portfolio_id: uuid.UUID) -> PortfolioState:
    """Current holdings vs target weights, with L1 drift and best-effort metrics.

    ``current`` is the ticker→weight map from the ``positions`` snapshot; ``target``
    is the latest ``completed`` run's ``weights`` (fallback: the latest allocator
    ``optimize_portfolio`` proposal). ``metrics`` is read from the latest allocator
    decision when it carries a ``metrics`` block, else ``{}`` — Phase 7 persists no
    dedicated metrics column, and metrics are never fabricated.
    """
    from fund.audit.positions_repository import PositionRepository
    from fund.audit.repository import AgentRunRepository

    holdings = PositionRepository(session).get_holdings(portfolio_id)
    current = {p.ticker: p.weight for p in holdings}
    run_repo = AgentRunRepository(session)
    runs = run_repo.list_runs_for_portfolio(portfolio_id)
    target = _target_weights(run_repo, runs)
    return PortfolioState(
        portfolio_id=portfolio_id,
        current=current,
        target=target,
        drift_l1=drift_l1(current, target),
        metrics=_metrics_from_runs(runs),
    )


def get_mandate(session: Any, portfolio_id: uuid.UUID) -> PortfolioMandate | None:
    """Rehydrate the pydantic ``PortfolioMandate`` from its JSON column, or ``None``.

    ``MandateRepository.get`` returns the DB row (JSON source of truth + scalar
    mirror); JSON→pydantic rehydration lives here, on the read side, so the
    state/report panels get the typed mandate.
    """
    from fund.audit.mandate_repository import MandateRepository
    from fund.schemas.mandate import PortfolioMandate

    row = MandateRepository(session).get(portfolio_id)
    if row is None:
        return None
    return PortfolioMandate.model_validate(row.mandate)


# --- internals --------------------------------------------------------------


def _run_summary(run: Any, *, awaiting_hitl: bool) -> RunSummary:
    return RunSummary(
        run_id=run.id,
        portfolio_id=run.portfolio_id,
        asof=run.asof,
        status=run.status,
        created_at=run.created_at,
        finished_at=run.finished_at,
        n_weights=len(run.weights or {}),
        awaiting_hitl=awaiting_hitl,
    )


def _thread_config(thread_id: str) -> dict[str, Any]:
    """The top-level (``checkpoint_ns=""``) config the saver reads a run's thread by."""
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _has_pending_interrupt(saver: Any, thread_id: str | None) -> bool:
    """Whether ``thread_id``'s checkpoint carries a pending ``__interrupt__`` write."""
    if not thread_id:
        return False
    tup = saver.get_tuple(_thread_config(thread_id))
    if tup is None:
        return False
    return any(
        channel == _INTERRUPT_CHANNEL
        for (_task_id, channel, _value) in (tup.pending_writes or [])
    )


def _messages_for(saver: Any, thread_id: str | None) -> list[Any]:
    """The checkpoint's ``messages`` channel for ``thread_id`` (empty if absent)."""
    if not thread_id:
        return []
    tup = saver.get_tuple(_thread_config(thread_id))
    if tup is None:
        return []
    return tup.checkpoint.get("channel_values", {}).get("messages", []) or []


def _message_entry(message: Any, index: int) -> TranscriptEntry:
    return TranscriptEntry(
        source="message",
        agent=_message_agent(message),
        step=None,
        index=index,
        text=_message_text(message),
        payload=None,
    )


def _message_agent(message: Any) -> str:
    """A named subagent message reports its ``name``; else the message ``type``."""
    name = getattr(message, "name", None)
    if name:
        return str(name)
    return str(getattr(message, "type", None) or type(message).__name__)


def _message_text(message: Any) -> str:
    """Flatten a message's ``content`` (str or a list of content blocks)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def _message_tokens(message: Any) -> set[str]:
    """Anchor tokens for a message: its tool-call names + any ``subagent_type`` arg."""
    tokens: set[str] = set()
    for call in getattr(message, "tool_calls", None) or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if name:
            tokens.add(str(name))
        args = call.get("args")
        if isinstance(args, dict):
            subagent = args.get("subagent_type")
            if subagent:
                tokens.add(str(subagent))
    return tokens


def _decision_matches(decision: Any, tokens: set[str]) -> bool:
    return decision.agent in tokens or decision.step in tokens


def _decision_entry(decision: Any) -> TranscriptEntry:
    payload: dict[str, Any] = {}
    if decision.constraint_set:
        payload["constraint_set"] = decision.constraint_set
    if decision.views:
        payload["views"] = decision.views
    if decision.hitl_decision:
        payload["hitl_decision"] = decision.hitl_decision
    weights = _weights_from_response(decision.llm_response)
    if weights:
        payload["weights"] = weights
    return TranscriptEntry(
        source="decision",
        agent=decision.agent,
        step=decision.step,
        index=decision.decision_index,
        text=f"{decision.agent} · {decision.step}",
        payload=payload or None,
    )


def _target_weights(run_repo: Any, runs: list[Any]) -> dict[str, float]:
    """Latest completed run's weights, else the latest allocator proposal, else {}."""
    for run in runs:
        if run.status == "completed" and run.weights:
            return {str(k): float(v) for k, v in run.weights.items()}
    for run in runs:
        weights = run_repo.latest_optimizer_weights(run.id)
        if weights:
            return weights
    return {}


def _metrics_from_runs(runs: list[Any]) -> dict[str, float]:
    """Best-effort metrics from the newest allocator ``optimize_portfolio`` decision.

    ``run.decisions`` is ordered ``decision_index`` ASC, so iterate it reversed to
    take the *newest* allocator proposal within a run — aligning the metrics source
    with ``AgentRunRepository.latest_optimizer_weights`` (which feeds ``target``) so
    the state panel never shows fresh weights beside a stale run's metrics.
    """
    for run in runs:
        for decision in reversed(run.decisions):
            if decision.agent == "allocator" and decision.step == "optimize_portfolio":
                metrics = _metrics_from_response(decision.llm_response)
                if metrics:
                    return metrics
    return {}


def _weights_from_response(raw: str | None) -> dict[str, Any] | None:
    parsed = _parse_json_object(raw)
    if parsed is None:
        return None
    weights = parsed.get("weights")
    return weights if isinstance(weights, dict) else None


def _metrics_from_response(raw: str | None) -> dict[str, float]:
    parsed = _parse_json_object(raw)
    if parsed is None:
        return {}
    metrics = parsed.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        str(k): float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _parse_json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = [
    "PortfolioState",
    "RunSummary",
    "TranscriptEntry",
    "drift_l1",
    "get_mandate",
    "list_portfolio_runs",
    "load_run_transcript",
    "pending_hitl",
    "portfolio_state",
]
