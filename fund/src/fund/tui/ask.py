"""``fund.tui.ask`` — the model-free classify / target-resolve / retrieve helpers.

The Ask view's deterministic core: deciding whether a turn is a read-only *query* or a
decision *action*, resolving which gate / portfolio / scope an action targets, and
assembling a compact grounded context from the agent-stack-free read model. Pure
functions over an injected session + checkpointer — no Textual, no model, no agent
stack. The lazy LLM summarize call and all ``apply_action`` dispatch live in
``fund.tui.app`` workers; this module only decides and retrieves, and never guesses a
target (an ambiguous turn asks a follow-up instead).

At module scope this imports only the stdlib and the agent-stack-free ``fund.observe``
read model (whose optimizer/skfolio tracker is itself lazy), so ``import fund.tui.ask``
— and therefore ``import fund.tui.app`` — stays agent-stack-light and
optimizer/skfolio-free.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from fund import observe

if TYPE_CHECKING:
    from fund.observe import RunSummary

# The recognized decision verbs → the ``apply_action`` verb each drives. Form-heavy
# actions (onboard / profile / mandate edit / guardrail / weight edits) are deliberately
# absent: they stay in their dedicated views, so Ask only ever drives a decision verb.
_ACTION_VERBS: dict[str, str] = {
    "approve": "approve",
    "reject": "reject",
    "killswitch": "killswitch",
    "kill": "killswitch",
    "freeze": "freeze",
    "rearm": "rearm",
    "re-arm": "rearm",
    "arm": "rearm",
    "run": "run",
    "rebalance": "run",
    "steer": "respond",
    "respond": "respond",
}

# Resume verbs target a specific paused gate (a run); scope verbs target an autonomy
# scope; ``run`` (rebalance) targets the focused portfolio.
_RESUME_VERBS: frozenset[str] = frozenset(
    {"approve", "reject", "killswitch", "respond"}
)
_SCOPE_VERBS: frozenset[str] = frozenset({"freeze", "rearm"})

Status = Literal["resolved", "ambiguous", "unresolved"]


@dataclass(frozen=True)
class ResolvedAction:
    """A confirm-ready decision parsed from an Ask turn.

    Attributes:
        verb: The ``apply_action`` verb (approve / reject / killswitch / freeze /
            rearm / run / respond).
        run_id: The paused run a resume verb resolves to, else ``None``.
        portfolio_id: The affected portfolio, else ``None`` (fund scope).
        scope: The autonomy scope (``"portfolio"`` / ``"fund"``) for freeze/rearm,
            else ``None``.
    """

    verb: str
    run_id: uuid.UUID | None
    portfolio_id: uuid.UUID | None
    scope: str | None


@dataclass(frozen=True)
class TargetResolution:
    """The outcome of resolving an action turn's target.

    Attributes:
        status: ``"resolved"`` (``action`` is confirm-ready), ``"ambiguous"`` (needs a
            disambiguation follow-up — ``prompt`` asks it and ``candidates`` lists the
            gates when the ambiguity is between gates), or ``"unresolved"`` (a dead end
            with nothing to act on; ``prompt`` says why).
        action: The confirm-ready action when ``status == "resolved"``, else ``None``.
        candidates: The candidate gates for an ambiguous resume verb, else empty.
        prompt: A human-readable follow-up question or explanation.
    """

    status: Status
    action: ResolvedAction | None
    candidates: tuple[RunSummary, ...]
    prompt: str


def classify(turn: str) -> Literal["query", "action"]:
    """Classify a turn as a decision ``action`` or a read-only ``query``.

    Deterministic — no model: a turn whose **leading** word is a recognized decision
    verb (approve / reject / killswitch / freeze / re-arm / run / steer, plus the
    synonyms kill / arm / rebalance / respond) is an ``action``; everything else is a
    ``query`` — including a question that merely *mentions* a verb mid-sentence ("why
    did the allocator trim energy?"), so a read is never mistaken for a mutation.

    Args:
        turn: The adviser's natural-language turn.

    Returns:
        ``"action"`` when the leading word is a decision verb, else ``"query"``.
    """
    return "action" if _leading_verb(turn) is not None else "query"


def _leading_verb(turn: str) -> str | None:
    """The ``apply_action`` verb a turn's leading word maps to, or ``None``."""
    match = re.match(r"[a-z][a-z-]*", turn.strip().lower())
    return _ACTION_VERBS.get(match.group(0)) if match else None


def resolve_target(
    turn: str, session: Any, saver: Any, focus: uuid.UUID | None
) -> TargetResolution:
    """Resolve which gate / portfolio / scope an action turn targets.

    Routes on the verb kind — never guessing a target:

    * a resume verb (approve / reject / killswitch / respond) resolves against the
      pending-HITL gates: a single gate (or one named by its short id in the turn) is
      ``resolved``, two or more unnamed are ``ambiguous`` (the candidates), none is
      ``unresolved``;
    * a scope verb (freeze / rearm) resolves to ``fund`` when the turn says so, else the
      focused portfolio, else ``ambiguous`` (ask which);
    * ``run`` (rebalance) resolves to the focused portfolio, else ``ambiguous``.

    Args:
        turn: The adviser's action turn.
        session: An injected read-only sync session; not owned here.
        saver: The bootstrapped LangGraph checkpointer (opaque), for the gate reads.
        focus: The focused portfolio, or ``None`` for a fund-wide turn.

    Returns:
        A :class:`TargetResolution` whose ``status`` drives the Ask view's confirm vs
        disambiguation flow.
    """
    verb = _leading_verb(turn)
    if verb is None:
        return TargetResolution("unresolved", None, (), "not a recognized action")
    if verb in _RESUME_VERBS:
        return _resolve_gate(verb, turn, session, saver)
    if verb in _SCOPE_VERBS:
        return _resolve_scope(verb, turn, focus)
    return _resolve_rebalance(focus)  # verb == "run"


def _resolve_gate(verb: str, turn: str, session: Any, saver: Any) -> TargetResolution:
    """Resolve a resume verb against the pending-HITL gates (never on a guess)."""
    gates = observe.pending_hitl(session, saver, None)
    if not gates:
        return TargetResolution(
            "unresolved", None, (), "no gate is awaiting a decision"
        )
    named = _match_gate(turn, gates)
    if named is not None:
        return _resolved_gate(verb, named)
    if len(gates) == 1:
        return _resolved_gate(verb, gates[0])
    return TargetResolution("ambiguous", None, tuple(gates), _gate_prompt(verb, gates))


def _match_gate(turn: str, gates: list[RunSummary]) -> RunSummary | None:
    """The single gate named by its short run/portfolio id in ``turn``, else ``None``.

    Returns ``None`` when zero or more than one gate matches, so a bare "approve the
    gate" (no id) and an ambiguous "approve both" both fall through to the gate count.
    """
    lowered = turn.lower()
    matched = [
        gate
        for gate in gates
        if _short(gate.run_id) in lowered
        or (gate.portfolio_id is not None and _short(gate.portfolio_id) in lowered)
    ]
    return matched[0] if len(matched) == 1 else None


def _resolved_gate(verb: str, gate: RunSummary) -> TargetResolution:
    """A ``resolved`` resolution for a resume ``verb`` against ``gate``."""
    action = ResolvedAction(verb, gate.run_id, gate.portfolio_id, None)
    return TargetResolution(
        "resolved", action, (), f"{verb} gate #{_short(gate.run_id)}"
    )


def _gate_prompt(verb: str, gates: list[RunSummary]) -> str:
    """The disambiguation follow-up listing the candidate gates for ``verb``."""
    listed = " or ".join(
        f"#{_short(gate.run_id)} "
        f"({_short(gate.portfolio_id) if gate.portfolio_id else 'fund'})"
        for gate in gates
    )
    return f"which gate to {verb} — {listed}?"


def _resolve_scope(verb: str, turn: str, focus: uuid.UUID | None) -> TargetResolution:
    """Resolve a scope verb to fund / focused-portfolio scope, else ask which."""
    if "fund" in turn.lower():
        action = ResolvedAction(verb, None, None, "fund")
        return TargetResolution("resolved", action, (), f"{verb} the whole fund")
    if focus is not None:
        action = ResolvedAction(verb, focus, focus, "portfolio")
        return TargetResolution(
            "resolved", action, (), f"{verb} portfolio {_short(focus)}"
        )
    return TargetResolution(
        "ambiguous", None, (), f"{verb} what — the fund, or focus a portfolio first?"
    )


def _resolve_rebalance(focus: uuid.UUID | None) -> TargetResolution:
    """Resolve a ``run`` (rebalance) to the focused portfolio, else ask which."""
    if focus is None:
        return TargetResolution(
            "ambiguous", None, (), "run which portfolio — focus one first"
        )
    action = ResolvedAction("run", focus, focus, None)
    return TargetResolution(
        "resolved", action, (), f"run a rebalance for {_short(focus)}"
    )


@dataclass(frozen=True)
class PendingAction:
    """A resolved-but-unconfirmed action carried across turns for disambiguation.

    Attributes:
        verb: The decision verb the ambiguous turn invoked.
        candidates: Candidate gate run ids for a resume verb (empty for a scope /
            rebalance verb, whose disambiguation is re-resolved from the next turn).
    """

    verb: str
    candidates: tuple[uuid.UUID, ...]


def pending_action(turn: str, resolution: TargetResolution) -> PendingAction:
    """Build the pending action to carry from an ambiguous ``turn`` + ``resolution``."""
    return PendingAction(
        verb=_leading_verb(turn) or "",
        candidates=tuple(gate.run_id for gate in resolution.candidates),
    )


def resolve_pending(
    pending: PendingAction,
    turn: str,
    session: Any,
    saver: Any,
    focus: uuid.UUID | None,
) -> TargetResolution:
    """Resolve a carried pending action against the next ``turn`` — never on a guess.

    A resume-verb pending resolves to the single still-live candidate gate the next turn
    names (by short id); a scope/rebalance pending re-resolves from the next turn's
    target words (e.g. "the fund") or the now-focused portfolio. An unmatched next turn
    stays non-``resolved`` so the caller can treat it as a fresh, unrelated turn (which
    clears the pending action).

    Args:
        pending: The carried :class:`PendingAction` from the prior ambiguous turn.
        turn: The adviser's follow-up turn.
        session: An injected read-only sync session; not owned here.
        saver: The bootstrapped LangGraph checkpointer (opaque), for the gate reads.
        focus: The focused portfolio, or ``None``.

    Returns:
        A :class:`TargetResolution`; ``resolved`` only when the follow-up names the
        target unambiguously.
    """
    if pending.verb in _RESUME_VERBS:
        live = [
            gate
            for gate in observe.pending_hitl(session, saver, None)
            if gate.run_id in pending.candidates
        ]
        named = _match_gate(turn, live)
        if named is not None:
            return _resolved_gate(pending.verb, named)
        return TargetResolution("unresolved", None, (), "could not match a gate")
    if pending.verb in _SCOPE_VERBS:
        return _resolve_scope(pending.verb, turn, focus)
    return _resolve_rebalance(focus)


def build_context(
    session: Any, saver: Any, focus: uuid.UUID | None, question: str
) -> str:
    """Assemble a compact, grounded context from the read model for ``question``.

    Selects the relevant ``fund.observe`` reads — the fund roster size, the pending
    gates and, when a portfolio is focused, its drift/holdings count, NAV-series
    returns, and recent decisions — formatted as a short plain-text block the summarizer
    grounds its answer in. A pure read: opens no transaction, builds no model.
    ``question`` is accepted for future relevance selection; today the snapshot is
    fixed and compact.

    Args:
        session: An injected read-only sync session; not owned here.
        saver: The bootstrapped LangGraph checkpointer (opaque), for the gate reads.
        focus: The focused portfolio, or ``None`` for a fund-wide question.
        question: The adviser's natural-language question (reserved).

    Returns:
        A compact grounded-context string — never empty (at minimum the fund summary).
    """
    portfolios = observe.list_portfolios(session)
    gates = observe.pending_hitl(session, saver, None)
    lines = [
        "FUND STATE",
        f"portfolios: {len(portfolios)} · pending gates: {len(gates)}",
    ]
    for gate in gates[:5]:
        pf = _short(gate.portfolio_id) if gate.portfolio_id else "fund"
        lines.append(f"  gate #{_short(gate.run_id)} · pf {pf} · asof {gate.asof}")
    if focus is not None:
        lines.extend(_focus_lines(session, focus))
    return "\n".join(lines)


def _focus_lines(session: Any, focus: uuid.UUID) -> list[str]:
    """The focused-portfolio grounding block: drift, NAV returns, recent decisions."""
    state = observe.portfolio_state(session, focus)
    summary = observe.performance_summary(session, focus)
    lines = [
        f"FOCUS portfolio {_short(focus)}",
        f"  drift L1 {state.drift_l1:.4f} · "
        f"{len(state.current)} held → {len(state.target)} target",
        f"  returns: max {_fmt_pct(summary.returns.get('max'))} · "
        f"1D {_fmt_pct(summary.returns.get('1D'))}",
    ]
    timeline = observe.audit_timeline(session, portfolio_id=focus, limit=5)
    if timeline:
        lines.append("  recent decisions:")
        lines.extend(
            f"    {entry.when:%m-%d} {entry.actor} {entry.action}" for entry in timeline
        )
    return lines


def _fmt_pct(value: float | None) -> str:
    """A signed-percent cell, or ``n/a`` when the metric is unavailable."""
    return f"{value:+.2%}" if value is not None else "n/a"


def _short(value: Any) -> str:
    """First hex group of a UUID (or its string) — matches the view's compact id."""
    return str(value).split("-", 1)[0]


__all__ = [
    "PendingAction",
    "ResolvedAction",
    "TargetResolution",
    "build_context",
    "classify",
    "pending_action",
    "resolve_pending",
    "resolve_target",
]
