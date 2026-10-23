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

# Fund-wide scope is named by the standalone word "fund". A bare substring test would
# misroute a focused freeze/rearm onto the whole fund on words like "refund"/"funding".
_FUND_WORD = re.compile(r"\bfund\b")

# Words that make a turn name "all of them" (a blanket target) or exclude something.
# Neither can be honoured by a single focused portfolio, so a turn using one asks which
# target is meant instead of quietly narrowing to the focus.
_BLANKET_WORD = re.compile(r"\b(?:all|every|everything|each|both)\b")
_NEGATION = re.compile(r"\b(?:not|never|nor|neither|except|without)\b|n['\u2019]t\b")

# A typed id: the 8-hex short form the views render (optionally ``#``-prefixed) or a
# ``#``-prefixed partial of it. Delimited so a word that merely contains hex letters is
# not read as one; a hyphen may follow so a pasted full UUID is read by its first group.
_ID_TOKEN = re.compile(r"(?<![0-9a-z])(?:#[0-9a-f]{4,8}|[0-9a-f]{8})(?![0-9a-z])")

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
      ``unresolved`` — as is a turn naming an id that matches no live gate (already
      decided, or a typo), which is never swapped for another gate;
    * a scope verb (freeze / rearm) resolves to ``fund`` when the turn says so, to the
      portfolio whose short id it names, else to the focused portfolio, else
      ``ambiguous`` (ask which). A turn that names an id matching no portfolio is
      ``unresolved``; one using a blanket word ("everything", "all portfolios") or a
      negation ("not the fund") is ``ambiguous`` rather than narrowed to the focus;
    * ``run`` (rebalance) resolves like a scope verb's portfolio branch: the named
      portfolio, else the focused one, else ``ambiguous``. A rebalance is per portfolio,
      so a turn naming "the fund" (or a blanket word) is ``ambiguous`` too, never
      narrowed to the focus;

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
        return _resolve_scope(verb, turn, session, focus)
    return _resolve_rebalance(turn, session, focus)  # verb == "run"


def _resolve_gate(verb: str, turn: str, session: Any, saver: Any) -> TargetResolution:
    """Resolve a resume verb against the pending-HITL gates (never on a guess)."""
    gates = observe.pending_hitl(session, saver, None)
    if not gates:
        return TargetResolution(
            "unresolved", None, (), "no gate is awaiting a decision"
        )
    stale = _stale_ids(turn, gates)
    if stale:
        typed = ", ".join(f"#{token}" for token in stale)
        return TargetResolution(
            "unresolved",
            None,
            (),
            f"no gate awaiting a decision matches {typed} — it may already be decided",
        )
    named = _match_gate(turn, gates)
    if named is not None:
        return _resolved_gate(verb, named)
    if len(gates) == 1:
        return _resolved_gate(verb, gates[0])
    return TargetResolution("ambiguous", None, tuple(gates), _gate_prompt(verb, gates))


def _id_tokens(turn: str) -> list[str]:
    """The id-like tokens typed in ``turn``, lower-cased and without the ``#``."""
    return [match.group(0).lstrip("#") for match in _ID_TOKEN.finditer(turn.lower())]


def _gate_ids(gate: RunSummary) -> list[str]:
    """The short ids a gate answers to: its run id and its portfolio id, if any."""
    ids = [_short(gate.run_id)]
    if gate.portfolio_id is not None:
        ids.append(_short(gate.portfolio_id))
    return ids


def _match_gate(turn: str, gates: list[RunSummary]) -> RunSummary | None:
    """The single gate named by its short run/portfolio id in ``turn``, else ``None``.

    Returns ``None`` when zero or more than one gate matches, so a bare "approve the
    gate" (no id) and an ambiguous "approve both" both fall through to the gate count.
    """
    tokens = _id_tokens(turn)
    matched = [
        gate
        for gate in gates
        if any(gid.startswith(token) for token in tokens for gid in _gate_ids(gate))
    ]
    return matched[0] if len(matched) == 1 else None


def _stale_ids(turn: str, gates: list[RunSummary]) -> list[str]:
    """The ids typed in ``turn`` that name none of ``gates``.

    A non-empty result means the adviser pointed at something that is not a live gate,
    so the turn must not fall back to "the only gate" — that would act on a proposal
    they never named.
    """
    return [
        token
        for token in _id_tokens(turn)
        if not any(gid.startswith(token) for gate in gates for gid in _gate_ids(gate))
    ]


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


def _resolve_scope(
    verb: str, turn: str, session: Any, focus: uuid.UUID | None
) -> TargetResolution:
    """Resolve a scope verb to fund / named / focused scope, else ask which."""
    lowered = turn.lower()
    if _NEGATION.search(lowered):
        return _negated(verb)
    named, stale = _named_portfolios(turn, session)
    if stale:
        return _unknown_portfolio(stale)
    names_fund = _FUND_WORD.search(lowered) is not None
    if len(named) > 1 or (named and names_fund):
        return TargetResolution(
            "ambiguous",
            None,
            (),
            f"{verb} which — name one portfolio by id, or say the whole fund?",
        )
    if named:
        return _resolved_portfolio_scope(verb, named[0])
    if names_fund:
        action = ResolvedAction(verb, None, None, "fund")
        return TargetResolution("resolved", action, (), f"{verb} the whole fund")
    if _BLANKET_WORD.search(lowered):
        return TargetResolution(
            "ambiguous",
            None,
            (),
            f"{verb} what — the whole fund, or one portfolio (name its id)?",
        )
    if focus is not None:
        return _resolved_portfolio_scope(verb, focus)
    return TargetResolution(
        "ambiguous", None, (), f"{verb} what — the fund, or focus a portfolio first?"
    )


def _resolved_portfolio_scope(verb: str, portfolio_id: uuid.UUID) -> TargetResolution:
    """A ``resolved`` portfolio-scope freeze / re-arm of ``portfolio_id``."""
    action = ResolvedAction(verb, None, portfolio_id, "portfolio")
    return TargetResolution(
        "resolved", action, (), f"{verb} portfolio {_short(portfolio_id)}"
    )


def _resolve_rebalance(
    turn: str, session: Any, focus: uuid.UUID | None
) -> TargetResolution:
    """Resolve a ``run`` (rebalance) to the named / focused portfolio, else ask."""
    if _NEGATION.search(turn.lower()):
        return _negated("run")
    named, stale = _named_portfolios(turn, session)
    if stale:
        return _unknown_portfolio(stale)
    lowered = turn.lower()
    names_fund = _FUND_WORD.search(lowered) is not None
    if len(named) > 1 or names_fund or (not named and _BLANKET_WORD.search(lowered)):
        return TargetResolution(
            "ambiguous",
            None,
            (),
            "run which portfolio — a rebalance is per portfolio, name one by id",
        )
    target = named[0] if named else focus
    if target is None:
        return TargetResolution(
            "ambiguous", None, (), "run which portfolio — focus one first"
        )
    action = ResolvedAction("run", None, target, None)
    return TargetResolution(
        "resolved", action, (), f"run a rebalance for {_short(target)}"
    )


def _named_portfolios(turn: str, session: Any) -> tuple[list[uuid.UUID], list[str]]:
    """The known portfolios ``turn`` names by short id, and the ids that name none.

    Reads the fund roster only when the turn actually typed an id.
    """
    tokens = _id_tokens(turn)
    if not tokens:
        return [], []
    known = observe.list_portfolios(session)
    named: list[uuid.UUID] = []
    stale: list[str] = []
    for token in tokens:
        hits = [pid for pid in known if _short(pid).startswith(token)]
        if not hits:
            stale.append(token)
        named.extend(pid for pid in hits if pid not in named)
    return named, stale


def _unknown_portfolio(stale: list[str]) -> TargetResolution:
    """The dead end for a turn that typed ids matching no known portfolio."""
    typed = ", ".join(f"#{token}" for token in stale)
    return TargetResolution(
        "unresolved", None, (), f"no portfolio matches {typed} — check the id"
    )


def _negated(verb: str) -> TargetResolution:
    """The follow-up for a turn whose target is mixed up with a negation."""
    return TargetResolution(
        "ambiguous",
        None,
        (),
        f"{verb} which? A negation is too easy to misread — name the target "
        "outright (the fund, or a portfolio id)",
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
    clears the pending action). A follow-up that opens with a *different* decision verb
    ("reject #44ae61f1" after an "approve" question) is a new command, not an answer, so
    it is left unresolved rather than executed under the carried verb.

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
    follow_up_verb = _leading_verb(turn)
    if follow_up_verb is not None and follow_up_verb != pending.verb:
        return TargetResolution("unresolved", None, (), "a different action")
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
        return _resolve_scope(pending.verb, turn, session, focus)
    return _resolve_rebalance(turn, session, focus)


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
