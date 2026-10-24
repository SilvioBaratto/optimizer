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
# scope; ``run`` (rebalance) targets the focused portfolio. ``killswitch`` is a resume
# verb only at run level: naming a scope turns it into the freeze at that scope.
_RESUME_VERBS: frozenset[str] = frozenset(
    {"approve", "reject", "killswitch", "respond"}
)
_SCOPE_VERBS: frozenset[str] = frozenset({"freeze", "rearm"})

# Fund-wide scope is named by the standalone word "fund". A bare substring test would
# misroute a focused freeze/rearm onto the whole fund on words like "refund"/"funding".
_FUND_WORD = re.compile(r"\bfund\b")
_PORTFOLIO_WORD = re.compile(r"\bportfolios?\b")

# "run" opens many ordinary questions ("run me through…", "run the numbers…"), so it
# only means a rebalance when a rebalance word follows it, past a few filler words.
_RUN_REBALANCE = re.compile(
    r"run\b(?:\s+(?:a|an|the|my|our|one|another|new|fresh|paper|manual|full|quick|now"
    r"|please))*\s+re-?balanc(?:e|ing)\b"
)

# "rebalance" is also a noun that heads many reads ("rebalance impact analysis", "the
# rebalancing numbers", "rebalance history"), so it only commands a rebalance when
# nothing but a target (an id, "the fund", "all portfolios") or politeness follows it.
_REBALANCE_WORD = re.compile(r"[#\w-]+")
_REBALANCE_FILLER: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "my",
        "our",
        "this",
        "that",
        "it",
        "them",
        "for",
        "of",
        "on",
        "in",
        "at",
        "to",
        "and",
        "or",
        "now",
        "today",
        "again",
        "please",
        "asap",
        "immediately",
        "portfolio",
        "portfolios",
        "fund",
        "whole",
        "entire",
        "all",
        "every",
        "each",
        "both",
        "everything",
    }
)

# Words that make a turn name "all of them" (a blanket target) or exclude something.
# Neither can be honoured by a single focused portfolio, so a turn using one asks which
# target is meant instead of quietly narrowing to the focus.
_BLANKET_WORD = re.compile(r"\b(?:all|every|everything|each|both)\b")
_NEGATION = re.compile(r"\b(?:not|never|nor|neither|except|without)\b|n['\u2019]t\b")

# A turn opening with one of these is a question or a read request, never the answer
# to "which gate / which scope?".
_QUESTION_START = re.compile(
    r"(?:what|why|how|when|where|who|whom|whose|is|are|was|were|does|did|do|can|could"
    r"|should|would|will|show|tell|explain)\b"
)

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
    did the allocator trim energy?"), so a read is never mistaken for a mutation. A
    turn ending in "?" is a question whatever its first word ("Reject the null: is drift
    significant?"). A leading "run" counts only when a rebalance word follows ("run a
    rebalance"); "run me through why…" is a question. A rebalance word counts only when
    nothing but a target follows it: "rebalance impact analysis" and "run the rebalance
    numbers" are reads.

    Args:
        turn: The adviser's natural-language turn.

    Returns:
        ``"action"`` when the leading word is a decision verb, else ``"query"``.
    """
    return "action" if _leading_verb(turn) is not None else "query"


def _leading_verb(turn: str) -> str | None:
    """The ``apply_action`` verb a turn's leading word maps to, or ``None``."""
    lowered = turn.strip().lower()
    if lowered.endswith("?"):
        return None
    match = re.match(r"[a-z][a-z-]*", lowered)
    if match is None:
        return None
    word = match.group(0)
    verb = _ACTION_VERBS.get(word)
    if word == "run":
        opened = _RUN_REBALANCE.match(lowered)
        if opened is None:
            return None
        return verb if _is_rebalance_command(lowered[opened.end() :]) else None
    if verb == "run":  # the leading synonym "rebalance"
        return verb if _is_rebalance_command(lowered[match.end() :]) else None
    return verb


def _is_rebalance_command(rest: str) -> bool:
    """Whether what follows a rebalance word is only a target or politeness.

    A negation counts as part of the command: the resolver asks about a negated target,
    which is safer than quietly answering the turn as a question.
    """
    if _NEGATION.search(rest):
        return True
    return all(
        word in _REBALANCE_FILLER or _ID_TOKEN.match(word) is not None
        for word in _REBALANCE_WORD.findall(rest)
    )


def resolve_target(
    turn: str, session: Any, saver: Any, focus: uuid.UUID | None
) -> TargetResolution:
    """Resolve which gate / portfolio / scope an action turn targets.

    Routes on the verb kind — never guessing a target:

    * a resume verb (approve / reject / respond) resolves against the pending-HITL
      gates: a single gate (or one named by its short id in the turn) is ``resolved``,
      two or more unnamed are ``ambiguous`` (the candidates), none is ``unresolved`` —
      as is a turn naming an id that matches no live gate (already decided, or a
      typo), which is never swapped for another gate. With a portfolio focused, an
      unnamed turn means that portfolio's gate; when none is paused there, the other
      portfolios' gates are ``ambiguous`` rather than offered as "the" gate. The focus
      only fills in a turn that names no gate and uses no blanket word: two typed ids
      stay ``ambiguous`` between exactly those gates. A negation ("approve not #id")
      is ``ambiguous`` over every live gate, whatever the verb or focus: the excluded
      gate is never the one acted on;
    * ``killswitch`` resolves the same way at run level, but a turn that names a scope
      ("the fund", "the portfolio") or a blanket target ("everything") without naming a
      live gate is the emergency stop at that scope: it resolves like ``freeze``, so the
      stop never shrinks to rejecting one paused run, nor dead-ends when none is paused;
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
    if verb == "killswitch":
        return _resolve_killswitch(turn, session, saver, focus)
    if verb in _RESUME_VERBS:
        return _resolve_gate(verb, turn, session, saver, focus)
    if verb in _SCOPE_VERBS:
        return _resolve_scope(verb, turn, session, focus)
    return _resolve_rebalance(turn, session, focus)  # verb == "run"


def _resolve_killswitch(
    turn: str, session: Any, saver: Any, focus: uuid.UUID | None
) -> TargetResolution:
    """Resolve a killswitch to a scoped freeze, or to a paused gate when no scope.

    The Decisions view offers run / portfolio / fund for the killswitch; typing a
    scope word is the same choice. Only a turn that names a live gate's own id keeps the
    run-level meaning once a scope word is present.
    """
    lowered = turn.lower()
    names_scope = (
        _FUND_WORD.search(lowered) is not None
        or _PORTFOLIO_WORD.search(lowered) is not None
        or _BLANKET_WORD.search(lowered) is not None
    )
    if names_scope and not _names_gate_run(turn, session, saver):
        return _resolve_scope("freeze", turn, session, focus)
    return _resolve_gate("killswitch", turn, session, saver, focus)


def _names_gate_run(turn: str, session: Any, saver: Any) -> bool:
    """Whether ``turn`` types the short id of a live gate's own run."""
    tokens = _id_tokens(turn)
    if not tokens:
        return False
    return any(
        _short(gate.run_id).startswith(token)
        for gate in observe.pending_hitl(session, saver, None)
        for token in tokens
    )


def _resolve_gate(
    verb: str, turn: str, session: Any, saver: Any, focus: uuid.UUID | None
) -> TargetResolution:
    """Resolve a resume verb against the pending-HITL gates (never on a guess)."""
    gates = observe.pending_hitl(session, saver, None)
    if not gates:
        hint = (
            " — to halt live runs say 'killswitch the fund' or 'killswitch the "
            "portfolio'"
            if verb == "killswitch"
            else ""
        )
        return TargetResolution(
            "unresolved", None, (), f"no gate is awaiting a decision{hint}"
        )
    if _NEGATION.search(turn.lower()):
        prompt = "a negation is too easy to misread — " + _gate_prompt(verb, gates)
        return TargetResolution("ambiguous", None, tuple(gates), prompt)
    stale = _stale_ids(turn, gates)
    if stale:
        typed = ", ".join(f"#{token}" for token in stale)
        return TargetResolution(
            "unresolved",
            None,
            (),
            f"no gate awaiting a decision matches {typed} — it may already be decided",
        )
    named = _named_gates(turn, gates)
    if len(named) == 1:
        return _resolved_gate(verb, named[0])
    if named:
        # The turn typed several gates; the focus must not pick one of them for it.
        return TargetResolution(
            "ambiguous", None, tuple(named), _gate_prompt(verb, named)
        )
    if focus is not None and _BLANKET_WORD.search(turn.lower()) is None:
        focused = [gate for gate in gates if gate.portfolio_id == focus]
        if not focused:
            prompt = (
                f"no gate is paused for portfolio {_short(focus)}; "
                + _gate_prompt(verb, gates)
            )
            return TargetResolution("ambiguous", None, tuple(gates), prompt)
        gates = focused
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


def _named_gates(turn: str, gates: list[RunSummary]) -> list[RunSummary]:
    """The gates whose short run/portfolio id ``turn`` types (empty for a bare turn)."""
    tokens = _id_tokens(turn)
    return [
        gate
        for gate in gates
        if any(gid.startswith(token) for token in tokens for gid in _gate_ids(gate))
    ]


def _match_gate(turn: str, gates: list[RunSummary]) -> RunSummary | None:
    """The single gate named by its short run/portfolio id in ``turn``, else ``None``.

    Returns ``None`` when zero or more than one gate matches, so a bare "approve the
    gate" (no id) and an ambiguous "approve both" both fall through to the gate count.
    """
    matched = _named_gates(turn, gates)
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
    verb = _leading_verb(turn) or ""
    if verb == "killswitch" and not resolution.candidates:
        # Only a scope question is ambiguous without gate candidates, and the answer
        # ("the fund" / a portfolio id) completes a freeze, not a gate pick.
        verb = "freeze"
    return PendingAction(
        verb=verb,
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

    The follow-up must *answer* the question that was asked. It is not an answer — so
    stays non-``resolved`` and the caller handles it as a fresh turn — when it is a
    question, carries a negation, or opens with a different decision verb (a pending
    ``approve`` followed by "reject #id" is a new reject request, not an approve). An
    answer to a resume-verb pending names a still-live candidate gate by short id; an
    answer to a scope pending names "the fund" or a portfolio id; an answer to a ``run``
    pending names a portfolio id. The focus is never an answer on its own: whatever is
    focused when the follow-up arrives does not make an unrelated turn a target.

    Args:
        pending: The carried :class:`PendingAction` from the prior ambiguous turn.
        turn: The adviser's follow-up turn.
        session: An injected read-only sync session; not owned here.
        saver: The bootstrapped LangGraph checkpointer (opaque), for the gate reads.
        focus: The focused portfolio, or ``None``. Not consulted (see above); kept so
            callers pass the same context as :func:`resolve_target`.

    Returns:
        A :class:`TargetResolution`; ``resolved`` only when the follow-up names the
        target unambiguously.
    """
    if not _answers_pending(pending.verb, turn):
        return TargetResolution("unresolved", None, (), "not an answer to the question")
    if pending.verb in _RESUME_VERBS:
        live = [
            gate
            for gate in observe.pending_hitl(session, saver, None)
            if gate.run_id in pending.candidates
        ]
        named = _match_gate(turn, live)
        if named is not None and not _stale_ids(turn, live):
            return _resolved_gate(pending.verb, named)
        return TargetResolution("unresolved", None, (), "could not match a gate")
    if pending.verb in _SCOPE_VERBS:
        if _FUND_WORD.search(turn.lower()) is None and not _id_tokens(turn):
            return TargetResolution("unresolved", None, (), "no target named")
        return _resolve_scope(pending.verb, turn, session, None)
    if not _id_tokens(turn):
        return TargetResolution("unresolved", None, (), "no portfolio named")
    return _resolve_rebalance(turn, session, None)


def _answers_pending(verb: str, turn: str) -> bool:
    """Whether ``turn`` can be the answer to a pending ``verb`` question.

    A question, a negation, or a turn opening with a different decision verb is a new
    request, not an answer.
    """
    lowered = turn.strip().lower()
    if lowered.endswith("?") or _QUESTION_START.match(lowered):
        return False
    if _NEGATION.search(lowered):
        return False
    leading = _leading_verb(lowered)
    return leading is None or leading == verb


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
