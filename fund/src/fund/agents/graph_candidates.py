"""The adviser's choice of one candidate at the rebalance gate.

A rebalance round computes four candidate portfolios, the ``place_orders`` gate
shows all of them, and the adviser approves by naming one. The software never picks
one: there is no default and no automatic choice, and only the named candidate's
weights reach the paper ticket. The one exception is an adviser edit: weights the
adviser typed after the set was computed are a choice too, so they approve without
an id.

Every refusal is a ``ValueError`` raised before anything is written, so the caller's
rollback leaves the gate open for a corrected decision. Only the adviser's approve
reaches :func:`approve_weights` (through :func:`fund.agents.graph.resume_fund` and
:meth:`fund.agents.graph.FundRun.resume`); no agent, tool, scheduler or worker path
calls into this module, which ``test_candidate_selection_writers`` keeps true.

The repositories, the candidate reader and the live check are imported inside the
functions, so a bare ``import fund.agents.graph_candidates`` stays light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fund.schemas.candidates import CANDIDATE_IDS

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.orm import Session

    from fund.agents.toolsets import RunContext
    from fund.schemas.candidates import Candidate, CandidateSet

__all__ = [
    "CANDIDATE_SELECTION_SOURCE",
    "NO_CANDIDATE_SET",
    "NO_FEASIBLE_CANDIDATE",
    "ApprovalChoice",
    "approve_weights",
    "gate_without_candidates",
    "record_candidate_selection",
    "refuse_candidate_at_profiler_gate",
    "refuse_candidate_outside_approve",
    "resolve_approval",
]

# The ``source`` of the risk verdict recorded when the adviser names a candidate: the
# verdict ``place_orders`` requires comes from that fresh check, not from an agent.
CANDIDATE_SELECTION_SOURCE = "candidate_selection"

# Why a round that reached ``place_orders`` is closed without a gate.
NO_CANDIDATE_SET = "no_candidate_set"
NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"

_ORDER_GATE = "place_orders"
_APPROVE = "approve"
_ID_LIST = ", ".join(CANDIDATE_IDS)
_NO_SET = "agent_run {run_id} has no candidate set; reject it and run a fresh round"


@dataclass(frozen=True)
class ApprovalChoice:
    """What an approve of a run's gate commits, before anything is written.

    Attributes:
        weights: The weights the paper ticket is placed from. Empty when the run has
            computed nothing to approve yet: a gate the mandate added before the
            allocator ran, which an approve only lets the PM continue past.
        candidate: The named candidate, ``None`` when an adviser edit is the choice
            or nothing was computed yet.
        candidate_set_hash: Digest of the set ``candidate`` was chosen from.
    """

    weights: dict[str, float]
    candidate: Candidate | None = None
    candidate_set_hash: str | None = None


def _edit_in_force(session: Session, run_id: uuid.UUID) -> dict[str, float]:
    """The weights of an adviser edit still in force for the run, else ``{}``."""
    from fund.audit.repository import EDIT_DECISION, AgentRunRepository

    repo = AgentRunRepository(session)
    choice = repo.adviser_choice_in_force(run_id)
    if choice is None or (choice.hitl_decision or {}).get("decision") != EDIT_DECISION:
        return {}
    return repo.decode_weights(choice)


def _without_a_set(
    session: Session, run_id: uuid.UUID, candidate_id: str | None
) -> ApprovalChoice:
    """The approve of a run that has no candidate set.

    A run that already proposed weights reached its gate before candidates existed,
    and so did any run whose candidate is named: neither can be approved. A run that
    computed nothing yet is at a gate the mandate added before the allocator, which
    commits no weights.
    """
    from fund.audit.repository import AgentRunRepository

    proposed = AgentRunRepository(session).latest_optimizer_weights(run_id)
    if candidate_id is None and not proposed:
        return ApprovalChoice(weights={})
    raise ValueError(_NO_SET.format(run_id=run_id))


def _named_candidate(
    candidate_set: CandidateSet, run_id: uuid.UUID, candidate_id: str | None
) -> Candidate:
    """The feasible candidate ``candidate_id`` names, or the refusal that explains."""
    from fund.agents.toolsets import describe_breaches

    selectable = candidate_set.selectable_ids()
    if not selectable:
        raise ValueError(
            f"agent_run {run_id}: no candidate is feasible; reject the run"
        )
    feasible = ", ".join(selectable)
    if candidate_id is None:
        raise ValueError(
            "approving a rebalance gate requires a candidate id: one of "
            f"{_ID_LIST} (feasible: {feasible})"
        )
    candidate = candidate_set.candidate(candidate_id)
    if candidate is None:
        raise ValueError(
            f"candidate {candidate_id!r} is not in the run's candidate set; name one "
            f"of {_ID_LIST} (feasible: {feasible})"
        )
    if not candidate.feasible:
        raise ValueError(
            f"candidate {candidate_id!r} is not feasible "
            f"({describe_breaches(list(candidate.breaches))}); feasible: {feasible}"
        )
    return candidate


def resolve_approval(
    session: Session, run_id: uuid.UUID, candidate_id: str | None
) -> ApprovalChoice:
    """Resolve what approving the run's gate with ``candidate_id`` would commit.

    Reads the run's newest candidate set and adviser choice and writes nothing, so a
    caller can refuse an approve before any audit row exists.

    Args:
        session: A sync session; read-only here.
        run_id: The run whose gate is approved.
        candidate_id: The candidate the adviser named, or ``None``.

    Returns:
        The named candidate with its weights and set digest; the edited weights when
        ``candidate_id`` is ``None`` and an adviser edit newer than the set is in
        force; or empty weights when the run has computed nothing to approve yet.

    Raises:
        ValueError: The run has no candidate set to choose from, nothing in it is
            feasible, no id was named, or the id is unknown or infeasible.
    """
    from fund import observe

    if candidate_id is None and (edited := _edit_in_force(session, run_id)):
        return ApprovalChoice(weights=edited)
    candidate_set = observe.candidate_set(session, run_id)
    if candidate_set is None:
        return _without_a_set(session, run_id, candidate_id)
    candidate = _named_candidate(candidate_set, run_id, candidate_id)
    return ApprovalChoice(
        weights=dict(candidate.weights),
        candidate=candidate,
        candidate_set_hash=candidate_set.digest(),
    )


def _require_fresh_pass(
    ctx: RunContext, candidate_id: str, weights: dict[str, float]
) -> None:
    """Refuse the candidate unless the live check passes it under today's mandate."""
    from fund.agents.toolsets import check_weights_against_mandate, describe_breaches

    check = check_weights_against_mandate(ctx, weights)
    if not check.get("ok"):
        raise ValueError(
            f"candidate {candidate_id!r} could not be checked: {check.get('error')}"
        )
    if not check["data"]["passed"]:
        raise ValueError(
            f"candidate {candidate_id!r} breaches the current mandate: "
            f"{describe_breaches(check['data']['violations'])}; choose another "
            "candidate or reject the run"
        )


def record_candidate_selection(ctx: RunContext, candidate_id: str) -> dict[str, float]:
    """Re-check the named candidate under the current mandate and record the choice.

    The set was judged when it was computed; the mandate, the eligible universe or
    the price history may have moved since, so the candidate is judged again by the
    live check the risk gate applies before the choice is recorded. The passing
    verdict is recorded first, then the choice, which supersedes any earlier one.

    Args:
        ctx: The resumed run's binding context.
        candidate_id: The candidate the adviser named.

    Returns:
        The chosen candidate's weights, the only weights ``place_orders`` will place.

    Raises:
        ValueError: The run has no candidate set, nothing in it is feasible, the id
            is unknown or infeasible, or the fresh check fails or cannot run;
            nothing was written.
    """
    from fund import observe
    from fund.audit.repository import AgentRunRepository

    candidate_set = observe.candidate_set(ctx.session, ctx.run_id)
    if candidate_set is None:
        raise ValueError(_NO_SET.format(run_id=ctx.run_id))
    candidate = _named_candidate(candidate_set, ctx.run_id, candidate_id)
    weights = dict(candidate.weights)
    _require_fresh_pass(ctx, candidate_id, weights)
    repo = AgentRunRepository(ctx.session)
    repo.append_risk_verdict(
        ctx.run_id,
        weights=weights,
        passed=True,
        violations=[],
        source=CANDIDATE_SELECTION_SOURCE,
    )
    repo.append_candidate_selection(
        ctx.run_id,
        candidate_id=candidate.candidate_id,
        candidate_set_hash=candidate_set.digest(),
        weights=weights,
        metrics=candidate.metrics.model_dump() if candidate.metrics else {},
        settings=candidate.settings,
    )
    return weights


def approve_weights(ctx: RunContext, candidate_id: str | None) -> dict[str, float]:
    """The weights an approve of the run's gate commits, recording a named choice.

    Args:
        ctx: The resumed run's binding context.
        candidate_id: The candidate the adviser named, or ``None`` (an adviser edit
            in force, or a gate reached before any candidate set).

    Returns:
        The weights the approve places: the named candidate's, the edited ones, or
        ``{}`` when nothing was computed yet.

    Raises:
        ValueError: The approve is refused (see :func:`resolve_approval` and
            :func:`record_candidate_selection`); nothing was written.
    """
    if candidate_id is None:
        return resolve_approval(ctx.session, ctx.run_id, None).weights
    return record_candidate_selection(ctx, candidate_id)


def refuse_candidate_outside_approve(verb: str, candidate_id: str | None) -> None:
    """Refuse a candidate id on any decision but an approve.

    Naming a candidate on a reject, killswitch or respond would read as a choice the
    adviser never made.

    Args:
        verb: The decision or logged action (``reject``, ``respond``, ...).
        candidate_id: The id it named, or ``None``.

    Raises:
        ValueError: ``candidate_id`` is given and ``verb`` is not ``approve``.
    """
    if candidate_id is not None and verb != _APPROVE:
        raise ValueError(
            f"a candidate id is named only when approving, not on {verb!r}"
        )


def refuse_candidate_at_profiler_gate(
    run_id: uuid.UUID, candidate_id: str | None
) -> None:
    """Refuse a candidate id on the approve of a profiler gate.

    A profile gate saves a suitability profile and has no candidates, so an id there
    is a mistaken target, never something to ignore.

    Args:
        run_id: The run paused at the profiler's ``save_profile`` gate.
        candidate_id: The id the approve named, or ``None``.

    Raises:
        ValueError: ``candidate_id`` is given.
    """
    if candidate_id is not None:
        raise ValueError(
            f"agent_run {run_id} is at a profiler gate, which has no candidates; "
            "approve it without a candidate id"
        )


def gate_without_candidates(
    session: Session, run_id: uuid.UUID, interrupt: dict[str, Any] | None
) -> str | None:
    """Why a ``place_orders`` gate must not open for the run, or ``None``.

    A gate with no candidate set, or with no feasible candidate, gives the adviser
    nothing to choose, so the round is closed incomplete instead.

    Args:
        session: A sync session; read-only here.
        run_id: The run whose round reached a gate.
        interrupt: The pending HITL payload the round ended on.

    Returns:
        ``"no_candidate_set"`` or ``"no_feasible_candidate"`` when ``place_orders``
        is among the gated calls and there is nothing to choose, else ``None``.
    """
    from fund import observe

    requests = (interrupt or {}).get("action_requests") or []
    if not any(
        isinstance(request, dict) and request.get("name") == _ORDER_GATE
        for request in requests
    ):
        return None
    candidate_set = observe.candidate_set(session, run_id)
    if candidate_set is None:
        return NO_CANDIDATE_SET
    return None if candidate_set.selectable_ids() else NO_FEASIBLE_CANDIDATE
