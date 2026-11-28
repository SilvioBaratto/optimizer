"""The user's choice of one candidate at the rebalance gate.

A rebalance round computes four candidate portfolios, the ``place_orders`` gate
shows all of them, and the user approves by naming one. The software never picks
one: there is no default and no automatic choice, and only the named candidate's
weights reach the paper ticket. The one exception is a user edit: weights the
user typed after the set was computed are a choice too, so they approve without
an id. The rule follows the gate that is pending, not the stored set: an approve of
any other gate (one a mandate adds ahead of ``place_orders``) shows no candidates, so
it takes no id and commits no weights.

Every refusal is a ``ValueError`` raised before anything is written, so the caller's
rollback leaves the gate open for a corrected decision. The choice writers,
:func:`approve_weights` and :func:`record_candidate_selection`, are reached only by
the user's approve (through :func:`fund.agents.graph.resume_fund` and
:meth:`fund.agents.graph.FundRun.resume`); no agent, tool, scheduler or worker path
records a choice, which ``test_candidate_selection_writers`` keeps true. The
read-only checks here also run where a round or a user action is settled.

The repositories, the candidate reader and the live check are imported inside the
functions, so a bare ``import fund.agents.graph_candidates`` stays light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fund.schemas.candidates import CANDIDATE_IDS

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

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
    "is_order_gate",
    "pending_gate_names",
    "record_candidate_selection",
    "refuse_candidate_at_profiler_gate",
    "refuse_candidate_outside_approve",
    "resolve_approval",
]

# The ``source`` of the risk verdict recorded when the user names a candidate: the
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
        weights: The weights the paper ticket is placed from. Empty for an approve of
            a gate other than ``place_orders``, which only lets the PM continue.
        candidate: The named candidate, ``None`` when a user edit is the choice
            or the gate is not the order gate.
        candidate_set_hash: Digest of the set ``candidate`` was chosen from.
    """

    weights: dict[str, float]
    candidate: Candidate | None = None
    candidate_set_hash: str | None = None


def _edit_in_force(session: Session, run_id: uuid.UUID) -> dict[str, float]:
    """The weights of a user edit still in force for the run, else ``{}``."""
    from fund.audit.repository import EDIT_DECISION, AgentRunRepository

    repo = AgentRunRepository(session)
    choice = repo.adviser_choice_in_force(run_id)
    if choice is None or (choice.hitl_decision or {}).get("decision") != EDIT_DECISION:
        return {}
    return repo.decode_weights(choice)


def pending_gate_names(interrupt: dict[str, Any] | None) -> tuple[str, ...]:
    """The gated tool names a pending HITL payload asks the user about.

    Args:
        interrupt: The payload a run is paused on (``action_requests``), or ``None``.

    Returns:
        The names in request order; empty when nothing is pending.
    """
    requests = (interrupt or {}).get("action_requests") or []
    return tuple(
        str(request.get("name")) for request in requests if isinstance(request, dict)
    )


def is_order_gate(pending: Sequence[str]) -> bool:
    """Whether an approve with ``pending`` gates is the approve of ``place_orders``.

    A thread paused at nothing readable has had its gate consumed (or cannot be read),
    which only the order gate's retry reaches, so it is held to the order gate's rule
    and fails closed.

    Args:
        pending: The gated tools the thread is paused on.

    Returns:
        ``True`` when ``place_orders`` is pending or nothing is.
    """
    return not pending or _ORDER_GATE in pending


def _refuse_candidate_off_the_order_gate(
    run_id: uuid.UUID, pending: Sequence[str], candidate_id: str | None
) -> None:
    """Refuse a candidate id on the approve of a gate that shows no candidates."""
    if candidate_id is not None:
        raise ValueError(
            f"agent_run {run_id} is at the {', '.join(pending)} gate, which shows no "
            "candidates; approve it without a candidate id"
        )


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
    session: Session,
    run_id: uuid.UUID,
    candidate_id: str | None,
    *,
    pending: Sequence[str],
) -> ApprovalChoice:
    """Resolve what approving the run's gate with ``candidate_id`` would commit.

    Reads the run's newest candidate set and user choice and writes nothing, so a
    caller can refuse an approve before any audit row exists.

    Args:
        session: A sync session; read-only here.
        run_id: The run whose gate is approved.
        candidate_id: The candidate the user named, or ``None``.
        pending: The gated tools the thread is paused on (see :func:`is_order_gate`).

    Returns:
        For the order gate: the named candidate with its weights and set digest, or
        the edited weights when ``candidate_id`` is ``None`` and a user edit
        newer than the set is in force. For any other gate: empty weights.

    Raises:
        ValueError: At the order gate, the run has no candidate set, nothing in it is
            feasible, no id was named, or the id is unknown or infeasible; at any
            other gate, an id was named.
    """
    from fund import observe

    if not is_order_gate(pending):
        _refuse_candidate_off_the_order_gate(run_id, pending, candidate_id)
        return ApprovalChoice(weights={})
    if candidate_id is None and (edited := _edit_in_force(session, run_id)):
        return ApprovalChoice(weights=edited)
    candidate_set = observe.candidate_set(session, run_id)
    if candidate_set is None:
        raise ValueError(_NO_SET.format(run_id=run_id))
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
            f"candidate {candidate_id!r} breaches the current constraints: "
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
        candidate_id: The candidate the user named.

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


def approve_weights(
    ctx: RunContext, candidate_id: str | None, *, pending: Sequence[str]
) -> dict[str, float]:
    """The weights an approve of the run's gate commits, recording a named choice.

    Only an approve of the ``place_orders`` gate names and records a candidate; any
    other gate shows no candidates, so it records nothing and commits no weights.

    Args:
        ctx: The resumed run's binding context.
        candidate_id: The candidate the user named, or ``None`` (a user edit
            in force, or a gate other than the order gate).
        pending: The gated tools the thread is paused on.

    Returns:
        The weights the approve places: the named candidate's, the edited ones, or
        ``{}`` for a gate other than the order gate.

    Raises:
        ValueError: The approve is refused (see :func:`resolve_approval` and
            :func:`record_candidate_selection`); nothing was written.
    """
    if candidate_id is None or not is_order_gate(pending):
        return resolve_approval(
            ctx.session, ctx.run_id, candidate_id, pending=pending
        ).weights
    return record_candidate_selection(ctx, candidate_id)


def refuse_candidate_outside_approve(verb: str, candidate_id: str | None) -> None:
    """Refuse a candidate id on any decision but an approve.

    Naming a candidate on a reject, killswitch or respond would read as a choice the
    user never made.

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

    A profile gate saves optimisation inputs and has no candidates, so an id there
    is a mistaken target, never something to ignore.

    Args:
        run_id: The run paused at the profiler's ``save_profile`` gate.
        candidate_id: The id the approve named, or ``None``.

    Raises:
        ValueError: ``candidate_id`` is given.
    """
    if candidate_id is not None:
        raise ValueError(
            f"agent_run {run_id} is at an inputs (profile) gate, which has no "
            "candidates; approve it without a candidate id"
        )


def gate_without_candidates(
    session: Session, run_id: uuid.UUID, interrupt: dict[str, Any] | None
) -> str | None:
    """Why a ``place_orders`` gate must not open for the run, or ``None``.

    A gate with no candidate set, or with no feasible candidate, gives the user
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

    if _ORDER_GATE not in pending_gate_names(interrupt):
        return None
    candidate_set = observe.candidate_set(session, run_id)
    if candidate_set is None:
        return NO_CANDIDATE_SET
    return None if candidate_set.selectable_ids() else NO_FEASIBLE_CANDIDATE
