"""A failed round: its record, the first-round budget hand-back, cleanup."""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# Records keep the logger name these functions had in ``fund.agents.graph``.
_log = logging.getLogger("fund.agents.graph")


# Cap on the error text stored with a failed run; provider errors can embed whole
# response bodies.
_RUN_ERROR_MAX_CHARS = 2000


@contextmanager
def _own_session(session: Any) -> Iterator[Any]:
    """A short session on the round session's bind, committed by its user.

    Bookkeeping that must not live and die with the round's transaction (the
    first-round gate, the record of a failed round) goes through this, so it commits
    at once and holds no row for the length of an LLM round.
    """
    from sqlalchemy.orm import Session

    with Session(bind=session.get_bind()) as own:
        yield own


def _consume_first_round(session: Any, portfolio_id: uuid.UUID) -> bool:
    """Spend the portfolio's one-time first-round gate in a session of its own.

    The gate is a row of ``fund_autonomy_state``. Updating it in the round's
    transaction would hold that row until the round ends, minutes later, and an
    operator's freeze of the same portfolio would wait on it. Committing at once also
    keeps the row creation out of the round: a round that is rolled back neither
    leaves the gate spent nor, on a fresh install, takes the new row down with it.

    Returns:
        ``True`` when this call spent the gate (the round is the portfolio's first),
        ``False`` when an earlier round already had.
    """
    from fund.audit.autonomy_repository import AutonomyRepository

    with _own_session(session) as own:
        consumed = AutonomyRepository(own).consume_first_round(portfolio_id)
        own.commit()
    return consumed


def _restore_first_round(session: Any, portfolio_id: uuid.UUID) -> None:
    """Hand back a gate :func:`_consume_first_round` spent, in a session of its own.

    The spend is already committed, so the hand-back has to be as well: one made inside
    the round's transaction would be rolled back with a failed round and leave the gate
    gone.
    """
    from fund.audit.autonomy_repository import AutonomyRepository

    with _own_session(session) as own:
        AutonomyRepository(own).restore_first_round(portfolio_id)
        own.commit()


def _hand_back_first_round(session: Any, portfolio_id: uuid.UUID) -> bool:
    """Restore a spent first-round gate; ``False`` (logged) when that failed."""
    try:
        _restore_first_round(session, portfolio_id)
    except Exception:
        _log.exception("could not hand back the first-round gate of %s", portfolio_id)
        return False
    return True


@dataclass(frozen=True)
class FailedRound:
    """What a failed round leaves for :func:`record_failed_round` to write durably.

    Attributes:
        run: The ``agent_runs`` row's columns, as finalised ``failed``.
        decisions: The columns of each of the run's ``agent_decisions`` rows, in order.
        gate_portfolio_id: The portfolio whose first-round gate the round spent and
            could not hand back at the time, or ``None`` when there is nothing left to
            hand back.
    """

    run: dict[str, Any] = field(repr=False)
    decisions: list[dict[str, Any]] = field(repr=False)
    gate_portfolio_id: uuid.UUID | None = None


# The attribute :func:`_finalize_failed` sets on the exception it was handed, so the
# code that rolls the round back can still find what the round recorded.
_FAILED_ROUND_ATTR = "fund_failed_round"


def failed_round_of(exc: BaseException) -> FailedRound | None:
    """The :class:`FailedRound` a failed fund round attached to ``exc``, if any."""
    found = getattr(exc, _FAILED_ROUND_ATTR, None)
    return found if isinstance(found, FailedRound) else None


def _attach_failed_round(
    audit: Any,
    run_id: uuid.UUID,
    exc: Exception,
    *,
    gate_portfolio_id: uuid.UUID | None,
) -> None:
    """Snapshot the failed run and its decisions onto ``exc`` for the rollback."""
    run = audit.get_run(run_id)
    if run is None:
        return
    decisions = sorted(run.decisions, key=lambda d: d.decision_index)
    failed = FailedRound(
        run=run.to_dict(),
        decisions=[decision.to_dict() for decision in decisions],
        gate_portfolio_id=gate_portfolio_id,
    )
    try:
        setattr(exc, _FAILED_ROUND_ATTR, failed)
    except AttributeError:  # pragma: no cover - exceptions carry a __dict__
        _log.warning("could not attach the failed round %s to %r", run_id, exc)


def record_failed_round(session: Any, exc: BaseException) -> bool:
    """Write a failed round that its caller rolled back, in a session of its own.

    :func:`_finalize_failed` marks the run ``failed`` inside the round's transaction,
    and the callers that drive rounds on a shared session (the scheduler sweeps, the
    CLI and TUI through ``get_session``) roll that transaction back when the error
    propagates, taking the run row and its error with it. Called after that rollback,
    this re-creates the run and its decisions from the snapshot the exception carries,
    so the failure stays visible and the drift monitor's cooldown (which counts
    ``failed`` among the outcomes nobody acted on) still applies. It also hands back a
    first-round gate that :func:`_finalize_failed` could not. Best effort: never
    raises, and a run that already exists (the caller committed after all) is left as
    it is.

    Args:
        session: The round's session, after its rollback; only its bind is used.
        exc: The exception the failed round raised.

    Returns:
        ``True`` when the failed run is now on record, ``False`` when ``exc`` carries
        no failed round or writing it failed.
    """
    failed = failed_round_of(exc)
    if failed is None:
        return False
    from portopt_db.models import AgentDecision, AgentRun

    if failed.gate_portfolio_id is not None:
        _hand_back_first_round(session, failed.gate_portfolio_id)
    try:
        with _own_session(session) as own:
            if own.get(AgentRun, failed.run["id"]) is None:
                own.add(AgentRun(**failed.run))
                own.flush()
                own.add_all(AgentDecision(**row) for row in failed.decisions)
            own.commit()
    except Exception:
        _log.exception("could not record failed fund run %s", failed.run.get("id"))
        return False
    return True


def _drop_checkpoint_thread(
    checkpointer: Any, run_id: uuid.UUID, thread_id: str | None
) -> None:
    """Delete the run's LangGraph thread; best effort, and only a thread the run owns.

    A failed run is terminal, so nothing can resume its thread, and the thread is keyed
    by a run id that is gone once the caller rolls back. A thread the caller supplied
    is not the run's to delete.
    """
    if checkpointer is None or thread_id != str(run_id):
        return
    try:
        checkpointer.delete_thread(thread_id)
    except Exception:
        _log.exception("could not delete the checkpoint thread of run %s", run_id)


def _finalize_failed(
    audit: Any,
    run_id: uuid.UUID,
    exc: Exception,
    *,
    session: Any,
    portfolio_id: uuid.UUID,
    first_round: bool,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
) -> None:
    """Finalise a round that errored before reaching the gate as ``failed``.

    Without it the run row stays ``pending`` for good: nothing else ever closes it. The
    error text lands in ``agent_runs.error``, inside the round's transaction: the caller
    rolls that transaction back when the error propagates, so it survives only where the
    caller commits, or calls :func:`record_failed_round` after its rollback, which the
    snapshot attached to ``exc`` serves. A first-round gate the round consumed
    (``first_round``) is handed back in a session of its own, since no adviser reviewed
    anything and the spend was committed on its own too. The run's checkpoint thread is
    deleted: nothing can resume a failed run. Best effort: a broken session (a failed
    flush awaits a rollback) must not mask the error being propagated. No commit of
    ``session``.
    """
    reason = f"{type(exc).__name__}: {exc}"[:_RUN_ERROR_MAX_CHARS]
    finalised = True
    try:
        audit.append_decision(
            run_id,
            agent="orchestrator",
            step="place_orders",
            hitl_decision={"decision": "failed", "reason": type(exc).__name__},
        )
        audit.finalize_run(run_id, weights={}, status="failed", error=reason)
    except Exception:
        finalised = False
        _log.exception("could not finalise failed fund run %s", run_id)
    handed_back = _hand_back_first_round(session, portfolio_id) if first_round else True
    if finalised:
        try:
            _attach_failed_round(
                audit,
                run_id,
                exc,
                gate_portfolio_id=None if handed_back else portfolio_id,
            )
        except Exception:
            _log.exception("could not snapshot failed fund run %s", run_id)
    _drop_checkpoint_thread(checkpointer, run_id, thread_id)
