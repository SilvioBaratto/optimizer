"""A run that re-gates after an approve must be approvable again.

The idempotency key of a gate resolution used to be ``<run_id>:<decision>``, which names
the run, not the gate. A run that re-gates (an approve whose ``place_orders`` is refused
and retried, or a barge-in) opens a second gate under the same run id, so the adviser's
second approve was replayed from the first one's audit row and never resumed the run.
"""

from __future__ import annotations

import uuid

import pytest
from _fund_fakes import ASOF, PORTFOLIO_ID
from _gate_env import make_gate_env
from portopt_db.models.market_data.yfinance_data import PriceHistory
from sqlalchemy import insert, select

from fund import observe
from fund.actions import apply_action
from fund.audit import AgentRunRepository, HumanAuditLogRepository
from fund.tui.app import ShellApp


@pytest.fixture
def env(tmp_path):
    """One run paused at its gate; the PM may place orders three times to re-gate."""
    harness = make_gate_env(tmp_path / "gate.db", max_orders=3)
    yield harness
    harness.dispose()


def _remove_bars_after_asof(env) -> list[dict]:
    """Delete the fill bars so ``place_orders`` refuses; return them for restoring."""
    with env.get_session() as session:
        rows = session.execute(
            select(PriceHistory).where(PriceHistory.date > ASOF)
        ).scalars()
        saved = [
            {col.name: getattr(row, col.name) for col in PriceHistory.__table__.columns}
            for row in rows
        ]
        session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
        session.commit()
    return saved


def _restore_bars(env, saved: list[dict]) -> None:
    with env.get_session() as session:
        session.execute(insert(PriceHistory), saved)
        session.commit()


@pytest.fixture
def regating_run(env, monkeypatch):
    """Let an approve reach ``place_orders`` with no fill bar, so it re-gates."""
    from fund.agents import graph as graph_mod

    monkeypatch.setattr(graph_mod, "_require_fill_bars", lambda *_a, **_k: None)
    return _remove_bars_after_asof(env)


def _approve_rows(env):
    with env.get_session() as session:
        return HumanAuditLogRepository(session).list_recent(action="approve")


@pytest.mark.asyncio
async def test_tui_second_approve_resumes_a_run_that_regated(env, regating_run) -> None:
    """The approve worker resumes the second gate instead of replaying the first."""
    app = ShellApp(
        None,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=env.model_factory,
        poll_interval=30.0,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        worker = app._action_worker(env.run_id, PORTFOLIO_ID, "approve")
        await worker.wait()
        await pilot.pause()
        with env.get_session() as session:
            first_status = AgentRunRepository(session).get_run(env.run_id).status
        assert first_status == "responded", "the first approve must re-gate the run"

        _restore_bars(env, regating_run)
        worker = app._action_worker(env.run_id, PORTFOLIO_ID, "approve")
        await worker.wait()
        await pilot.pause()

        assert "already applied" not in app.last_status
        assert "action failed" not in app.last_status
    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "completed"
    keys = [row.idempotency_key for row in _approve_rows(env)]
    assert len(keys) == 2
    assert len(set(keys)) == 2


def test_two_approves_through_apply_action_each_resume_their_own_gate(
    env, regating_run
) -> None:
    """Approve, re-gate, approve: each resolution carries a key for its own gate."""
    from fund.actions import resolution_key

    with env.get_session() as session:
        first_key = resolution_key(session, env.saver, env.run_id, "approve")
        apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            idempotency_key=first_key,
        )
        session.commit()
    _restore_bars(env, regating_run)

    with env.get_session() as session:
        second_key = resolution_key(session, env.saver, env.run_id, "approve")
        assert second_key != first_key
        result = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            idempotency_key=second_key,
        )
        session.commit()

    assert "already applied" not in result.message
    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "completed"


def test_a_repeat_of_the_same_gate_resolution_still_replays(env) -> None:
    """A double submit once the gate is consumed replays the logged approve."""
    from fund.actions import resolution_key

    with env.get_session() as session:
        key = resolution_key(session, env.saver, env.run_id, "approve")
        first = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            idempotency_key=key,
        )
        session.commit()

    with env.get_session() as session:
        # The gate is gone now, so the key falls back to the resolution already logged.
        again_key = resolution_key(session, env.saver, env.run_id, "approve")
        again = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            idempotency_key=again_key,
        )
        session.commit()

    assert again.audit_id == first.audit_id
    assert "already applied" in again.message
    assert len(_approve_rows(env)) == 1


def test_the_key_is_stable_while_the_gate_stays_pending(env) -> None:
    """Two concurrent submits of one gate share a key, so one is turned away."""
    from fund.actions import resolution_key

    with env.get_session() as session:
        first = resolution_key(session, env.saver, env.run_id, "approve")
        second = resolution_key(session, env.saver, env.run_id, "approve")
        assert first == second
        assert str(env.run_id) in first
        assert resolution_key(session, env.saver, env.run_id, "reject") != first


def test_each_gate_of_one_run_has_its_own_id(env, regating_run) -> None:
    """The gate id changes when an approve re-gates the run."""
    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        first_gate = observe.live_gate_id(env.saver, run)
        apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()
        run = AgentRunRepository(session).get_run(env.run_id)
        second_gate = observe.live_gate_id(env.saver, run)

    assert first_gate is not None
    assert second_gate is not None
    assert first_gate != second_gate


def test_a_run_that_is_not_at_a_gate_has_no_gate_id(env) -> None:
    """A finalised run, a missing run or an unknown thread has no pending gate."""
    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        run.status = "completed"
        assert observe.live_gate_id(env.saver, run) is None
        assert observe.live_gate_id(env.saver, None) is None
        run.status = "paused"
        run.thread_id = f"missing-{uuid.uuid4()}"
        assert observe.live_gate_id(env.saver, run) is None


@pytest.mark.asyncio
async def test_tui_second_edit_resumes_a_run_that_regated(
    env, regating_run, monkeypatch
) -> None:
    """The edit worker keys each edit on its gate, so a re-gated run is editable."""
    from fund.tools import orders

    monkeypatch.setattr(orders, "fill_shortfalls", lambda *_a, **_k: [])
    app = ShellApp(
        None,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=env.model_factory,
        poll_interval=30.0,
    )
    first_edit = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    second_edit = {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        worker = app._edit_worker(env.run_id, PORTFOLIO_ID, first_edit)
        await worker.wait()
        await pilot.pause()
        with env.get_session() as session:
            assert AgentRunRepository(session).get_run(env.run_id).status == "responded"

        _restore_bars(env, regating_run)
        worker = app._edit_worker(env.run_id, PORTFOLIO_ID, second_edit)
        await worker.wait()
        await pilot.pause()

        assert "already applied" not in app.last_status
        assert "action failed" not in app.last_status
    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "completed"
        rows = HumanAuditLogRepository(session).list_recent(action="edit")
    assert len({row.idempotency_key for row in rows}) == 2


@pytest.mark.parametrize("action", ["reject", "killswitch"])
def test_the_emergency_stop_key_needs_no_checkpoint_read(env, action) -> None:
    """Reject and killswitch finalise the run at any gate, so the run id is enough."""
    from fund.actions import resolution_key

    with env.get_session() as session:
        key = resolution_key(session, None, env.run_id, action)

    assert key == f"{env.run_id}:{action}"
