"""An approve resolves the gate the adviser was shown, not whichever gate is live.

The Decisions view renders a gate's orders and risk verdict, and the key of the approve
that follows used to be read from the checkpoint at keypress time. A run that re-gated
in between (another process, a barge-in) was approved at its NEW gate, a decision on
orders the adviser had never seen. The view now remembers the gate id it rendered and
the approve is refused when the run is parked at a different one.
"""

from __future__ import annotations

import pytest
from _fund_fakes import PORTFOLIO_ID, make_mandate
from _gate_env import make_gate_env
from test_gate_resolution_key import _remove_bars_after_asof
from textual.widgets import ContentSwitcher

from fund import observe
from fund.actions import GateMovedError, apply_action, resolution_key
from fund.audit import AgentRunRepository, HumanAuditLogRepository, MandateRepository
from fund.tui.app import ShellApp
from fund.tui.views import DecisionsView


@pytest.fixture
def env(tmp_path):
    """One run paused at its gate; the PM may place orders three times to re-gate."""
    harness = make_gate_env(tmp_path / "binding.db", max_orders=3)
    with harness.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    yield harness
    harness.dispose()


@pytest.fixture
def regating_run(env, monkeypatch):
    """Let an approve reach ``place_orders`` with no fill bar, so it re-gates."""
    from fund.agents import graph as graph_mod

    monkeypatch.setattr(graph_mod, "_require_fill_bars", lambda *_a, **_k: None)
    return _remove_bars_after_asof(env)


def _live_gate(env) -> str | None:
    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        return observe.live_gate_id(env.saver, run)


def _regate(env) -> None:
    """Approve once with no fill bar, which parks the run at a second gate."""
    with env.get_session() as session:
        apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()


def test_a_key_bound_to_the_live_gate_is_the_usual_key(env) -> None:
    """Naming the gate the run is parked at changes nothing about the key."""
    gate = _live_gate(env)
    with env.get_session() as session:
        bound = resolution_key(
            session, env.saver, env.run_id, "approve", expected_gate=gate
        )
        plain = resolution_key(session, env.saver, env.run_id, "approve")
    assert bound == plain


def test_a_key_bound_to_a_gate_the_run_has_left_is_refused(env, regating_run) -> None:
    """The adviser saw gate 1; the run now waits at gate 2, which they have not seen."""
    shown = _live_gate(env)
    _regate(env)
    assert _live_gate(env) != shown

    with env.get_session() as session, pytest.raises(GateMovedError):
        resolution_key(session, env.saver, env.run_id, "approve", expected_gate=shown)


def test_a_finished_run_still_replays_whatever_gate_was_shown(env) -> None:
    """A late double press must reach the replay path, not a refusal."""
    shown = _live_gate(env)
    with env.get_session() as session:
        AgentRunRepository(session).get_run(env.run_id).status = "completed"
        key = resolution_key(
            session, env.saver, env.run_id, "approve", expected_gate=shown
        )
    assert str(env.run_id) in key


@pytest.mark.parametrize("action", ["reject", "killswitch"])
def test_the_emergency_stop_never_waits_on_the_shown_gate(env, action: str) -> None:
    """Reject and killswitch settle the run whatever gate it is at."""
    with env.get_session() as session:
        key = resolution_key(
            session, env.saver, env.run_id, action, expected_gate="some-old-gate"
        )
    assert key == f"{env.run_id}:{action}"


def _shell(env) -> ShellApp:
    return ShellApp(
        PORTFOLIO_ID,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=env.model_factory,
        poll_interval=30.0,
    )


@pytest.mark.asyncio
async def test_the_decisions_view_remembers_the_gate_it_rendered(env) -> None:
    """The id of the gate on screen is kept next to its text."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(ContentSwitcher).current = "view-decisions"
        app._refresh()
        await pilot.pause()
        decisions = app.query_one("#view-decisions", DecisionsView)
        decisions.select_run(env.run_id)
        await pilot.pause()

        assert decisions.rendered_gate_id == _live_gate(env)
        assert decisions.rendered_gate_id is not None


@pytest.mark.asyncio
async def test_an_approve_after_the_run_moved_to_another_gate_is_not_sent(
    env, regating_run
) -> None:
    """Pressing approve on a stale view reports the move and resolves nothing."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(ContentSwitcher).current = "view-decisions"
        app._refresh()
        await pilot.pause()
        decisions = app.query_one("#view-decisions", DecisionsView)
        decisions.select_run(env.run_id)
        await pilot.pause()
        shown = decisions.rendered_gate_id

        _regate(env)  # another process re-gates the run; this view is not repainted
        assert _live_gate(env) != shown

        app._decide("approve")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert "moved" in app.last_status
    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "responded"
        approvals = HumanAuditLogRepository(session).list_recent(action="approve")
    assert len(approvals) == 1, "only the setup approve is logged"
