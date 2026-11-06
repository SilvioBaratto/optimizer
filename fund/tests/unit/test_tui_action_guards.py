"""Guards on the TUI's action workers: rounds, lost steers and blind approves."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from _fund_fakes import PORTFOLIO_ID, make_mandate
from _gate_env import make_gate_env
from textual.widgets import ContentSwitcher, Static

from fund.audit import MandateRepository
from fund.tui import app as tui_app
from fund.tui.app import ConfirmScreen, ShellApp
from fund.tui.barge_in import BargeInScreen
from fund.tui.questionnaire import QuestionnaireScreen
from fund.tui.views import AskView, DecisionsView


@pytest.fixture
def env(tmp_path):
    """A run paused at its gate, with the portfolio's mandate persisted."""
    harness = make_gate_env(tmp_path / "guards.db", max_orders=2)
    with harness.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    yield harness
    harness.dispose()


def _shell(env) -> ShellApp:
    return ShellApp(
        PORTFOLIO_ID,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=env.model_factory,
        poll_interval=30.0,
    )


async def _wait_until(pilot, predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await pilot.pause(0.02)


@pytest.fixture
def blocking_run(monkeypatch):
    """Make the ``run`` action block until released; every other action returns."""
    import fund.actions as actions_mod

    entered, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def stub(action, **_kwargs):
        """Record the action, holding a ``run`` open until the test releases it."""
        calls.append(action)
        if action == "run":
            entered.set()
            release.wait(timeout=5.0)
        return SimpleNamespace(message=f"{action} done", detail=None)

    monkeypatch.setattr(actions_mod, "apply_action", stub)
    yield SimpleNamespace(entered=entered, release=release, calls=calls)
    release.set()


@pytest.mark.asyncio
async def test_a_profile_does_not_start_beside_a_round_in_flight(
    env, blocking_run
) -> None:
    """A round's thread outlives a cancelled Textual worker, so the thread counts."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        round_worker = app._launch_run(PORTFOLIO_ID, None)
        await _wait_until(pilot, blocking_run.entered.is_set)

        profile = app._onboard_worker(PORTFOLIO_ID, "{}", None, "EUR")
        await profile.wait()
        await pilot.pause()

        assert blocking_run.calls == ["run"]
        assert app._round_in_flight()
        assert "in progress" in app.last_status

        blocking_run.release.set()
        await round_worker.wait()
        await _wait_until(pilot, lambda: not app._round_in_flight())


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["action_new_portfolio", "action_profile_portfolio"])
async def test_the_questionnaire_does_not_open_during_a_round(
    env, blocking_run, key
) -> None:
    """Posture's ``n`` / ``p`` say why instead of collecting answers they cannot run."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._launch_run(PORTFOLIO_ID, None)
        await _wait_until(pilot, blocking_run.entered.is_set)

        getattr(app, key)()
        await pilot.pause()

        assert not isinstance(app.screen, QuestionnaireScreen)
        assert "in progress" in app.last_status
        blocking_run.release.set()


@pytest.mark.asyncio
async def test_a_steer_refused_behind_another_action_is_echoed(env) -> None:
    """The adviser's text is echoed, not lost, when the single-flight guard refuses."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app._action_lock.acquire(blocking=False)
        try:
            worker = app._barge_in_worker(
                env.run_id, PORTFOLIO_ID, "Trim AAA by two points.", False
            )
            await worker.wait()
            await pilot.pause()
        finally:
            app._action_lock.release()

        assert "in progress" in app.last_status
        assert "Trim AAA by two points." in app.last_status
        assert "not sent" in app.last_status


async def _open_ask(app, pilot) -> AskView:
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-ask"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-ask", AskView)


@pytest.mark.asyncio
async def test_an_ask_approve_shows_the_orders_it_would_execute(env) -> None:
    """The confirm carries the gate's orders and risk verdict, not just a gate id."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()

        assert isinstance(app.screen, ConfirmScreen)
        prompt = str(app.screen.query_one(".confirm-prompt", Static).render())
        assert prompt.startswith("Approve and execute gate #")
        assert "orders (audited optimizer weights" in prompt
        for ticker in env.expected:
            assert f"{ticker}:" in prompt


@pytest.mark.asyncio
async def test_an_ask_approve_is_refused_when_the_gate_cannot_be_read(
    env, monkeypatch
) -> None:
    """Like Decisions, no approve is offered for orders the adviser could not see."""

    def unreadable(*_args, **_kwargs):
        raise RuntimeError("checkpointer pool timeout")

    monkeypatch.setattr(tui_app, "_gate_text", unreadable)
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()

        assert not isinstance(app.screen, ConfirmScreen)
        assert "cannot approve" in app.last_status
        assert app.last_worker is None


@pytest.mark.asyncio
async def test_an_ask_steer_names_the_gate_the_directive_goes_to(env) -> None:
    """The overlay says which gate it targets, since Ask picked it, not the cursor."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("steer the gate")
        await pilot.pause()

        assert isinstance(app.screen, BargeInScreen)
        where = str(app.screen.query_one(".bargein-where", Static).render())
        assert f"gate #{str(env.run_id)[:8]}" in where
        assert f"portfolio {str(PORTFOLIO_ID)[:8]}" in where


@pytest.mark.asyncio
async def test_a_decisions_steer_names_the_selected_gate(env) -> None:
    """The Decisions steer overlay carries the same gate label."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(ContentSwitcher).current = "view-decisions"
        app._refresh()
        await pilot.pause()
        decisions = app.query_one("#view-decisions", DecisionsView)
        decisions.select_run(env.run_id)
        await pilot.pause()

        app.action_steer_decision()
        await pilot.pause()

        assert isinstance(app.screen, BargeInScreen)
        where = str(app.screen.query_one(".bargein-where", Static).render())
        assert f"gate #{str(env.run_id)[:8]}" in where
