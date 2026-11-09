"""Ask shows the saved conversation when opened, and ``/clear`` deletes it for good."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from _fund_fakes import PORTFOLIO_ID, threaded_sqlite_engine
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import PrivateAttr
from sqlalchemy.orm import sessionmaker
from textual.widgets import ContentSwitcher, Static

from fund.agents.ask_agent import ask_thread_id
from fund.tui.app import ConfirmScreen, ShellApp
from fund.tui.ask_history import AskHistory, scope_of
from fund.tui.views import AskView

OTHER = uuid.UUID("99999999-9999-9999-9999-999999999999")


class CountingModel(BaseChatModel):
    """A network-free model that answers fixed text and counts the messages it gets."""

    answer: str = "all good"
    _seen: list[int] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "counting"

    @property
    def seen(self) -> list[int]:
        """How many messages each model call received."""
        return self._seen

    def bind_tools(self, tools: Any, **_: Any) -> CountingModel:
        """No tools are ever called, so binding changes nothing."""
        return self

    def _generate(self, messages, stop=None, run_manager=None, **_):
        self._seen.append(len(messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )


@pytest.fixture
def parts(tmp_path):
    """The shared saver and store, and a factory for shells that use them."""
    engine = threaded_sqlite_engine(tmp_path / "hist.db")
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    saver, store = MemorySaver(), InMemoryStore()

    @contextmanager
    def get_session():
        """Yield a session on the shared engine, closed on exit."""
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    def make(model=None, portfolio_id=PORTFOLIO_ID) -> ShellApp:
        """Build a shell over the shared database, saver and store."""
        return ShellApp(
            portfolio_id,
            session_factory=get_session,
            persistence=SimpleNamespace(saver=saver, store=store, pool=None),
            model_factory=lambda: model,
            poll_interval=30.0,
        )

    yield SimpleNamespace(make=make, saver=saver, store=store)
    engine.dispose()


async def _open_ask(app: ShellApp, pilot) -> AskView:
    """Switch to the Ask view and let the shell paint and load its history."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-ask"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-ask", AskView)


def _shown(view: AskView) -> str:
    return str(view.query_one("#ask-log", Static).render())


@pytest.mark.asyncio
async def test_opening_ask_shows_the_saved_conversation(parts) -> None:
    """What was said last time is on screen the moment Ask opens."""
    history = AskHistory(parts.store, scope_of(PORTFOLIO_ID))
    history.append("you", "how are we doing?")
    history.append("ask", "all good")
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)

        shown = _shown(view)

    assert "you: how are we doing?" in shown
    assert "ask: all good" in shown


@pytest.mark.asyncio
async def test_a_new_question_and_its_answer_are_saved(parts) -> None:
    """Each message is written to the history as it appears on screen."""
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.append_log("you", "what is the cap?")
        view.append_log("ask", "ten percent")

    saved = AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load()
    assert saved == [("you", "what is the cap?"), ("ask", "ten percent")]


@pytest.mark.asyncio
async def test_the_conversation_is_still_there_after_a_restart(parts) -> None:
    """A second cockpit over the same database shows the first one's conversation."""
    first = parts.make()
    async with first.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(first, pilot)
        view.append_log("you", "remember this")
    second = parts.make()

    async with second.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(second, pilot)

        assert "you: remember this" in _shown(view)


@pytest.mark.asyncio
async def test_switching_portfolio_switches_the_conversation(parts) -> None:
    """Each portfolio, and the fund-wide view, shows only its own history."""
    AskHistory(parts.store, scope_of(PORTFOLIO_ID)).append("you", "about the first")
    AskHistory(parts.store, scope_of(OTHER)).append("you", "about the other")
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        assert "about the first" in _shown(view)

        app._portfolio_id = OTHER
        app._refresh()
        await pilot.pause()

        assert "about the other" in _shown(view)
        assert "about the first" not in _shown(view)


@pytest.mark.asyncio
async def test_typing_clear_asks_before_deleting_anything(parts) -> None:
    """``/clear`` opens a confirmation, deletes nothing yet, and is not itself saved."""
    history = AskHistory(parts.store, scope_of(PORTFOLIO_ID))
    history.append("you", "keep me")
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)

        view.submit_turn("/clear")
        await pilot.pause()

        assert isinstance(app.screen, ConfirmScreen)
        assert history.load() == [("you", "keep me")]
        assert "/clear" not in _shown(view)


@pytest.mark.asyncio
async def test_cancelling_the_confirmation_keeps_the_conversation(parts) -> None:
    """Cancel leaves every message where it was."""
    history = AskHistory(parts.store, scope_of(PORTFOLIO_ID))
    history.append("you", "keep me")
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("/clear")
        await pilot.pause()

        await pilot.click("#confirm-no")
        await pilot.pause()

        assert not isinstance(app.screen, ConfirmScreen)
        assert "keep me" in _shown(view)
    assert history.load() == [("you", "keep me")]


@pytest.mark.asyncio
async def test_confirming_deletes_the_conversation_for_good(parts) -> None:
    """Confirm erases the saved messages, the agent's memory, and the screen."""
    history = AskHistory(parts.store, scope_of(PORTFOLIO_ID))
    history.append("you", "erase me")
    parts.saver.put(
        {
            "configurable": {
                "thread_id": ask_thread_id(scope_of(PORTFOLIO_ID)),
                "checkpoint_ns": "",
            }
        },
        {
            "v": 1,
            "id": "1",
            "ts": "2026-01-01T00:00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
        },
        {},
        {},
    )
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("/clear")
        await pilot.pause()

        await pilot.click("#confirm-yes")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()

        assert "erase me" not in _shown(view)
    assert history.load() == []
    thread = {"configurable": {"thread_id": ask_thread_id(scope_of(PORTFOLIO_ID))}}
    assert parts.saver.get_tuple(thread) is None


@pytest.mark.asyncio
async def test_clearing_one_portfolio_leaves_the_others(parts) -> None:
    """Only the portfolio on screen is wiped."""
    AskHistory(parts.store, scope_of(PORTFOLIO_ID)).append("you", "mine")
    other = AskHistory(parts.store, scope_of(OTHER))
    other.append("you", "theirs")
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("/clear")
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()

    assert other.load() == [("you", "theirs")]


@pytest.mark.asyncio
async def test_a_question_is_answered_by_the_agent_and_saved(parts) -> None:
    """The answer comes from the deep agent, appears, and is written to history."""
    model = CountingModel(answer="the fund looks healthy")
    app = parts.make(model)

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)

        view.submit_turn("how are we doing?")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()

        assert "the fund looks healthy" in _shown(view)
    saved = AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load()
    assert saved == [
        ("you", "how are we doing?"),
        ("ask", "the fund looks healthy"),
    ]


@pytest.mark.asyncio
async def test_a_follow_up_is_answered_with_the_conversation_in_mind(parts) -> None:
    """The second question reaches the model together with the first exchange."""
    model = CountingModel()
    app = parts.make(model)

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        for question in ("how are we doing?", "and the second one?"):
            view.submit_turn(question)
            await pilot.pause()
            worker = app.last_worker
            assert worker is not None
            await worker.wait()
            await pilot.pause()

    assert model.seen[1] > model.seen[0]


@pytest.mark.asyncio
async def test_clearing_is_refused_while_an_answer_is_being_written(parts) -> None:
    """A delete cannot race an agent that is still writing its memory."""
    import threading

    release = threading.Event()

    class SlowModel(CountingModel):
        """Holds the answer back until the test lets it go."""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            release.wait(timeout=10)
            return super()._generate(messages, stop, run_manager, **kwargs)

    app = parts.make(SlowModel())
    AskHistory(parts.store, scope_of(PORTFOLIO_ID)).append("you", "keep me")

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("a slow question")
        await pilot.pause()

        view.submit_turn("/clear")
        await pilot.pause()

        assert not isinstance(app.screen, ConfirmScreen)
        release.set()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()


@pytest.mark.asyncio
async def test_opening_a_long_conversation_shows_its_newest_message(parts) -> None:
    """Ask opens scrolled to the end, so the latest answer is on screen, not the first."""
    history = AskHistory(parts.store, scope_of(PORTFOLIO_ID))
    for index in range(60):
        history.append("you", f"question {index}")
        history.append("ask", f"answer {index}: " + "word " * 40)
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        await pilot.pause()
        await pilot.pause()

        wrap = view.query_one("#ask-log-wrap")

        assert wrap.max_scroll_y > 0
        assert wrap.scroll_y == wrap.max_scroll_y
