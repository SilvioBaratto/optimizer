"""Ask shows the saved conversation when opened, and ``/clear`` deletes it for good."""

from __future__ import annotations

import asyncio
import threading
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar

import pytest
from _fund_fakes import PORTFOLIO_ID, NoToolsChatModel, threaded_sqlite_engine
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


class CountingModel(NoToolsChatModel):
    """A network-free model that answers fixed text and counts the messages it gets."""

    answer: str = "all good"
    _seen: list[int] = PrivateAttr(default_factory=list)
    _texts: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "counting"

    @property
    def seen(self) -> list[int]:
        """How many messages each model call received."""
        return self._seen

    @property
    def texts(self) -> list[str]:
        """The joined text of the messages each model call received."""
        return self._texts

    def _generate(self, messages, stop=None, run_manager=None, **_):
        self._seen.append(len(messages))
        self._texts.append(" | ".join(str(m.content) for m in messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )


class ControlledStore(InMemoryStore):
    """An in-memory store whose deletes and reads can be held or failed, and that
    remembers which thread each call came from."""

    hold_delete: threading.Event | None = None
    delete_started: threading.Event | None = None
    hold_search: threading.Event | None = None
    search_failures: int = 0
    threads: ClassVar[dict[str, set[str]]] = {}

    def _note(self, call: str) -> None:
        """Record the calling thread's name for ``call``."""
        self.threads.setdefault(call, set()).add(threading.current_thread().name)

    def put(self, namespace, key, value, *args, **kwargs):
        """Record the thread, then store."""
        self._note("put")
        super().put(namespace, key, value, *args, **kwargs)

    def delete(self, namespace, key):
        """Wait on ``hold_delete`` (when set) before deleting."""
        self._hold_deletes()
        super().delete(namespace, key)

    def batch(self, ops):
        """Hold a batch that deletes, as a single delete would be held."""
        if any(getattr(op, "value", 1) is None for op in ops):
            self._hold_deletes()
        return super().batch(ops)

    def _hold_deletes(self) -> None:
        """Signal that a delete started and wait for the test to let it go."""
        if self.delete_started is not None:
            self.delete_started.set()
        if self.hold_delete is not None:
            self.hold_delete.wait(timeout=10)

    def search(self, namespace_prefix, /, **kwargs):
        """Hold, record the thread, and raise for the next ``search_failures`` calls."""
        self._note("search")
        if self.hold_search is not None:
            self.hold_search.wait(timeout=10)
        if self.search_failures > 0:
            self.search_failures -= 1
            raise ConnectionError("store down")
        return super().search(namespace_prefix, **kwargs)


class _FlushOnExit:
    """Runs a shell's ``run_test`` and lets its queued history work finish on exit.

    The saved conversation is read and written on a background thread, so a test that
    reads the store after leaving the block must not race it.
    """

    def __init__(self, app, context) -> None:
        self._app = app
        self._context = context

    async def __aenter__(self):
        """Enter the wrapped test context."""
        return await self._context.__aenter__()

    async def __aexit__(self, *exc):
        """Flush the history thread, then leave the wrapped context."""
        await asyncio.to_thread(self._app.flush_ask_history)
        return await self._context.__aexit__(*exc)


class GatedModel(CountingModel):
    """Signals when it is called and holds its answer until released."""

    entered: threading.Event | None = None
    release: threading.Event | None = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(timeout=10)
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.fixture
def parts(tmp_path):
    """The shared saver and store, and a factory for shells that use them."""
    engine = threaded_sqlite_engine(tmp_path / "hist.db")
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    saver, store = MemorySaver(), ControlledStore()

    @contextmanager
    def get_session():
        """Yield a session on the shared engine, closed on exit."""
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    def make(model=None, portfolio_id=PORTFOLIO_ID, factory=None) -> ShellApp:
        """Build a shell over the shared database, saver and store."""
        app = ShellApp(
            portfolio_id,
            session_factory=get_session,
            persistence=SimpleNamespace(saver=saver, store=store, pool=None),
            model_factory=factory or (lambda: model),
            poll_interval=30.0,
        )
        real_run_test = app.run_test
        app.run_test = lambda *a, **k: _FlushOnExit(app, real_run_test(*a, **k))
        return app

    yield SimpleNamespace(make=make, saver=saver, store=store)
    engine.dispose()


async def _open_ask(app: ShellApp, pilot) -> AskView:
    """Switch to the Ask view and let the shell paint and load its history."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-ask"
    app._refresh()
    await pilot.pause()
    await asyncio.to_thread(app.flush_ask_history)
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
    assert "how are we doing?" in model.texts[1]
    assert "the fund" in model.texts[1] or "all good" in model.texts[1]


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
    """Ask opens scrolled to the end, so the newest answer is on screen."""
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


async def _finish(app: ShellApp, pilot) -> None:
    """Wait for the shell's latest worker, then let the screen settle."""
    worker = app.last_worker
    assert worker is not None
    await worker.wait()
    await pilot.pause()


@pytest.mark.asyncio
async def test_an_answer_is_saved_with_the_portfolio_it_was_asked_about(parts) -> None:
    """Switching portfolio while an answer is written does not move the answer."""
    entered, release = threading.Event(), threading.Event()
    app = parts.make(
        GatedModel(answer="ANSWER-FOR-A", entered=entered, release=release)
    )
    AskHistory(parts.store, scope_of(OTHER)).append("you", "the other's own question")

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("question about A")
        await pilot.pause()
        assert await asyncio.to_thread(entered.wait, 5)

        app._portfolio_id = OTHER
        app._refresh()
        await pilot.pause()
        release.set()
        await _finish(app, pilot)

        assert "ANSWER-FOR-A" not in _shown(view)
        assert "the other's own question" in _shown(view)
    assert AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load() == [
        ("you", "question about A"),
        ("ask", "ANSWER-FOR-A"),
    ]
    assert AskHistory(parts.store, scope_of(OTHER)).load() == [
        ("you", "the other's own question")
    ]


@pytest.mark.asyncio
async def test_clearing_is_refused_while_the_model_is_still_being_built(parts) -> None:
    """A question that has not reached the agent yet still blocks a clear."""
    started, gate = threading.Event(), threading.Event()

    def slow_factory():
        """Build the model slowly, as a first lazy provider import does."""
        started.set()
        gate.wait(timeout=10)
        return CountingModel()

    app = parts.make(factory=slow_factory)

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("a question")
        await pilot.pause()
        assert await asyncio.to_thread(started.wait, 5)

        view.submit_turn("/clear")
        await pilot.pause()

        assert not isinstance(app.screen, ConfirmScreen)
        assert "wait for the answer" in app.last_status
        gate.set()
        await _finish(app, pilot)


@pytest.mark.asyncio
async def test_a_question_during_a_clear_is_refused_and_nothing_is_lost(parts) -> None:
    """While the history is being deleted, a new question is not accepted."""
    parts.store.hold_delete = threading.Event()
    parts.store.delete_started = threading.Event()
    AskHistory(parts.store, scope_of(PORTFOLIO_ID)).append("you", "old")
    app = parts.make(CountingModel())

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("/clear")
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        assert await asyncio.to_thread(parts.store.delete_started.wait, 5)

        view.submit_turn("a question during the clear")
        await pilot.pause()

        assert "being deleted" in app.last_status
        parts.store.hold_delete.set()
        await _finish(app, pilot)

        assert "a question during the clear" not in _shown(view)
    assert AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load() == []


@pytest.mark.asyncio
async def test_switching_portfolio_during_a_clear_keeps_the_other_screen(parts) -> None:
    """The clear's screen update applies only to the portfolio that was cleared."""
    parts.store.hold_delete = threading.Event()
    parts.store.delete_started = threading.Event()
    AskHistory(parts.store, scope_of(PORTFOLIO_ID)).append("you", "mine")
    AskHistory(parts.store, scope_of(OTHER)).append("you", "theirs")
    app = parts.make(CountingModel())

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("/clear")
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        assert await asyncio.to_thread(parts.store.delete_started.wait, 5)

        app._portfolio_id = OTHER
        app._refresh()
        await pilot.pause()
        parts.store.hold_delete.set()
        await _finish(app, pilot)

        assert "theirs" in _shown(view)
    assert AskHistory(parts.store, scope_of(OTHER)).load() == [("you", "theirs")]


@pytest.mark.asyncio
async def test_a_clear_that_half_fails_still_empties_the_screen_and_says_so(
    parts, monkeypatch
) -> None:
    """The saved messages are gone, so the screen shows it, and the failure is named."""
    import fund.agents.ask_agent as ask_mod

    def failing_clear(saver, scope, **kwargs):
        raise RuntimeError("checkpointer down")

    monkeypatch.setattr(ask_mod, "clear_ask_state", failing_clear)
    AskHistory(parts.store, scope_of(PORTFOLIO_ID)).append("you", "erase me")
    app = parts.make(CountingModel())

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("/clear")
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        await _finish(app, pilot)

        assert "erase me" not in _shown(view)
        assert "checkpointer down" in app.last_status
    assert AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load() == []


@pytest.mark.asyncio
async def test_a_store_that_fails_at_first_open_is_tried_again_later(parts) -> None:
    """A brief outage at open does not switch saving off for the whole session."""
    parts.store.search_failures = 1
    app = parts.make(CountingModel())

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        assert view.history is None
        assert "unavailable" in app.last_status

        app._ask_history_retry_at = 0.0
        app._refresh()
        await pilot.pause()
        view.append_log("you", "saved after the outage")

        assert view.history is not None
    assert AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load() == [
        ("you", "saved after the outage")
    ]


@pytest.mark.asyncio
async def test_a_second_question_says_it_is_waiting_for_the_first(parts) -> None:
    """A queued question shows why nothing is happening."""
    entered, release = threading.Event(), threading.Event()
    app = parts.make(GatedModel(entered=entered, release=release))

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("first")
        await pilot.pause()
        assert await asyncio.to_thread(entered.wait, 5)

        view.submit_turn("second")
        await pilot.pause()
        await pilot.pause()

        assert "waiting for the previous answer" in app.last_status
        release.set()
        await _finish(app, pilot)


@pytest.mark.asyncio
async def test_an_error_note_is_shown_but_not_saved_as_an_answer(parts) -> None:
    """A failed question leaves its question in the history, not the error text."""

    class Failing(CountingModel):
        """A model that always fails."""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise ValueError("provider said no")

    app = parts.make(Failing())

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("will fail")
        await pilot.pause()
        await _finish(app, pilot)

        assert "ask failed" in _shown(view)
    assert AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load() == [
        ("you", "will fail")
    ]


@pytest.mark.asyncio
async def test_the_status_line_cannot_carry_terminal_control_codes(parts) -> None:
    """Text a model chose for a status message is stripped before it is shown."""
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._set_status("reading \x1b]0;pwned\x07file")

    assert app.last_status == "reading ]0;pwnedfile"


@pytest.mark.asyncio
async def test_the_clear_confirmation_says_exports_are_kept(parts) -> None:
    """The dialog is honest about what a clear does not touch."""
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("/clear")
        await pilot.pause()

        assert isinstance(app.screen, ConfirmScreen)
        assert "exported" in app.screen._prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/CLEAR", "  /clear  ", "/Clear"])
async def test_the_clear_command_ignores_case_and_spaces(parts, typed) -> None:
    """However it is typed, ``/clear`` opens the confirmation."""
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn(typed)
        await pilot.pause()

        assert isinstance(app.screen, ConfirmScreen)


@pytest.mark.asyncio
async def test_a_failing_save_does_not_lose_the_line_on_screen(
    parts, monkeypatch
) -> None:
    """If the store rejects a message, it still shows and the cockpit keeps running."""

    def failing_append(self, speaker, text):
        raise ConnectionError("store down")

    monkeypatch.setattr(AskHistory, "append", failing_append)
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.append_log("you", "still visible")

        assert "still visible" in _shown(view)
        assert app.is_running


@pytest.mark.asyncio
async def test_a_long_conversation_shows_its_tail_and_exports_all_of_it(
    parts, tmp_path
) -> None:
    """Only the newest messages are drawn, with a note; the export has everything."""
    history = AskHistory(parts.store, scope_of(PORTFOLIO_ID))
    for index in range(150):
        history.append("you", f"question {index}")
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)

        shown = _shown(view)
        path = view.export_markdown(str(tmp_path / "all.md"))

    assert "question 149" in shown
    assert "question 0" not in shown
    assert "earlier messages" in shown
    exported = open(path, encoding="utf-8").read()  # noqa: PTH123, SIM115
    assert "question 0" in exported
    assert "question 149" in exported


@pytest.mark.asyncio
async def test_the_fund_wide_view_keeps_its_own_conversation(parts) -> None:
    """With no portfolio focused, Ask uses the fund-wide history."""
    AskHistory(parts.store, scope_of(None)).append("you", "about the whole fund")
    app = parts.make(portfolio_id=None)

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)

        assert "about the whole fund" in _shown(view)


@pytest.mark.asyncio
async def test_the_saved_conversation_is_read_off_the_event_loop(parts) -> None:
    """Opening Ask never queries the store on the screen's own thread."""
    AskHistory(parts.store, scope_of(PORTFOLIO_ID)).append("you", "old")
    parts.store.threads.clear()
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        await _open_ask(app, pilot)

    names = parts.store.threads["search"]
    assert names and all(name.startswith("ask-history") for name in names)


@pytest.mark.asyncio
async def test_saving_a_message_never_blocks_the_event_loop(parts) -> None:
    """Writes happen on the history thread."""
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        parts.store.threads.clear()
        view.append_log("you", "a line")
        await asyncio.to_thread(app.flush_ask_history)

    names = parts.store.threads["put"]
    assert names and all(name.startswith("ask-history") for name in names)


@pytest.mark.asyncio
async def test_lines_typed_while_the_history_loads_are_kept_in_order(parts) -> None:
    """A slow read shows a placeholder, and nothing typed meanwhile is lost."""
    AskHistory(parts.store, scope_of(PORTFOLIO_ID)).append("you", "old question")
    parts.store.hold_search = threading.Event()
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(ContentSwitcher).current = "view-ask"
        app._refresh()
        await pilot.pause()
        view = app.query_one("#view-ask", AskView)
        assert "loading the saved conversation" in _shown(view)

        view.append_log("you", "typed while loading")
        parts.store.hold_search.set()
        await asyncio.to_thread(app.flush_ask_history)
        await pilot.pause()

        shown = _shown(view)
        assert shown.index("old question") < shown.index("typed while loading")
    assert AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load() == [
        ("you", "old question"),
        ("you", "typed while loading"),
    ]


@pytest.mark.asyncio
async def test_a_clear_does_not_leave_messages_queued_before_it(parts) -> None:
    """Messages saved just before ``/clear`` are deleted too, not resurrected."""
    app = parts.make(CountingModel())

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        for index in range(20):
            view.append_log("you", f"line {index}")
        view.submit_turn("/clear")
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        await _finish(app, pilot)

    assert AskHistory(parts.store, scope_of(PORTFOLIO_ID)).load() == []


@pytest.mark.asyncio
async def test_an_export_has_no_control_codes_and_no_remote_images(
    parts, tmp_path
) -> None:
    """What is written to a file is safe to open in a previewer."""
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.append_log("ask", "chart ![c](https://evil.example/?d=1) \x1b]0;t\x07 end")
        path = view.export_markdown(str(tmp_path / "out.md"))

    exported = open(path, encoding="utf-8").read()  # noqa: PTH123, SIM115
    assert "evil.example" not in exported
    assert "\x1b" not in exported
    assert "[image: c]" in exported


@pytest.mark.asyncio
async def test_a_database_error_note_has_no_sql_or_parameters(parts) -> None:
    """The error shown for a failed question is one safe line."""
    from sqlalchemy.exc import DBAPIError

    class Failing(CountingModel):
        """A model whose call fails with a driver error."""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise DBAPIError(
                "SELECT token FROM clients WHERE id = %s",
                ("hunter2",),
                Exception("boom"),
            )

    app = parts.make(Failing())

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("will fail")
        await pilot.pause()
        await _finish(app, pilot)

        shown = _shown(view)
        assert "ask failed" in shown
        assert "hunter2" not in shown
        assert "SELECT" not in shown
        assert "hunter2" not in app.last_status


@pytest.mark.asyncio
async def test_a_repaint_keeps_a_pending_choice_but_a_new_portfolio_drops_it(
    parts,
) -> None:
    """A disambiguation belongs to its own conversation and survives a refresh."""
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.pending_action = object()

        app._refresh()
        await pilot.pause()
        assert view.pending_action is not None

        app._portfolio_id = OTHER
        app._refresh()
        await pilot.pause()
        assert view.pending_action is None


@pytest.mark.asyncio
async def test_two_questions_never_run_in_the_model_at_the_same_time(parts) -> None:
    """The turns of one scope share one agent thread, so they cannot overlap."""
    active = {"now": 0, "most": 0}
    guard = threading.Lock()

    class Overlap(CountingModel):
        """Counts how many calls are inside the model at once."""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            import time

            with guard:
                active["now"] += 1
                active["most"] = max(active["most"], active["now"])
            time.sleep(0.3)
            with guard:
                active["now"] -= 1
            return super()._generate(messages, stop, run_manager, **kwargs)

    app = parts.make(Overlap())

    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("first")
        await pilot.pause()
        view.submit_turn("second")
        await pilot.pause()
        await _finish(app, pilot)
        await asyncio.sleep(0.5)

    assert active["most"] == 1


@pytest.mark.asyncio
async def test_a_plain_repaint_does_not_read_the_store_again(parts) -> None:
    """The saved conversation is read once per scope, not on every tick."""
    app = parts.make()

    async with app.run_test(size=(120, 40)) as pilot:
        await _open_ask(app, pilot)
        reads = parts.store.threads.get("search", set())
        calls = {"n": 0}
        original = parts.store.search

        def counting(namespace_prefix, /, **kwargs):
            """Count reads, then read."""
            calls["n"] += 1
            return original(namespace_prefix, **kwargs)

        parts.store.search = counting
        for _ in range(5):
            app._refresh()
            await pilot.pause()
        await asyncio.to_thread(app.flush_ask_history)

        assert reads
        assert calls["n"] == 0
