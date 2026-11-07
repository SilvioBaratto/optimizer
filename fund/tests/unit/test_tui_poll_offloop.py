"""The 2-second poll must not mark every portfolio's book on the event loop.

``observe.portfolio_state`` marks a portfolio's holdings against its price history. The
attention feed and the Posture roster both call it for every portfolio on each tick, so
doing that on the event loop made key presses and redraws stutter as the roster grew.
"""

from __future__ import annotations

import datetime as dt
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from _fund_fakes import PORTFOLIO_ID, make_mandate, threaded_sqlite_engine
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy.orm import sessionmaker
from textual.widgets import ContentSwitcher

from fund import observe
from fund.audit import MandateRepository
from fund.tui.app import ShellApp


@pytest.fixture
def shell(tmp_path):
    """A shell over a database holding one mandated portfolio."""
    engine = threaded_sqlite_engine(tmp_path / "poll.db")
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def get_session():
        """Yield a session on the shared engine, closed on exit."""
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    with get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = ShellApp(
        PORTFOLIO_ID,
        session_factory=get_session,
        persistence=SimpleNamespace(
            saver=MemorySaver(), store=InMemoryStore(), pool=None
        ),
        model_factory=lambda: None,
        poll_interval=30.0,
    )
    yield app
    engine.dispose()


@pytest.fixture
def marking_threads(monkeypatch):
    """Record the thread every ``portfolio_state`` book marking runs on."""
    threads: list[threading.Thread] = []
    real = observe.portfolio_state

    def spy(session, portfolio_id):
        """Note the calling thread, then mark the book as usual."""
        threads.append(threading.current_thread())
        return real(session, portfolio_id)

    monkeypatch.setattr(observe, "portfolio_state", spy)
    return threads


@pytest.mark.asyncio
@pytest.mark.parametrize("view_id", ["view-posture", "view-audit"])
async def test_a_poll_tick_marks_the_books_off_the_event_loop(
    shell, marking_threads, view_id
) -> None:
    """Whatever view is open, the attention feed and the roster read on a worker."""
    async with shell.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        shell.query_one(ContentSwitcher).current = view_id
        await pilot.pause()
        await shell.workers.wait_for_complete()
        marking_threads.clear()

        shell._poll()
        await shell.workers.wait_for_complete()
        await pilot.pause()

    assert marking_threads, "the tick marked no book at all"
    assert threading.main_thread() not in marking_threads


@pytest.mark.asyncio
async def test_the_attention_feed_still_counts_a_gate_found_off_the_loop(
    shell, monkeypatch
) -> None:
    """The off-loop read is applied on the loop: header counter and seen-set update."""
    from fund.audit import AgentRunRepository

    with shell._session_factory() as session:
        run = AgentRunRepository(session).create_run(
            portfolio_id=PORTFOLIO_ID,
            asof=dt.date(2024, 1, 30),
            seed=1,
            universe=[],
            optimizer_config={"step": "fund"},
        )
        run.status = "paused"
        run.thread_id = "t-1"
        session.commit()
        run_id = run.id
    monkeypatch.setattr(observe, "_has_pending_interrupt", lambda *_a: True)

    async with shell.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        shell._poll()
        await shell.workers.wait_for_complete()
        await pilot.pause()

        assert shell._awaiting_count == 1
        assert ("gate", run_id) in shell._seen_attention
