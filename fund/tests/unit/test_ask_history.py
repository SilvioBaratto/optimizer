"""The Ask conversation is kept per portfolio, in order, and can be wiped for good."""

from __future__ import annotations

import uuid

import pytest
from langgraph.store.memory import InMemoryStore

from fund.tui.ask_history import FUND_WIDE, MAX_MESSAGES, AskHistory, scope_of

PID_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
PID_B = uuid.UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def store() -> InMemoryStore:
    """An empty in-memory LangGraph store."""
    return InMemoryStore()


def test_a_portfolio_and_the_whole_fund_have_separate_scopes() -> None:
    """The scope key is the portfolio id, or one fixed key for the fund-wide view."""
    assert scope_of(PID_A) == str(PID_A)
    assert scope_of(None) == FUND_WIDE
    assert scope_of(PID_A) != scope_of(PID_B)


def test_messages_come_back_in_the_order_they_were_said(store) -> None:
    """History reads oldest first, whatever order the store returns rows in."""
    history = AskHistory(store, scope_of(PID_A))
    history.append("you", "how are we doing?")
    history.append("ask", "all good")
    history.append("you", "and the second one?")

    assert history.load() == [
        ("you", "how are we doing?"),
        ("ask", "all good"),
        ("you", "and the second one?"),
    ]


def test_an_empty_history_loads_as_nothing(store) -> None:
    """A portfolio that was never asked anything has no history, not an error."""
    assert AskHistory(store, scope_of(PID_A)).load() == []


def test_each_portfolio_keeps_its_own_history(store) -> None:
    """Switching portfolio shows that portfolio's conversation, not another's."""
    AskHistory(store, scope_of(PID_A)).append("you", "about A")
    AskHistory(store, scope_of(PID_B)).append("you", "about B")
    AskHistory(store, scope_of(None)).append("you", "about the fund")

    assert AskHistory(store, scope_of(PID_A)).load() == [("you", "about A")]
    assert AskHistory(store, scope_of(PID_B)).load() == [("you", "about B")]
    assert AskHistory(store, scope_of(None)).load() == [("you", "about the fund")]


def test_text_is_stored_exactly_as_said(store) -> None:
    """Markdown, brackets and newlines survive the round trip untouched."""
    text = "## Status\n**ok** [AAPL] — see [/policy]\n- one"
    history = AskHistory(store, scope_of(PID_A))
    history.append("ask", text)

    assert history.load() == [("ask", text)]


def test_clear_removes_only_that_portfolios_history(store) -> None:
    """Clearing one portfolio leaves every other conversation alone."""
    mine = AskHistory(store, scope_of(PID_A))
    other = AskHistory(store, scope_of(PID_B))
    mine.append("you", "a")
    mine.append("ask", "b")
    other.append("you", "keep me")

    removed = mine.clear()

    assert removed == 2
    assert mine.load() == []
    assert other.load() == [("you", "keep me")]


def test_clear_on_an_empty_history_is_a_no_op(store) -> None:
    """Clearing nothing removes nothing and does not fail."""
    assert AskHistory(store, scope_of(PID_A)).clear() == 0


def test_history_longer_than_one_store_page_is_read_in_full(store) -> None:
    """A store caps one search at a few rows; every message must still come back."""
    history = AskHistory(store, scope_of(PID_A))
    for index in range(130):
        history.append("you", f"question {index}")

    loaded = history.load()

    assert len(loaded) == 130
    assert loaded[0] == ("you", "question 0")
    assert loaded[-1] == ("you", "question 129")


def test_the_history_is_bounded_and_the_oldest_messages_go_first(store) -> None:
    """Past the cap, loading trims the oldest so a long-lived portfolio stays quick."""
    history = AskHistory(store, scope_of(PID_A))
    for index in range(MAX_MESSAGES + 5):
        history.append("you", f"q{index}")

    loaded = history.load()

    assert len(loaded) == MAX_MESSAGES
    assert loaded[0] == ("you", "q5")
    assert loaded[-1] == ("you", f"q{MAX_MESSAGES + 4}")
    # The trim is a real delete, so the next load does not re-read the old rows.
    assert len(AskHistory(store, scope_of(PID_A)).load()) == MAX_MESSAGES


def test_a_recent_window_can_be_read_for_the_model(store) -> None:
    """The last few messages are available on their own, still oldest first."""
    history = AskHistory(store, scope_of(PID_A))
    for index in range(6):
        history.append("you", f"q{index}")

    assert history.recent(3) == [("you", "q3"), ("you", "q4"), ("you", "q5")]
    assert history.recent(0) == []


def test_two_messages_in_the_same_instant_keep_their_order(store, monkeypatch) -> None:
    """Even with a frozen clock, append order is the read order."""
    monkeypatch.setattr("fund.tui.ask_history.time.time_ns", lambda: 1_000)
    history = AskHistory(store, scope_of(PID_A))
    history.append("you", "first")
    history.append("ask", "second")
    history.append("you", "third")

    assert [text for _, text in history.load()] == ["first", "second", "third"]


def test_messages_are_saved_without_being_embedded() -> None:
    """A transcript is never sent to an embedding model, whatever the store."""
    calls: list[dict] = []

    class RecordingStore:
        """Records the keyword arguments of every put."""

        def put(self, namespace, key, value, **kwargs):
            """Remember how the message was stored."""
            calls.append(kwargs)

    AskHistory(RecordingStore(), scope_of(PID_A)).append("you", "hello")

    assert calls == [{"index": False}]
