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
    # The trim is a real delete: the store itself holds no more than the cap.
    assert len(_rows(store, PID_A)) == MAX_MESSAGES


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


def _rows(store, pid):
    """Every row the store holds for one portfolio's history, read directly."""
    return store.search(("ask", "history", scope_of(pid)), limit=10_000)


def test_clear_removes_every_row_even_past_one_store_page(store) -> None:
    """Clearing 130 messages leaves nothing behind in the store."""
    history = AskHistory(store, scope_of(PID_A))
    for index in range(130):
        history.append("you", f"q{index}")

    removed = history.clear()

    assert removed == 130
    assert _rows(store, PID_A) == []


def test_a_store_that_returns_newest_first_still_reads_oldest_first(store) -> None:
    """The Postgres store lists newest first; the order comes from the keys."""

    class NewestFirstStore(InMemoryStore):
        """A store whose search results come back in reverse."""

        def search(self, namespace_prefix, /, **kwargs):
            """Return the rows newest first, as the Postgres store does."""
            return list(reversed(super().search(namespace_prefix, **kwargs)))

    reversed_store = NewestFirstStore()
    history = AskHistory(reversed_store, scope_of(PID_A))
    for text in ("first", "second", "third"):
        history.append("you", text)

    assert [text for _, text in history.load()] == ["first", "second", "third"]


def test_a_row_listed_twice_across_pages_is_read_once() -> None:
    """Rows shifted by a concurrent writer between two pages do not appear twice."""

    class DoubledStore(InMemoryStore):
        """A store that lists every row twice, as overlapping pages can."""

        def search(self, namespace_prefix, /, **kwargs):
            """Return each row two times."""
            rows = super().search(namespace_prefix, **kwargs)
            return rows + rows

    doubled = DoubledStore()
    history = AskHistory(doubled, scope_of(PID_A))
    history.append("you", "only")

    assert history.load() == [("you", "only")]


def test_terminal_control_characters_are_never_saved(store) -> None:
    """Whatever a model echoed, an escape sequence is not stored to replay later."""
    history = AskHistory(store, scope_of(PID_A))
    history.append("ask", "ok \x1b]0;pwned\x07 \x1b[2J done")

    [(_, text)] = history.load()

    assert "\x1b" not in text
    assert text.startswith("ok ")


def test_a_runaway_message_is_cut_before_it_is_saved(store) -> None:
    """One enormous answer cannot slow every later open of Ask."""
    history = AskHistory(store, scope_of(PID_A))
    history.append("ask", "x" * 200_000)

    [(_, text)] = history.load()

    assert len(text) < 25_000
    assert "more characters" in text


def test_messages_from_two_writers_keep_the_order_they_were_written(store) -> None:
    """Two cockpits on one database interleave without scrambling the conversation."""
    first = AskHistory(store, scope_of(PID_A))
    second = AskHistory(store, scope_of(PID_A))

    first.append("you", "one")
    second.append("ask", "two")
    first.append("you", "three")
    second.append("ask", "four")

    assert [text for _, text in first.load()] == ["one", "two", "three", "four"]


def test_a_clock_that_steps_back_does_not_reorder_messages(store, monkeypatch) -> None:
    """Keys never go backwards within a process, whatever the wall clock does."""
    ticks = iter([5_000, 4_000, 3_000, 9_000])
    monkeypatch.setattr("fund.tui.ask_history.time.time_ns", lambda: next(ticks))
    history = AskHistory(store, scope_of(PID_A))

    for text in ("a", "b", "c", "d"):
        history.append("you", text)

    assert [text for _, text in history.load()] == ["a", "b", "c", "d"]


def test_clearing_deletes_in_one_batch_not_row_by_row() -> None:
    """A long history is removed in a single round trip to the store."""

    class CountingStore(InMemoryStore):
        """Counts single deletes and batch calls."""

        deletes = 0
        batches = 0

        def delete(self, namespace, key):
            """Count a single-row delete."""
            type(self).deletes += 1
            super().delete(namespace, key)

        def batch(self, ops):
            """Count a batch call."""
            type(self).batches += 1
            return super().batch(ops)

    counting = CountingStore()
    history = AskHistory(counting, scope_of(PID_A))
    for index in range(50):
        history.append("you", f"q{index}")
    CountingStore.deletes = CountingStore.batches = 0

    history.clear()

    assert CountingStore.deletes == 0
    assert 1 <= CountingStore.batches <= 2
    assert _rows(counting, PID_A) == []


def test_the_history_cap_is_five_hundred_messages() -> None:
    """The limit is part of the contract with the deletion and rendering code."""
    assert MAX_MESSAGES == 500
