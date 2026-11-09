"""The Ask conversation, kept per portfolio in the LangGraph store.

Ask used to forget everything when the cockpit closed. This keeps the exact words of
every turn, one store item per message, so opening Ask shows the conversation again
and a clear deletes it for good. It is the transcript a person reads, so it is never
summarized or trimmed to fit a model: the agent's own working memory (which is
compacted) lives in a separate checkpointer thread, see :mod:`fund.agents.ask_agent`.

Agent-stack-free on purpose, like the rest of ``fund.tui``: the store is any object
with the LangGraph ``put`` / ``search`` / ``delete`` methods, so importing this module
pulls no ``langgraph``.
"""

from __future__ import annotations

import datetime as dt
import itertools
import time
import uuid
from typing import Any

__all__ = ["FUND_WIDE", "MAX_MESSAGES", "AskHistory", "scope_of"]

# The scope key of the fund-wide view, where no portfolio is focused.
FUND_WIDE = "fund"
# A long-lived portfolio would otherwise grow without bound and make every open of Ask
# page through all of it; past this the oldest messages are deleted.
MAX_MESSAGES = 500
_NAMESPACE_ROOT = ("ask", "history")
# One store search returns at most `limit` rows (10 by default), so reads page.
_PAGE = 100
# Breaks ties between two messages written in the same clock tick, so append order is
# always read order within one process.
_SEQUENCE = itertools.count()


def scope_of(portfolio_id: uuid.UUID | None) -> str:
    """The history scope of a focused portfolio, or the fund-wide scope for ``None``."""
    return str(portfolio_id) if portfolio_id is not None else FUND_WIDE


def _new_key() -> str:
    """A store key that sorts in the order messages were written."""
    return f"{time.time_ns():020d}-{next(_SEQUENCE):09d}"


class AskHistory:
    """One scope's saved Ask conversation (a portfolio's, or the fund-wide one).

    Args:
        store: A LangGraph store (``put`` / ``search`` / ``delete``), shared with the
            rest of the app; ``InMemoryStore`` in tests, ``PostgresStore`` live.
        scope: The history to read and write, from :func:`scope_of`.
    """

    def __init__(self, store: Any, scope: str) -> None:
        self._store = store
        self._namespace = (*_NAMESPACE_ROOT, scope)

    def append(self, speaker: str, text: str) -> None:
        """Save one message, in the exact words it was said.

        Args:
            speaker: ``you`` for the adviser, ``ask`` for the answer.
            text: The message, markdown and brackets untouched.
        """
        self._store.put(
            self._namespace,
            _new_key(),
            {
                "speaker": speaker,
                "text": text,
                "at": dt.datetime.now(dt.UTC).isoformat(),
            },
            # A store configured for semantic search would otherwise send every
            # message to an embedding model; a transcript is read in order.
            index=False,
        )

    def load(self) -> list[tuple[str, str]]:
        """Every saved message, oldest first, as ``(speaker, text)`` pairs.

        Past :data:`MAX_MESSAGES` the oldest are deleted here, so the history a long
        conversation reads back stays bounded.
        """
        items = self._items()
        excess = len(items) - MAX_MESSAGES
        if excess > 0:
            for key, _ in items[:excess]:
                self._store.delete(self._namespace, key)
            items = items[excess:]
        return [(value["speaker"], value["text"]) for _, value in items]

    def recent(self, count: int) -> list[tuple[str, str]]:
        """The last ``count`` messages, oldest first; empty for a count of zero."""
        if count <= 0:
            return []
        return self.load()[-count:]

    def clear(self) -> int:
        """Delete every message of this scope for good.

        Returns:
            How many messages were removed.
        """
        items = self._items()
        for key, _ in items:
            self._store.delete(self._namespace, key)
        return len(items)

    def _items(self) -> list[tuple[str, dict[str, Any]]]:
        """All ``(key, value)`` rows of the scope in write order, read page by page.

        A store's own result order is not part of its contract (the Postgres store
        returns most recently updated first), so the keys, which sort by time, decide.
        """
        rows: list[tuple[str, dict[str, Any]]] = []
        offset = 0
        while True:
            page = self._store.search(self._namespace, limit=_PAGE, offset=offset)
            rows.extend((item.key, dict(item.value)) for item in page)
            if len(page) < _PAGE:
                break
            offset += _PAGE
        rows.sort(key=lambda row: row[0])
        return rows
