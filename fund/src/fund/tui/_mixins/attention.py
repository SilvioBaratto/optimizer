"""The shell's notification bus: poll-diff toasts and the header counters."""

from __future__ import annotations

import logging
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import work
from textual.css.query import NoMatches
from textual.widgets import Static

from fund import observe
from fund.tui._mixins.host import ShellHost

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# Logs under ``fund.tui.app``, the shell's name that operators and tests filter on.
logger = logging.getLogger("fund.tui.app")


# ``App.notify`` has no "never expire" value: ``timeout=None`` falls back to the
# default (5s). A day outlasts any unattended stretch, and the adviser can still click
# a toast away.
_ATTENTION_TOAST_TIMEOUT_S = 24 * 60 * 60.0

_AttentionKey = tuple[str, uuid.UUID]


@dataclass(frozen=True)
class _AttentionRead:
    """One tick's read of the fund-wide attention feed, with what could not be read.

    Attributes:
        items: ``(key, message)`` for every gate / breach the tick could see.
        unknown_kinds: Item kinds (``"gate"`` / ``"breach"``) whose whole read failed,
            so their prior state is neither confirmed nor cleared.
        unreadable_breaches: Portfolios whose drift check failed on its own while the
            rest of the roster read fine.
    """

    items: list[tuple[_AttentionKey, str]]
    unknown_kinds: frozenset[str] = frozenset()
    unreadable_breaches: frozenset[uuid.UUID] = frozenset()

    @property
    def degraded(self) -> bool:
        """Whether any part of the feed could not be read this tick."""
        return bool(self.unknown_kinds or self.unreadable_breaches)

    def is_unknown(self, key: _AttentionKey) -> bool:
        """Whether ``key``'s state is undetermined, so it must not be cleared."""
        kind, ident = key
        return kind in self.unknown_kinds or (
            kind == "breach" and ident in self.unreadable_breaches
        )


class AttentionMixin(ShellHost):
    """The shell's notification bus: poll-diff toasts and the header counters.

    Mixed into :class:`~fund.tui.app.ShellApp`, whose ``__init__`` sets the state
    these methods share (``_seen_attention``, ``_awaiting_count``,
    ``_attention_degraded``, ``_logged_attention_failures``, ``last_attention``).
    """

    def _poll_attention_off_loop(self) -> None:
        """Start the attention feed's read on a worker; one read in flight at a time.

        The feed marks every mandated portfolio's book, so on the event loop it cost
        more per tick as the roster grew. :meth:`_refresh` keeps the synchronous read
        for the repaint that follows an adviser's action.
        """
        if self._attention_read_inflight:
            return
        self._attention_read_inflight = True
        self._attention_read_worker()

    @work(thread=True, group="attention")
    def _attention_read_worker(self) -> None:
        """Read the fund-wide attention items off the loop, then apply them on it."""
        try:
            read: _AttentionRead | None = self._read_attention()
        except Exception:  # keep the poll loop alive on any read error
            logger.warning("attention feed read failed", exc_info=True)
            read = None
        self.call_from_thread(self._finish_attention_read, read)

    def _finish_attention_read(self, read: _AttentionRead | None) -> None:
        """Free the attention slot and apply the read (``None`` means it died)."""
        self._attention_read_inflight = False
        if read is not None:
            self._apply_attention(read)

    def _poll_attention(self) -> None:
        """Diff fund-wide attention items against the seen-set; toast the new ones.

        Each tick reads the cross-portfolio awaiting gates + drift breaches, raises a
        persistent (stays until dismissed) toast for any **newly-appeared** item, and
        bumps the header ``⏸N awaiting`` counter. A still-pending item stays in the
        seen-set, so it is not re-toasted; a resolved item drops out and would re-alert
        only if it genuinely recurs.

        A read that failed is *unknown*, not *empty*: the items it would have covered
        keep their prior seen-state and the awaiting counter keeps its last value, so a
        DB hiccup neither re-toasts every still-pending gate on the next good tick nor
        flickers the counter to zero. Reads fail independently — one portfolio's
        unreadable mandate must not silence the gates or the other portfolios — and
        the header flags the feed as degraded while any of them is failing.
        """
        self._apply_attention(self._read_attention())

    def _apply_attention(self, read: _AttentionRead) -> None:
        """Toast the newly-appeared items of ``read``; repaint the header counters."""
        for key, message in read.items:
            if key not in self._seen_attention:
                self._raise_attention(message)
        carried = {key for key in self._seen_attention if read.is_unknown(key)}
        self._seen_attention = {key for key, _ in read.items} | carried
        if "gate" not in read.unknown_kinds:
            self._awaiting_count = sum(
                1 for (kind, _), _ in read.items if kind == "gate"
            )
        self._attention_degraded = read.degraded
        self._paint_attention()

    def _read_attention(self) -> _AttentionRead:
        """The current fund-wide attention items: awaiting gates + drift breaches.

        Each item is ``((kind, id), message)`` where ``kind`` is ``"gate"`` (keyed by
        run id) or ``"breach"`` (keyed by portfolio id). The gate read, the roster read
        and each portfolio's drift check are isolated from one another; whatever
        failed is reported on the result instead of being folded into "no items", and
        logged (once per failure episode) so the header's degraded flag can be traced
        to the read behind it.
        """
        items: list[tuple[_AttentionKey, str]] = []
        unknown: set[str] = set()
        unreadable: set[uuid.UUID] = set()
        failing: set[str] = set()
        try:
            with self._session_factory() as session:
                saver = self._persistence.saver
                try:
                    for row in observe.pending_hitl(session, saver, None):
                        short = str(row.run_id)[:8]
                        items.append(
                            (("gate", row.run_id), f"⏸ gate awaiting · {short}")
                        )
                except Exception:  # unknown gates, not "no gates"
                    session.rollback()
                    unknown.add("gate")
                    self._log_attention_failure(failing, "gate", "pending gates")
                try:
                    roster = observe.list_portfolios(session)
                except Exception:  # unknown breaches, not "no breaches"
                    session.rollback()
                    unknown.add("breach")
                    self._log_attention_failure(failing, "roster", "portfolio roster")
                    roster = []
                for pid in roster:
                    try:
                        if self._is_drift_breach(session, pid):
                            items.append(
                                (("breach", pid), f"◐ drift breach · {str(pid)[:8]}")
                            )
                    except Exception:  # only this portfolio (e.g. a bad mandate)
                        session.rollback()
                        unreadable.add(pid)
                        self._log_attention_failure(
                            failing, f"breach:{pid}", f"drift check for portfolio {pid}"
                        )
                session.rollback()
        except Exception:  # the session itself failed: nothing read can be trusted
            self._log_attention_failure(failing, "session", "database session")
            self._logged_attention_failures &= failing
            return _AttentionRead([], frozenset({"gate", "breach"}))
        self._logged_attention_failures &= failing
        return _AttentionRead(items, frozenset(unknown), frozenset(unreadable))

    def _log_attention_failure(self, failing: set[str], key: str, what: str) -> None:
        """Note a failed feed read for this tick; log it if it just started failing.

        Called from inside the ``except`` that caught the error, so ``exc_info`` carries
        the traceback. A read that stops failing drops out of the logged set (see
        :meth:`_read_attention`), so a recurrence is logged again.

        Args:
            failing: This tick's failing-read keys, extended with ``key``.
            key: Stable identity of the read (the gate read, the roster read, or one
                portfolio's drift check).
            what: Human description of the read, naming the portfolio where relevant.
        """
        failing.add(key)
        if key not in self._logged_attention_failures:
            self._logged_attention_failures.add(key)
            logger.warning("attention feed: %s unreadable", what, exc_info=True)

    def _is_drift_breach(self, session: Session, portfolio_id: uuid.UUID) -> bool:
        """Whether ``portfolio_id``'s L1 drift breaches its mandate threshold.

        Unmandated portfolios (no threshold) never breach; the drift read is skipped
        for them so the bus does not compute state for every roster row needlessly.
        """
        mandate = observe.get_mandate(session, portfolio_id)
        if mandate is None:
            return False
        state = observe.portfolio_state(session, portfolio_id)
        return state.drift_l1 >= mandate.drift_l1_threshold

    def _raise_attention(self, message: str) -> None:
        """Raise an attention toast that stays until dismissed + record it."""
        self.last_attention.append(message)
        self.notify(message, timeout=_ATTENTION_TOAST_TIMEOUT_S)

    def _paint_attention(self) -> None:
        """Repaint the header ``⏸N awaiting`` counter and the feed-degraded flag."""
        with suppress(NoMatches):
            self.query_one("#hdr-attention", Static).update(
                f"⏸{self._awaiting_count} awaiting"
            )
            self.query_one("#hdr-feed", Static).display = self._attention_degraded
