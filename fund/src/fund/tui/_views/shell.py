"""The placeholder base view and the seam every data-view reads the App through.

Each data-view subclasses :class:`ShellView` and reaches the hosting
:class:`~fund.tui.app.ShellApp` only through the narrowed :class:`_ShellContext`
protocol, so no view module imports ``fund.tui.app``.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast
from weakref import WeakKeyDictionary

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Label, Static

from fund.tui.ask_history import AskHistory

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session

    from fund.config import FundConfig

# Kept under the logger name the views had before they moved here, so a filter
# or handler set for ``fund.tui.views`` still sees their records.
logger = logging.getLogger("fund.tui.views")


class _ShellContext(Protocol):
    """The slice of :class:`~fund.tui.app.ShellApp` the data-views read each refresh.

    A *structural* view of the hosting App (not an import of it), so
    ``fund.tui.views`` need not import ``fund.tui.app`` — which would be a circular
    import and would risk dragging heavier collaborators into this agent-stack-light
    view module.

    Attributes:
        _session_factory: Opens a context-managed sync session for a read tick.
        _persistence: LangGraph bundle (``.saver`` / ``.store``) the reads cross-check.
        _portfolio_id: The focused portfolio, or ``None`` for the fund-wide landing.
        _streaming: Whether a live-streamed round is still pushing steps to the shell.
        _config: The runtime settings, read for the baseline HITL gate set.
    """

    _session_factory: Callable[[], AbstractContextManager[Session]]
    _persistence: Any
    _portfolio_id: uuid.UUID | None
    _streaming: bool
    _config: FundConfig

    def focus_portfolio(self, portfolio_id: uuid.UUID) -> None:
        """Focus the shell on ``portfolio_id`` (set the header context from a view)."""
        ...

    def ask_submit(self, turn: str) -> None:
        """Handle a submitted Ask turn: resolve a pending action, else route it."""
        ...

    def ask_clear(self) -> None:
        """Ask for confirmation, then delete the focused scope's Ask conversation."""
        ...

    def ask_persist(self, history: AskHistory, speaker: str, text: str) -> None:
        """Save one Ask message to ``history`` off the event loop, in order."""
        ...

    def ask_turn_refused(self) -> bool:
        """Whether a typed turn must wait (the history is being deleted); says why."""
        ...


# The dataset each polled table last painted, so a repaint can tell a poll of the same
# data (hold the cursor) from a swap to different data (start at the top). Weak keys let
# a table's entry die with its view.
_PAINTED_SCOPE: WeakKeyDictionary[DataTable, Hashable] = WeakKeyDictionary()


@contextmanager
def _keep_cursor(table: DataTable, scope: Hashable = None) -> Generator[None]:
    """Hold ``table``'s cursor row and scroll offset across a clear-and-refill repaint.

    ``DataTable.clear()`` resets the cursor to row 0 and scrolls to the top, so a table
    repainted on every poll tick would yank the adviser's highlight and viewport back to
    the top and a following Enter would act on the wrong row. The cursor is re-seated on
    the same row key when that row survives the refill, else on the same index clamped
    to the new row count (rows added without a key get a fresh random key each paint and
    so take that fallback). The old viewport is re-applied only while the cursor stays
    on the same row index; a cursor that moved is scrolled into view instead.

    Only a repaint of the *same* dataset is held. ``scope`` identifies the dataset the
    table shows (a filter set, the focused portfolio, the picked run); when it differs
    from the last paint the new rows are a different list, so they start at the top
    rather than inheriting a row index and viewport that belonged to the old one
    (positional keys and unkeyed rows would land the cursor on an arbitrary row).
    """
    held = _PAINTED_SCOPE.get(table, scope) == scope
    _PAINTED_SCOPE[table] = scope
    index = table.cursor_row
    previous_index = index
    scroll_y = table.scroll_y
    key = (
        table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        if table.row_count
        else None
    )
    yield
    if not held or not table.row_count:
        return
    if key is not None and key in table.rows:
        index = table.get_row_index(key)
    index = min(index, table.row_count - 1)
    table.move_cursor(row=index)
    if index != previous_index:
        # The row moved (a reordered book, a row that left): the table scrolls the
        # cursor into view, and restoring the old viewport would strand it off-screen.
        return
    # Queued after the cursor move's own scroll-into-view so the previous viewport
    # wins; the virtual height is only settled after the next refresh.
    table.call_after_refresh(table.scroll_to, y=scroll_y, animate=False, force=True)


@contextmanager
def _short_session(
    session_factory: Callable[[], AbstractContextManager[Session]],
) -> Generator[Session]:
    """Yield a session from ``session_factory`` and roll it back on exit.

    A free function (not a view method) so an off-loop read can use it from a worker
    thread without touching ``self.app``, which is only resolvable on the event loop.
    """
    with session_factory() as session:
        try:
            yield session
        finally:
            session.rollback()


class ShellView(Vertical):
    """A dumb placeholder view: a title line, a body line, and a refresh counter.

    Attributes:
        view_key: Stable id fragment, shared with the sidebar entry key.
        view_title: Human label shown at the top of the view.
        render_count: Incremented by :meth:`refresh_view` — a headless-test hook so a
            test can prove the active view was repainted without scraping the DOM.
    """

    # CSS selector of the control a sidebar pick lands on. Views are not focusable
    # themselves, so their view-scoped keys only fire once focus is inside them.
    primary_focus: ClassVar[str | None] = None

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.view_key = view_key
        self.view_title = view_title
        self.render_count = 0
        self._read_failures: dict[str, str] = {}

    def focus_primary(self) -> None:
        """Move focus into the view so its view-scoped keys and arrows act on it.

        A focusable view takes focus itself; otherwise the widget named by
        :attr:`primary_focus` does. A no-op for a view that declares neither.
        """
        if self.can_focus:
            self.focus()
        elif self.primary_focus is not None:
            self.query_one(self.primary_focus).focus()

    def compose(self) -> ComposeResult:
        """Yield the view's title label and its placeholder body line."""
        yield Label(self.view_title, classes="view-title")
        yield Static(
            f"[{self.view_key}] — placeholder (Phase 0 scaffold)",
            classes="view-body",
            id=f"body-{self.view_key}",
        )

    def refresh_view(self) -> None:
        """Repaint the view from fresh read-model state.

        A Phase-0 no-op beyond bumping :attr:`render_count`; later phases override
        this to pull the view's data through ``fund.observe`` on each poll tick.
        """
        self.render_count += 1

    # --- context seam -------------------------------------------------------

    @property
    def shell(self) -> _ShellContext:
        """The hosting ``ShellApp``, narrowed to the collaborators a view reads.

        A typed accessor over ``self.app`` so a data-view's :meth:`refresh_view` can
        read ``self.shell._portfolio_id`` / ``._persistence`` without importing the
        App (see :class:`_ShellContext`).
        """
        return cast("_ShellContext", self.app)

    def _read_failed(self, source: str, exc: BaseException) -> str:
        """Describe a failed read in one line, logging it once per distinct reason.

        The poll loop retries every tick, so a persistent fault (an FX gap, a dead
        checkpoint) would otherwise write the same traceback every two seconds.

        Args:
            source: Which read failed; keys the once-per-reason bookkeeping.
            exc: The exception the read raised.

        Returns:
            A single-line ``Type: message`` the view can show next to its data.
        """
        error = _error_text(exc)
        if self._read_failures.get(source) != error:
            logger.warning(
                "%s %s read failed: %s", self.view_key, source, error, exc_info=exc
            )
        self._read_failures[source] = error
        return error

    def _read_recovered(self, source: str) -> None:
        """Forget a read failure so the next one logs afresh."""
        self._read_failures.pop(source, None)

    @contextmanager
    def read_session(self) -> Generator[Session]:
        """Yield a short read-only session from the shell, rolled back on exit.

        The read discipline every data-view shares: open a session from the shell's
        injected factory, read the model, and roll back — the UI never holds a
        transaction across poll ticks. The caller does not commit.

        Yields:
            An injected sync session, valid only inside the ``with`` block.
        """
        with _short_session(self.shell._session_factory) as session:
            yield session

    def poll_task(self) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
        """Split a poll tick into an off-loop read and an on-loop paint, if worthwhile.

        A view whose read is heavy enough to stall key presses and redraws when run on
        the event loop (full price-history loads, a regression fit) overrides this to
        return a ``(read, apply)`` pair. ``read`` runs on a worker thread, so it must
        not touch the DOM or ``self.app`` (capture what it needs here, on the loop);
        its result is handed to ``apply`` back on the loop, which drops it when the
        view has moved on meanwhile.

        Returns:
            The ``(read, apply)`` pair, or ``None`` (the default) to keep the cheap
            synchronous :meth:`refresh_view` path.
        """
        return None


def _short(value: Any) -> str:
    """First hex group of a UUID (or its string) — a compact table cell."""
    return str(value).split("-", 1)[0]


_ERROR_TEXT_MAX = 120


def _error_text(exc: BaseException) -> str:
    """One ``Type: message`` line for a subtitle; SQL errors run to many lines."""
    first = next(iter(str(exc).splitlines()), "")
    text = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
    if len(text) <= _ERROR_TEXT_MAX:
        return text
    return f"{text[: _ERROR_TEXT_MAX - 1]}…"
