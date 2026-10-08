"""View bodies for the multi-view shell + the data-view context seam.

Each view is a panel the :class:`~fund.tui.app.ShellApp` mounts in its
``ContentSwitcher``. Phase 0 renders labeled placeholders exposing a ``render_count``
test counter (mirroring the cockpit panels' ``entry_count`` / ``event_count`` hooks);
later phases fill them with real content — Deliberation with the live transcript,
Decisions with the HITL queue, and so on.

Data-views reach the App's injected collaborators through the
:class:`ShellView` context seam — the narrowed :attr:`ShellView.shell` accessor and
the :meth:`ShellView.read_session` short-session helper — rather than importing
``fund.tui.app`` (a circular import, and the App module is heavier). The module
imports only Textual + the stdlib, so ``import fund.tui.app`` stays agent-stack-light
(the import-hygiene guard in ``test_tui.py``).
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, cast

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, Static

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session


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
    """

    _session_factory: Callable[[], AbstractContextManager[Session]]
    _persistence: Any
    _portfolio_id: uuid.UUID | None


# The six shell views in sidebar order: (key, human title). The key is the id
# fragment for both the sidebar entry (``nav-<key>``) and the view (``view-<key>``).
VIEW_SPECS: tuple[tuple[str, str], ...] = (
    ("posture", "Posture"),
    ("deliberation", "Deliberation"),
    ("decisions", "Decisions"),
    ("mandate", "Mandate"),
    ("audit", "Audit"),
    ("ask", "Ask"),
)


class ShellView(Vertical):
    """A dumb placeholder view: a title line, a body line, and a refresh counter.

    Attributes:
        view_key: Stable id fragment, shared with the sidebar entry key.
        view_title: Human label shown at the top of the view.
        render_count: Incremented by :meth:`refresh_view` — a headless-test hook so a
            test can prove the active view was repainted without scraping the DOM.
    """

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.view_key = view_key
        self.view_title = view_title
        self.render_count = 0

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

    @contextmanager
    def read_session(self) -> Generator[Session]:
        """Yield a short read-only session from the shell, rolled back on exit.

        The read discipline every data-view shares (mirrors ``FundTUI._refresh``):
        open a session from the shell's injected factory, read the model, and roll
        back — the UI never holds a transaction across poll ticks. The caller does
        not commit.

        Yields:
            An injected sync session, valid only inside the ``with`` block.
        """
        with self.shell._session_factory() as session:
            try:
                yield session
            finally:
                session.rollback()


def build_views() -> list[ShellView]:
    """Build one :class:`ShellView` per :data:`VIEW_SPECS`, id'd ``view-<key>``."""
    return [ShellView(key, title, id=f"view-{key}") for key, title in VIEW_SPECS]


__all__ = ["VIEW_SPECS", "ShellView", "build_views"]
