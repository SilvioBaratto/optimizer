"""Placeholder view bodies for the multi-view shell (Phase 0 scaffold).

Each view is a dumb, model-free panel the :class:`~fund.tui.app.ShellApp` mounts in
its ``ContentSwitcher``. For Phase 0 they render a labeled placeholder and expose a
``render_count`` test counter (mirroring the cockpit panels' ``entry_count`` /
``event_count`` hooks); later phases fill them with real content — Deliberation with
the live transcript, Decisions with the HITL queue, and so on. They import only
Textual, so ``import fund.tui.app`` stays agent-stack-light (the ``:341`` guard).
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, Static

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


def build_views() -> list[ShellView]:
    """Build one :class:`ShellView` per :data:`VIEW_SPECS`, id'd ``view-<key>``."""
    return [ShellView(key, title, id=f"view-{key}") for key, title in VIEW_SPECS]


__all__ = ["VIEW_SPECS", "ShellView", "build_views"]
