"""The Daily brief view: one day's P&L, attribution, events and economist note."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Label, ListView, Static

from fund import observe
from fund.observe import DailyBrief
from fund.tui._views.shell import ShellView, _short_session
from fund.tui.prose import format_market_digest
from fund.tui.readouts import attribution_bar

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.binding import BindingType


# Daily-brief P&L tiers as (dot, label) — the word travels with the glyph so the day's
# direction never rides on colour alone (mirrors _HEALTH_*). The two ● tiers (up/down)
# read apart by their label; ◐ marks a flat day.
_BRIEF_UP: tuple[str, str] = ("●", "up")
_BRIEF_FLAT: tuple[str, str] = ("◐", "flat")
_BRIEF_DOWN: tuple[str, str] = ("●", "down")

_NO_DATA = "  no data"
# Worded rather than coloured so the caveat survives monochrome terminals; a fit this
# close to saturation shows an R2 and betas that look more precise than they are.
_SHORT_HISTORY_NOTE = "short history: thin window, OLS fallback; read with caution"


class DailyBriefView(ShellView):
    """The per-day recap: today's P&L, macro attribution, events, the economist read.

    Reads ``observe.daily_brief(session, saver, pid, asof=self.asof)`` each tick and
    renders its five sections into one text pane — the P&L strip (dot + label
    direction), the macro-attribution meters, the ``market_journal`` narrative, the
    carried-forward economist read (dated ``(from MM-DD)`` when it predates the day),
    and the ``portfolio_journal`` overlays. A missing section degrades to an explicit
    "no data" line, never a crash. ``self.asof`` is the viewed day (``None`` opens on
    the latest NAV mark); day navigation and export arrive in later slices.

    Attributes:
        asof: The viewed day, or ``None`` to open on the latest NAV-marked day.
        brief: The last-assembled :class:`~fund.observe.DailyBrief` for the focused
            portfolio, or ``None`` — a headless-test hook. A failed read leaves it in
            place so the day arrows keep working through a transient outage.
        detail_text: Mirror of the rendered detail pane — a headless-test hook.
    """

    # Focusable so ← / → reach the view's bindings when the brief pane holds focus —
    # a view-scoped scope, never a global arrow hijack (the day state is view-local).
    can_focus = True
    # ``e`` is view-scoped so it exports the brief here and leaves the global Audit
    # export ``e`` untouched (mirrors DecisionsView overriding ``e`` for edit).
    BINDINGS: ClassVar[list[BindingType]] = [
        ("left", "prev_day", "Prev day"),
        ("right", "next_day", "Next day"),
        ("enter", "app.open_deliberation", "Open deliberation"),
        ("e", "app.export_brief", "Export"),
        # Arrow keys scroll the pane, so the sidebar is otherwise reachable only by
        # shift+tab; esc is the way back a user expects from a focus-holding pane.
        ("escape", "focus_sidebar", "Sidebar"),
        # The view keeps focus (for the keys above), so it forwards the scroll keys to
        # the recap pane rather than making the pane a second focus stop.
        Binding("up", "scroll_brief('scroll_up')", "Scroll up", show=False),
        Binding("down", "scroll_brief('scroll_down')", "Scroll down", show=False),
        Binding("pageup", "scroll_brief('page_up')", "Page up", show=False),
        Binding("pagedown", "scroll_brief('page_down')", "Page down", show=False),
        Binding("home", "scroll_brief('scroll_home')", "Top", show=False),
        Binding("end", "scroll_brief('scroll_end')", "Bottom", show=False),
    ]

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.asof: dt.date | None = None
        # The portfolio ``asof`` was chosen for; see _drop_day_pinned_for_another.
        self._asof_pid: uuid.UUID | None = None
        self.brief: DailyBrief | None = None
        self.detail_text = ""

    def compose(self) -> ComposeResult:
        """Yield the title, the five-section detail pane, and the key hint."""
        yield Label(self.view_title, classes="view-title")
        # The recap outgrows a normal terminal (journal rows + a paragraph-long
        # economist read), so it scrolls inside the pane and the key hint stays put.
        with VerticalScroll(id="brief-scroll", classes="brief-scroll", can_focus=False):
            # The economist read and journal narratives are LLM prose; see delib-detail.
            yield Static(
                "(loading…)", id="brief-detail", classes="brief-detail", markup=False
            )
        yield Static(
            "← prev day · → next · ↵ open deliberation · e export · esc sidebar",
            id="brief-hint",
            classes="brief-hint",
        )

    def refresh_view(self) -> None:
        """Repaint the recap for the viewed day from a fresh assembler read."""
        self.render_count += 1
        self._drop_day_pinned_for_another()
        self._paint(self._read())

    def poll_task(self) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
        """Assemble the recap off the loop: it fits a regression and reads a checkpoint.

        The painted result is dropped when the portfolio or the viewed day changed
        while the read was in flight, so a slow poll never clobbers a day the adviser
        stepped to meanwhile.
        """
        self._drop_day_pinned_for_another()
        pid = self.shell._portfolio_id
        if pid is None:
            return None
        asof = self.asof
        session_factory = self.shell._session_factory
        persistence = self.shell._persistence

        def read() -> tuple[uuid.UUID, dt.date | None, _BriefRead]:
            return pid, asof, _read_brief(session_factory, persistence, pid, asof)

        def apply(result: tuple[uuid.UUID, dt.date | None, _BriefRead]) -> None:
            read_pid, read_asof, outcome = result
            if (read_pid, read_asof) != (self.shell._portfolio_id, self.asof):
                return
            self.render_count += 1
            self._paint(outcome)

        return read, apply

    def _paint(self, outcome: _BriefRead) -> None:
        if outcome.error is None:
            self.brief = outcome.brief
            text = (
                _render_brief_detail(outcome.brief)
                if outcome.brief is not None
                else "(no portfolio focused)"
            )
        else:
            # A brief that belongs to another portfolio is no use as a fallback.
            if self.brief is not None and (
                self.brief.portfolio_id != self.shell._portfolio_id
            ):
                self.brief = None
            text = f"brief unavailable: {outcome.error}"
            if self.brief is not None:
                stale = _render_brief_detail(self.brief)
                text = f"{text} — showing the last good read\n\n{stale}"
        self.detail_text = text
        self.query_one("#brief-detail", Static).update(text)

    def action_focus_sidebar(self) -> None:
        """Hand focus back to the sidebar (the ``esc`` binding)."""
        self.app.query_one("#sidebar", ListView).focus()

    def action_scroll_brief(self, scroll_action: str) -> None:
        """Scroll the recap pane (``up`` / ``down`` / ``pageup`` / ``pagedown`` keys).

        Args:
            scroll_action: Name of the ``VerticalScroll`` action to run, e.g.
                ``"page_down"`` or ``"scroll_end"``.
        """
        pane = self.query_one("#brief-scroll", VerticalScroll)
        getattr(pane, f"action_{scroll_action}")()

    def _read(self) -> _BriefRead:
        """Assemble the brief in one short session (empty outcome on no focus)."""
        pid = self.shell._portfolio_id
        if pid is None:
            return _BriefRead()
        return _read_brief(
            self.shell._session_factory, self.shell._persistence, pid, self.asof
        )

    def action_prev_day(self) -> None:
        """Step to the previous NAV-marked day; a no-op at the earliest mark."""
        self._step_day(self.brief.prev_day if self.brief else None)

    def action_next_day(self) -> None:
        """Step to the next NAV-marked day; a no-op at the latest mark."""
        self._step_day(self.brief.next_day if self.brief else None)

    def _drop_day_pinned_for_another(self) -> None:
        """Reopen on the latest NAV mark when the focused portfolio changed.

        A day stepped to for one portfolio need not be on another's NAV grid, so
        carrying it over would show the new portfolio's older mark under that date
        (or no data at all) instead of its own latest day.
        """
        pid = self.shell._portfolio_id
        if pid != self._asof_pid:
            self.asof = None
            self._asof_pid = pid

    def _step_day(self, target: dt.date | None) -> None:
        """Re-read the recap for ``target``; a None target (a grid end) is a no-op."""
        if target is None:
            return
        self.asof = target
        self.refresh_view()

    def export_markdown(self, path: str) -> str:
        """Write the current brief to ``path`` as markdown; return ``path``.

        Re-export overwrites (idempotent). The caller guards on a loaded brief; a
        missing brief raises ``ValueError`` rather than writing an empty file. Any
        write failure surfaces as ``OSError`` for the caller to report.
        """
        from fund.tui.exports import open_private

        if self.brief is None:
            raise ValueError("no brief loaded to export")
        with open_private(path) as handle:
            handle.write(_render_brief_markdown(self.brief))
        return path


@dataclass(frozen=True)
class _BriefRead:
    """One brief assembly: the brief, or why it could not be read.

    Attributes:
        brief: The assembled brief; ``None`` when no portfolio is focused or the read
            failed.
        error: One-line description of the failure, or ``None`` on a successful read —
            what tells "nothing focused" apart from "the read raised".
    """

    brief: DailyBrief | None = None
    error: str | None = None


def _read_brief(
    session_factory: Callable[[], AbstractContextManager[Session]],
    persistence: Any,
    pid: uuid.UUID,
    asof: dt.date | None,
) -> _BriefRead:
    """Assemble the brief in one short session; a read error is returned, not raised.

    Safe to run on a worker thread: it takes its collaborators as arguments rather
    than reaching for the App.
    """
    try:
        with _short_session(session_factory) as session:
            brief = observe.daily_brief(session, persistence.saver, pid, asof=asof)
    except Exception as exc:  # keep the poll loop alive on any read error
        # Driver errors embed the failing SQL on later lines; the pane shows one.
        first_line = next(iter(str(exc).splitlines()), "")
        name = type(exc).__name__
        return _BriefRead(error=f"{name}: {first_line}" if first_line else name)
    return _BriefRead(brief=brief)


def _render_brief_detail(brief: DailyBrief) -> str:
    """Render a DailyBrief's five sections as detail-pane text."""
    lines = [
        f"Daily brief · {brief.asof.isoformat()}",
        _brief_pnl_line(brief.performance),
        "",
        "MACRO ATTRIBUTION (today)",
        *_brief_attribution_lines(brief.attribution),
        "",
        "WHAT HAPPENED (market_journal)",
        *_brief_event_lines(brief.market_events),
        "",
        "ECONOMIST READ",
        *_brief_economist_lines(brief),
        "",
        "FUND DID (portfolio_journal)",
        *_brief_event_lines(brief.fund_events),
    ]
    return "\n".join(lines)


def _sorted_contributions(attribution: Any) -> list[tuple[str, float]]:
    """Return contributions plus the ``idio`` residual, descending by magnitude."""
    items = [*attribution.contributions.items(), ("idio", attribution.residual)]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items


def _carried_suffix(brief: DailyBrief) -> str:
    """Return the ``(from MM-DD)`` tag when the read predates the day, else ``""``."""
    read_date = brief.economist_read_date
    if read_date is not None and read_date != brief.asof:
        return f" (from {read_date:%m-%d})"
    return ""


def _brief_pnl_line(perf: Any) -> str:
    """Render the P&L strip with a dot + label direction, or a no-data line."""
    if perf is None:
        return f"TODAY {_NO_DATA.strip()}"
    day_return = perf.returns.get("1D")
    if day_return is None:
        return f"TODAY {_NO_DATA.strip()}"
    if day_return > 0:
        dot, label = _BRIEF_UP
    elif day_return < 0:
        dot, label = _BRIEF_DOWN
    else:
        dot, label = _BRIEF_FLAT
    excess = perf.excess.get("1D")
    vs_bench = f" · vs bench {excess:+.2%}" if excess is not None else ""
    return f"TODAY  {dot} {label}  P&L {day_return:+.2%}{vs_bench}"


def _brief_attribution_lines(attribution: Any) -> list[str]:
    """Render per-factor contributions (+ ``idio`` residual) as sorted bar meters."""
    if attribution is None or not attribution.contributions:
        return [_NO_DATA]
    items = _sorted_contributions(attribution)
    scale = max(abs(value) for _, value in items)
    lines = [
        f"  {name:<6} {value:+.2%}  {attribution_bar(value, scale)}"
        for name, value in items
    ]
    lines.append(f"  {'total':<6} {attribution.total_return:+.2%}")
    if attribution.insufficient_history:
        lines.append(f"  ({_SHORT_HISTORY_NOTE})")
    return lines


def _brief_event_lines(events: tuple[dict[str, Any], ...]) -> list[str]:
    """Render journal rows as dated narrative bullets, or a no-data line."""
    if not events:
        return [_NO_DATA]
    lines: list[str] = []
    for event in events:
        narrative = event.get("narrative", "")
        # The market digest is one machine-written line of ~150 words; laid out as
        # short labelled lines it can be read, and its own header carries the day.
        digest = format_market_digest(narrative)
        if digest is None:
            lines.append(f"  • {narrative}  ({event.get('as_of', '')})")
            continue
        lines.append(f"  • {digest[0]}")
        lines.extend(f"    {row}" for row in digest[1:])
    return lines


def _brief_economist_lines(brief: DailyBrief) -> list[str]:
    """Render the economist read, dated ``(from MM-DD)`` when carried forward."""
    if brief.economist_read is None:
        return [_NO_DATA]
    # The read is multi-paragraph LLM prose: indent every row so it stays one block.
    rows = brief.economist_read.strip().splitlines() or [""]
    rows[0] = f'"{rows[0]}'
    rows[-1] = f'{rows[-1]}"{_carried_suffix(brief)}'
    return [f"  {row}" if row else "" for row in rows]


def _render_brief_markdown(brief: DailyBrief) -> str:
    """Serialize a DailyBrief to a markdown recap, one heading per section."""
    lines = [
        f"# Daily brief · {brief.portfolio_id} · {brief.asof.isoformat()}",
        "",
        "## Today's P&L",
        _brief_pnl_line(brief.performance),
        "",
        "## Macro attribution (today)",
        *_brief_attribution_table(brief.attribution),
        "",
        "## What happened",
        *_brief_markdown_events(brief.market_events),
        "",
        "## Economist read",
        *_brief_markdown_economist(brief),
        "",
        "## Fund did",
        *_brief_markdown_events(brief.fund_events),
        "",
    ]
    return "\n".join(lines)


def _brief_attribution_table(attribution: Any) -> list[str]:
    """Render attribution contributions (+ ``idio`` residual) as a markdown table."""
    if attribution is None or not attribution.contributions:
        return ["_no data_"]
    rows = _sorted_contributions(attribution)
    table = ["| factor | contribution |", "| --- | --- |"]
    table += [f"| {name} | {value:+.2%} |" for name, value in rows]
    table.append(f"| **total** | **{attribution.total_return:+.2%}** |")
    if attribution.insufficient_history:
        table += ["", f"_{_SHORT_HISTORY_NOTE}_"]
    return table


def _brief_markdown_events(events: tuple[dict[str, Any], ...]) -> list[str]:
    """Render journal rows as markdown bullets, or a no-data line."""
    if not events:
        return ["_no data_"]
    return [
        f"- {event.get('narrative', '')} ({event.get('as_of', '')})" for event in events
    ]


def _brief_markdown_economist(brief: DailyBrief) -> list[str]:
    """Render the economist read as a dated markdown blockquote."""
    if brief.economist_read is None:
        return ["_no data_"]
    # Quote every line, blanks included: a bare line would end the blockquote and let
    # a later "#" or "-" in the prose become a real heading or bullet of the export.
    rows = brief.economist_read.strip().splitlines() or [""]
    quote = [f"> {row}" if row else ">" for row in rows]
    carried = _carried_suffix(brief).strip()
    if carried:
        quote.append(f"> {carried}")
    return quote
