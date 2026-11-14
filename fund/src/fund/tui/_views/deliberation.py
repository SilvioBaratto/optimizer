"""The Deliberation view: recorded run transcripts and the live stream of a round."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.widgets import DataTable, Label, Static

from fund import observe
from fund.observe import RunSummary
from fund.tui._views.posture import _RUN_AWAITING, _RUN_RUNNING
from fund.tui._views.shell import ShellView, _keep_cursor, _short
from fund.tui.readouts import render_readout
from fund.tui.stream import StepStatus, StreamEvent

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from textual.binding import BindingType


class DeliberationView(ShellView):
    """Replay a run's deliberation: a run picker, a step timeline, and step detail.

    Per-portfolio (header focus): lists the focused portfolio's runs, and for a picked
    run renders ``observe.load_run_transcript`` — the checkpoint's PM/subagent turns
    interleaved with the ``agent_decisions`` — as an ordered step timeline (left);
    selecting a step reveals its detail (right, with the decision payload where one
    exists).

    While a round streams it switches to a **live tail**: :meth:`start_live` swaps the
    replay split for a single timeline that :meth:`ingest_event` fills in emission
    order, flipping each prior step's ●/✓ marker as the next event lands; auto-follow
    glues the view to the newest row until the adviser moves the cursor to inspect
    (``f`` / :meth:`action_follow_live` re-snaps). :meth:`end_live` drops back to the
    replay transcript on the terminal event.

    Attributes:
        selected_run_id: The run whose transcript is shown, or ``None``.
        selected_step: The focused step's position, or ``None``.
        detail_text: Mirror of the rendered step detail — a headless-test hook.
        live_run_id: The streaming run while live, else ``None`` (replay mode).
        live_portfolio_id: The live run's portfolio while streaming, else ``None``.
        following: Whether the live tail auto-follows; a cursor move pauses it.
        live_text: Mirror of the rendered live-step detail — a headless-test hook.
        read_error: Why the last replay read failed, or ``None`` when it succeeded; the
            rows and selection from the last good read stay put meanwhile.
    """

    # View-scoped so ``f`` means follow-live only while Deliberation holds focus; the
    # App's global ``f`` (Audit overrides filter) is untouched (R2). ``x`` routes to the
    # App's recompute action (the worker lives there, mirroring the stream worker).
    BINDINGS: ClassVar[list[BindingType]] = [
        ("f", "follow_live", "Follow live"),
        ("x", "app.expand_readout", "Expand read-out"),
    ]

    primary_focus: ClassVar[str | None] = "#delib-runs"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.selected_run_id: uuid.UUID | None = None
        self.selected_step: int | None = None
        self.detail_text = ""
        self._entries: list[Any] = []
        self.live_run_id: uuid.UUID | None = None
        # The portfolio the live run belongs to, pinned at start_live so a barge-in
        # addresses the streaming run's portfolio even if the header focus moves.
        self.live_portfolio_id: uuid.UUID | None = None
        self.following = True
        self.live_text = ""
        self.read_error: str | None = None
        # The portfolio whose rows the run/step tables hold, so a failed read never
        # leaves another portfolio's runs on screen after a focus change.
        self._read_pid: uuid.UUID | None = None
        self._live_events: list[StreamEvent] = []
        # The worker's recompute for the expanded step: (run_id, agent, key, result),
        # where key is the allocator decision_index (None for risk). Held so it survives
        # poll ticks until the adviser selects a different run/step.
        self._readout: tuple[uuid.UUID, str, int | None, Any] | None = None

    def compose(self) -> ComposeResult:
        """Yield the title, subtitle, the runs/steps/live split, detail, and hint."""
        yield Label(self.view_title, classes="view-title")
        # The subtitle carries read-error text, which can contain ``[...]``.
        yield Label(
            "(no portfolio focused)",
            id="delib-subtitle",
            classes="view-subtitle",
            markup=False,
        )
        with Horizontal(classes="delib-body"):
            with Vertical(classes="delib-left"):
                yield DataTable(
                    id="delib-runs", cursor_type="row", classes="delib-runs"
                )
                yield DataTable(
                    id="delib-steps", cursor_type="row", classes="delib-steps"
                )
                yield DataTable(
                    id="delib-live", cursor_type="row", classes="delib-live"
                )
            # Agent prose routinely carries "[k=v, k=v]" runs, which Textual would parse
            # as markup and raise on; every free-text pane below renders literally.
            with VerticalScroll(classes="delib-detail-wrap"):
                yield Static(
                    "(select a step)",
                    id="delib-detail",
                    classes="delib-detail",
                    markup=False,
                )
        yield Static(
            "↑↓ select · ↵ run then step · f follow live",
            id="delib-hint",
            classes="delib-hint",
        )

    def on_mount(self) -> None:
        """Seed the run/step/live tables' columns; hide the live tail until a stream."""
        self.query_one("#delib-runs", DataTable).add_columns("run", "asof", "status")
        self.query_one("#delib-steps", DataTable).add_columns("idx", "agent", "step")
        live = self.query_one("#delib-live", DataTable)
        live.add_columns("status", "agent", "step")
        live.display = False

    def refresh_view(self) -> None:
        """Repaint the run picker and step detail, or the live subtitle while streaming.

        In live mode the timeline is push-driven by :meth:`ingest_event`, so a poll
        tick only refreshes the follow/inspect subtitle and must not overwrite the
        streamed rows with the replay read. A finished stream's tail that no longer
        has a gate to act on is retired first (:meth:`retire_spent_live_tail`).
        """
        self.render_count += 1
        self.retire_spent_live_tail()
        if self.live_run_id is not None:
            self._paint_live_subtitle()
            return
        pid = self.shell._portfolio_id
        read = self._read()
        if read is None:
            if self._read_pid == pid:
                self._paint_subtitle()
                return
            read = ([], [])
        self._read_pid = pid
        runs, entries = read
        self._paint_subtitle()
        self._paint_runs(runs)
        self._paint_steps(entries)
        self._paint_detail()

    # --- live tail ----------------------------------------------------------

    def start_live(self, run_id: uuid.UUID, portfolio_id: uuid.UUID) -> None:
        """Enter live mode for ``run_id``: swap the replay split for an empty timeline.

        Clears any prior stream, re-arms auto-follow, and hides the run/step picker so
        the streamed steps fill the pane. Idempotent per run — a fresh call restarts
        the timeline.

        Args:
            run_id: The streaming run whose steps fill the live tail.
            portfolio_id: The run's portfolio, pinned so a barge-in targets it rather
                than whatever the header happens to be focused on at keypress time.
        """
        self.live_run_id = run_id
        self.live_portfolio_id = portfolio_id
        self._live_events = []
        self.following = True
        self.live_text = ""
        self.query_one("#delib-runs", DataTable).display = False
        self.query_one("#delib-steps", DataTable).display = False
        live = self.query_one("#delib-live", DataTable)
        live.clear()
        live.display = True
        self._paint_live_subtitle()
        self.query_one("#delib-detail", Static).update("(waiting for the first step…)")

    def ingest_event(self, event: StreamEvent) -> None:
        """Append one streamed step, flipping the prior LIVE marker to DONE.

        A no-op unless a stream is live, and for an event of any run other than the
        live one (a superseded stream still draining its thread must not interleave
        its rows, whose indexes collide, into the current timeline). The prior row's ●
        becomes ✓ before the new row lands, so markers need no second producer pass;
        the viewport snaps to the tail while :attr:`following`, and the detail pane
        mirrors the newest step.
        While the adviser is inspecting an earlier row the cursor and the detail pane
        stay where they are.
        """
        if self.live_run_id is None or event.run_id != self.live_run_id:
            return
        table = self.query_one("#delib-live", DataTable)
        if self._live_events and self._live_events[-1].status is StepStatus.LIVE:
            table.update_cell_at(
                Coordinate(len(self._live_events) - 1, 0), StepStatus.DONE.marker
            )
        self._live_events.append(event)
        table.add_row(
            event.status_text,
            event.agent,
            event.step or event.kind,
            key=str(event.index),
        )
        if self.following:
            # Keeping the cursor on the tail makes ``following`` mean "cursor on the
            # last row", so a highlight message that outlives a later event cannot
            # read as the adviser moving away.
            table.move_cursor(row=table.row_count - 1)
            table.scroll_end(animate=False)
            self._paint_live_detail(event)

    def end_live(self) -> None:
        """Leave live mode and fall back to the replay transcript (terminal event)."""
        self._leave_live()
        self.refresh_view()

    def retire_spent_live_tail(self) -> bool:
        """Leave live mode once a finished stream's tail has nothing left to act on.

        A round that ends at its approval gate keeps its timeline standing, and only
        the stream worker ends live mode. Once the gate is decided (or the header moves
        to another portfolio) the tail would otherwise keep claiming ``LIVE`` over a
        finalised run: the run picker stays hidden, read-out expansion is dead and
        barge-in targets the closed run. The watched run is selected on the way out so
        its final steps stay on screen in the replay transcript.

        Returns:
            ``True`` when live mode was left, ``False`` when the tail still applies.
        """
        watched = self.live_run_id
        if watched is None or not self._live_tail_is_spent():
            return False
        if self.shell._portfolio_id == self.live_portfolio_id:
            self.selected_run_id = watched
            self.selected_step = None
        self._leave_live()
        return True

    def _live_tail_is_spent(self) -> bool:
        """Whether the live tail no longer matches a gate the adviser can act on.

        Never spent while the stream is running. Afterwards the tail is spent when the
        header moved to another portfolio, or when the watched run is readable and has
        left both the live and the awaiting states (approved, rejected, failed...). A
        run that is not readable keeps the tail rather than dropping it on a transient
        read error, and a pending/running row (visible to a reader that shares the
        stream's transaction) is still in flight, not decided.
        """
        shell = self.shell
        if shell._streaming:
            return False
        pid = self.live_portfolio_id
        if pid is None or shell._portfolio_id != pid:
            return True
        try:
            with self.read_session() as session:
                runs = observe.list_portfolio_runs(session, pid)
        except Exception:  # a failed read must not drop a tail that may still apply
            return False
        watched = next((run for run in runs if run.run_id == self.live_run_id), None)
        return watched is not None and watched.status not in (
            _RUN_AWAITING | _RUN_RUNNING
        )

    def _leave_live(self) -> None:
        self.live_run_id = None
        self.live_portfolio_id = None
        self.query_one("#delib-live", DataTable).display = False
        self.query_one("#delib-runs", DataTable).display = True
        self.query_one("#delib-steps", DataTable).display = True

    def action_follow_live(self) -> None:
        """Re-arm auto-follow and re-snap to the live tail (the view-scoped ``f``).

        Inert in replay: the binding stays active whenever Deliberation has focus, and
        repainting the live subtitle then would claim a round is streaming.
        """
        if self.live_run_id is None:
            return
        self.following = True
        table = self.query_one("#delib-live", DataTable)
        if table.row_count:
            table.move_cursor(row=table.row_count - 1)
            table.scroll_end(animate=False)
        self._paint_live_subtitle()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track the live cursor: off the tail pauses follow, and the row is shown.

        Reads the table's current cursor rather than the message's row, which can be
        stale by the time the loop handles it.
        """
        if event.data_table.id != "delib-live":
            return
        table = event.data_table
        row = table.cursor_row
        self.following = row == table.row_count - 1
        self._paint_live_subtitle()
        if 0 <= row < len(self._live_events):
            self._paint_live_detail(self._live_events[row])

    def _paint_live_subtitle(self) -> None:
        follow = "following" if self.following else "inspecting (f to follow)"
        self.query_one("#delib-subtitle", Label).update(f"▶ LIVE · {follow}")

    def _paint_live_detail(self, event: StreamEvent) -> None:
        text = _stream_detail(event)
        self.live_text = text
        self.query_one("#delib-detail", Static).update(text)

    def held_step_label(self) -> str:
        """Header line of the newest streamed step, or ``"the gate"`` before any.

        The stream is held at its tail whatever row the adviser is inspecting, so the
        barge-in overlay reads this rather than :attr:`live_text`, which mirrors the
        inspected row.
        """
        if not self._live_events:
            return "the gate"
        return _stream_detail(self._live_events[-1]).splitlines()[0]

    def select_run(self, run_id: uuid.UUID) -> None:
        """Focus a run's transcript; a new run clears the step, then repaint."""
        self.selected_run_id = run_id
        self.selected_step = None
        self.refresh_view()

    def select_step(self, index: int) -> None:
        """Focus a transcript step by position, then repaint the detail pane."""
        self.selected_step = index
        self.refresh_view()

    def expand_target(self) -> tuple[uuid.UUID, str, int | None] | None:
        """The ``(run_id, agent, key)`` the expand key should recompute, or ``None``.

        Replay-only — recompute never runs on a ``LIVE`` step, so the walk-forward
        backtest is gated off the live stream. ``key`` is the selected allocator
        decision's ``decision_index`` so the frontier/backtest reflect *that* proposal
        (a run may re-optimize), and ``None`` for a risk step, which the worker checks
        against the run's authoritative allocation.

        Returns:
            The focused run, the selected step's agent, and the per-step key when a
            recompute applies, else ``None`` (live mode, nothing selected, a non-
            ``optimize_portfolio`` allocator row, or an unmapped step).
        """
        if self.live_run_id is not None or self.selected_run_id is None:
            return None
        if self.selected_step is None or self.selected_step >= len(self._entries):
            return None
        entry = self._entries[self.selected_step]
        if entry.agent == "allocator":
            if entry.source != "decision" or entry.step != "optimize_portfolio":
                return None
            return self.selected_run_id, "allocator", entry.index
        if entry.agent == "risk":
            return self.selected_run_id, "risk", None
        return None

    def paint_readout(
        self, run_id: uuid.UUID, agent: str, key: int | None, result: Any
    ) -> None:
        """Store a worker's recompute and repaint the detail pane.

        Called via ``call_from_thread`` from the recompute worker. The result is stored
        keyed by ``(run_id, agent, key)`` and only painted while that matches the
        selection (see :meth:`_compute_for`). Skips the repaint in live mode — a replay
        recompute must never overwrite the streamed detail (mirrors
        :meth:`refresh_view`'s live guard); it stays stored for when replay resumes.
        """
        self._readout = (run_id, agent, key, result)
        if self.live_run_id is not None:
            return
        self._paint_detail()

    def _compute_for(self, entry: Any) -> Any:
        """The stored recompute for ``entry``, or ``None`` for another run/step.

        An allocator result matches only the exact proposal it was computed for (by
        ``decision_index``), so an earlier step never shows a later re-optimize's
        frontier; a risk result matches the run's risk step. A ``message`` row never
        matches an allocator result: message and decision indices are separate
        numberings that merely look alike.
        """
        if self._readout is None:
            return None
        run_id, agent, key, result = self._readout
        if run_id != self.selected_run_id or agent != entry.agent:
            return None
        if agent == "allocator" and (entry.source != "decision" or key != entry.index):
            return None
        return result

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Dispatch: a run row loads its transcript, a step row shows its detail."""
        key = event.row_key.value
        if key is None:
            return
        if event.data_table.id == "delib-runs":
            self.select_run(uuid.UUID(key))
        elif event.data_table.id == "delib-steps":
            self.select_step(int(key))

    def _read(self) -> tuple[list[RunSummary], list[Any]] | None:
        """Read the focused portfolio's runs + the selected run's transcript.

        Returns:
            ``([], [])`` when nothing is focused, else ``(runs, entries)``. ``None`` on
            any read error — distinct from an empty portfolio, so the caller keeps the
            previous rows and selection instead of reading a hiccup as "the run is
            gone"; the reason is left in :attr:`read_error`. The poll loop stays alive.
        """
        pid = self.shell._portfolio_id
        if pid is None:
            self.read_error = None
            return [], []
        try:
            with self.read_session() as session:
                runs = observe.list_portfolio_runs(session, pid)
                read = runs, self._read_transcript(session, runs)
        except Exception as exc:  # keep the poll loop alive on any read error
            self.read_error = self._read_failed("replay", exc)
            return None
        self._read_recovered("replay")
        self.read_error = None
        return read

    def _read_transcript(self, session: Session, runs: list[RunSummary]) -> list[Any]:
        """Load the selected run's transcript while it is still in the live list.

        Imports the run repo from its submodule (never the ``fund.audit`` root, which
        pulls the langgraph bootstrap); needs the ORM run row for its ``thread_id`` +
        ``decisions``.
        """
        if self.selected_run_id is None or all(
            run.run_id != self.selected_run_id for run in runs
        ):
            return []
        from fund.audit.repository import AgentRunRepository

        run_row = AgentRunRepository(session).get_run(self.selected_run_id)
        if run_row is None:
            return []
        saver = self.shell._persistence.saver
        return observe.load_run_transcript(session, saver, run_row)

    def _paint_subtitle(self) -> None:
        pid = self.shell._portfolio_id
        subtitle = self.query_one("#delib-subtitle", Label)
        if pid is None:
            subtitle.update("(no portfolio focused — pick one in the header)")
        else:
            note = f" · read failed: {self.read_error}" if self.read_error else ""
            subtitle.update(f"portfolio {_short(pid)}{note}")

    def _paint_runs(self, runs: list[RunSummary]) -> None:
        table = self.query_one("#delib-runs", DataTable)
        with _keep_cursor(table, self.shell._portfolio_id):
            table.clear()
            for run in runs:
                table.add_row(
                    _short(run.run_id),
                    run.asof.isoformat(),
                    run.status,
                    key=str(run.run_id),
                )
        if self.selected_run_id is not None and all(
            run.run_id != self.selected_run_id for run in runs
        ):
            self.selected_run_id = None
            self.selected_step = None

    def _paint_steps(self, entries: list[Any]) -> None:
        self._entries = entries
        table = self.query_one("#delib-steps", DataTable)
        with _keep_cursor(table, self.selected_run_id):
            table.clear()
            for position, entry in enumerate(entries):
                table.add_row(
                    str(entry.index),
                    entry.agent,
                    entry.step or entry.source,
                    key=str(position),
                )
        if self.selected_step is not None and self.selected_step >= len(entries):
            self.selected_step = None

    def _paint_detail(self) -> None:
        if self.selected_step is not None and self.selected_step < len(self._entries):
            entry = self._entries[self.selected_step]
            text = _transcript_detail(entry, self._compute_for(entry))
        else:
            text = "(select a step)"
        self.detail_text = text
        self.query_one("#delib-detail", Static).update(text)


def _transcript_detail(entry: Any, compute: Any = None) -> str:
    """Render a transcript step: header, text, then a read-out or the raw payload.

    An allocator/risk step routes its payload through :func:`render_readout` for the
    build/results/risk sub-panels; any other step keeps the raw JSON dump. ``compute``
    is the worker's recompute for this step, or ``None``.
    """
    lines = [f"{entry.agent} · {entry.step or entry.source} · #{entry.index}"]
    if entry.text:
        lines.append(entry.text)
    body = _detail_payload(entry.agent, entry.step, entry.payload, compute)
    if body:
        lines.append(body)
    return "\n".join(lines)


def _stream_detail(event: StreamEvent) -> str:
    """Render a live step: header, summary, then a read-out or the raw payload.

    Mirrors :func:`_transcript_detail` so live and replay render alike.
    """
    lines = [f"{event.agent} · {event.step or event.kind} · #{event.index}"]
    if event.text:
        lines.append(event.text)
    body = _detail_payload(event.agent, event.step, event.payload, expandable=False)
    if body:
        lines.append(body)
    return "\n".join(lines)


def _detail_payload(
    agent: str,
    step: str | None,
    payload: dict[str, Any] | None,
    compute: Any = None,
    *,
    expandable: bool = True,
) -> str:
    """Structured read-out for an allocator/risk step, else the raw JSON dump (or "").

    A risk step carries no payload, so :func:`render_readout` is always consulted; only
    an unmapped step with a payload falls back to the JSON dump. ``compute`` fills the
    recomputed frontier/backtest/verdict/sector lines when present. ``expandable`` is
    ``False`` on the live stream, where the expand key does nothing.
    """
    readout = render_readout(agent, step, payload, compute, expandable=expandable)
    if readout is not None:
        return readout
    if payload:
        return json.dumps(payload, indent=2, sort_keys=True)
    return ""
