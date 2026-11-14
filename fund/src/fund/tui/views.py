"""View bodies for the multi-view shell + the data-view context seam.

Each view is a panel the :class:`~fund.tui.app.ShellApp` mounts in its
``ContentSwitcher``. Phase 0 renders labeled placeholders exposing a ``render_count``
test counter (mirroring the cockpit panels' ``entry_count`` / ``event_count`` hooks);
later phases fill them with real content — Deliberation with the live transcript,
Decisions with the HITL queue, and so on.

Data-views reach the App's injected collaborators through the
:class:`ShellView` context seam — the narrowed :attr:`ShellView.shell` accessor and
the :meth:`ShellView.read_session` short-session helper — rather than importing
``fund.tui.app`` (a circular import, and the App module is heavier). At module scope
this imports only Textual, the stdlib, and the agent-stack-free ``fund.observe`` read
model, so ``import fund.tui.app`` stays agent-stack-light (the import-hygiene guard in
``test_tui.py``); the agent stack (``fund.actions`` / audit repos) is imported lazily
inside method bodies.

The view classes and their helpers live in :mod:`fund.tui._views`, one module
per view. This module holds the sidebar specs and :func:`build_views`, and
re-exports every name the view modules define so imports from
``fund.tui.views`` resolve. Patching ``fund.tui.views.<name>`` reaches only
:func:`build_views`; a test that patches a view's helper (``_read_brief``, say)
must patch the ``fund.tui._views`` module that defines it.
"""

from __future__ import annotations

from fund.tui._views.ask import _ASK_RENDER_LIMIT as _ASK_RENDER_LIMIT
from fund.tui._views.ask import CLEAR_COMMAND as CLEAR_COMMAND
from fund.tui._views.ask import AskView
from fund.tui._views.audit import _AUDIT_CSV_HEADER as _AUDIT_CSV_HEADER
from fund.tui._views.audit import _AUDIT_VIEW_LIMIT as _AUDIT_VIEW_LIMIT
from fund.tui._views.audit import _CSV_FORMULA_TRIGGERS as _CSV_FORMULA_TRIGGERS
from fund.tui._views.audit import AuditView
from fund.tui._views.audit import _audit_cell as _audit_cell
from fund.tui._views.audit import _audit_csv_row as _audit_csv_row
from fund.tui._views.audit import _audit_detail as _audit_detail
from fund.tui._views.audit import _audit_label as _audit_label
from fund.tui._views.audit import _csv_safe as _csv_safe
from fund.tui._views.daily_brief import _BRIEF_DOWN as _BRIEF_DOWN
from fund.tui._views.daily_brief import _BRIEF_FLAT as _BRIEF_FLAT
from fund.tui._views.daily_brief import _BRIEF_UP as _BRIEF_UP
from fund.tui._views.daily_brief import _NO_DATA as _NO_DATA
from fund.tui._views.daily_brief import _SHORT_HISTORY_NOTE as _SHORT_HISTORY_NOTE
from fund.tui._views.daily_brief import DailyBriefView
from fund.tui._views.daily_brief import (
    _brief_attribution_lines as _brief_attribution_lines,
)
from fund.tui._views.daily_brief import (
    _brief_attribution_table as _brief_attribution_table,
)
from fund.tui._views.daily_brief import _brief_economist_lines as _brief_economist_lines
from fund.tui._views.daily_brief import _brief_event_lines as _brief_event_lines
from fund.tui._views.daily_brief import (
    _brief_markdown_economist as _brief_markdown_economist,
)
from fund.tui._views.daily_brief import _brief_markdown_events as _brief_markdown_events
from fund.tui._views.daily_brief import _brief_pnl_line as _brief_pnl_line
from fund.tui._views.daily_brief import _BriefRead as _BriefRead
from fund.tui._views.daily_brief import _carried_suffix as _carried_suffix
from fund.tui._views.daily_brief import _read_brief as _read_brief
from fund.tui._views.daily_brief import _render_brief_detail as _render_brief_detail
from fund.tui._views.daily_brief import _render_brief_markdown as _render_brief_markdown
from fund.tui._views.daily_brief import _sorted_contributions as _sorted_contributions
from fund.tui._views.decisions import _RISK_GATE_LABELS as _RISK_GATE_LABELS
from fund.tui._views.decisions import (
    _WEIGHT_DISPLAY_TOLERANCE as _WEIGHT_DISPLAY_TOLERANCE,
)
from fund.tui._views.decisions import DecisionsView
from fund.tui._views.decisions import _format_decision_gate as _format_decision_gate
from fund.tui._views.decisions import _format_risk_gate as _format_risk_gate
from fund.tui._views.decisions import _gate_text as _gate_text
from fund.tui._views.decisions import _live_gate_id as _live_gate_id
from fund.tui._views.decisions import _pm_weights_differ as _pm_weights_differ
from fund.tui._views.deliberation import DeliberationView
from fund.tui._views.deliberation import _detail_payload as _detail_payload
from fund.tui._views.deliberation import _stream_detail as _stream_detail
from fund.tui._views.deliberation import _transcript_detail as _transcript_detail
from fund.tui._views.mandate import _RISK_BANDS as _RISK_BANDS
from fund.tui._views.mandate import _SAVED_FIELDS as _SAVED_FIELDS
from fund.tui._views.mandate import MandateView
from fund.tui._views.mandate import _exact_percent as _exact_percent
from fund.tui._views.mandate import _format_caps as _format_caps
from fund.tui._views.mandate import _tick as _tick
from fund.tui._views.performance import _PERF_DOWN as _PERF_DOWN
from fund.tui._views.performance import _PERF_FLAT as _PERF_FLAT
from fund.tui._views.performance import _PERF_HORIZONS as _PERF_HORIZONS
from fund.tui._views.performance import _PERF_UP as _PERF_UP
from fund.tui._views.performance import PerformanceView
from fund.tui._views.performance import _amt as _amt
from fund.tui._views.performance import _drawdown_pct as _drawdown_pct
from fund.tui._views.performance import _holding_detail as _holding_detail
from fund.tui._views.performance import _pct as _pct
from fund.tui._views.performance import _perf_direction as _perf_direction
from fund.tui._views.performance import _PerformanceRead as _PerformanceRead
from fund.tui._views.performance import _ratio as _ratio
from fund.tui._views.performance import _read_performance as _read_performance
from fund.tui._views.performance import _unsigned_pct as _unsigned_pct
from fund.tui._views.posture import _HEALTH_BREACH as _HEALTH_BREACH
from fund.tui._views.posture import _HEALTH_OK as _HEALTH_OK
from fund.tui._views.posture import _HEALTH_UNREADABLE as _HEALTH_UNREADABLE
from fund.tui._views.posture import _HEALTH_WARN as _HEALTH_WARN
from fund.tui._views.posture import _RUN_AWAITING as _RUN_AWAITING
from fund.tui._views.posture import _RUN_FAILED as _RUN_FAILED
from fund.tui._views.posture import _RUN_RUNNING as _RUN_RUNNING
from fund.tui._views.posture import PostureView
from fund.tui._views.posture import _fund_strip as _fund_strip
from fund.tui._views.posture import _load_posture_detail as _load_posture_detail
from fund.tui._views.posture import _load_roster as _load_roster
from fund.tui._views.posture import _mandate_warning_lines as _mandate_warning_lines
from fund.tui._views.posture import _posture_health as _posture_health
from fund.tui._views.posture import _posture_nav_line as _posture_nav_line
from fund.tui._views.posture import _read_autonomy as _read_autonomy
from fund.tui._views.posture import _render_posture_detail as _render_posture_detail
from fund.tui._views.posture import _roster_row as _roster_row
from fund.tui._views.posture import _roster_row_or_error as _roster_row_or_error
from fund.tui._views.posture import _RosterRow as _RosterRow
from fund.tui._views.posture import _run_glyph as _run_glyph
from fund.tui._views.shell import _ERROR_TEXT_MAX as _ERROR_TEXT_MAX
from fund.tui._views.shell import _PAINTED_SCOPE as _PAINTED_SCOPE
from fund.tui._views.shell import ShellView
from fund.tui._views.shell import _error_text as _error_text
from fund.tui._views.shell import _keep_cursor as _keep_cursor
from fund.tui._views.shell import _ShellContext as _ShellContext
from fund.tui._views.shell import _short as _short
from fund.tui._views.shell import _short_session as _short_session
from fund.tui._views.shell import logger as logger

# The shell views in sidebar order: (key, human title). The key is the id
# fragment for both the sidebar entry (``nav-<key>``) and the view (``view-<key>``).
VIEW_SPECS: tuple[tuple[str, str], ...] = (
    ("posture", "Posture"),
    ("deliberation", "Deliberation"),
    ("decisions", "Decisions"),
    ("mandate", "Mandate"),
    ("audit", "Audit"),
    ("performance", "Performance"),
    ("ask", "Ask"),
    ("brief", "Daily brief"),
)

# Two-letter nav labels for the collapsed rail, keyed by view key. Leading letters
# collide (Deliberation / Decisions / Daily brief), and the Decisions entry is the
# approve/kill gate, so every view needs its own.
RAIL_LABELS: dict[str, str] = {
    "posture": "Po",
    "deliberation": "Dl",
    "decisions": "Dc",
    "mandate": "Ma",
    "audit": "Au",
    "performance": "Pf",
    "ask": "As",
    "brief": "Br",
}


# Per-key view class; keys absent here fall back to the placeholder ``ShellView``.
_VIEW_CLASSES: dict[str, type[ShellView]] = {
    "posture": PostureView,
    "deliberation": DeliberationView,
    "decisions": DecisionsView,
    "mandate": MandateView,
    "audit": AuditView,
    "performance": PerformanceView,
    "ask": AskView,
    "brief": DailyBriefView,
}


def build_views() -> list[ShellView]:
    """Build one view per :data:`VIEW_SPECS`, id'd ``view-<key>``.

    Picks the real view class for keys in :data:`_VIEW_CLASSES` (currently
    ``decisions``); every other key still renders the placeholder ``ShellView``.
    """
    return [
        (_VIEW_CLASSES.get(key) or ShellView)(key, title, id=f"view-{key}")
        for key, title in VIEW_SPECS
    ]


__all__ = [
    "RAIL_LABELS",
    "VIEW_SPECS",
    "AskView",
    "AuditView",
    "DailyBriefView",
    "DecisionsView",
    "DeliberationView",
    "MandateView",
    "PerformanceView",
    "PostureView",
    "ShellView",
    "build_views",
]
