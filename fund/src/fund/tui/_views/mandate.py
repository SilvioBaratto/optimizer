"""The Mandate view: the run policy, guardrails and directives an adviser edits."""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox, Input, Label, ListItem, ListView, Select, Static

from fund import observe
from fund.tui._views.shell import ShellView, _short

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from textual.binding import BindingType


# RiskToleranceBand values least→most risk-tolerant (mirrors fund.schemas.enums and
# fund.actions._BAND_RANK); kept local so the module scope stays Textual + observe.
_RISK_BANDS: tuple[str, ...] = (
    "defensive",
    "conservative",
    "balanced",
    "growth",
    "aggressive",
)


def _tick(flag: bool) -> str:
    """A compact on/off glyph for a boolean trigger."""
    return "✓" if flag else "✗"


def _exact_percent(value: float) -> str:
    """A fraction as a percentage with one or two decimals (``0.15`` -> ``15.0%``).

    Two decimals keep a 1.5x stop of a 7.5% ceiling (11.25%) from reading as 11.2%;
    the second is dropped when it is a zero, so round figures stay short.
    """
    text = f"{value:.2%}"
    return f"{text[:-2]}%" if text.endswith("0%") else text


def _format_caps(constraint_set: dict[str, Any]) -> str:
    """One-line summary of the numeric ConstraintSet caps, for display.

    Reads the raw ``constraint_set`` JSON (no schema rehydration) so a missing nested
    key degrades gracefully rather than raising inside the poll loop.
    """
    parts: list[str] = []
    # Labelled because the tiers differ in kind: nu2 is the client's stated loss
    # tolerance (the hard ceiling), nu1 a soft warning below it and nu3 the 1.5x
    # absolute stop; one bare "drawdown" figure would read as the ceiling.
    tiers = [
        f"{label} ≤{_exact_percent(value)}"
        for label, key in (("warn", "nu1"), ("ceiling", "nu2"), ("stop", "nu3"))
        if isinstance(value := constraint_set.get(key), int | float)
    ]
    if tiers:
        parts.append("drawdown " + ", ".join(tiers))
    name_cap = (constraint_set.get("universe_filters") or {}).get("max_position_cap")
    if isinstance(name_cap, int | float):
        parts.append(f"name ≤{_exact_percent(name_cap)}")
    exclusions = (constraint_set.get("esg") or {}).get("exclusions") or []
    parts.append(f"ESG exclude {len(exclusions)}")
    return " · ".join(parts)


# Edit-guard key of each run-policy widget -> the ``PortfolioMandate`` field a save
# writes it to; the band has no entry because a mandate save never persists it.
_SAVED_FIELDS: tuple[tuple[str, str], ...] = (
    ("capital", "capital"),
    ("base_currency", "base_currency"),
    ("drift", "drift_l1_threshold"),
    ("benchmark", "benchmark"),
    ("eligible", "eligible_tickers"),
    ("tradeable", "tradeable_only"),
    ("gate", "hitl_gates"),
    ("directives", "remembered_directives"),
)


class MandateView(ShellView):
    """Per-portfolio policy editor: run-policy (persisted) + guardrails (audit-only).

    RUN POLICY mirrors the editable ``PortfolioMandate`` fields (capital, base currency,
    drift threshold, benchmark, eligible tickers, the Trading 212 tradeable-only scope);
    saving upserts through the unified action path (``mandate_edit``), which also warns
    about an unpriced eligible ticker.
    GUARDRAILS are read from the active MiFID profile — the
    suitability ``band`` is the ceiling, and proposing a *riskier* band logs a
    ``guardrail_override`` flagged ⚠ (audit-only this phase; enforcement is
    ``03-copilot``). The HITL gates mirror what a run actually pauses at: the config
    baseline always gates ``place_orders`` (a mandate can only widen it), so that box
    shows as always on (editable only when the config does not gate it), and
    ``save_profile`` is the profiler's always-on gate, an indicator only. Remembered
    directives render in a removable list (``x`` drops the highlighted one). Both fold
    back into the mandate on save.

    Reads through the context seam (``observe.get_mandate`` +
    ``MifidProfileRepository.get_active`` over a short read session); the writes go
    through ``fund.actions`` on a worker thread dispatched by the App.

    The poll repaints the editor every tick, but an editable widget the adviser has
    changed since its last paint keeps that unsaved value: the save and override
    actions read the live widgets, so a repaint would silently revert them to the
    persisted values. A field is released as soon as it matches the persisted value
    again, once :meth:`mark_saved` reports that a save landed it, or when the focused
    portfolio changes.

    Attributes:
        mandate: The focused portfolio's current run-policy, or ``None``.
        suitability_band: The active assessment's ``RiskToleranceBand`` value (the
            ceiling), or ``None`` when the portfolio has no profile yet.
        currency_locked: ``True`` once the portfolio has NAV history, positions or
            paper orders; the base currency is then read-only (none of them stores a
            currency).
        read_error: Text of the last failed read of the focused portfolio's mandate,
            or ``None`` when the latest read succeeded.
    """

    # View-scoped so ``x`` removes a directive only while Mandate holds focus.
    BINDINGS: ClassVar[list[BindingType]] = [
        ("x", "remove_directive", "Remove directive")
    ]

    # The directives list, not the first input: an Input would swallow the advertised
    # ``x``/``o`` as typed characters, while the list leaves every hinted key live.
    primary_focus: ClassVar[str | None] = "#mandate-directives"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.mandate: Any = None
        self.suitability_band: str | None = None
        self.currency_locked = False
        self.read_error: str | None = None
        self._directives: list[str] = []
        # Value each editable widget was last painted with, to tell an adviser edit
        # from a stale repaint; reset whenever the focused portfolio changes.
        self._painted: dict[str, Any] = {}
        self._painted_pid: uuid.UUID | None = None

    def compose(self) -> ComposeResult:
        """Yield the title, the run-policy | guardrails split, and the key hint."""
        yield Label(self.view_title, classes="view-title")
        yield Label(
            "(no portfolio focused)", id="mandate-subtitle", classes="view-subtitle"
        )
        with Horizontal(classes="mandate-body"):
            with Vertical(classes="mandate-policy"):
                yield Label("RUN POLICY", classes="mandate-section")
                yield Input(placeholder="capital", id="mandate-capital")
                yield Input(placeholder="base currency", id="mandate-base-currency")
                yield Input(placeholder="drift L1 threshold (0-2)", id="mandate-drift")
                yield Input(placeholder="benchmark", id="mandate-benchmark")
                yield Input(
                    placeholder="eligible tickers (comma-separated)",
                    id="mandate-eligible",
                )
                yield Checkbox("Tradeable on Trading 212 only", id="mandate-tradeable")
                yield Static(
                    "", id="mandate-triggers", classes="mandate-readonly", markup=False
                )
            with Vertical(classes="mandate-guardrails"):
                yield Label(
                    "GUARDRAILS (from MiFID profile)", classes="mandate-section"
                )
                yield Static(
                    "(no profile)", id="mandate-suitability", classes="mandate-readonly"
                )
                yield Select(
                    [(band, band) for band in _RISK_BANDS],
                    id="mandate-band",
                    allow_blank=True,
                )
                yield Static("", id="mandate-caps", classes="mandate-readonly")
                yield Label("HITL GATES", classes="mandate-section")
                yield Checkbox("Before execution", id="gate-place-orders")
                yield Checkbox(
                    "Save profile (always on)",
                    value=True,
                    disabled=True,
                    id="gate-save-profile",
                )
                yield Label("REMEMBERED DIRECTIVES", classes="mandate-section")
                yield ListView(id="mandate-directives", classes="mandate-directives")
        yield Static(
            "ctrl+s save · o override guardrail · x remove directive · "
            "eligible tickers: blank = unrestricted",
            id="mandate-hint",
            classes="mandate-hint",
        )

    def refresh_view(self) -> None:
        """Repaint the editor from the focused portfolio's mandate + active profile."""
        self.render_count += 1
        pid = self.shell._portfolio_id
        switched = pid != self._painted_pid
        if switched:
            self._painted.clear()
            self._painted_pid = pid
        subtitle = self.query_one("#mandate-subtitle", Label)
        if pid is None:
            subtitle.update("(no portfolio focused — pick one in the header)")
            self.mandate = None
            self.suitability_band = None
            self.currency_locked = False
            self.read_error = None
            self._paint_policy(None)
            self._paint_guardrails(None, "")
            self._paint_gates(None)
            self._paint_directives(None)
            return
        subtitle.update(f"portfolio {_short(pid)}")
        read = self._read(pid)
        if read is None:
            self._paint_read_failure(switched)
            return
        mandate, band, caps, marked = read
        self.read_error = None
        self.mandate = mandate
        self.suitability_band = band
        self.currency_locked = marked
        self._paint_policy(mandate)
        self._paint_guardrails(band, caps)
        self._paint_gates(mandate)
        self._paint_directives(mandate)

    def proposed_mandate(self) -> Any:
        """Build a ``PortfolioMandate`` from the edited run-policy inputs, re-validated.

        Returns ``None`` when there is no current mandate to edit. Raises
        ``pydantic.ValidationError`` on an out-of-contract edit (e.g. a non-positive
        capital) and ``ValueError`` when the base currency is changed on a portfolio
        that already has ledger history — the App surfaces either as a status line
        instead of saving.
        """
        if self.mandate is None:
            return None
        from fund.schemas.mandate import PortfolioMandate

        base_currency = self.query_one("#mandate-base-currency", Input).value
        if self.currency_locked and base_currency != self.mandate.base_currency:
            raise ValueError(
                "base currency cannot change once the portfolio has NAV history, "
                "positions or paper orders (none of them carries a currency)"
            )
        data = {
            **self.mandate.model_dump(mode="json"),
            "capital": self.query_one("#mandate-capital", Input).value,
            "base_currency": base_currency,
            "drift_l1_threshold": self.query_one("#mandate-drift", Input).value,
            "benchmark": self.query_one("#mandate-benchmark", Input).value or None,
            "eligible_tickers": self._proposed_eligible(),
            "tradeable_only": self.query_one("#mandate-tradeable", Checkbox).value,
            "hitl_gates": self._proposed_gates(),
            "remembered_directives": list(self._directives),
        }
        return PortfolioMandate.model_validate(data)

    def _proposed_eligible(self) -> list[str]:
        """The typed eligible tickers: comma- or space-separated, a blank field empty.

        Normalisation (case, duplicates) and the width limit are the mandate schema's,
        so a bad entry surfaces as the same validation error as any other field.
        """
        raw = self.query_one("#mandate-eligible", Input).value
        return [ticker for ticker in re.split(r"[,\s]+", raw.upper()) if ticker]

    def _proposed_gates(self) -> list[str]:
        """The mandate's gates with ``place_orders`` set from the checkbox.

        Only ``place_orders`` is editable here (``save_profile`` is not a mandate
        gate), and only while the config baseline does not already gate it. Any other
        gate on the mandate (e.g. one added through the CLI's ``--interrupt-on``) is
        carried over untouched, so saving never drops an approval requirement the
        editor cannot show.
        """
        current = list(self.mandate.hitl_gates) if self.mandate is not None else []
        if self._place_orders_always_gated():
            return current
        checked = self.query_one("#gate-place-orders", Checkbox).value
        if not checked:
            return [gate for gate in current if gate != "place_orders"]
        return current if "place_orders" in current else [*current, "place_orders"]

    def _place_orders_always_gated(self) -> bool:
        """Whether the config baseline gates ``place_orders`` regardless of the mandate.

        The agent graph runs under ``config.interrupt_on | mandate.hitl_gates``, so a
        mandate cannot lift a baseline gate.
        """
        return "place_orders" in self.shell._config.interrupt_on

    def _effective_gates(self, mandate: Any) -> list[str]:
        """The gates a run pauses at: config baseline, then the mandate's extras."""
        baseline = list(self.shell._config.interrupt_on)
        return [*baseline, *(g for g in mandate.hitl_gates if g not in baseline)]

    def action_remove_directive(self) -> None:
        """Drop the highlighted remembered directive (persisted on the next save)."""
        listview = self.query_one("#mandate-directives", ListView)
        index = listview.index
        if index is None or index >= len(self._directives):
            return
        del self._directives[index]
        self._refill_directives()

    def proposed_band(self) -> str | None:
        """The target band selected in the guardrail editor, or ``None`` when blank."""
        value = self.query_one("#mandate-band", Select).value
        return None if value is Select.NULL else str(value)

    def _read(self, pid: uuid.UUID) -> tuple[Any, str | None, str, bool] | None:
        """Read the mandate, active guardrails and ledger-history flag in one session.

        Returns ``None`` on any read error (recorded on :attr:`read_error`) so the
        poll loop stays alive and a failure stays distinguishable from a portfolio
        that simply has no mandate.
        """
        try:
            with self.read_session() as session:
                mandate = observe.get_mandate(session, pid)
                band, caps = self._read_guardrails(session, pid)
                return mandate, band, caps, self._has_ledger_history(session, pid)
        except Exception as exc:  # keep the poll loop alive on any read error
            self.read_error = str(exc) or type(exc).__name__
            return None

    def _paint_read_failure(self, switched: bool) -> None:
        """Report a failed read without wiping what the adviser is working on.

        The inputs and the last-known mandate are left alone, so an unsaved edit
        survives a transient error. After a switch to another portfolio they would
        belong to the previous one, so the editor is blanked instead — saving them
        would write that portfolio's policy onto this one.
        """
        if switched:
            self.mandate = None
            self.suitability_band = None
            self.currency_locked = False
            self._paint_policy(None)
            self._paint_guardrails(None, "")
            self._paint_gates(None)
            self._paint_directives(None)
        self.query_one("#mandate-triggers", Static).update(
            f"(read failed: {self.read_error})"
        )

    @staticmethod
    def _has_ledger_history(session: Session, pid: uuid.UUID) -> bool:
        """Whether ``pid`` has NAV points, positions or paper orders (no currency).

        Imports the repo from its submodule so the view stays agent-stack-free.
        """
        from fund.audit.mandate_repository import MandateRepository

        return MandateRepository(session).has_ledger_history(pid)

    def _read_guardrails(
        self, session: Session, pid: uuid.UUID
    ) -> tuple[str | None, str]:
        """The active suitability band + a caps summary, read lazily off the profile.

        Imports the repo from its submodule, keeping ``fund.tui.views`` free of the
        ``fund.audit`` package at import time.
        """
        from fund.audit.mifid_repository import MifidProfileRepository

        profile = MifidProfileRepository(session).get_active(pid)
        if profile is None:
            return None, ""
        band = (profile.suitability or {}).get("band")
        return band, _format_caps(profile.constraint_set or {})

    def _take_persisted(self, key: str, current: Any, persisted: Any) -> bool:
        """Whether ``key``'s widget may be repainted with the ``persisted`` value.

        ``False`` only when the adviser changed the widget since it was last painted
        and it still differs from ``persisted`` — that unsaved edit must survive the
        poll. Otherwise the field is clean (or just saved), so it is released and its
        painted baseline advances to ``persisted``.
        """
        edited = key in self._painted and current != self._painted[key]
        if edited and current != persisted:
            return False
        self._painted[key] = persisted
        return True

    def mark_saved(self, saved: Any) -> None:
        """Release the run-policy widgets that still hold what ``saved`` persisted.

        A save stores the typed value but the next read can render it differently
        (``0.20`` comes back as ``0.2``), so comparing text with the persisted text
        would keep the field "unsaved" forever: an external change could never reach
        it and a later save would overwrite that change. Equality is therefore judged
        on the validated model, once, at save time — not per poll, where treating
        ``0.20`` as equal to ``0.2`` would repaint the field mid-keystroke. A widget
        the adviser edited again after pressing save keeps its protection, and the
        pending band pick is untouched (the band is not part of the mandate).

        Args:
            saved: The ``PortfolioMandate`` whose upsert just committed.
        """
        from pydantic import ValidationError

        try:
            current = self.proposed_mandate()
        except ValidationError:
            return
        if current is None:
            return
        for key, attribute in _SAVED_FIELDS:
            if getattr(current, attribute) == getattr(saved, attribute):
                self._painted.pop(key, None)

    def _paint_policy(self, mandate: Any) -> None:
        capital = self.query_one("#mandate-capital", Input)
        base_currency = self.query_one("#mandate-base-currency", Input)
        drift = self.query_one("#mandate-drift", Input)
        benchmark = self.query_one("#mandate-benchmark", Input)
        eligible = self.query_one("#mandate-eligible", Input)
        triggers = self.query_one("#mandate-triggers", Static)
        persisted = (
            ("capital", capital, ""),
            ("base_currency", base_currency, ""),
            ("drift", drift, ""),
            ("benchmark", benchmark, ""),
            ("eligible", eligible, ""),
        )
        if mandate is not None:
            persisted = (
                ("capital", capital, str(mandate.capital)),
                ("base_currency", base_currency, mandate.base_currency),
                ("drift", drift, str(mandate.drift_l1_threshold)),
                ("benchmark", benchmark, mandate.benchmark or ""),
                ("eligible", eligible, ", ".join(mandate.eligible_tickers)),
            )
        for key, field, text in persisted:
            if self._take_persisted(key, field.value, text):
                field.value = text
        scope = self.query_one("#mandate-tradeable", Checkbox)
        persisted_scope = mandate is not None and mandate.tradeable_only
        if self._take_persisted("tradeable", scope.value, persisted_scope):
            scope.value = persisted_scope
        base_currency.disabled = self.currency_locked
        if mandate is None:
            triggers.update("(no mandate — profile the portfolio first)")
            return
        triggers.update(
            f"triggers cron {_tick(mandate.triggers.cron)} "
            f"drift {_tick(mandate.triggers.drift)} · "
            f"gates {', '.join(self._effective_gates(mandate))}"
        )

    def _paint_guardrails(self, band: str | None, caps: str) -> None:
        suitability = self.query_one("#mandate-suitability", Static)
        select = self.query_one("#mandate-band", Select)
        caps_widget = self.query_one("#mandate-caps", Static)
        if band is None:
            suitability.update("(no profile — p to profile)")
            caps_widget.update("")
            return
        suitability.update(f"suitability band {band} (ceiling)")
        if band in _RISK_BANDS and self._take_persisted(
            "band", self.proposed_band(), band
        ):
            select.value = band
        caps_widget.update(caps)

    def _paint_gates(self, mandate: Any) -> None:
        """Reflect the effective ``place_orders`` gate into its checkbox.

        A baseline-gated ``place_orders`` renders checked, disabled and labelled as
        always on; otherwise the box follows the mandate's own gate set and stays
        editable.
        """
        box = self.query_one("#gate-place-orders", Checkbox)
        locked = self._place_orders_always_gated()
        box.disabled = locked
        box.label = "Before execution (always on)" if locked else "Before execution"
        persisted = locked or (
            mandate is not None and "place_orders" in mandate.hitl_gates
        )
        if self._take_persisted("gate", box.value, persisted):
            box.value = persisted

    def _paint_directives(self, mandate: Any) -> None:
        """Repopulate the removable directives list from the mandate.

        The rows are rebuilt only when the list actually changes: a rebuild resets the
        ListView highlight, which ``x`` relies on to pick the directive to drop.
        """
        persisted = list(mandate.remembered_directives) if mandate else []
        if not self._take_persisted("directives", self._directives, persisted):
            return
        if self._directives != persisted:
            self._directives = list(persisted)
            self._refill_directives()

    def _refill_directives(self) -> None:
        """Rebuild the directives ``ListView`` rows from :attr:`_directives`."""
        listview = self.query_one("#mandate-directives", ListView)
        listview.clear()
        for directive in self._directives:
            # Adviser free text: parsed as markup it would hide `[bracketed]` words or
            # raise MarkupError on a stray `[/x]` on every repaint.
            listview.append(ListItem(Label(directive, markup=False)))
