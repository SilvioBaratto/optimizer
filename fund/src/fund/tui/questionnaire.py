"""In-TUI MiFID questionnaire — collects a typed ``MiFIDAnswers`` for onboarding.

A modal the shell pushes from Posture's ``n`` (new portfolio) / ``p`` (profile)
bindings. It renders the ESMA pillars as a small form and, on submit, builds a
validated :class:`~fund.schemas.questionnaire.MiFIDAnswers` plus the initial capital
for a new portfolio's mandate (:class:`OnboardingSubmission`).

Agent-stack-light by construction: module scope is Textual + stdlib only; the schema
types and enums are imported lazily inside :meth:`QuestionnaireScreen.compose` /
:meth:`QuestionnaireScreen.build_submission`, so ``import fund.tui.app`` stays light
(the import-hygiene guard in ``test_tui.py``). A bad entry is surfaced inline and the
modal stays open — nothing is dismissed until the answers validate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, SelectionList, Static

if TYPE_CHECKING:
    from textual.binding import BindingType

    from fund.schemas.questionnaire import MiFIDAnswers


@dataclass(frozen=True)
class OnboardingSubmission:
    """A validated questionnaire result: the typed answers + the mandate capital.

    Attributes:
        answers: The MiFID answers the profiler consumes.
        capital: The initial capital for a new portfolio's mandate (``0`` and unused
            when profiling an existing portfolio).
    """

    answers: MiFIDAnswers
    capital: Decimal


class QuestionnaireScreen(ModalScreen["OnboardingSubmission | None"]):
    """The onboarding form: render the ESMA pillars, build a typed ``MiFIDAnswers``.

    Dismisses an :class:`OnboardingSubmission` on a valid submit, or ``None`` on
    Cancel / ``escape``. A validation error (an out-of-range Likert, a missing choice,
    a non-positive capital) is shown inline and the modal stays open.

    Args:
        ask_capital: Render the capital field (a new portfolio needs an initial
            mandate); ``False`` when profiling an existing portfolio.
    """

    DEFAULT_CSS = """
    QuestionnaireScreen { align: center middle; }
    .q-dialog {
        width: 70; height: auto; max-height: 90%; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    .q-title { text-style: bold; color: $accent; }
    .q-form { height: 1fr; }
    .q-error { height: auto; color: $error; }
    .q-actions { height: auto; align-horizontal: center; padding-top: 1; }
    .q-dialog Button { margin: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

    def __init__(self, *, ask_capital: bool = True) -> None:
        super().__init__()
        self._ask_capital = ask_capital

    def compose(self) -> ComposeResult:
        """Yield the questionnaire form — one widget per pillar sub-datum."""
        from fund.schemas.enums import (
            GicsSector,
            Horizon,
            KnowledgeLevel,
            LossReaction,
            ObjectiveChoice,
        )

        with Vertical(classes="q-dialog"):
            yield Label("MiFID suitability questionnaire", classes="q-title")
            # Only the fields scroll; the error line and buttons sit outside the
            # scroll area so a short terminal can never push them out of reach.
            with VerticalScroll(classes="q-form"):
                if self._ask_capital:
                    yield Input(placeholder="capital (e.g. 100000)", id="q-capital")
                yield Input(
                    value="EUR", placeholder="base currency", id="q-base-currency"
                )
                yield Select(
                    [(level.value, level.value) for level in KnowledgeLevel],
                    id="q-knowledge",
                    prompt="knowledge & experience",
                )
                yield Input(placeholder="max 1yr loss (fraction 0-1)", id="q-max-loss")
                yield Input(placeholder="cash buffer (months)", id="q-buffer")
                yield Select(
                    [(goal.value, goal.value) for goal in ObjectiveChoice],
                    id="q-goal",
                    prompt="primary objective",
                )
                yield Select(
                    [(horizon.value, horizon.value) for horizon in Horizon],
                    id="q-horizon",
                    prompt="horizon",
                )
                yield Input(
                    placeholder="risk likert 1-7 (comma-separated)", id="q-likert"
                )
                yield Select(
                    [(reaction.value, reaction.value) for reaction in LossReaction],
                    id="q-loss-reaction",
                    prompt="reaction to a severe loss",
                )
                yield SelectionList[str](
                    *((sector.value, sector.value) for sector in GicsSector), id="q-esg"
                )
            yield Static("", id="q-error", classes="q-error", markup=False)
            with Horizontal(classes="q-actions"):
                yield Button("Submit", variant="primary", id="q-submit")
                yield Button("Cancel", id="q-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Submit builds + dismisses the answers; Cancel dismisses ``None``."""
        if event.button.id == "q-submit":
            self._submit()
        elif event.button.id == "q-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        """``escape`` dismisses the questionnaire as a cancel (a caller no-op)."""
        self.dismiss(None)

    def _submit(self) -> None:
        """Validate the form; dismiss the submission, or surface the error inline."""
        try:
            submission = self.build_submission()
        except Exception as exc:  # pydantic/parse errors → an inline message, stay open
            self.query_one("#q-error", Static).update(f"invalid: {exc}")
            return
        self.dismiss(submission)

    def build_submission(self) -> OnboardingSubmission:
        """Build a validated :class:`OnboardingSubmission` from the form widgets.

        Returns:
            The typed answers + the entered capital.

        Raises:
            ValueError: A required choice is blank, the Likert field is empty, or the
                capital is not strictly positive.
            pydantic.ValidationError: An answer is out of its MiFID contract.
            decimal.InvalidOperation: The capital field is not a number.
        """
        from fund.schemas.questionnaire import (
            CapacityAnswers,
            EsgAnswers,
            KnowledgeAnswers,
            MiFIDAnswers,
            ObjectivesAnswers,
        )

        answers = MiFIDAnswers(
            base_currency=self.query_one("#q-base-currency", Input).value.strip(),
            knowledge=KnowledgeAnswers(level=self._choice("#q-knowledge")),
            capacity=CapacityAnswers(
                max_1yr_loss_pct=float(self.query_one("#q-max-loss", Input).value),
                buffer_months=float(self.query_one("#q-buffer", Input).value),
            ),
            objectives=ObjectivesAnswers(
                goal=self._choice("#q-goal"),
                horizon=self._choice("#q-horizon"),
                likert_items=self._likert(),
                loss_reaction=self._choice("#q-loss-reaction"),
            ),
            esg=EsgAnswers(
                exclusions=tuple(self.query_one("#q-esg", SelectionList).selected)
            ),
        )
        return OnboardingSubmission(answers=answers, capital=self._capital())

    def _choice(self, selector: str) -> Any:
        """The picked value of a required ``Select``; raise when left blank."""
        value = self.query_one(selector, Select).value
        if value is Select.NULL:
            raise ValueError(f"{selector.lstrip('#q-')} is required")
        return value

    def _likert(self) -> tuple[int, ...]:
        """Parse the comma-separated Likert field into a non-empty int tuple."""
        raw = self.query_one("#q-likert", Input).value
        items = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
        if not items:
            raise ValueError("at least one risk-tolerance (likert) answer is required")
        return items

    def _capital(self) -> Decimal:
        """The entered capital for a new portfolio (``Decimal``); ``0`` when not asked.

        Raises:
            decimal.InvalidOperation: The capital field is not a number.
            ValueError: The capital is not strictly positive — a new portfolio needs
                real capital. Raised here (not left to downstream mandate validation) so
                ``_submit`` catches it inline and the modal keeps the entered answers.
        """
        if not self._ask_capital:
            return Decimal("0")
        capital = Decimal(self.query_one("#q-capital", Input).value.strip())
        if capital <= 0:
            raise ValueError("capital must be a positive number")
        return capital


__all__ = ["OnboardingSubmission", "QuestionnaireScreen"]
