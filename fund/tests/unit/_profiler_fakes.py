"""Shared, fully-local fakes for the Task-7 profiler agent tests.

The profiler drives ONE chat model down two paths: ``structured_call`` (answer
normalisation via ``.with_structured_output``) and the ``deepagents`` agent
(tool-calling via ``.bind_tools`` + ``._generate``). :class:`ScriptedProfilerModel`
scripts BOTH from in-file queues so no live LLM is touched and "no network" is
structurally guaranteed (SPEC §5) — the only calls it can make are the scripted
ones it counts.

Not a ``test_*`` module, so pytest does not collect it; the two profiler agent
test files import the helpers from here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from fund.schemas.enums import (
    Horizon,
    KnowledgeLevel,
    LossReaction,
    ObjectiveChoice,
)
from fund.schemas.questionnaire import (
    CapacityAnswers,
    EsgAnswers,
    KnowledgeAnswers,
    MiFIDAnswers,
    ObjectivesAnswers,
)


class _ScriptedRunnable:
    """What ``with_structured_output`` returns — a runnable with ``.invoke``."""

    def __init__(self, model: ScriptedProfilerModel) -> None:
        self._model = model

    def invoke(self, messages: Any, /) -> Any:
        return self._model._next_structured(messages)


class ScriptedProfilerModel(BaseChatModel):
    """Scripted stand-in for the DeepSeek chat model — zero network.

    ``structured_outcomes`` is consumed one-per ``with_structured_output().invoke``
    (a ``BaseModel``/``dict`` is returned, a ``BaseException`` is raised — driving
    ``structured_call``'s retry / fallback paths). ``chat_responses`` is consumed
    one-per ``._generate`` and feeds the agent's model node (script a
    ``save_profile`` tool call, then a final message).
    """

    structured_outcomes: list[Any] = Field(default_factory=list)
    chat_responses: list[Any] = Field(default_factory=list)
    with_structured_output_calls: int = 0
    structured_invoke_calls: int = 0
    generate_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-profiler"

    # --- structured-output path (structured_call) ---------------------------
    def with_structured_output(
        self, schema: type[Any], *, method: str = "function_calling", **_: Any
    ) -> _ScriptedRunnable:
        object.__setattr__(
            self, "with_structured_output_calls", self.with_structured_output_calls + 1
        )
        return _ScriptedRunnable(self)

    def _next_structured(self, _messages: Any) -> Any:
        object.__setattr__(
            self, "structured_invoke_calls", self.structured_invoke_calls + 1
        )
        outcome = self.structured_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    # --- tool-calling path (deepagents agent model node) --------------------
    def bind_tools(self, tools: Any, **_: Any) -> ScriptedProfilerModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **_: Any,
    ) -> ChatResult:
        object.__setattr__(self, "generate_calls", self.generate_calls + 1)
        message = self.chat_responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])


def save_tool_call(portfolio_id: str) -> AIMessage:
    """An assistant turn that calls ``save_profile`` once (arg is advisory only —
    the tool persists from its closure, never from an LLM-supplied knob)."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "save_profile",
                "args": {"portfolio_id": portfolio_id},
                "id": "call_save_profile",
            }
        ],
    )


def final_reply() -> AIMessage:
    """The agent's closing turn after the tool result / rejection."""
    return AIMessage(content="Suitability profile handled.")


def make_answers(
    *,
    likert: tuple[int, ...] = (5, 6, 4),
    max_loss: float = 0.25,
    buffer: float = 6.0,
    goal: ObjectiveChoice = ObjectiveChoice.GROWTH,
    horizon: Horizon = Horizon.LONG,
    reaction: LossReaction = LossReaction.HOLD,
    knowledge: KnowledgeLevel = KnowledgeLevel.INFORMED,
    currency: str = "EUR",
    exclusions: tuple[Any, ...] = (),
) -> MiFIDAnswers:
    """A valid ``MiFIDAnswers`` (defaults land in the Balanced band, no flags)."""
    return MiFIDAnswers(
        base_currency=currency,
        knowledge=KnowledgeAnswers(level=knowledge),
        capacity=CapacityAnswers(max_1yr_loss_pct=max_loss, buffer_months=buffer),
        objectives=ObjectivesAnswers(
            goal=goal,
            horizon=horizon,
            likert_items=likert,
            loss_reaction=reaction,
        ),
        esg=EsgAnswers(exclusions=exclusions),
    )


def make_model(
    answers: MiFIDAnswers,
    *,
    portfolio_id: str,
    structured_prefix: tuple[Any, ...] = (),
) -> ScriptedProfilerModel:
    """A model that normalises to ``answers`` (after any malformed ``prefix``),
    then drives the agent to call ``save_profile`` once and reply."""
    return ScriptedProfilerModel(
        structured_outcomes=[*structured_prefix, answers],
        chat_responses=[save_tool_call(portfolio_id), final_reply()],
    )
