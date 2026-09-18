"""Shared, fully-local fake for the Task-6 fund-run tests.

The fund PM deep agent drives ONE chat model across five roles — the PM plus the
four subagents (economist → allocator → risk → executor) — all sharing the model
instance (deepagents compiles each subagent with the same ``model``, invoked
synchronously and nested). :class:`ScriptedFundModel` keys its reply on **which
role is calling** (detected from the role's system prompt) and on **how far that
role has progressed** (counted from the tool calls already in the message
history), so no live LLM is touched and "no network" is structurally guaranteed
(SPEC §5) — the only work it does is emit the scripted tool calls and reports.

The scripted pipeline:

* **PM** — delegate via ``task`` in fixed order economist → allocator → risk, then
  (only if ``risk_passes``) executor, then commit via ``place_orders(weights)``.
  With ``risk_passes=False`` the PM stops after risk (executor un-invoked, no
  order — the blocking gate). With ``overrun=True`` it re-delegates forever, so the
  recursion limit trips (the round-cap path).
* **allocator** — calls the real bound ``optimize_portfolio(universe)`` once (which
  computes the load-bearing weights + logs the audit decision), then reports.
* **economist / risk / executor** — return a narrative report (no side effects
  needed; the risk gate is governed by the PM's ``risk_passes`` flag).

Not a ``test_*`` module, so pytest does not collect it; the fund-run test file
imports the helpers from here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

# Distinctive opening-line markers of each role's system prompt (see
# ``fund.agents.prompts``); matched as substrings so middleware appends do not
# break detection.
_ROLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("portfolio manager (PM)", "pm"),
    ("fund's economist", "economist"),
    ("fund's allocator", "allocator"),
    ("fund's risk officer", "risk"),
    ("fund's executor", "executor"),
)


def _system_text(messages: list[BaseMessage]) -> str:
    """Flatten the first ``SystemMessage`` content (str or content-block list)."""
    for message in messages:
        if isinstance(message, SystemMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
    return ""


def _role_of(messages: list[BaseMessage]) -> str:
    """Which fund role is calling the model, from its system prompt."""
    system = _system_text(messages)
    for marker, role in _ROLE_MARKERS:
        if marker in system:
            return role
    return "?"


def _count_tool_calls(messages: list[BaseMessage], name: str) -> int:
    """How many ``name`` tool calls already appear in the message history."""
    return sum(
        1
        for message in messages
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
        if call["name"] == name
    )


class ScriptedFundModel(BaseChatModel):
    """Role-aware, network-free stand-in for the DeepSeek chat model.

    Configured per run with the allocator's ``universe`` and the optimizer
    ``weights`` the PM commits (the test computes both from the same deterministic
    ``optimize_portfolio``, so the ticket, the audit, and the finalised run all
    agree). ``risk_passes`` and ``overrun`` select the non-happy paths.
    """

    universe: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    risk_passes: bool = True
    overrun: bool = False
    generate_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fund"

    def bind_tools(self, tools: Any, **_: Any) -> ScriptedFundModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **_: Any,
    ) -> ChatResult:
        object.__setattr__(self, "generate_calls", self.generate_calls + 1)
        role = _role_of(messages)
        message = self._reply_for(role, messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # --- per-role scripts ---------------------------------------------------

    def _reply_for(self, role: str, messages: list[BaseMessage]) -> AIMessage:
        if role == "pm":
            return self._pm_turn(messages)
        if role == "allocator":
            return self._allocator_turn(messages)
        return AIMessage(content=f"{role} report: analysis complete.")

    def _pm_turn(self, messages: list[BaseMessage]) -> AIMessage:
        n_task = _count_tool_calls(messages, "task")
        n_orders = _count_tool_calls(messages, "place_orders")
        if self.overrun:
            # Never make progress: keep re-delegating so the recursion limit trips.
            return _task_call("economist", n_task)
        # economist (0) -> allocator (1) -> risk (2) -> executor (3).
        stages = ("economist", "allocator", "risk")
        if n_task < len(stages):
            return _task_call(stages[n_task], n_task)
        if n_task == len(stages):
            if not self.risk_passes:
                # Blocking gate failed: do not reach the executor or place_orders.
                return AIMessage(content="Risk gate failed; halting. No order placed.")
            return _task_call("executor", n_task)
        if n_orders == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "place_orders",
                        "args": {"weights": self.weights},
                        "id": "call_place_orders",
                    }
                ],
            )
        return AIMessage(content="Run complete; paper ticket handled.")

    def _allocator_turn(self, messages: list[BaseMessage]) -> AIMessage:
        # Call the real bound optimizer once (weights + audit), then report.
        if _count_tool_calls(messages, "optimize_portfolio") == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "optimize_portfolio",
                        "args": {"universe": self.universe},
                        "id": "call_optimize",
                    }
                ],
            )
        return AIMessage(content="allocator report: candidate weights ready.")


def _task_call(subagent: str, index: int) -> AIMessage:
    """A PM assistant turn delegating to ``subagent`` via the built-in ``task``."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {
                    "description": f"Run the {subagent} step.",
                    "subagent_type": subagent,
                },
                "id": f"call_task_{index}",
            }
        ],
    )
