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
* **risk** — calls the real bound ``risk_check`` once (which records the verdict
  ``place_orders`` requires), then reports; the PM's ``risk_passes`` flag decides
  whether the run proceeds past it. With ``risk_checks=False`` it only reports, so
  no verdict exists — a risk step that was skipped.
* **economist / executor** — return a narrative report (no side effects needed).

Not a ``test_*`` module, so pytest does not collect it; the fund-run test file
imports the helpers from here.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from decimal import Decimal
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument
from pydantic import Field
from sqlalchemy import select

from fund.schemas import ConstraintSet
from fund.schemas.constraint_set import EsgPolicy
from fund.schemas.enums import GicsSector, Horizon, ObjectiveChoice, RiskMeasureChoice
from fund.schemas.mandate import PortfolioMandate, RunTriggers
from fund.tools.optimize import optimize_portfolio

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
    agree). ``risk_passes``, ``risk_checks`` and ``overrun`` select the non-happy
    paths; ``max_orders`` (default 1) lets the PM re-gate — call ``place_orders``
    again after a respond barge-in or a refused approval — so a re-gating resume can
    be scripted. ``parallel_orders`` (default 1) makes each ``place_orders`` turn emit
    that many tool calls in one message, the parallel-tool-calling shape that gives
    the HITL gate several action requests.
    """

    universe: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    risk_passes: bool = True
    risk_checks: bool = True
    overrun: bool = False
    max_orders: int = 1
    parallel_orders: int = 1
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
        if role == "risk":
            return self._risk_turn(messages)
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
        if n_orders < self.max_orders:
            # Re-gate once per prior place_orders call so a respond barge-in (which
            # threads the gate's result back and lets the PM continue) drives another
            # place_orders → a fresh HITL interrupt; max_orders=1 keeps the default
            # single-gate behaviour.
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "place_orders",
                        "args": {"weights": self.weights},
                        "id": f"call_place_orders_{n_orders}_{k}",
                    }
                    for k in range(self.parallel_orders)
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

    def _risk_turn(self, messages: list[BaseMessage]) -> AIMessage:
        # Call the real bound risk gate once (it records the verdict), then report.
        if self.risk_checks and _count_tool_calls(messages, "risk_check") == 0:
            return AIMessage(
                content="",
                tool_calls=[{"name": "risk_check", "args": {}, "id": "call_risk"}],
            )
        return AIMessage(content="risk report: proposal checked.")


class PmFailsAfterOrderModel(ScriptedFundModel):
    """A :class:`ScriptedFundModel` whose PM turn raises once ``place_orders`` ran.

    Models a provider failure (429 / timeout) on the PM's follow-up turn after an
    approved gate: the interrupt is already consumed and the paper ticket written when
    the call blows up, so the resume raises *after* the checkpoint has moved on.
    """

    def _pm_turn(self, messages: list[BaseMessage]) -> AIMessage:
        if _count_tool_calls(messages, "place_orders"):
            raise RuntimeError("429 rate limited")
        return super()._pm_turn(messages)


class FreezeMidRunModel(ScriptedFundModel):
    """A :class:`ScriptedFundModel` that fires ``on_first_call`` on its first turn.

    Stands in for an operator hitting the killswitch while the round is still in
    flight: the hook runs after the run's own FROZEN pre-check but before the run
    reaches its gate.
    """

    on_first_call: Any = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.on_first_call is not None and self.generate_calls == 0:
            self.on_first_call()
        return super()._generate(messages, stop, run_manager, **kwargs)


class AllocatorTimesOutFundModel(ScriptedFundModel):
    """A provider outage: the allocator's model call times out mid-pipeline."""

    def _allocator_turn(self, messages: list[BaseMessage]) -> AIMessage:
        raise TimeoutError("provider timed out")


class BrokenPlaceOrdersFundModel(ScriptedFundModel):
    """The PM's ``place_orders`` call arrives unusable.

    Langchain surfaces a truncated or unparsable tool call only as
    ``invalid_tool_calls``, so the turn carries no valid tool call and ``create_agent``
    ends the run as if it were a final answer.

    ``response_metadata`` is what the provider reported for the turn: a provider output
    cap shows up as ``stop_reason="max_tokens"`` / ``finish_reason="length"``; a model
    that wrote malformed JSON stops normally.
    """

    stop_metadata: dict[str, Any] = Field(default_factory=dict)

    def _pm_turn(self, messages: list[BaseMessage]) -> AIMessage:
        message = super()._pm_turn(messages)
        if not any(call["name"] == "place_orders" for call in message.tool_calls):
            return message
        return AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "place_orders",
                    "args": '{"weights": {"AAA": 0.5076, "BBB": 0.0',
                    "id": "call_place_orders_broken",
                    "error": "Function place_orders arguments are not valid JSON",
                    "type": "invalid_tool_call",
                }
            ],
            response_metadata=dict(self.stop_metadata),
        )


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


# ---------------------------------------------------------------------------
# Shared run-seeding helpers used by the fund-run and resume test modules.
#
# A deterministic ``N_DAYS`` x 3 close panel extending past ``ASOF`` so the real
# bound ``optimize_portfolio`` computes stable weights over a seeded SQLite panel
# (the optimizer output, never the LLM). ``PORTFOLIO_ID`` bears hex letters: an
# all-digit UUID gets coerced to a float by SQLite's numeric affinity when a UUID
# column round-trips through ``refresh`` (see fund-test-uuid-sqlite-affinity).
# ---------------------------------------------------------------------------

START = dt.date(2024, 1, 1)
N_DAYS = 40
ASOF = dt.date(2024, 1, 30)  # decision bar (index 29); bars 30..39 are future
UNIVERSE = ["AAA", "BBB", "CCC"]
PORTFOLIO_ID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
_SERIES = {"AAA": (100.0, 0.0), "BBB": (50.0, 1.3), "CCC": (25.0, 2.6)}


def _close_on(start: float, phase: float, i: int) -> float:
    """Deterministic close for day ``i``: a phased oscillation off ``start``."""
    return round(start * (1.0 + 0.02 * math.sin(0.5 * i + phase)), 6)


def seed_panel(session: Any, n_days: int = N_DAYS) -> None:
    """Seed an ``n_days`` x 3 close panel (default extends past ``ASOF``)."""
    for ticker, (start, phase) in _SERIES.items():
        ex = Exchange(name=f"EX-{ticker}")
        session.add(ex)
        session.flush()
        inst = Instrument(
            ticker=ticker,
            short_name=ticker,
            exchange_id=ex.id,
            instrument_type="EQUITY",
            asset_class="equity",
            yfinance_ticker=ticker,
        )
        session.add(inst)
        session.flush()
        for i in range(n_days):
            session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=START + dt.timedelta(days=i),
                    close=_close_on(start, phase, i),
                    volume=1000,
                )
            )
    session.flush()


def make_constraint_set(portfolio_id: uuid.UUID = PORTFOLIO_ID) -> ConstraintSet:
    """A minimal active ``ConstraintSet`` for ``portfolio_id`` (growth / variance)."""
    return ConstraintSet(
        portfolio_id=str(portfolio_id),
        base_currency="EUR",
        a_gamma=2.5,
        objective=ObjectiveChoice.GROWTH,
        risk_measure=RiskMeasureChoice.VARIANCE,
        beta=0.95,
        nu1=0.05,
        nu2=0.10,
        nu3=0.20,
        horizon=Horizon.LONG,
    )


def make_energy_exclusion_constraint_set(
    portfolio_id: uuid.UUID = PORTFOLIO_ID,
) -> ConstraintSet:
    """``make_constraint_set`` re-profiled with an Energy ESG exclusion."""
    return make_constraint_set(portfolio_id).model_copy(
        update={"esg": EsgPolicy(exclusions=(GicsSector.ENERGY,))}
    )


def classify_instruments(session: Any, sectors: dict[str, str]) -> None:
    """Give each seeded instrument its profile sector (the mandate screen reads it)."""
    for ticker, sector in sectors.items():
        instrument = session.execute(
            select(Instrument).where(Instrument.yfinance_ticker == ticker)
        ).scalar_one()
        session.add(TickerProfile(instrument_id=instrument.id, sector=sector))
    session.flush()


def make_mandate(portfolio_id: uuid.UUID = PORTFOLIO_ID) -> PortfolioMandate:
    """A minimal ``PortfolioMandate`` for ``portfolio_id`` (100k EUR, cron+drift)."""
    return PortfolioMandate(
        portfolio_id=str(portfolio_id),
        capital=Decimal("100000"),
        base_currency="EUR",
        drift_l1_threshold=0.1,
        triggers=RunTriggers(cron=True, drift=True),
    )


def expected_weights(
    session: Any, constraint_set: ConstraintSet | None = None
) -> dict[str, float]:
    """The optimizer's weights the allocator will produce over the seeded panel.

    Computed under the same mapped ``ConstraintSet`` (``make_constraint_set()`` unless
    the run resolves another) so the ticket, the audit trail, and the finalised run
    must all match these — the load-bearing "skfolio, not the LLM, computes the
    weights" invariant.
    """
    from fund.agents.toolsets import _optimizer_constraints

    result = optimize_portfolio(
        session,
        ASOF,
        UNIVERSE,
        constraints=_optimizer_constraints(constraint_set or make_constraint_set()),
    )
    assert result["ok"] is True
    return result["data"]["weights"]


def seed_paused_fund_run(
    session_factory: Any, saver: Any, store: Any
) -> tuple[uuid.UUID, dict[str, float]]:
    """Seed a ``run_fund`` paused at ``place_orders``; return ``(run_id, weights)``.

    The shared setup the cockpit and the action-path tests both need: a deterministic
    price panel plus an active ``ConstraintSet`` in ``store``, then one ``run_fund``
    driven by a :class:`ScriptedFundModel` to the HITL gate. ``session_factory`` opens
    a context-managed session over the test's shared in-memory engine; ``saver`` /
    ``store`` are an in-memory ``MemorySaver`` / ``InMemoryStore`` the caller owns. The
    returned weights are the optimizer's — the ticket, audit, and finalised run agree.

    Imports the agent stack lazily so a bare ``import _fund_fakes`` stays light.
    """
    from fund.agents.graph import run_fund
    from fund.audit import put_constraint_set
    from fund.config import settings

    with session_factory() as session:
        seed_panel(session)
        session.commit()
    put_constraint_set(
        store, make_constraint_set(), store_key=settings.constraint_set_store_key
    )
    with session_factory() as session:
        expected = expected_weights(session)
    with session_factory() as session:
        run = run_fund(
            ScriptedFundModel(universe=UNIVERSE, weights=expected),
            make_mandate(),
            portfolio_id=PORTFOLIO_ID,
            asof=ASOF,
            session=session,
            checkpointer=saver,
            store=store,
        )
        session.commit()
        return run.run_id, expected
