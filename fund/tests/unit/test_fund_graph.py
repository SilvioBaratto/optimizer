"""Structural tests for ``fund.agents.graph``.

Assert on the **constructed shapes**, never an LLM call (SPEC §5 — no network in
CI). The four subagent builders return well-formed ``SubAgent`` dicts
(name/description/system_prompt/tools/skills) that never set ``interrupt_on``; the
critical roles (allocator → risk → executor) carry a ``ModelFallbackMiddleware``
when a fallback is supplied while the economist never does, and every subagent
(plus the explicit ``general-purpose`` spec) carries the model-call cap.
``build_fund_agent``
wires the PM ``create_deep_agent`` with the orchestrator toolset (the top-level
gated ``place_orders``), the four subagents in pipeline order, the root-relative
skills (Risk R1), the ``virtual_mode`` backend, the checkpointer/store, and
``interrupt_on={"place_orders": True}`` — captured through a spy on
``create_deep_agent`` (no model is ever invoked).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid
from typing import Any

import deepagents
import pytest
from _fund_fakes import ScriptedFundModel
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

from fund.agents import graph
from fund.agents.prompts import (
    ALLOCATOR_SYSTEM_PROMPT,
    ECONOMIST_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    PM_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
)
from fund.agents.skills import skill_sources
from fund.agents.toolsets import TOOLS_BY_AGENT, RunContext
from fund.config import settings

_ASOF = dt.date(2024, 1, 30)
_PORTFOLIO_ID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
# A bare string id; ModelFallbackMiddleware accepts str|BaseChatModel and resolves
# it lazily (no network at construction), so the build stays env-free.
_FALLBACK = "ollama:fallback-model"


def _ctx() -> RunContext:
    """A construction-only RunContext: the bound closures capture it but are never
    invoked here, so a sentinel session/run_id suffices (no DB, no network)."""
    return RunContext(
        session=object(),  # type: ignore[arg-type]
        asof=_ASOF,
        store=None,
        config=settings,
        run_id=uuid.uuid4(),
        portfolio_id=_PORTFOLIO_ID,
    )


# role → (builder, prompt, is_critical)
_ROLES: dict[str, tuple[Any, str, bool]] = {
    "economist": (graph.build_economist_subagent, ECONOMIST_SYSTEM_PROMPT, False),
    "allocator": (graph.build_allocator_subagent, ALLOCATOR_SYSTEM_PROMPT, True),
    "risk": (graph.build_risk_subagent, RISK_SYSTEM_PROMPT, True),
    "executor": (graph.build_executor_subagent, EXECUTOR_SYSTEM_PROMPT, True),
}


def _build(role: str, *, fallback: Any | None = None) -> dict[str, Any]:
    builder, _, is_critical = _ROLES[role]
    if is_critical:
        return builder(_ctx(), fallback=fallback)
    return builder(_ctx())


# --- subagent dict shape -------------------------------------------------------


def test_subagent_dicts_are_well_formed() -> None:
    for role, (_, prompt, _crit) in _ROLES.items():
        sub = _build(role, fallback=_FALLBACK)
        assert sub["name"] == role
        assert isinstance(sub["description"], str) and sub["description"].strip()
        assert sub["system_prompt"] == prompt
        # tools are per-run BaseTool closures matching the role's tool partition,
        # and never expose the bound Session.
        tools = sub["tools"]
        assert all(isinstance(t, BaseTool) for t in tools)
        assert {t.name for t in tools} == set(TOOLS_BY_AGENT[role])
        assert all("session" not in t.args for t in tools)
        # skills are the root-relative staged sources (Risk R1), not absolute paths.
        assert sub["skills"] == skill_sources(role)


def test_no_subagent_sets_interrupt_on() -> None:
    # HITL is PM-level only; a subagent must never carry interrupt_on (avoids
    # relying on subagent-tool interrupt propagation — SPEC).
    for role in _ROLES:
        sub = _build(role, fallback=_FALLBACK)
        assert "interrupt_on" not in sub


def test_executor_subagent_binds_no_tools() -> None:
    # The executor proposes; the PM commits via place_orders. It owns no tool from
    # the frozen eight.
    sub = _build("executor", fallback=_FALLBACK)
    assert sub["tools"] == []


# --- fallback middleware on critical roles only --------------------------------


def _fallbacks(sub: dict[str, Any]) -> list[Any]:
    return [m for m in sub["middleware"] if isinstance(m, ModelFallbackMiddleware)]


def _call_caps(sub: dict[str, Any]) -> list[ModelCallLimitMiddleware]:
    return [m for m in sub["middleware"] if isinstance(m, ModelCallLimitMiddleware)]


def test_critical_roles_attach_fallback_middleware_when_supplied() -> None:
    for role in ("allocator", "risk", "executor"):
        sub = _build(role, fallback=_FALLBACK)
        assert len(_fallbacks(sub)) == 1


def test_economist_never_attaches_fallback_middleware() -> None:
    sub = graph.build_economist_subagent(_ctx())
    assert _fallbacks(sub) == []


def test_critical_roles_without_fallback_have_no_fallback_middleware() -> None:
    for role in ("allocator", "risk", "executor"):
        sub = _build(role, fallback=None)
        assert _fallbacks(sub) == []


def test_every_role_carries_the_configured_model_call_cap() -> None:
    for role in _ROLES:
        sub = _build(role, fallback=_FALLBACK)
        (cap,) = _call_caps(sub)
        assert cap.run_limit == settings.max_subagent_model_calls
        assert cap.exit_behavior == "end"


# --- build_fund_agent: PM create_deep_agent kwargs (spy, no LLM) ---------------


def test_build_fund_agent_wires_the_pm_create_deep_agent(tmp_path, monkeypatch) -> None:
    from deepagents.backends import FilesystemBackend

    captured: dict[str, Any] = {}

    def _spy(*args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "PM_AGENT"

    monkeypatch.setattr(deepagents, "create_deep_agent", _spy)
    # Avoid staging the gitignored theory tree in CI: return a real virtual-mode
    # backend rooted at a temp dir so the virtual_mode assertion stays honest.
    fake_backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(graph, "build_backend", lambda config: fake_backend)

    model = object()
    agent = graph.build_fund_agent(
        model,
        checkpointer="CKPT",
        store="STORE",
        ctx=_ctx(),
        fallback=_FALLBACK,
    )

    assert agent == "PM_AGENT"
    assert captured["model"] is model
    assert captured["system_prompt"] == PM_SYSTEM_PROMPT
    # PM's own tools are the orchestrator toolset: the top-level gated place_orders
    # plus the read-only get_portfolio_history (Phase-4 T9).
    pm_tools = captured["tools"]
    assert {t.name for t in pm_tools} == set(TOOLS_BY_AGENT["orchestrator"])
    assert {t.name for t in pm_tools} == {"place_orders", "get_portfolio_history"}
    # HITL gate at the PM level.
    assert captured["interrupt_on"] == {"place_orders": True}
    assert captured["interrupt_on"] == settings.interrupt_on_map()
    assert captured["skills"] == skill_sources("orchestrator")
    assert captured["backend"] is fake_backend
    assert captured["backend"].virtual_mode is True
    assert captured["checkpointer"] == "CKPT"
    assert captured["store"] == "STORE"


def test_build_fund_agent_wires_four_subagents_in_pipeline_order(
    tmp_path, monkeypatch
) -> None:
    from deepagents.backends import FilesystemBackend

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda *a, **k: captured.update(k)
    )
    fake_backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(graph, "build_backend", lambda config: fake_backend)

    graph.build_fund_agent(
        object(),
        checkpointer="CKPT",
        store=None,
        ctx=_ctx(),
        fallback=_FALLBACK,
    )

    subs = captured["subagents"]
    # The explicit general-purpose spec replaces deepagents' uncapped auto-added one.
    assert [s["name"] for s in subs] == [
        "economist",
        "allocator",
        "risk",
        "executor",
        "general-purpose",
    ]
    # No subagent gates itself; the critical three carry the fallback middleware.
    assert all("interrupt_on" not in s for s in subs)
    by_name = {s["name"]: s for s in subs}
    assert _fallbacks(by_name["economist"]) == []
    assert _fallbacks(by_name["general-purpose"]) == []
    for role in ("allocator", "risk", "executor"):
        assert len(_fallbacks(by_name[role])) == 1
    # Every delegation target, the general-purpose one included, is call-capped.
    assert all(len(_call_caps(s)) == 1 for s in subs)


# --- subagent runaway bound ------------------------------------------------------

_PM_SEED = "start the round"
_SUB_SEED = "loop forever"


class _LoopingSubagentModel(BaseChatModel):
    """The PM delegates once; the subagent then calls ``loop_tool`` until stopped.

    The subagent only stops on its own after ``runaway_after`` model calls, so a
    missing bound shows up as a call count far above the cap instead of a hang.
    ``sub_calls`` is a one-element list so the (frozen) model can count in place.
    """

    subagent_type: str
    loop_tool: str
    runaway_after: int = 200
    sub_calls: list[int] = Field(default_factory=lambda: [0])

    @property
    def _llm_type(self) -> str:
        return "looping-subagent"

    def bind_tools(self, tools: Any, **_: Any) -> _LoopingSubagentModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **_: Any,
    ) -> ChatResult:
        first = next(m for m in messages if isinstance(m, HumanMessage))
        if first.content == _PM_SEED:
            reply = self._pm_turn(messages)
        else:
            reply = self._subagent_turn()
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _pm_turn(self, messages: list[BaseMessage]) -> AIMessage:
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="delegation finished")
        call = {
            "name": "task",
            "args": {"description": _SUB_SEED, "subagent_type": self.subagent_type},
            "id": "call_task",
        }
        return AIMessage(content="", tool_calls=[call])

    def _subagent_turn(self) -> AIMessage:
        self.sub_calls[0] += 1
        n = self.sub_calls[0]
        if n > self.runaway_after:
            return AIMessage(content="giving up")
        call = {"name": self.loop_tool, "args": {}, "id": f"call_loop_{n}"}
        return AIMessage(content="", tool_calls=[call])


@pytest.mark.parametrize(
    ("subagent_type", "loop_tool"),
    [
        ("economist", "get_recent_events"),
        ("allocator", "universe_filter"),
        ("general-purpose", "get_portfolio_history"),
    ],
)
def test_a_looping_subagent_is_stopped_by_the_model_call_cap(
    subagent_type: str, loop_tool: str
) -> None:
    """The PM's ``recursion_limit`` and round cap count only PM supersteps, and a
    subagent graph carries langgraph's own 9999 bound, so without its own cap a
    subagent that keeps retrying a tool spins until the model gives up."""
    model = _LoopingSubagentModel(subagent_type=subagent_type, loop_tool=loop_tool)
    agent = graph.build_fund_agent(
        model, checkpointer=MemorySaver(), store=None, ctx=_ctx()
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": _PM_SEED}]},
        config={
            "configurable": {"thread_id": f"loop-{subagent_type}"},
            "recursion_limit": settings.recursion_limit,
        },
    )

    assert model.sub_calls[0] <= settings.max_subagent_model_calls
    assert result["messages"][-1].content == "delegation finished"


# --- summarization override: FUND_SUMMARY_* bound the PM and every subagent ---


_SUMMARIZATION = "SummarizationMiddleware"


def _summarizers(middleware: list[Any]) -> list[Any]:
    return [m for m in middleware if m.name == _SUMMARIZATION]


def _summary_config() -> Any:
    return dataclasses.replace(
        settings, summary_token_threshold=24000, summary_messages_to_keep=4
    )


def test_summary_override_reaches_the_pm_and_every_subagent(tmp_path, monkeypatch):
    # deepagents' built-in trigger (fraction of the context, else 170k tokens)
    # overflows a small-context model; the operator's override must replace it in
    # the PM stack and in each declarative subagent, which inherits nothing.
    captured = _spy_create_deep_agent(monkeypatch, tmp_path)

    graph.build_fund_agent(
        ScriptedFundModel(),
        checkpointer="CKPT",
        store=None,
        ctx=_ctx(),
        config=_summary_config(),
        fallback=_FALLBACK,
    )

    stacks = {"pm": captured["middleware"]}
    stacks.update({s["name"]: s.get("middleware", []) for s in captured["subagents"]})
    assert set(stacks) == {
        "pm",
        "economist",
        "allocator",
        "risk",
        "executor",
        "general-purpose",
    }
    for name, stack in stacks.items():
        (summarizer,) = _summarizers(stack)
        assert summarizer._lc_helper.trigger == ("tokens", 24000), name
        assert summarizer._lc_helper.keep == ("messages", 4), name
        assert summarizer._lc_helper.trim_tokens_to_summarize is None, name
    for role in ("allocator", "risk", "executor"):
        assert any(isinstance(m, ModelFallbackMiddleware) for m in stacks[role])


def test_without_a_summary_override_deepagents_defaults_stay(tmp_path, monkeypatch):
    captured = _spy_create_deep_agent(monkeypatch, tmp_path)

    graph.build_fund_agent(
        ScriptedFundModel(),
        checkpointer="CKPT",
        store=None,
        ctx=_ctx(),
        fallback=_FALLBACK,
    )

    assert _summarizers(captured["middleware"]) == []
    for sub in captured["subagents"]:
        assert _summarizers(sub.get("middleware", [])) == []


# --- T5: effective interrupt_on = config.interrupt_on ∪ mandate.hitl_gates ------


def _spy_create_deep_agent(monkeypatch, tmp_path) -> dict[str, Any]:
    """Patch ``create_deep_agent`` + ``build_backend``; return the captured kwargs."""
    from deepagents.backends import FilesystemBackend

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda *a, **k: captured.update(k) or "PM"
    )
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(graph, "build_backend", lambda config: backend)
    return captured


def test_build_fund_agent_uses_explicit_interrupt_on_override(tmp_path, monkeypatch):
    """An explicit ``interrupt_on`` override wins over the config default."""
    captured = _spy_create_deep_agent(monkeypatch, tmp_path)
    override = {"place_orders": True, "before_risk": True}

    graph.build_fund_agent(
        object(),
        checkpointer="CKPT",
        store=None,
        ctx=_ctx(),
        interrupt_on=override,
    )

    assert captured["interrupt_on"] == override


def test_build_fund_agent_defaults_interrupt_on_to_config_map(tmp_path, monkeypatch):
    """Without an override, ``build_fund_agent`` falls back to the config map."""
    captured = _spy_create_deep_agent(monkeypatch, tmp_path)

    graph.build_fund_agent(object(), checkpointer="CKPT", store=None, ctx=_ctx())

    assert captured["interrupt_on"] == settings.interrupt_on_map()
    assert captured["interrupt_on"] == {"place_orders": True}


def _store_with_cs() -> Any:
    """An in-memory store carrying the fake portfolio's active ConstraintSet."""
    from _fund_fakes import make_constraint_set
    from langgraph.store.memory import InMemoryStore

    from fund.audit import put_constraint_set

    store = InMemoryStore()
    put_constraint_set(
        store, make_constraint_set(), store_key=settings.constraint_set_store_key
    )
    return store


def test_prepare_fund_run_unions_mandate_gates_into_interrupt_on(
    db_session, monkeypatch
):
    """A mandate's extra ``hitl_gates`` union into the effective ``interrupt_on``."""
    from _fund_fakes import ASOF, PORTFOLIO_ID, make_mandate

    from fund.audit import AutonomyRepository

    captured: dict[str, Any] = {}
    monkeypatch.setattr(graph, "build_fund_agent", lambda *a, **k: captured.update(k))
    mandate = make_mandate().model_copy(
        update={"hitl_gates": ("place_orders", "before_risk")}
    )
    # Steady state: consume the one-time first-round forcing up front so this asserts
    # the mandate-union path, not the round-0 all-gates override (T7).
    AutonomyRepository(db_session).consume_first_round(PORTFOLIO_ID)

    graph._prepare_fund_run(
        object(),
        mandate,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer="CKPT",
        store=_store_with_cs(),
        config=settings,
        fallback=None,
        thread_id=None,
        seed=1,
    )

    assert captured["interrupt_on"] == {"place_orders": True, "before_risk": True}


def test_prepare_fund_run_default_mandate_interrupt_on_is_place_orders_only(
    db_session, monkeypatch
):
    """The default mandate leaves the effective gate set at ``place_orders`` only."""
    from _fund_fakes import ASOF, PORTFOLIO_ID, make_mandate

    from fund.audit import AutonomyRepository

    captured: dict[str, Any] = {}
    monkeypatch.setattr(graph, "build_fund_agent", lambda *a, **k: captured.update(k))
    AutonomyRepository(db_session).consume_first_round(PORTFOLIO_ID)

    graph._prepare_fund_run(
        object(),
        make_mandate(),
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer="CKPT",
        store=_store_with_cs(),
        config=settings,
        fallback=None,
        thread_id=None,
        seed=1,
    )

    assert captured["interrupt_on"] == {"place_orders": True}


def test_resume_fund_rereads_mandate_gates_into_interrupt_on(db_session, monkeypatch):
    """On resume, the gate set is re-read from the persisted mandate ``hitl_gates``."""
    from _fund_fakes import ASOF, PORTFOLIO_ID, make_mandate

    from fund.audit import AgentRunRepository, MandateRepository

    mandate = make_mandate().model_copy(
        update={"hitl_gates": ("place_orders", "before_risk")}
    )
    MandateRepository(db_session).upsert(mandate)
    run = AgentRunRepository(db_session).create_run(
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        seed=1,
        universe=[],
        optimizer_config={"step": "fund"},
    )

    captured: dict[str, Any] = {}

    class _FakeAgent:
        def invoke(self, *a: Any, **k: Any) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(
        graph, "build_fund_agent", lambda *a, **k: captured.update(k) or _FakeAgent()
    )

    graph.resume_fund(
        run.id,
        "reject",
        session=db_session,
        checkpointer="CKPT",
        store=_store_with_cs(),
        model=object(),
    )

    assert captured["interrupt_on"] == {"place_orders": True, "before_risk": True}


# --- T7: round-0 gate forcing + FROZEN manual-run block --------------------------


def _prepare(db_session, mandate, monkeypatch) -> dict[str, Any]:
    """Run ``_prepare_fund_run`` with a spied ``build_fund_agent``; return its kw."""
    from _fund_fakes import ASOF, PORTFOLIO_ID

    captured: dict[str, Any] = {}
    monkeypatch.setattr(graph, "build_fund_agent", lambda *a, **k: captured.update(k))
    graph._prepare_fund_run(
        object(),
        mandate,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer="CKPT",
        store=_store_with_cs(),
        config=settings,
        fallback=None,
        thread_id=None,
        seed=1,
    )
    return captured


def test_first_round_forces_all_orchestrator_gates(db_session, monkeypatch):
    """A portfolio's first round forces the full ``_ALL_ORCHESTRATOR_GATES`` set,
    overriding the mandate, and flips ``first_round_gated`` off."""
    from _fund_fakes import PORTFOLIO_ID, make_mandate

    from fund.audit import AutonomyRepository

    mandate = make_mandate().model_copy(
        update={"hitl_gates": ("place_orders", "before_risk")}
    )
    captured = _prepare(db_session, mandate, monkeypatch)

    assert captured["interrupt_on"] == dict.fromkeys(
        graph._ALL_ORCHESTRATOR_GATES, True
    )
    assert (
        AutonomyRepository(db_session).portfolio_state(PORTFOLIO_ID).first_round_gated
        is False
    )


def test_first_round_gate_flips_so_later_rounds_respect_the_mandate(
    db_session, monkeypatch
):
    """Once the first round is consumed, later rounds use the mandate's gate union."""
    from _fund_fakes import PORTFOLIO_ID, make_mandate

    from fund.audit import AutonomyRepository

    mandate = make_mandate().model_copy(
        update={"hitl_gates": ("place_orders", "before_risk")}
    )
    # First round consumes the one-time forcing...
    _prepare(db_session, mandate, monkeypatch)
    assert (
        AutonomyRepository(db_session).portfolio_state(PORTFOLIO_ID).first_round_gated
        is False
    )
    # ...so the next round respects the mandate's gate set.
    captured = _prepare(db_session, mandate, monkeypatch)
    assert captured["interrupt_on"] == {"place_orders": True, "before_risk": True}


def test_prepare_fund_run_leaves_no_fund_autonomy_row_behind(db_session, monkeypatch):
    """A round reads the fund posture but never inserts the fund row.

    The round's transaction stays open for minutes: a fund row inserted inside it is
    invisible to a concurrent fund freeze, which would INSERT its own, wait on the
    singleton index until the round ends, and then fail with a unique violation.
    """
    from _fund_fakes import make_mandate
    from portopt_db.models import FundAutonomyState
    from sqlalchemy import select

    _prepare(db_session, make_mandate(), monkeypatch)

    scopes = set(db_session.execute(select(FundAutonomyState.scope)).scalars())
    assert scopes == {"portfolio"}  # the first-round flag still needs its own row


def test_prepare_fund_run_raises_when_portfolio_frozen(db_session, monkeypatch):
    """A FROZEN portfolio blocks a manual run before any agent is built."""
    from _fund_fakes import PORTFOLIO_ID, make_mandate

    from fund.audit import AutonomyRepository, AutonomyScope, AutonomyState

    AutonomyRepository(db_session).set_state(
        AutonomyScope.PORTFOLIO, PORTFOLIO_ID, AutonomyState.FROZEN, reason="halt"
    )
    monkeypatch.setattr(
        graph,
        "build_fund_agent",
        lambda *a, **k: pytest.fail("agent must not be built for a frozen portfolio"),
    )

    with pytest.raises(RuntimeError, match="frozen"):
        _prepare(db_session, make_mandate(), monkeypatch)


def test_prepare_fund_run_raises_when_fund_frozen(db_session, monkeypatch):
    """A FROZEN fund blocks a manual run for any portfolio before building an agent."""
    from _fund_fakes import make_mandate

    from fund.audit import AutonomyRepository, AutonomyScope, AutonomyState

    AutonomyRepository(db_session).set_state(
        AutonomyScope.FUND, None, AutonomyState.FROZEN
    )
    with pytest.raises(RuntimeError, match="frozen"):
        _prepare(db_session, make_mandate(), monkeypatch)
