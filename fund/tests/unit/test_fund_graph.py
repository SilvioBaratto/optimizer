"""Task 5 — Phase-7 subagents + the PM deep agent (``fund.agents.graph``).

Assert on the **constructed shapes**, never an LLM call (SPEC §5 — no network in
CI). The four subagent builders return well-formed ``SubAgent`` dicts
(name/description/system_prompt/tools/skills) that never set ``interrupt_on``; the
critical roles (allocator → risk → executor) carry a ``ModelFallbackMiddleware``
when a fallback is supplied while the economist never does. ``build_fund_agent``
wires the PM ``create_deep_agent`` with the orchestrator toolset (the top-level
gated ``place_orders``), the four subagents in pipeline order, the root-relative
skills (Risk R1), the ``virtual_mode`` backend, the checkpointer/store, and
``interrupt_on={"place_orders": True}`` — captured through a spy on
``create_deep_agent`` (no model is ever invoked).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import deepagents
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.tools import BaseTool

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


def test_critical_roles_attach_fallback_middleware_when_supplied() -> None:
    for role in ("allocator", "risk", "executor"):
        sub = _build(role, fallback=_FALLBACK)
        middleware = sub["middleware"]
        assert len(middleware) == 1
        assert isinstance(middleware[0], ModelFallbackMiddleware)


def test_economist_never_attaches_fallback_middleware() -> None:
    sub = graph.build_economist_subagent(_ctx())
    assert "middleware" not in sub


def test_critical_roles_without_fallback_have_no_middleware() -> None:
    for role in ("allocator", "risk", "executor"):
        sub = _build(role, fallback=None)
        assert "middleware" not in sub


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
    # PM's own tool is the top-level gated place_orders (the orchestrator toolset).
    pm_tools = captured["tools"]
    assert {t.name for t in pm_tools} == set(TOOLS_BY_AGENT["orchestrator"])
    assert {t.name for t in pm_tools} == {"place_orders"}
    # HITL gate at the PM level.
    assert captured["interrupt_on"] == {"place_orders": True}
    assert captured["interrupt_on"] == settings.interrupt_on_map()
    # Root-relative orchestrator skills + virtual-mode backend + persistence.
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
    assert [s["name"] for s in subs] == ["economist", "allocator", "risk", "executor"]
    # No subagent gates itself; the critical three carry the fallback middleware.
    assert all("interrupt_on" not in s for s in subs)
    by_name = {s["name"]: s for s in subs}
    assert "middleware" not in by_name["economist"]
    for role in ("allocator", "risk", "executor"):
        assert isinstance(by_name[role]["middleware"][0], ModelFallbackMiddleware)
