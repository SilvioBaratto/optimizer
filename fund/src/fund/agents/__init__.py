"""Fund deep-agent layer — the MiFID profiler (Fase 5) plus the Phase-7 fund
orchestration graph, and their shared public surface.

Re-exports the profiler entry points and the Phase-7 orchestration entry points
so callers use ``from fund.agents import run_fund`` / ``run_profiler`` without
reaching into ``fund.agents.graph`` / ``fund.agents.profiler``. The deterministic
mapping helpers stay in ``fund.agents.profiler`` (imported directly where needed).

Both source modules keep the agent stack (``deepagents`` / ``langchain`` /
``langgraph``) lazy — imported inside their builders, never at module top level —
so a bare ``import fund.agents`` drags in no agent runtime and needs no
environment.
"""

from __future__ import annotations

from fund.agents.graph import (
    FundRun,
    build_fund_agent,
    run_fund,
)
from fund.agents.profiler import (
    ProfilerRun,
    SuitabilityBreachError,
    build_constraint_set,
    build_profiler_agent,
    run_mapping,
    run_profiler,
)

__all__ = [
    "FundRun",
    "ProfilerRun",
    "SuitabilityBreachError",
    "build_constraint_set",
    "build_fund_agent",
    "build_profiler_agent",
    "run_fund",
    "run_mapping",
    "run_profiler",
]
