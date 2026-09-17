"""Fund deep-agent layer — the MiFID profiler (Fase 5) and its public surface.

Re-exports the profiler entry points so callers use ``from fund.agents import
run_profiler`` without reaching into ``fund.agents.profiler``. The deterministic
mapping helpers stay in ``fund.agents.profiler`` (imported directly where needed).
"""

from __future__ import annotations

from fund.agents.profiler import (
    ProfilerRun,
    SuitabilityBreachError,
    build_constraint_set,
    build_profiler_agent,
    run_mapping,
    run_profiler,
)

__all__ = [
    "ProfilerRun",
    "SuitabilityBreachError",
    "build_constraint_set",
    "build_profiler_agent",
    "run_mapping",
    "run_profiler",
]
