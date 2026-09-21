"""Task 7 — ``fund.agents`` public surface (Phase-7 orchestration entry points).

The package ``__init__`` re-exports the three Phase-7 orchestration entry points
— ``run_fund``, ``build_fund_agent``, ``FundRun`` (from ``fund.agents.graph``) —
alongside the pre-existing MiFID profiler surface, so callers import from one
place instead of reaching into ``fund.agents.graph``.

The load-bearing invariant (SPEC §5 / plan Task 7): a *bare* ``import fund.agents``
must stay **agent-stack-free** — pulling in no ``deepagents`` / ``langchain`` /
``langgraph`` runtime and needing no environment. ``graph.py`` keeps the agent
stack lazy (imported inside its builders), so re-exporting its symbols must not
regress that. The subprocess test asserts the *transitive* closure stays clean in
a fresh interpreter (order-independent — a sibling test importing ``deepagents``
into this process cannot mask a real leak).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import fund.agents as agents

# The three Phase-7 orchestration names §7 promises callers import directly.
_ORCHESTRATION_SURFACE = ("FundRun", "build_fund_agent", "run_fund")
# The pre-existing MiFID profiler surface must keep resolving (regression guard).
_PROFILER_SURFACE = (
    "ProfilerRun",
    "SuitabilityBreachError",
    "build_constraint_set",
    "build_profiler_agent",
    "run_mapping",
    "run_profiler",
)


def test_orchestration_surface_resolves():
    from fund.agents import FundRun, build_fund_agent, run_fund

    assert all(obj is not None for obj in (FundRun, build_fund_agent, run_fund))


def test_profiler_surface_still_resolves():
    # Task 7 adds names; it must not drop the Fase-5 profiler exports.
    for name in _PROFILER_SURFACE:
        assert hasattr(agents, name), f"profiler export {name!r} went missing"


def test_orchestration_reexports_are_identical_objects():
    # A re-export must be the same object, not a copy/shadow.
    from fund.agents import graph

    assert agents.FundRun is graph.FundRun
    assert agents.build_fund_agent is graph.build_fund_agent
    assert agents.run_fund is graph.run_fund


def test_all_has_no_dangling_names():
    # Every name advertised in ``__all__`` must be a real attribute.
    for name in agents.__all__:
        assert hasattr(agents, name), f"__all__ lists {name!r} but it is not exported"


def test_all_covers_both_surfaces():
    for name in (*_ORCHESTRATION_SURFACE, *_PROFILER_SURFACE):
        assert name in agents.__all__, f"{name!r} is missing from fund.agents.__all__"


def test_all_is_sorted_and_unique():
    assert agents.__all__ == sorted(agents.__all__)
    assert len(agents.__all__) == len(set(agents.__all__))


def test_bare_import_stays_agent_stack_free():
    # The agent stack lives lazily inside graph.py's builders; a bare
    # ``import fund.agents`` (touching every exported name) must drag none of it
    # into a fresh interpreter's sys.modules. Fresh process ⇒ order-independent.
    code = textwrap.dedent(
        """
        import sys
        import fund.agents as agents
        # Touch every re-exported name so a lazy trigger would fire if present.
        for name in agents.__all__:
            getattr(agents, name)
        forbidden = {
            "deepagents",
            "langchain",
            "langchain_core",
            "langchain_ollama",
            "langgraph",
            "app",
        }
        leaked = sorted({m.split(".")[0] for m in sys.modules} & forbidden)
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
