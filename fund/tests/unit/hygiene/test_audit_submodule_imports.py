"""Guard: importing a ``fund.audit`` repository submodule stays agent-stack-free.

The scheduler/worker paths import audit repositories lazily so they run without
``langgraph``/``psycopg_pool``. Python executes ``fund/audit/__init__.py`` before any
submodule, so that package init must not eagerly pull the LangGraph checkpointer
bootstrap — otherwise the "import the submodule directly" advice is unmeetable.

Each probe runs in a fresh interpreter so a stack already loaded by another test
cannot mask a leak.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_REPOSITORY_SUBMODULES = (
    "autonomy_repository",
    "fund_job_repository",
    "human_audit_repository",
    "mandate_repository",
    "mifid_repository",
    "nav_history_repository",
    "orders_repository",
    "positions_repository",
    "repository",
)

_FORBIDDEN = ("deepagents", "langgraph", "langchain_core", "psycopg", "psycopg_pool")


def _run(code: str) -> None:
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)  # noqa: S603


@pytest.mark.parametrize("submodule", _REPOSITORY_SUBMODULES)
def test_importing_an_audit_submodule_loads_no_agent_stack(submodule: str) -> None:
    """Importing ``fund.audit.<submodule>`` leaves the agent stack unloaded."""
    _run(
        f"""
        import importlib
        import sys

        importlib.import_module("fund.audit.{submodule}")
        forbidden = {_FORBIDDEN!r}
        leaked = sorted({{m.split(".")[0] for m in sys.modules}} & set(forbidden))
        assert not leaked, leaked
        """
    )


def test_persistence_names_still_resolve_from_the_package_on_demand() -> None:
    """``from fund.audit import setup_langgraph`` keeps working, loaded lazily."""
    _run(
        """
        import sys

        import fund.audit as audit

        assert "langgraph" not in sys.modules
        from fund.audit import LangGraphPersistence, setup_langgraph
        from fund.audit import persistence

        assert setup_langgraph is persistence.setup_langgraph
        assert LangGraphPersistence is persistence.LangGraphPersistence
        assert {"LangGraphPersistence", "setup_langgraph"} <= set(audit.__all__)
        """
    )


def test_unknown_package_attribute_still_raises_attribute_error() -> None:
    """The lazy package ``__getattr__`` does not swallow typos."""
    _run(
        """
        import fund.audit as audit

        try:
            audit.no_such_name
        except AttributeError:
            pass
        else:
            raise AssertionError("expected AttributeError")
        """
    )
