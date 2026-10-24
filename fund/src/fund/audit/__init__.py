"""Audit layer: run/decision persistence + LangGraph checkpointer bootstrap.

The ``agent_runs`` / ``agent_decisions`` models live in ``portopt_db``; this
package holds the fund-side ``AgentRunRepository`` behavior over them.

Note:
    ``LangGraphPersistence`` and ``setup_langgraph`` live in
    ``fund.audit.persistence``, which pulls in the full langgraph stack. They are
    resolved lazily on first attribute access, so importing this package or any
    repository submodule stays agent-stack-free (scheduler, worker entrypoints).
"""

from typing import TYPE_CHECKING, Any

from fund.audit.autonomy_repository import (
    AutonomyRepository,
    AutonomyScope,
    AutonomyState,
)
from fund.audit.fund_job_repository import FUND_JOB_TYPES, FundJobRepository
from fund.audit.human_audit_repository import HumanAuditLogRepository
from fund.audit.mandate_repository import MandateRepository
from fund.audit.mifid_repository import (
    MifidProfileRepository,
    put_constraint_set,
    put_constraint_set_after_commit,
    resolve_constraint_set,
)
from fund.audit.orders_repository import OrderRepository
from fund.audit.positions_repository import PositionRepository
from fund.audit.repository import AgentRunRepository

if TYPE_CHECKING:
    from fund.audit.persistence import LangGraphPersistence, setup_langgraph

_LAZY_PERSISTENCE_NAMES = frozenset({"LangGraphPersistence", "setup_langgraph"})


def __getattr__(name: str) -> Any:
    """Resolve the persistence names on demand so the package init loads no langgraph.

    Python runs this module before any ``fund.audit.<submodule>``, so an eager
    ``persistence`` import would defeat every caller that imports a repository
    submodule precisely to avoid the agent stack.

    Args:
        name: The attribute being looked up on the package.

    Returns:
        The named object from ``fund.audit.persistence``, cached on the package.

    Raises:
        AttributeError: ``name`` is not a lazily exported persistence name.
    """
    if name in _LAZY_PERSISTENCE_NAMES:
        from fund.audit import persistence

        value = getattr(persistence, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FUND_JOB_TYPES",
    "AgentRunRepository",
    "AutonomyRepository",
    "AutonomyScope",
    "AutonomyState",
    "FundJobRepository",
    "HumanAuditLogRepository",
    "LangGraphPersistence",
    "MandateRepository",
    "MifidProfileRepository",
    "OrderRepository",
    "PositionRepository",
    "put_constraint_set",
    "put_constraint_set_after_commit",
    "resolve_constraint_set",
    "setup_langgraph",
]
