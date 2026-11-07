"""A SQLite + in-memory LangGraph harness with one run paused at its approval gate.

Shared by the tests that drive ``fund.actions`` and the TUI workers against a real
``MemorySaver`` checkpoint. Imports the agent stack at module load, like the other
``_*_fakes`` helpers, so only agent-stack tests import it.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING

from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    ScriptedFundModel,
    expected_weights,
    make_constraint_set,
    make_mandate,
    seed_panel,
    threaded_sqlite_engine,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy.orm import sessionmaker

from fund.agents.graph import run_fund
from fund.audit import put_constraint_set
from fund.config import settings

if TYPE_CHECKING:
    from pathlib import Path


def make_gate_env(path: Path, *, max_orders: int = 1) -> SimpleNamespace:
    """Seed a price panel and drive one ``run_fund`` to its ``place_orders`` gate.

    Args:
        path: The SQLite file to create; discarded with the test's ``tmp_path``.
        max_orders: How many times the resume model may call ``place_orders``; more
            than one lets a refused or responded gate re-gate.

    Returns:
        A namespace with ``get_session`` (a context-managed session factory),
        ``persistence`` (saver, store and a ``None`` pool), ``saver``, ``store``,
        ``run_id``, ``expected`` (the optimizer weights), ``model_factory`` and
        ``dispose`` (release the engine).
    """
    engine = threaded_sqlite_engine(path)
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    saver = MemorySaver()
    store = InMemoryStore()

    @contextmanager
    def get_session():
        """Yield a session on the shared engine, closed on exit."""
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    with get_session() as session:
        seed_panel(session)
        session.commit()
    put_constraint_set(
        store, make_constraint_set(), store_key=settings.constraint_set_store_key
    )
    with get_session() as session:
        expected = expected_weights(session)
    with get_session() as session:
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
        run_id = run.run_id

    return SimpleNamespace(
        get_session=get_session,
        persistence=SimpleNamespace(saver=saver, store=store, pool=None),
        saver=saver,
        store=store,
        run_id=run_id,
        expected=expected,
        model_factory=lambda: ScriptedFundModel(
            universe=UNIVERSE, weights=expected, max_orders=max_orders
        ),
        dispose=engine.dispose,
    )
