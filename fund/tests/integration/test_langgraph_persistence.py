"""T2.3 — D3 LangGraph persistence, integration slice (Postgres-only).

Promoted from ``.phase0-probes/probe_d3_pg_schema.py``. Proves the SPEC D3
contract against a real database: ``setup_langgraph`` lands all six LangGraph
tables in the dedicated ``langgraph`` schema with **zero** leak into Alembic's
``public``, and a checkpoint + a namespaced store item round-trip.

Marked ``integration`` and skipped unless ``DATABASE_URL`` is set, so the default
unit run (and CI without Postgres) stays green. The test owns its schema: it
drops ``langgraph`` before and after so a developer's real DB is left untouched.
"""

from __future__ import annotations

import os
import re
from collections.abc import Generator
from typing import TypedDict

import psycopg
import pytest
from psycopg.rows import dict_row

from fund.audit import persistence
from fund.audit.persistence import LANGGRAPH_TABLES
from fund.config import FundConfig, load_config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="D3 integration test requires a live Postgres DATABASE_URL",
    ),
]

SCHEMA = "langgraph"


def _conninfo() -> str:
    url = os.environ["DATABASE_URL"]
    return re.sub(r"^postgresql\+\w+://", "postgresql://", url)


def _tables_in(schema: str) -> set[str]:
    with (
        psycopg.connect(_conninfo(), autocommit=True) as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname=%s", (schema,))
        return {r["tablename"] for r in cur.fetchall()}


@pytest.fixture
def clean_langgraph_schema() -> Generator[FundConfig, None, None]:
    """Snapshot ``public``, drop any stale ``langgraph``, yield config, restore."""
    with psycopg.connect(_conninfo(), autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    cfg = load_config()
    try:
        yield cfg
    finally:
        with psycopg.connect(_conninfo(), autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


def test_setup_lands_six_tables_in_langgraph_and_none_leak_into_public(
    clean_langgraph_schema: FundConfig,
) -> None:
    public_before = _tables_in("public")

    result = persistence.setup_langgraph(clean_langgraph_schema)
    try:
        lg_tables = _tables_in(SCHEMA)
        public_after = _tables_in("public")

        assert set(LANGGRAPH_TABLES) <= lg_tables, (
            f"missing LangGraph tables in {SCHEMA}: {set(LANGGRAPH_TABLES) - lg_tables}"
        )
        assert public_after - public_before == set(), (
            f"LangGraph leaked tables into public: {public_after - public_before}"
        )
    finally:
        result.pool.close()


def test_checkpoint_and_namespaced_store_round_trip(
    clean_langgraph_schema: FundConfig,
) -> None:
    from langgraph.graph import END, START, StateGraph

    result = persistence.setup_langgraph(clean_langgraph_schema)
    try:

        class S(TypedDict):
            n: int

        def bump(state: S) -> S:
            return {"n": state["n"] + 1}

        graph = StateGraph(S)
        graph.add_node("bump", bump)
        graph.add_edge(START, "bump")
        graph.add_edge("bump", END)
        app = graph.compile(checkpointer=result.saver, store=result.store)

        cfg = {"configurable": {"thread_id": "t-d3"}}
        out = app.invoke({"n": 41}, cfg)
        assert out["n"] == 42
        assert app.get_state(cfg).values["n"] == 42

        # Store namespaced per portfolio_id (D1 scoping).
        ns = ("portfolio", "PF-TEST", "preferences")
        result.store.put(ns, "constraint_set", {"A_gamma": 3.5})
        item = result.store.get(ns, "constraint_set")
        assert item is not None
        assert item.value["A_gamma"] == 3.5
    finally:
        result.pool.close()
