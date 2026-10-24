"""D3 LangGraph persistence, integration slice (Postgres-only).

Proves the SPEC D3 contract against a real database: ``setup_langgraph`` lands
all six LangGraph tables in a dedicated schema with **zero** leak into Alembic's
``public``, and a checkpoint + a namespaced store item round-trip.

Marked ``integration`` and skipped unless ``FUND_TEST_DATABASE_URL`` names a
scratch Postgres, so the default unit run (and CI without Postgres) stays green.
The ambient ``DATABASE_URL`` is deliberately ignored: importing ``fund.config``
reloads the repo ``.env`` over it, so on a developer machine it is the live
compose database. The test owns a throwaway schema (never the fund's
``langgraph``) which it drops before and after, and it refuses a target that
resolves to the ambient database.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import replace
from typing import TypedDict

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier

from fund.audit import persistence
from fund.audit.persistence import LANGGRAPH_TABLES
from fund.config import FundConfig, load_config, settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("FUND_TEST_DATABASE_URL"),
        reason="D3 integration test requires FUND_TEST_DATABASE_URL (scratch Postgres)",
    ),
]

# Never "langgraph": that is the schema the running fund checkpoints into.
SCHEMA = "langgraph_it"

_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def _endpoint(url: str) -> tuple[str, str, str]:
    """Reduce a database URL to (host, port, dbname), folding local host aliases."""
    info = conninfo_to_dict(persistence.normalize_conninfo(url))
    host = info.get("host", "")
    host = "localhost" if host in _LOCAL_HOSTS else host
    dbname = info.get("dbname") or info.get("user") or ""
    return host, str(info.get("port") or "5432"), dbname


def _target_url() -> str:
    """Return the scratch database URL, refusing the ambient (live) database."""
    target = os.environ["FUND_TEST_DATABASE_URL"]
    ambient = {
        url for url in (os.environ.get("DATABASE_URL"), settings.database_url) if url
    }
    if any(_endpoint(url) == _endpoint(target) for url in ambient):
        pytest.fail(
            "refusing to run: FUND_TEST_DATABASE_URL points at the same database "
            "as DATABASE_URL; use a separate scratch database",
            pytrace=False,
        )
    return target


def _conninfo() -> str:
    return persistence.normalize_conninfo(_target_url())


def _drop_schema() -> None:
    with psycopg.connect(_conninfo(), autocommit=True) as conn:
        conn.execute(SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(Identifier(SCHEMA)))


def _tables_in(schema: str) -> set[str]:
    with (
        psycopg.connect(_conninfo(), autocommit=True) as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname=%s", (schema,))
        return {r["tablename"] for r in cur.fetchall()}


@pytest.fixture
def clean_langgraph_schema() -> Generator[FundConfig, None, None]:
    """Drop any stale throwaway schema, yield a config bound to it, drop again."""
    # An injected env skips the .env reload, so the config cannot be redirected
    # to the ambient database.
    cfg = replace(
        load_config(env={"DATABASE_URL": _target_url()}, load_dotenv_file=False),
        langgraph_schema=SCHEMA,
    )
    _drop_schema()
    try:
        yield cfg
    finally:
        _drop_schema()


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
