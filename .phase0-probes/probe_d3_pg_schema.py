"""Phase 0 exit probe — D3.

Verify LangGraph checkpointer + store land their tables in a dedicated `langgraph`
Postgres schema (NOT Alembic-managed `public`) using the search_path mechanism,
and that a checkpoint + a store item round-trip.

Run (isolated, no workspace mutation):
  uv run --isolated --no-project \
    --with langgraph --with langgraph-checkpoint-postgres \
    --with "psycopg[binary]" --with psycopg-pool \
    python .phase0-probes/probe_d3_pg_schema.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TypedDict

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = "langgraph"


def load_database_url() -> str:
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL=") and not line.startswith("#"):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            # psycopg wants postgresql:// (strip any +driver)
            return re.sub(r"^postgresql\+\w+://", "postgresql://", v)
    raise SystemExit("DATABASE_URL not found in .env")


def main() -> int:
    db_url = load_database_url()
    masked = re.sub(r"//[^@]+@", "//***@", db_url)
    print(f"[D3] DATABASE_URL = {masked}")

    # 1. Create the dedicated schema out-of-band (search_path only redirects).
    with psycopg.connect(db_url, autocommit=True) as conn:
        # snapshot public tables BEFORE, to prove we add nothing there
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public'"
            )
            public_before = {r["tablename"] for r in cur.fetchall()}
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        conn.execute(f"CREATE SCHEMA {SCHEMA}")
    print(f"[D3] created schema '{SCHEMA}'; public had {len(public_before)} tables before")

    # 2. Shared pool forcing search_path=langgraph (+ setup() requirements).
    pool = ConnectionPool(
        conninfo=db_url,
        min_size=1,
        max_size=4,
        open=True,
        kwargs={
            "options": f"-c search_path={SCHEMA}",
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
        },
    )

    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    saver = PostgresSaver(pool)
    store = PostgresStore(pool)
    saver.setup()
    store.setup()
    print("[D3] saver.setup() + store.setup() OK")

    # 3. Assert tables landed in langgraph, not public.
    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname=%s ORDER BY tablename",
                (SCHEMA,),
            )
            lg_tables = [r["tablename"] for r in cur.fetchall()]
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public'"
            )
            public_after = {r["tablename"] for r in cur.fetchall()}

    print(f"[D3] tables in '{SCHEMA}': {lg_tables}")
    leaked = public_after - public_before
    print(f"[D3] new tables leaked into public: {sorted(leaked) or 'NONE'}")

    have_ckpt = any("checkpoint" in t for t in lg_tables)
    have_store = any(t == "store" for t in lg_tables)
    assert have_ckpt, "checkpoint tables missing from langgraph schema"
    assert have_store, "store table missing from langgraph schema"
    assert not leaked, f"LangGraph leaked tables into public: {leaked}"

    # 4. Checkpoint round-trip through a trivial graph.
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        n: int

    def bump(state: S) -> S:
        return {"n": state["n"] + 1}

    g = StateGraph(S)
    g.add_node("bump", bump)
    g.add_edge(START, "bump")
    g.add_edge("bump", END)
    app = g.compile(checkpointer=saver, store=store)

    cfg = {"configurable": {"thread_id": "probe-d3"}}
    out = app.invoke({"n": 41}, cfg)
    snap = app.get_state(cfg)
    assert out["n"] == 42, out
    assert snap.values["n"] == 42, snap.values
    print(f"[D3] checkpoint round-trip OK (invoke={out['n']}, persisted={snap.values['n']})")

    # 5. Store round-trip, namespaced per portfolio_id (D1).
    ns = ("portfolio", "PF-TEST", "preferences")
    store.put(ns, "constraint_set", {"A_gamma": 3.5})
    item = store.get(ns, "constraint_set")
    assert item is not None and item.value["A_gamma"] == 3.5, item
    print(f"[D3] store round-trip OK (namespace={ns}, value={item.value})")

    # 6. Cleanup — drop the probe schema so the real DB is left untouched.
    pool.close()
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    print(f"[D3] cleaned up: dropped schema '{SCHEMA}'")

    print("\n[D3] PASS — search_path schema isolation + checkpoint/store round-trip verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
