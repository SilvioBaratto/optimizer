"""LangGraph checkpointer/store bootstrap on a dedicated ``langgraph`` schema (D3).

The Python ``langgraph-checkpoint-postgres`` package has **no** ``schema=`` param
(GitHub issue #7345); its DDL uses unqualified table names resolved by the
session ``search_path``. To keep the six LangGraph tables out of Alembic's
``public`` we:

1. ``CREATE SCHEMA IF NOT EXISTS langgraph`` **out-of-band** — ``search_path`` only
   *redirects* unqualified DDL, it does not create the schema.
2. Build one psycopg :class:`~psycopg_pool.ConnectionPool` forcing
   ``search_path=langgraph`` plus ``.setup()``'s requirements (``autocommit=True``
   so ``CREATE INDEX CONCURRENTLY`` works, ``row_factory=dict_row``,
   ``prepare_threshold=0``) — see :meth:`FundConfig.langgraph_pool_kwargs`.
3. Run ``.setup()`` for both ``PostgresSaver`` and ``PostgresStore`` on that one
   pool (SPEC D3: one pool hosts both).

Postgres-only; the round-trip is covered by the marked integration test. This
module deliberately imports nothing from ``optimizer`` — the ``fund`` boundary
allows it, but persistence needs only ``portopt_db``-adjacent primitives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg import Connection
from psycopg.rows import DictRow
from psycopg.sql import SQL, Identifier
from psycopg_pool import ConnectionPool

from fund.config import FundConfig, settings

# The six tables `.setup()` creates: four for the saver, two for the store.
# Asserted table-for-table by the D3 integration test (all in `langgraph`).
LANGGRAPH_TABLES: tuple[str, ...] = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
    "store",
    "store_migrations",
)


@dataclass(frozen=True)
class LangGraphPersistence:
    """The bootstrapped pool + both persistence handles (share one pool)."""

    pool: ConnectionPool[Connection[DictRow]]
    saver: PostgresSaver
    store: PostgresStore


def normalize_conninfo(database_url: str) -> str:
    """Strip a SQLAlchemy ``+driver`` so psycopg accepts the URL.

    ``DATABASE_URL`` may carry an SQLAlchemy driver suffix (e.g.
    ``postgresql+psycopg2://``); psycopg wants a bare ``postgresql://``.
    """
    return re.sub(r"^postgresql\+\w+://", "postgresql://", database_url)


def _require_conninfo(config: FundConfig) -> str:
    """Return the psycopg conninfo, or raise if ``DATABASE_URL`` is unset.

    Validation is at use-time (not import-time): ``fund.config`` keeps secrets
    ``None`` so a bare import never fails in CI.
    """
    if not config.database_url:
        raise RuntimeError(
            "DATABASE_URL is required to bootstrap LangGraph persistence; "
            "set it in the environment."
        )
    return normalize_conninfo(config.database_url)


def create_schema(conninfo: str, schema: str) -> None:
    """``CREATE SCHEMA IF NOT EXISTS`` on a throwaway autocommit connection.

    Out-of-band because ``search_path`` redirects unqualified DDL but does not
    create the target schema.
    """
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(SQL("CREATE SCHEMA IF NOT EXISTS {}").format(Identifier(schema)))


def build_pool(config: FundConfig) -> ConnectionPool[Connection[DictRow]]:
    """Open the shared pool forcing ``search_path=langgraph`` + setup requirements.

    ``row_factory=dict_row`` is set at runtime via ``kwargs`` (invisible to the
    type checker, which infers the default tuple-row pool), so the dict-row pool
    type is asserted with a ``cast``.
    """
    conninfo = _require_conninfo(config)
    pool = ConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=4,
        open=True,
        kwargs=config.langgraph_pool_kwargs(),
    )
    return cast("ConnectionPool[Connection[DictRow]]", pool)


def setup_langgraph(config: FundConfig | None = None) -> LangGraphPersistence:
    """Create the schema, open the pool, and ``.setup()`` the saver + store.

    Args:
        config: Fund config; defaults to the module-level ``settings``.

    Returns:
        The pool and both persistence handles. The caller owns the pool's
        lifetime (call ``result.pool.close()`` when done).
    """
    cfg = config if config is not None else settings
    conninfo = _require_conninfo(cfg)
    create_schema(conninfo, cfg.langgraph_schema)
    pool = build_pool(cfg)
    saver = PostgresSaver(pool)
    store = PostgresStore(pool)
    saver.setup()
    store.setup()
    return LangGraphPersistence(pool=pool, saver=saver, store=store)


__all__ = [
    "LANGGRAPH_TABLES",
    "LangGraphPersistence",
    "build_pool",
    "create_schema",
    "normalize_conninfo",
    "setup_langgraph",
]
