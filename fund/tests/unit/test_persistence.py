"""T2.3 — ``fund.audit.persistence`` LangGraph bootstrap (unit slice).

The Postgres-touching round-trip is proved by the marked integration test
(``fund/tests/integration/test_langgraph_persistence.py``). These unit tests
drive the pure/orchestration surface with fakes so no live database is needed:
conninfo normalisation, the missing-``DATABASE_URL`` use-time guard, the
out-of-band ``CREATE SCHEMA`` DDL, the pool kwargs, and the
``create_schema → build_pool → saver.setup → store.setup`` sequence.
"""

from __future__ import annotations

import pytest
from psycopg.sql import SQL, Identifier

from fund.audit import persistence
from fund.config import FundConfig


def test_normalize_conninfo_strips_sqlalchemy_driver():
    assert (
        persistence.normalize_conninfo("postgresql+psycopg2://u:p@h:54320/db")
        == "postgresql://u:p@h:54320/db"
    )


def test_normalize_conninfo_leaves_a_plain_postgresql_url_unchanged():
    assert (
        persistence.normalize_conninfo("postgresql://u:p@h:54320/db")
        == "postgresql://u:p@h:54320/db"
    )


def test_build_pool_without_database_url_raises():
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        persistence.build_pool(FundConfig(database_url=None))


def test_setup_langgraph_without_database_url_raises():
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        persistence.setup_langgraph(FundConfig(database_url=None))


def test_create_schema_issues_idempotent_create_schema_ddl(monkeypatch):
    executed: list[object] = []

    class _FakeConn:
        def __enter__(self) -> _FakeConn:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def execute(self, stmt: object) -> None:
            executed.append(stmt)

    connect_calls: list[dict] = []

    def _fake_connect(conninfo: str, **kwargs: object) -> _FakeConn:
        connect_calls.append({"conninfo": conninfo, **kwargs})
        return _FakeConn()

    monkeypatch.setattr(persistence.psycopg, "connect", _fake_connect)

    persistence.create_schema("postgresql://u:p@h:54320/db", "langgraph")

    # out-of-band connection must be autocommit (search_path only redirects).
    assert connect_calls == [
        {"conninfo": "postgresql://u:p@h:54320/db", "autocommit": True}
    ]
    expected = SQL("CREATE SCHEMA IF NOT EXISTS {}").format(Identifier("langgraph"))
    assert executed == [expected]


def test_build_pool_forces_search_path_kwargs(monkeypatch):
    captured: dict = {}

    class _FakePool:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(persistence, "ConnectionPool", _FakePool)

    cfg = FundConfig(database_url="postgresql+psycopg2://u:p@h:54320/db")
    persistence.build_pool(cfg)

    assert captured["conninfo"] == "postgresql://u:p@h:54320/db"
    assert captured["open"] is True
    assert captured["kwargs"] == cfg.langgraph_pool_kwargs()


def test_setup_langgraph_orchestrates_schema_then_saver_then_store(monkeypatch):
    order: list[str] = []
    sentinel_pool = object()

    def _fake_create_schema(conninfo: str, schema: str) -> None:
        order.append(f"schema:{conninfo}:{schema}")

    def _fake_build_pool(_config: FundConfig) -> object:
        order.append("pool")
        return sentinel_pool

    class _FakeSaver:
        def __init__(self, pool: object) -> None:
            self.pool = pool

        def setup(self) -> None:
            order.append("saver.setup")

    class _FakeStore:
        def __init__(self, pool: object) -> None:
            self.pool = pool

        def setup(self) -> None:
            order.append("store.setup")

    monkeypatch.setattr(persistence, "create_schema", _fake_create_schema)
    monkeypatch.setattr(persistence, "build_pool", _fake_build_pool)
    monkeypatch.setattr(persistence, "PostgresSaver", _FakeSaver)
    monkeypatch.setattr(persistence, "PostgresStore", _FakeStore)

    cfg = FundConfig(database_url="postgresql+psycopg2://u:p@h:54320/db")
    result = persistence.setup_langgraph(cfg)

    # schema created out-of-band FIRST, then the pooled saver + store setup.
    assert order == [
        "schema:postgresql://u:p@h:54320/db:langgraph",
        "pool",
        "saver.setup",
        "store.setup",
    ]
    assert result.pool is sentinel_pool
    assert isinstance(result.saver, _FakeSaver)
    assert isinstance(result.store, _FakeStore)
    assert result.saver.pool is sentinel_pool
    assert result.store.pool is sentinel_pool
