"""``fund.audit.persistence`` LangGraph bootstrap (unit slice).

The Postgres-touching round-trip is proved by the marked integration test
(``fund/tests/integration/test_langgraph_persistence.py``). These unit tests
drive the pure/orchestration surface with fakes so no live database is needed:
conninfo normalisation, the missing-``DATABASE_URL`` use-time guard, the
out-of-band ``CREATE SCHEMA`` DDL, the pool kwargs, and the
``create_schema → build_pool → saver.setup → store.setup`` sequence.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from psycopg.sql import SQL, Identifier

from fund.audit import persistence
from fund.config import FundConfig


@contextmanager
def _no_lock(_conninfo: str) -> Iterator[None]:
    yield


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
        check_connection = staticmethod(lambda _conn: None)

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
        def __init__(self, pool: object, **_kwargs: object) -> None:
            self.pool = pool

        def setup(self) -> None:
            order.append("saver.setup")

    class _FakeStore:
        def __init__(self, pool: object) -> None:
            self.pool = pool

        def setup(self) -> None:
            order.append("store.setup")

    monkeypatch.setattr(persistence, "_bootstrap_lock", _no_lock)
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


@pytest.mark.parametrize("failing", ["saver", "store"])
def test_setup_langgraph_closes_the_pool_when_setup_raises(monkeypatch, failing):
    class _Pool:
        closed = False

        def close(self) -> None:
            self.closed = True

    pool = _Pool()

    def _make(name: str):
        class _Handle:
            def __init__(self, _pool: object, **_kwargs: object) -> None:
                pass

            def setup(self) -> None:
                if name == failing:
                    raise RuntimeError(f"{name} migration failed")

        return _Handle

    monkeypatch.setattr(persistence, "_bootstrap_lock", _no_lock)
    monkeypatch.setattr(persistence, "create_schema", lambda *_a: None)
    monkeypatch.setattr(persistence, "build_pool", lambda _config: pool)
    monkeypatch.setattr(persistence, "PostgresSaver", _make("saver"))
    monkeypatch.setattr(persistence, "PostgresStore", _make("store"))

    cfg = FundConfig(database_url="postgresql+psycopg2://u:p@h:54320/db")
    with pytest.raises(RuntimeError, match=f"{failing} migration failed"):
        persistence.setup_langgraph(cfg)

    assert pool.closed


_tripwire_calls: list[object] = []


def _tripwire(arg: object) -> None:
    """Stands in for a code-execution gadget a forged checkpoint blob would name."""
    _tripwire_calls.append(arg)


def _forged_constructor_blob() -> tuple[str, bytes]:
    """A checkpoint blob whose msgpack ext asks the loader to call ``_tripwire``."""
    ormsgpack = pytest.importorskip("ormsgpack")
    payload = ormsgpack.packb((__name__, "_tripwire", "forged"))
    return "msgpack", ormsgpack.packb(ormsgpack.Ext(0, payload))


def test_the_checkpointer_does_not_execute_constructors_named_by_a_stored_blob(
    monkeypatch,
):
    # Anyone able to write the langgraph schema can plant a blob; the permissive
    # default serializer would import and call whatever it names when the TUI polls.
    monkeypatch.setattr(persistence, "_bootstrap_lock", _no_lock)
    monkeypatch.setattr(persistence, "create_schema", lambda *_a: None)
    monkeypatch.setattr(persistence, "build_pool", lambda _config: object())
    monkeypatch.setattr(persistence.PostgresSaver, "setup", lambda _self: None)
    monkeypatch.setattr(persistence.PostgresStore, "setup", lambda _self: None)
    _tripwire_calls.clear()

    cfg = FundConfig(database_url="postgresql+psycopg2://u:p@h:54320/db")
    saver = persistence.setup_langgraph(cfg).saver
    saver.serde.loads_typed(_forged_constructor_blob())

    assert _tripwire_calls == []


def test_the_checkpointer_still_round_trips_graph_state_messages(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    monkeypatch.setattr(persistence, "_bootstrap_lock", _no_lock)
    monkeypatch.setattr(persistence, "create_schema", lambda *_a: None)
    monkeypatch.setattr(persistence, "build_pool", lambda _config: object())
    monkeypatch.setattr(persistence.PostgresSaver, "setup", lambda _self: None)
    monkeypatch.setattr(persistence.PostgresStore, "setup", lambda _self: None)

    cfg = FundConfig(database_url="postgresql+psycopg2://u:p@h:54320/db")
    serde = persistence.setup_langgraph(cfg).saver.serde
    state = {"messages": [HumanMessage("hi"), AIMessage("ok")]}

    assert serde.loads_typed(serde.dumps_typed(state)) == state


def test_build_pool_health_checks_a_connection_before_handing_it_out(monkeypatch):
    # Without a check the pool returns a socket that died with a Postgres restart,
    # and the first checkpoint read or write after it raises OperationalError.
    from psycopg_pool import ConnectionPool

    captured: dict = {}

    class _FakePool:
        check_connection = staticmethod(ConnectionPool.check_connection)

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(persistence, "ConnectionPool", _FakePool)

    persistence.build_pool(FundConfig(database_url="postgresql://u:p@h:54320/db"))

    assert captured["check"] is ConnectionPool.check_connection


def test_setup_langgraph_serialises_bootstrap_across_processes(monkeypatch):
    # saver.setup() reads the migration version and then inserts it with no lock, so
    # two processes starting together both insert v=0 and the loser dies on a
    # UniqueViolation.
    events: list[str] = []

    class _FakeConn:
        def __enter__(self) -> _FakeConn:
            events.append("connect")
            return self

        def __exit__(self, *_exc: object) -> None:
            events.append("disconnect")

        def execute(self, stmt: object, *_args: object) -> None:
            events.append(f"execute:{stmt}")

    def _make(name: str):
        class _Handle:
            def __init__(self, _pool: object, **_kwargs: object) -> None:
                pass

            def setup(self) -> None:
                events.append(f"{name}.setup")

        return _Handle

    monkeypatch.setattr(persistence.psycopg, "connect", lambda *_a, **_k: _FakeConn())
    monkeypatch.setattr(
        persistence, "create_schema", lambda *_a: events.append("schema")
    )
    monkeypatch.setattr(persistence, "build_pool", lambda _config: object())
    monkeypatch.setattr(persistence, "PostgresSaver", _make("saver"))
    monkeypatch.setattr(persistence, "PostgresStore", _make("store"))

    persistence.setup_langgraph(FundConfig(database_url="postgresql://u:p@h:54320/db"))

    locked = next(i for i, e in enumerate(events) if "pg_advisory_lock" in e)
    released = events.index("disconnect")
    assert locked < events.index("schema")
    assert events.index("store.setup") < released
