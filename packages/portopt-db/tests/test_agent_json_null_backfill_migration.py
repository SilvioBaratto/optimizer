"""Backfill revision: JSON-literal ``null`` -> SQL NULL in the nullable agent columns.

Before the models set ``none_as_null=True`` the fund repos' explicit ``None`` was stored
as the JSON literal ``null``, so ``IS NULL`` filters missed every historical row. The
revision is replayed by path against in-memory SQLite (after the revision that creates
the tables), so it runs in CI without PostgreSQL; the ``CAST(col AS TEXT)`` predicate it
uses reads the literal back as ``null`` on both backends.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
CREATE_TABLES_FILENAME = "e9f0a1b2c3d4_add_agent_audit_tables.py"
MIGRATION_FILENAME = "f5c7e9a1b3d2_backfill_agent_json_sql_null.py"

EXPECTED_REVISION = "f5c7e9a1b3d2"
EXPECTED_DOWN_REVISION = "e4a6c8d0f2b1"

_JSON_NULL = "null"
_REAL_VALUE = '{"AAA": 1.0}'
_JSON_STRING_NULL = '"null"'

_RUN_LITERAL = "a" * 32
_RUN_SQL_NULL = "b" * 32
_RUN_VALUE = "c" * 32
_RUN_STRING = "d" * 32


def _load(filename: str) -> ModuleType:
    path = str((MIGRATION_DIR / filename).resolve())
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(engine, fn) -> None:
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _insert_run(conn, run_id: str, weights: str | None) -> None:
    conn.execute(
        text(
            "INSERT INTO agent_runs (id, asof, universe, optimizer_config, weights) "
            "VALUES (:id, '2026-01-02', '[]', '{}', :weights)"
        ),
        {"id": run_id, "weights": weights},
    )


def _insert_decision(
    conn, run_id: str, index: int, *, constraint_set, views, hitl_decision
) -> None:
    conn.execute(
        text(
            "INSERT INTO agent_decisions (id, run_id, decision_index, agent, step, "
            "constraint_set, views, hitl_decision) VALUES "
            "(:id, :run_id, :idx, 'pm', 'a', :constraint_set, :views, :hitl)"
        ),
        {
            "id": f"{index:032d}",
            "run_id": run_id,
            "idx": index,
            "constraint_set": constraint_set,
            "views": views,
            "hitl": hitl_decision,
        },
    )


@pytest.fixture
def engine():
    """SQLite with the audit tables, seeded with one row per storage shape."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _run(engine, _load(CREATE_TABLES_FILENAME).upgrade)
    with engine.begin() as conn:
        _insert_run(conn, _RUN_LITERAL, _JSON_NULL)
        _insert_run(conn, _RUN_SQL_NULL, None)
        _insert_run(conn, _RUN_VALUE, _REAL_VALUE)
        _insert_run(conn, _RUN_STRING, _JSON_STRING_NULL)
        # Decisions cover the four shapes a column can hold: legacy literal, SQL
        # NULL, a real value, and a JSON string spelling "null" that is not the literal.
        _insert_decision(
            conn,
            _RUN_LITERAL,
            0,
            constraint_set=_JSON_NULL,
            views=_JSON_NULL,
            hitl_decision=_JSON_NULL,
        )
        _insert_decision(
            conn,
            _RUN_LITERAL,
            1,
            constraint_set=None,
            views=None,
            hitl_decision=None,
        )
        _insert_decision(
            conn,
            _RUN_LITERAL,
            2,
            constraint_set=_REAL_VALUE,
            views=_REAL_VALUE,
            hitl_decision=_REAL_VALUE,
        )
        _insert_decision(
            conn,
            _RUN_LITERAL,
            3,
            constraint_set=_JSON_STRING_NULL,
            views=_JSON_STRING_NULL,
            hitl_decision=_JSON_STRING_NULL,
        )
    return engine


def _weights(engine) -> dict[str, str | None]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, weights FROM agent_runs")).all()
    return {row.id: row.weights for row in rows}


def _decision_columns(engine) -> dict[int, tuple[str | None, ...]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT decision_index, constraint_set, views, hitl_decision "
                "FROM agent_decisions"
            )
        ).all()
    return {row.decision_index: tuple(row[1:]) for row in rows}


def test_revision_chains_from_the_current_head() -> None:
    """The revision extends the single Alembic head."""
    mod = _load(MIGRATION_FILENAME)
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_turns_legacy_json_null_into_sql_null(engine) -> None:
    """Historical literal ``null`` rows become SQL NULL in every nullable column."""
    _run(engine, _load(MIGRATION_FILENAME).upgrade)

    weights = _weights(engine)
    assert weights[_RUN_LITERAL] is None
    assert weights[_RUN_SQL_NULL] is None
    decisions = _decision_columns(engine)
    assert decisions[0] == (None, None, None)
    assert decisions[1] == (None, None, None)


def test_upgrade_keeps_real_values_and_json_strings(engine) -> None:
    """Populated cells, including the JSON string ``"null"``, are left alone."""
    _run(engine, _load(MIGRATION_FILENAME).upgrade)

    weights = _weights(engine)
    assert weights[_RUN_VALUE] == _REAL_VALUE
    assert weights[_RUN_STRING] == _JSON_STRING_NULL
    decisions = _decision_columns(engine)
    assert decisions[2] == (_REAL_VALUE,) * 3
    assert decisions[3] == (_JSON_STRING_NULL,) * 3


def test_upgrade_is_idempotent(engine) -> None:
    """Running the backfill twice leaves the data as after the first pass."""
    mod = _load(MIGRATION_FILENAME)
    _run(engine, mod.upgrade)
    first = (_weights(engine), _decision_columns(engine))

    _run(engine, mod.upgrade)

    assert (_weights(engine), _decision_columns(engine)) == first


def test_downgrade_restores_the_legacy_literal_for_empty_values(engine) -> None:
    """Rolling back stores the legacy literal for every empty cell, real values stay."""
    mod = _load(MIGRATION_FILENAME)
    _run(engine, mod.upgrade)

    _run(engine, mod.downgrade)

    weights = _weights(engine)
    assert weights[_RUN_LITERAL] == _JSON_NULL
    assert weights[_RUN_SQL_NULL] == _JSON_NULL
    assert weights[_RUN_VALUE] == _REAL_VALUE
    decisions = _decision_columns(engine)
    assert decisions[0] == (_JSON_NULL,) * 3
    assert decisions[1] == (_JSON_NULL,) * 3
    assert decisions[2] == (_REAL_VALUE,) * 3
    assert decisions[3] == (_JSON_STRING_NULL,) * 3
