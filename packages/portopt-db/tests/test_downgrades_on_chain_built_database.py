"""Downgrades that must also work on a database built by the chain itself.

``alembic upgrade head`` from an empty database never runs the legacy-schema
branches of several early revisions (the initial revision already creates the
ORM shape, and the macro tables are not created until a later revision). Their
``downgrade()`` bodies nevertheless assumed the legacy shape, so stepping such a
database down through them aborted on a missing constraint or table, on a
cast PostgreSQL refuses to infer, or left a column behind that the matching
``upgrade()`` could no longer drop.

Offline rendering against the PostgreSQL dialect covers the statements that
need no inspection; a mocked ``op`` bound to a real SQLite connection covers the
guards that read the live schema. The upgrades of ``c3d4e5f6a7b8`` and
``b0b30daa80c5`` inspect the database, so only their downgrades can render
offline. An offline render has nothing to inspect and must therefore emit the
complete legacy reversal rather than silently dropping part of it: a script that
looks finished but is not would leave a legacy database half reverted.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.pool import StaticPool

_VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"

_MACRO_TABLES = ("bond_yields", "economic_indicators", "trading_economics_indicators")


def _load(filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"downgrade_probe_{filename[:-3]}", _VERSIONS / filename
    )
    assert spec and spec.loader, f"Cannot find migration {filename}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render_postgres_downgrade(mod: ModuleType) -> str:
    """Return the SQL ``mod.downgrade()`` renders for PostgreSQL in offline mode."""
    buffer = io.StringIO()
    ctx = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": buffer},
    )
    with Operations.context(ctx):
        mod.downgrade()
    return buffer.getvalue()


@pytest.fixture
def conn() -> Connection:
    """A connection to an empty in-memory SQLite database."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as connection:
        yield connection


def _run_downgrade_with_mocked_op(mod: ModuleType, conn: Connection) -> MagicMock:
    """Run ``mod.downgrade()`` with ``op`` mocked and bound to ``conn`` for inspection."""
    mock_op = MagicMock()
    mock_op.get_bind.return_value = conn
    mock_op.get_context.return_value.as_sql = False
    mod.downgrade.__globals__["op"] = mock_op
    mod.downgrade()
    return mock_op


def _ddl_calls(mock_op: MagicMock) -> list[str]:
    """Names of the ``op`` calls that emit DDL, ignoring bind/context lookups."""
    return [
        call[0]
        for call in mock_op.method_calls
        if call[0] not in {"get_bind", "get_context"}
    ]


# ---------------------------------------------------------------------------
# c3d4e5f6a7b8: align_exchanges_with_orm
# ---------------------------------------------------------------------------

_EXCHANGES_MIGRATION = "c3d4e5f6a7b8_align_exchanges_with_orm.py"


def test_exchanges_downgrade_is_a_noop_when_upgrade_found_the_orm_shape(
    conn: Connection,
) -> None:
    """A chain-built exchanges table carries an unnamed unique key, not ours."""
    conn.execute(
        text(
            "CREATE TABLE exchanges ("
            "id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, t212_id INTEGER)"
        )
    )

    mock_op = _run_downgrade_with_mocked_op(_load(_EXCHANGES_MIGRATION), conn)

    assert _ddl_calls(mock_op) == []


def test_exchanges_downgrade_still_reverses_a_legacy_conversion(
    conn: Connection,
) -> None:
    """When upgrade created ``uq_exchanges_name`` the downgrade must drop it."""
    conn.execute(
        text(
            "CREATE TABLE exchanges ("
            "id TEXT PRIMARY KEY, name TEXT NOT NULL, t212_id INTEGER, "
            "CONSTRAINT uq_exchanges_name UNIQUE (name))"
        )
    )

    mock_op = _run_downgrade_with_mocked_op(_load(_EXCHANGES_MIGRATION), conn)

    mock_op.drop_constraint.assert_called_once_with(
        "uq_exchanges_name", "exchanges", type_="unique"
    )
    assert "add_column" in _ddl_calls(mock_op)


def test_exchanges_downgrade_renders_the_full_legacy_reversal_offline() -> None:
    """Offline mode cannot inspect, so it must not emit an empty-looking script."""
    sql = " ".join(_render_postgres_downgrade(_load(_EXCHANGES_MIGRATION)).split())

    assert "DROP CONSTRAINT uq_exchanges_name" in sql
    assert "ADD COLUMN is_active" in sql
    assert "RENAME name TO exchange_name" in sql


# ---------------------------------------------------------------------------
# 1c95f3fd3da4: remove_jsonb_columns_raw_json_and_related_tickers
# ---------------------------------------------------------------------------


def test_related_tickers_downgrade_casts_text_back_to_jsonb_explicitly() -> None:
    """PostgreSQL refuses an implicit text-to-jsonb cast without a USING clause."""
    sql = _render_postgres_downgrade(
        _load("1c95f3fd3da4_remove_jsonb_columns_raw_json_and_.py")
    )

    assert "ALTER COLUMN related_tickers TYPE JSONB USING" in sql
    assert "::jsonb" in sql


# ---------------------------------------------------------------------------
# b0b30daa80c5: normalize_and_optimize_models
# ---------------------------------------------------------------------------

_NORMALIZE_MIGRATION = "b0b30daa80c5_normalize_and_optimize_models.py"


def test_normalize_downgrade_skips_macro_tables_a_chain_built_database_lacks(
    conn: Connection,
) -> None:
    """The upgrade skipped the macro tables, so the downgrade must not touch them."""
    mock_op = _run_downgrade_with_mocked_op(_load(_NORMALIZE_MIGRATION), conn)

    touched = repr(mock_op.method_calls)
    for table in _MACRO_TABLES:
        assert table not in touched


def test_normalize_downgrade_still_reverts_macro_tables_that_exist(
    conn: Connection,
) -> None:
    """A legacy database that has the macro tables gets them reverted as before."""
    for table in _MACRO_TABLES:
        conn.execute(text(f"CREATE TABLE {table} (country TEXT)"))

    mock_op = _run_downgrade_with_mocked_op(_load(_NORMALIZE_MIGRATION), conn)

    touched = repr(mock_op.method_calls)
    for table in _MACRO_TABLES:
        assert table in touched


def test_normalize_downgrade_renders_the_macro_table_reversal_offline() -> None:
    """Offline mode cannot inspect, so the macro sections must not be dropped."""
    sql = " ".join(_render_postgres_downgrade(_load(_NORMALIZE_MIGRATION)).split())

    for table in _MACRO_TABLES:
        assert f"ALTER TABLE {table} ADD COLUMN scraped_at" in sql
        assert f"DROP INDEX ix_{table}_country" in sql
    assert (
        "ALTER TABLE bond_yields ALTER COLUMN reference_date TYPE varchar(100)" in sql
    )
    assert (
        "ALTER TABLE economic_indicators ALTER COLUMN reference_date TYPE varchar(100)"
        in sql
    )


# ---------------------------------------------------------------------------
# f6a7b8c9d0e1: drop_ilsole_real_columns
# ---------------------------------------------------------------------------


def test_ilsole_downgrade_restores_the_st_rate_forecast_column_its_upgrade_dropped() -> (
    None
):
    """Without the column the matching upgrade cannot drop it a second time."""
    sql = _render_postgres_downgrade(_load("f6a7b8c9d0e1_drop_ilsole_real_columns.py"))

    assert "ADD COLUMN IF NOT EXISTS st_rate_forecast" in sql
