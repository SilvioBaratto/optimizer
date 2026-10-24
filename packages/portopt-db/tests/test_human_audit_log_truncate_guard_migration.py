"""Tests for migration 77f236075b2e: block TRUNCATE on ``human_audit_log``.

``f1a2b3c4d5e6`` guards the table with a row-level ``BEFORE UPDATE OR DELETE``
trigger, but PostgreSQL never fires row-level triggers for ``TRUNCATE``, so the
"append-only" audit trail could still be emptied in one statement. The revision
adds a statement-level ``BEFORE TRUNCATE`` trigger on the same function.

PostgreSQL DDL is checked by rendering the revision offline against the
PostgreSQL dialect; SQLite needs no guard because it has no ``TRUNCATE`` (its
``DELETE`` trigger already covers the statement).
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import ModuleType

from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "77f236075b2e_block_truncate_on_human_audit_log.py"
MIGRATION_PATH = str(
    (
        Path(__file__).parent / ".." / "alembic" / "versions" / MIGRATION_FILENAME
    ).resolve()
)

EXPECTED_REVISION = "77f236075b2e"
EXPECTED_DOWN_REVISION = "c8d0bb63d826"


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "human_audit_log_truncate_guard_migration", MIGRATION_PATH
    )
    assert spec and spec.loader, f"Cannot find migration at {MIGRATION_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render_postgres(step: str) -> str:
    """Return the SQL ``upgrade()`` or ``downgrade()`` renders for PostgreSQL."""
    buffer = io.StringIO()
    ctx = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": buffer},
    )
    mod = _load_migration()
    with Operations.context(ctx):
        getattr(mod, step)()
    return " ".join(buffer.getvalue().split())


def test_revision_chains_from_the_previous_revision() -> None:
    """The revision sits directly on the previous Alembic head."""
    mod = _load_migration()
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_installs_a_statement_level_before_truncate_trigger() -> None:
    """Only a ``FOR EACH STATEMENT`` trigger fires on ``TRUNCATE``."""
    sql = _render_postgres("upgrade")

    assert "CREATE TRIGGER human_audit_log_block_truncate" in sql
    assert "BEFORE TRUNCATE ON human_audit_log" in sql
    assert "FOR EACH STATEMENT" in sql


def test_upgrade_reuses_the_append_only_exception_function() -> None:
    """The refusal message and function stay shared with the row-level guard."""
    sql = _render_postgres("upgrade")

    assert "EXECUTE FUNCTION human_audit_log_append_only()" in sql


def test_upgrade_replaces_an_existing_trigger_so_it_can_be_replayed() -> None:
    """A stale trigger is dropped first, since CREATE TRIGGER has no IF NOT EXISTS."""
    sql = _render_postgres("upgrade")

    drop = "DROP TRIGGER IF EXISTS human_audit_log_block_truncate ON human_audit_log"
    assert drop in sql
    assert sql.index(drop) < sql.index("CREATE TRIGGER human_audit_log_block_truncate")


def test_downgrade_drops_only_the_truncate_trigger() -> None:
    """The row-level guard and its function belong to the earlier revision."""
    sql = _render_postgres("downgrade")

    assert (
        "DROP TRIGGER IF EXISTS human_audit_log_block_truncate ON human_audit_log"
        in sql
    )
    assert "human_audit_log_block_mutate" not in sql
    assert "DROP FUNCTION" not in sql


def test_both_steps_are_noops_on_sqlite() -> None:
    """SQLite has no TRUNCATE, and its DELETE trigger already blocks the statement."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    mod = _load_migration()

    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            mod.upgrade()
            mod.downgrade()
