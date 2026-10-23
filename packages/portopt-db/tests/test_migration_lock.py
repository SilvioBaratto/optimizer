"""Concurrent ``alembic upgrade head`` runs must serialise on a PostgreSQL lock.

The scheduler and fund containers both run ``alembic upgrade head`` from their
entrypoints and are started together by one ``docker compose up``. Without a
lock both read the same ``alembic_version`` and race on identical DDL: the
loser dies with a duplicate relation/type error and crashes the container.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from alembic import command
from portopt_db import migration_lock

_ENV_PY = Path(__file__).resolve().parents[1] / "alembic" / "env.py"

_NOOP_REVISION = '''\
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op revision; only the env.py wiring is under test."""


def downgrade() -> None:
    """No-op revision; only the env.py wiring is under test."""
'''


def _fake_connection(dialect_name: str) -> MagicMock:
    connection = MagicMock()
    connection.dialect.name = dialect_name
    return connection


def test_acquire_migration_lock_takes_a_transaction_scoped_advisory_lock_on_postgres():
    """The lock dies with the migration transaction, so a crash cannot leak it."""
    connection = _fake_connection("postgresql")

    migration_lock.acquire_migration_lock(connection)

    connection.execute.assert_called_once()
    statement, params = connection.execute.call_args.args
    assert "pg_advisory_xact_lock" in str(statement)
    assert params == {"key": migration_lock.MIGRATION_LOCK_KEY}


def test_migration_lock_key_fits_a_signed_bigint():
    """``pg_advisory_xact_lock(bigint)`` rejects keys outside the signed range."""
    assert 0 < migration_lock.MIGRATION_LOCK_KEY < 2**63


def test_acquire_migration_lock_is_a_noop_off_postgres():
    """SQLite has no advisory locks; migrating a dev/test database must still work."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as connection:
        migration_lock.acquire_migration_lock(connection)
        assert connection.execute(text("SELECT 1")).scalar() == 1


def test_env_py_takes_the_lock_before_alembic_reads_the_version_table(
    tmp_path, monkeypatch
):
    """A loser that locks only after reading ``alembic_version`` would still race."""
    script_dir = tmp_path / "alembic"
    (script_dir / "versions").mkdir(parents=True)
    shutil.copy(_ENV_PY, script_dir / "env.py")
    (script_dir / "versions" / "0001_noop.py").write_text(_NOOP_REVISION)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'migrate.db'}")
    events: list[str] = []

    def spy(connection) -> None:
        """Record when env.py asks for the lock."""
        events.append("lock")

    monkeypatch.setattr(migration_lock, "acquire_migration_lock", spy)

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        """Record every statement that touches the version table."""
        if "alembic_version" in statement:
            events.append("version-table")

    event.listen(Engine, "before_cursor_execute", record)
    try:
        cfg = Config()
        cfg.set_main_option("script_location", str(script_dir))
        command.upgrade(cfg, "head")
    finally:
        event.remove(Engine, "before_cursor_execute", record)

    assert events.count("lock") == 1
    assert "version-table" in events
    assert events.index("lock") < events.index("version-table")


@pytest.mark.parametrize("dialect_name", ["sqlite", "mysql"])
def test_acquire_migration_lock_never_executes_off_postgres(dialect_name):
    """Only the PostgreSQL dialect understands ``pg_advisory_xact_lock``."""
    connection = _fake_connection(dialect_name)

    migration_lock.acquire_migration_lock(connection)

    connection.execute.assert_not_called()
