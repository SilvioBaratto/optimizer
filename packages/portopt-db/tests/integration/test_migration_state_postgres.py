"""``portopt_db.migration_state`` against a real PostgreSQL server and the shipped tree.

Each test gets its own scratch database on the server ``PORTOPT_TEST_DATABASE_URL``
names, created empty and dropped afterwards, so the session database the other
integration tests migrate to head is never touched. The revisions are applied with
Alembic's own ``upgrade`` command, so the stamp the check reads is the one a real
container leaves behind.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, make_url, text

from alembic import command
from portopt_db.config import pin_postgres_driver
from portopt_db.migration_state import (
    EXIT_PENDING,
    EXIT_PROCEED,
    MigrationState,
    inspect_migration_state,
)

pytestmark = pytest.mark.integration

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_DIR = _PACKAGE_ROOT / "alembic"
# The last revision before the two that delete data; the tree is append-only, so
# both stay the first two pending revisions above it whatever is added later.
_BEFORE_THE_DESTRUCTIVE_PAIR = "b8d0f2a4c6e1"
_DESTRUCTIVE_PAIR = ["2266e24c509f", "8ed6504aee6a"]
_SUBPROCESS_TIMEOUT_SECONDS = 120


@pytest.fixture
def scratch_url(pg_url: str) -> Iterator[str]:
    """Create an empty database next to the test database and drop it afterwards."""
    name = f"migration_state_{uuid.uuid4().hex[:12]}_test"
    admin = create_engine(pin_postgres_driver(pg_url), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield make_url(pg_url).set(database=name).render_as_string(hide_password=False)
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _upgrade(url: str, revision: str) -> None:
    config = Config(str(_PACKAGE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", url)
        command.upgrade(config, revision)


def _run_cli(url: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "DATABASE_URL": url}
    env.pop("PORTOPT_AUTO_MIGRATE", None)
    return subprocess.run(  # noqa: S603 - this interpreter running the module under test
        [
            sys.executable,
            "-m",
            "portopt_db.migration_state",
            "--alembic-dir",
            str(_ALEMBIC_DIR),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )


@pytest.mark.criterion("REQ-DB-006")
def test_a_new_database_is_empty_and_lets_the_container_migrate(
    scratch_url: str,
) -> None:
    """Validates portopt_db.migration_state.inspect_migration_state and main on a
    database no container has touched yet."""
    assert inspect_migration_state(scratch_url, _ALEMBIC_DIR).state is (
        MigrationState.EMPTY
    )
    assert _run_cli(scratch_url).returncode == EXIT_PROCEED


@pytest.mark.criterion("REQ-DB-006")
def test_a_database_behind_the_destructive_pair_is_pending_and_refused(
    scratch_url: str,
) -> None:
    """Validates portopt_db.migration_state.inspect_migration_state and main on the
    upgrade the legal clean-up ships: both deleting revisions are listed first and
    the module exits 3 without changing the stamp."""
    _upgrade(scratch_url, _BEFORE_THE_DESTRUCTIVE_PAIR)

    status = inspect_migration_state(scratch_url, _ALEMBIC_DIR)
    result = _run_cli(scratch_url)

    assert status.state is MigrationState.PENDING
    assert status.current == (_BEFORE_THE_DESTRUCTIVE_PAIR,)
    assert [step.revision for step in status.pending][:2] == _DESTRUCTIVE_PAIR
    assert status.pending[-1].revision == status.head
    assert "public.instruments" in status.tables
    assert result.returncode == EXIT_PENDING, result.stderr
    assert "portopt migrate" in result.stderr
    after = inspect_migration_state(scratch_url, _ALEMBIC_DIR)
    assert after.current == (_BEFORE_THE_DESTRUCTIVE_PAIR,)


@pytest.mark.criterion("REQ-DB-006")
def test_a_database_at_head_proceeds(scratch_url: str) -> None:
    """Validates portopt_db.migration_state.inspect_migration_state and main once
    every revision is applied."""
    _upgrade(scratch_url, "head")

    status = inspect_migration_state(scratch_url, _ALEMBIC_DIR)

    assert status.state is MigrationState.AT_HEAD
    assert status.current == (status.head,)
    assert _run_cli(scratch_url).returncode == EXIT_PROCEED
