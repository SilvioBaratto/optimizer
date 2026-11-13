"""Fixtures for the PostgreSQL integration suite.

The suite needs a real server because the code under test is PostgreSQL-specific:
``ON CONFLICT ON CONSTRAINT`` (SQLite drops the target) and the append-only
triggers on ``human_audit_log``. It is keyed on its own ``PORTOPT_TEST_DATABASE_URL``
and refuses any database whose name does not end in ``_test``, so it can never
run against the live ``optimizer_db``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, make_url
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from alembic import command

_URL_VARIABLE = "PORTOPT_TEST_DATABASE_URL"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def pg_url() -> str:
    """Return the scratch database URL, or skip when none is configured."""
    url = os.environ.get(_URL_VARIABLE, "")
    if not url:
        pytest.skip(f"{_URL_VARIABLE} is not set")
    database = make_url(url).database or ""
    if not database.endswith("_test"):
        pytest.fail(
            f"{_URL_VARIABLE} must name a scratch database ending in '_test', "
            f"got {database!r}"
        )
    live = os.environ.get("DATABASE_URL")
    if live and make_url(live) == make_url(url):
        pytest.fail(f"{_URL_VARIABLE} must differ from DATABASE_URL")
    return url


@pytest.fixture(scope="session")
def pg_engine(pg_url: str) -> Iterator[Engine]:
    """Migrate the scratch database to head and yield an engine on it."""
    config = Config(str(_PACKAGE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_PACKAGE_ROOT / "alembic"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", pg_url)
        command.upgrade(config, "head")
    engine = create_engine(pg_url)
    yield engine
    engine.dispose()


@pytest.fixture
def pg_connection(pg_engine: Engine) -> Iterator[Connection]:
    """A connection inside an outer transaction that is rolled back after the test."""
    with pg_engine.connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


@pytest.fixture
def pg_session(pg_connection: Connection) -> Iterator[Session]:
    """A session bound to the rolled-back connection; ``_upsert`` savepoints nest in it."""
    with Session(
        bind=pg_connection, join_transaction_mode="create_savepoint"
    ) as session:
        yield session
