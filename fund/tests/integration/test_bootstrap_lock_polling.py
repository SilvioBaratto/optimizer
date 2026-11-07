"""The bootstrap lock must not make a concurrent index build deadlock (Postgres-only).

LangGraph's ``.setup()`` runs ``CREATE INDEX CONCURRENTLY``, which waits for every
older snapshot to end. A process waiting for the bootstrap lock with a blocking
``pg_advisory_lock`` keeps its statement snapshot open, so the holder's index build
and the waiter wait on each other and Postgres cancels the build with
``DeadlockDetected``: the first boot of a fresh database fails. Polling with
``pg_try_advisory_lock`` leaves no statement running while the waiter sleeps.

Marked ``integration`` and skipped unless ``FUND_TEST_DATABASE_URL`` names a scratch
Postgres; it refuses a target that resolves to the ambient database, like the other
integration tests, and works in a throwaway table of its own.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.sql import SQL, Identifier

from fund.audit import persistence
from fund.config import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("FUND_TEST_DATABASE_URL"),
        reason="requires FUND_TEST_DATABASE_URL (scratch Postgres)",
    ),
]

_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def _endpoint(url: str) -> tuple[str, str, str]:
    info = conninfo_to_dict(persistence.normalize_conninfo(url))
    host = info.get("host", "")
    host = "localhost" if host in _LOCAL_HOSTS else host
    return host, str(info.get("port") or "5432"), info.get("dbname") or ""


def _conninfo() -> str:
    target = os.environ["FUND_TEST_DATABASE_URL"]
    ambient = {
        url for url in (os.environ.get("DATABASE_URL"), settings.database_url) if url
    }
    if any(_endpoint(url) == _endpoint(target) for url in ambient):
        pytest.fail(
            "refusing to run: FUND_TEST_DATABASE_URL points at the same database "
            "as DATABASE_URL; use a separate scratch database",
            pytrace=False,
        )
    return persistence.normalize_conninfo(target)


def test_a_waiter_for_the_bootstrap_lock_does_not_block_a_concurrent_index_build():
    """The holder's concurrent index build finishes while another process waits."""
    conninfo = _conninfo()
    table = Identifier(f"bootstrap_lock_probe_{uuid.uuid4().hex[:8]}")
    index = Identifier(f"ix_{uuid.uuid4().hex[:8]}")
    waited: dict[str, float] = {}

    with psycopg.connect(conninfo, autocommit=True) as setup:
        setup.execute(SQL("CREATE TABLE {} (a int)").format(table))
        setup.execute(
            SQL("INSERT INTO {} SELECT generate_series(1, 1000)").format(table)
        )
    try:
        with persistence._bootstrap_lock(conninfo):
            started = threading.Event()

            def waiter() -> None:
                """Wait for the lock held by the main thread, then record how long."""
                started.set()
                begun = time.monotonic()
                with persistence._bootstrap_lock(conninfo):
                    waited["seconds"] = time.monotonic() - begun

            thread = threading.Thread(target=waiter, daemon=True)
            thread.start()
            started.wait(timeout=5)
            time.sleep(1.0)  # the waiter is now polling for the lock

            with psycopg.connect(conninfo, autocommit=True) as builder:
                builder.execute("SET statement_timeout = '20s'")
                builder.execute(
                    SQL("CREATE INDEX CONCURRENTLY {} ON {} (a)").format(index, table)
                )
        thread.join(timeout=15)

        assert not thread.is_alive(), "the waiter never acquired the lock"
        assert waited["seconds"] >= 1.0
    finally:
        with psycopg.connect(conninfo, autocommit=True) as cleanup:
            cleanup.execute(SQL("DROP TABLE IF EXISTS {}").format(table))
