"""``FundJobRepository.claim_or_create`` against real concurrent claimants.

``background_jobs`` has no unique guard on active rows, so the one-active-slot
invariant rests entirely on the per-``job_type`` advisory lock taken before the
guarded ``INSERT … WHERE NOT EXISTS``. Under PostgreSQL's READ COMMITTED two
claimants cannot see each other's uncommitted row, so without that lock several
of them insert and the slot runs twice. SQLite serialises writers and cannot show
the race, which is why the unit suite can only assert the statement order; this
test pins the behaviour itself.

Marked ``integration`` and skipped unless ``DATABASE_URL`` is set. The test owns a
private, uniquely named schema holding only the job tables, so a developer's real
``public.background_jobs`` is never read or written, and the schema is dropped on
teardown.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

import pytest
from portopt_db.models import BackgroundJob, BackgroundJobError
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from fund.audit import FundJobRepository

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="claim-race test requires a live Postgres DATABASE_URL",
    ),
]

CLAIMANTS = 8
# Unguarded, ~99% of rounds produced a double claim at this width, so a handful of
# rounds already makes a missing lock fail reliably without slowing the suite.
ROUNDS = 25


@pytest.fixture
def race_engine() -> Generator[Engine, None, None]:
    """An engine whose ``background_jobs`` lives in a private throwaway schema.

    ``schema_translate_map`` redirects every statement (the guarded insert, its
    ``EXISTS`` subquery, the transitions) so the repository code runs unchanged.
    The pool is sized to the claimant count so no thread waits on a connection
    instead of on the lock under test.
    """
    url = os.environ["DATABASE_URL"]
    schema = f"fund_claim_race_{uuid.uuid4().hex[:8]}"
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(
        url,
        pool_size=CLAIMANTS + 1,
        max_overflow=0,
        execution_options={"schema_translate_map": {None: schema}},
    )
    try:
        job_tables = [BackgroundJob.__table__, BackgroundJobError.__table__]
        BackgroundJob.metadata.create_all(
            bind=engine, tables=job_tables, checkfirst=False
        )
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()


def _claim_concurrently(engine: Engine, job_type: str) -> list[uuid.UUID | None]:
    """Have :data:`CLAIMANTS` threads claim *job_type* at the same instant.

    Each thread owns its session (as each daemon would) and holds an open
    connection before the barrier, so the release lands straight on the guarded
    statement rather than on pool checkout jitter.
    """
    barrier = threading.Barrier(CLAIMANTS, timeout=30)

    def _claim() -> uuid.UUID | None:
        with Session(bind=engine) as session:
            session.connection()
            barrier.wait()
            return FundJobRepository(session).claim_or_create(job_type)

    with ThreadPoolExecutor(max_workers=CLAIMANTS) as pool:
        futures = [pool.submit(_claim) for _ in range(CLAIMANTS)]
        return [future.result() for future in futures]


def _active_rows(engine: Engine, job_type: str) -> int:
    with Session(bind=engine) as session:
        return (
            session.query(BackgroundJob)
            .filter(
                BackgroundJob.job_type == job_type,
                BackgroundJob.status.in_(("pending", "running")),
            )
            .count()
        )


def test_concurrent_claimants_of_one_slot_produce_exactly_one_winner(
    race_engine: Engine,
) -> None:
    """Every round of simultaneous claims leaves exactly one active slot row.

    Fails with the per-round winner and row counts when two claimants both pass
    the ``NOT EXISTS`` guard, i.e. when the advisory lock is missing or wrong.
    """
    run_id = uuid.uuid4().hex[:8]
    double_claims: list[str] = []

    for round_no in range(ROUNDS):
        # A fresh job_type per round keeps rounds independent, and the run id keeps
        # the advisory-lock key clear of the real daemon's slot names.
        job_type = f"fund_claim_race_{run_id}_{round_no}"

        winners = [
            job_id
            for job_id in _claim_concurrently(race_engine, job_type)
            if job_id is not None
        ]
        active = _active_rows(race_engine, job_type)

        if len(winners) != 1 or active != 1:
            double_claims.append(
                f"round {round_no}: {len(winners)} winners, {active} active rows"
            )

    assert not double_claims, "slot claimed more than once: " + "; ".join(double_claims)
