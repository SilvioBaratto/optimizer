"""``FundJobRepository`` — atomic job-slot mechanics for the fund daemon.

The ``background_jobs`` *model* lives in ``portopt_db`` (shared schema,
Alembic-owned); the fund daemon needs its own *behavior* over it — a
boundary-clean mirror of the ingestion ``BackgroundJobRepository`` that never
imports ``ingestion.app``. Two fund-specific ``job_type``s
(:data:`FUND_JOB_TYPES`) each get at most one active row, so the rebalance sweep
and the drift monitor can never double-run across a restart or a second daemon.

Unlike the other ``fund.audit`` repositories (which open no session and leave the
commit to the caller, D1), this repo **owns its own short transactions**: the
atomic slot claim and every lifecycle transition (``mark_running`` →
``update_heartbeat`` → ``mark_done``) must *land* — be durable and visible to a
reaper or observer running in another process — the moment they succeed. That is
the one documented exception to the fund "no commit" rule. :meth:`reap_orphans`
is the exception's exception: it flushes but does **not** commit (its scheduler
caller owns the commit), mirroring the ingestion reaper.

Liveness is a **heartbeat lease**, not a host/PID identity check: a claim is
reaped only once ``last_heartbeat_at`` is NULL or older than the lease TTL, with
**no** ``worker_host``/``worker_pid`` clause (those columns are recorded for
observability only). Run exactly one daemon per DB.
"""

from __future__ import annotations

import os
import socket
import uuid
from datetime import UTC, datetime, timedelta

from portopt_db.models import BackgroundJob
from portopt_db.repository import RepositoryBase
from sqlalchemy import exists, func, insert, literal, or_, select, update

# The two fund daemon job slots. Each gets at most one active (pending|running)
# row; the scheduler's rebalance sweep and drift monitor claim these.
FUND_JOB_TYPES: tuple[str, ...] = ("fund_rebalance_sweep", "fund_drift_monitor")

_ACTIVE_STATUSES: tuple[str, ...] = ("pending", "running")


class FundJobRepository(RepositoryBase):
    """Atomic per-``job_type`` slot + heartbeat-lease reaper on ``background_jobs``."""

    def claim_or_create(self, job_type: str) -> uuid.UUID | None:
        """Atomically create a ``pending`` job iff no active row of *job_type* exists.

        A single ``INSERT … FROM SELECT … WHERE NOT EXISTS(active row)`` keeps at
        most one active row per ``job_type`` even under concurrent daemons: exactly
        one insert affects a row (``rowcount == 1`` → the new UUID), the rest see
        the existing row and no-op (``None``). Stamps ``worker_pid`` /
        ``worker_host`` / ``last_heartbeat_at = now`` (observability + the initial
        lease) and commits so the slot is immediately visible to other processes.
        Dialect-agnostic (PostgreSQL + SQLite).
        """
        new_id = uuid.uuid4()
        now = datetime.now(UTC)
        pid = os.getpid()
        host = socket.gethostname()
        tbl = BackgroundJob.__table__

        conflict_exists = exists().where(
            BackgroundJob.job_type == job_type,
            BackgroundJob.status.in_(_ACTIVE_STATUSES),
        )
        source = select(
            literal(new_id, type_=tbl.c.id.type).label("id"),
            literal(job_type, type_=tbl.c.job_type.type).label("job_type"),
            literal("pending", type_=tbl.c.status.type).label("status"),
            literal(0, type_=tbl.c.current.type).label("current"),
            literal(0, type_=tbl.c.total.type).label("total"),
            literal(now, type_=tbl.c.started_at.type).label("started_at"),
            literal(pid, type_=tbl.c.worker_pid.type).label("worker_pid"),
            literal(host, type_=tbl.c.worker_host.type).label("worker_host"),
            literal(now, type_=tbl.c.last_heartbeat_at.type).label("last_heartbeat_at"),
            literal(0, type_=tbl.c.attempt.type).label("attempt"),
            literal(now, type_=tbl.c.created_at.type).label("created_at"),
            literal(now, type_=tbl.c.updated_at.type).label("updated_at"),
        ).where(~conflict_exists)
        stmt = insert(tbl).from_select(
            [
                "id",
                "job_type",
                "status",
                "current",
                "total",
                "started_at",
                "worker_pid",
                "worker_host",
                "last_heartbeat_at",
                "attempt",
                "created_at",
                "updated_at",
            ],
            source,
        )
        result = self.session.execute(stmt)
        self.session.commit()
        return new_id if result.rowcount == 1 else None

    def mark_running(self, job_id: uuid.UUID) -> bool:
        """Flip a claimed slot from active → ``running``; ``True`` if a row moved.

        Status-guarded on the active statuses so a terminal row is never
        resurrected. Commits so the running state is visible to the reaper.
        """
        stmt = (
            update(BackgroundJob)
            .where(
                BackgroundJob.id == job_id,
                BackgroundJob.status.in_(_ACTIVE_STATUSES),
            )
            .values(status="running")
        )
        result = self.session.execute(stmt)
        self.session.commit()
        return (result.rowcount or 0) == 1

    def update_heartbeat(self, job_id: uuid.UUID) -> bool:
        """Stamp ``last_heartbeat_at = NOW()`` iff the row is still ``running``.

        Returns ``True`` when a running row was renewed, ``False`` otherwise
        (terminal, never started, or unknown ``job_id``). Commits so the reaper in
        another process sees the fresh lease and spares a live long-running step.
        """
        stmt = (
            update(BackgroundJob)
            .where(
                BackgroundJob.id == job_id,
                BackgroundJob.status == "running",
            )
            .values(last_heartbeat_at=func.now())
        )
        result = self.session.execute(stmt)
        self.session.commit()
        return (result.rowcount or 0) == 1

    def mark_done(
        self,
        job_id: uuid.UUID,
        *,
        status: str,
        error: str | None = None,
    ) -> bool:
        """Move an active slot to a terminal *status* and stamp ``finished_at``.

        Status-guarded on the active statuses so a slot is finalised at most once
        (a late failure can't overwrite an already-completed row). ``True`` if a
        row moved. Commits so the slot re-opens for the next scheduled run.
        """
        stmt = (
            update(BackgroundJob)
            .where(
                BackgroundJob.id == job_id,
                BackgroundJob.status.in_(_ACTIVE_STATUSES),
            )
            .values(status=status, error=error, finished_at=func.now())
        )
        result = self.session.execute(stmt)
        self.session.commit()
        return (result.rowcount or 0) == 1

    def reap_orphans(
        self,
        error_msg: str,
        *,
        heartbeat_timeout_seconds: int,
    ) -> int:
        """Fail every lease-expired fund slot; return how many were reaped.

        Liveness is a pure **heartbeat lease**: a ``pending``/``running`` row is
        failed once ``last_heartbeat_at`` is NULL or older than
        ``heartbeat_timeout_seconds`` — there is deliberately **no** host/PID
        clause, so a stale row is reaped even if stamped with this very process's
        live PID (the lease *renewal*, not an identity check, is what protects a
        live long-running step). Scoped to :data:`FUND_JOB_TYPES` so a stale
        ingestion job is never touched. Caller owns the commit.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=heartbeat_timeout_seconds)
        stale_active = (
            BackgroundJob.job_type.in_(FUND_JOB_TYPES),
            BackgroundJob.status.in_(_ACTIVE_STATUSES),
            or_(
                BackgroundJob.last_heartbeat_at.is_(None),
                BackgroundJob.last_heartbeat_at < cutoff,
            ),
        )
        stmt = (
            update(BackgroundJob)
            .where(*stale_active)
            .values(status="failed", error=error_msg, finished_at=func.now())
        )
        # A bulk maintenance sweep: don't synchronise the caller's identity map
        # (callers re-read after the caller-owned commit). This also keeps the
        # server-side heartbeat/timeout comparison out of the ORM evaluator.
        result = self.session.execute(
            stmt, execution_options={"synchronize_session": False}
        )
        self.session.flush()
        return result.rowcount or 0

    def get(self, job_id: uuid.UUID) -> BackgroundJob | None:
        """Return the job row by id, or ``None``."""
        return self.session.get(BackgroundJob, job_id)


__all__ = ["FUND_JOB_TYPES", "FundJobRepository"]
