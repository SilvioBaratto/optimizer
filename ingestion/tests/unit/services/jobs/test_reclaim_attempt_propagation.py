"""R3/§5.3 — the reclaim ``attempt`` must survive every hop of re-dispatch.

The reaper re-dispatches an orphan by calling a step wrapper with ``attempt =
prev + 1``; that must reach the new ``background_jobs`` row so the cap can bite.
This locks each hop of that chain:

    run_*_step(attempt) -> _run_step(attempt) -> create_job(attempt) -> row.attempt

Other reclaim pieces are covered elsewhere: reap_orphans returns job_type +
attempt (test_background_job_repository), and _redispatch_reclaimed submits a
scheduler one-shot job with attempt+1 and honours the cap (test_scheduler_pipeline).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.repositories.jobs.background_job_repository import BackgroundJobRepository
from app.services.jobs.background_job import BackgroundJobService

M = "app.services.jobs.scheduler"
_VALID_JOB_ID = "12345678-1234-1234-1234-123456789abc"


# ---------------------------------------------------------------------------
# Hop 1: create_job(attempt) -> repo.claim_or_create(attempt=...)
#
# Repo-level persistence (claim_or_create attempt -> row.attempt) is covered by
# test_background_job_repository::TestReclaimOrphans; here we lock the service
# forwarding with a mocked repo so no real row is committed (which would leak
# through the shared in-memory DB into other tests).
# ---------------------------------------------------------------------------


class TestCreateJobForwardsAttempt:
    _REPO = "app.services.jobs.background_job.BackgroundJobRepository"

    def _make_service(self) -> BackgroundJobService:
        return BackgroundJobService("yfinance_fetch", lambda: MagicMock())

    def test_create_job_forwards_attempt_to_claim_or_create(self) -> None:
        with patch(self._REPO) as repo_cls:
            repo = repo_cls.return_value
            repo.claim_or_create.return_value = uuid.uuid4()
            self._make_service().create_job(attempt=2)

        assert repo.claim_or_create.call_args.kwargs["attempt"] == 2

    def test_create_job_defaults_attempt_to_zero(self) -> None:
        with patch(self._REPO) as repo_cls:
            repo = repo_cls.return_value
            repo.claim_or_create.return_value = uuid.uuid4()
            self._make_service().create_job()

        assert repo.claim_or_create.call_args.kwargs["attempt"] == 0


# ---------------------------------------------------------------------------
# Hop 2: _run_step(attempt) -> create_job(attempt)
# ---------------------------------------------------------------------------


class TestRunStepForwardsAttempt:
    def test_run_step_passes_attempt_to_create_job(self) -> None:
        from app.services.jobs.scheduler import _run_step

        svc = MagicMock()
        svc.create_job.return_value = _VALID_JOB_ID
        svc._heartbeat_cadence = 30
        svc.get_job.return_value = {"status": "completed"}

        with patch(f"{M}.threading.Thread"):
            _run_step("label", svc, MagicMock(), attempt=3)

        svc.create_job.assert_called_once_with(attempt=3)


# ---------------------------------------------------------------------------
# Hop 3: run_*_step(attempt) -> _run_step(attempt)
# ---------------------------------------------------------------------------


class TestStepWrappersForwardAttempt:
    """Every step wrapper forwards ``attempt`` into ``_run_step``."""

    def _call_wrapper(self, name: str, attempt: int) -> int:
        from app.services.jobs import scheduler as sched

        with (
            patch.object(sched, "_run_step", return_value=True) as run_step,
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            getattr(sched, name)(attempt=attempt)

        assert run_step.call_count == 1
        return run_step.call_args.kwargs["attempt"]

    @pytest.mark.parametrize(
        "wrapper",
        [
            "run_yfinance_step",
            "run_macro_step",
            "run_fred_step",
            "run_news_step",
            "run_summarize_step",
            "run_calibrate_step",
        ],
    )
    def test_wrapper_forwards_attempt(self, wrapper: str) -> None:
        assert self._call_wrapper(wrapper, 7) == 7

    def test_run_universe_step_forwards_attempt(self) -> None:
        from app.services.jobs import scheduler as sched

        with (
            patch.object(sched, "_run_step", return_value=True) as run_step,
            patch(
                "app.services.universe.universe_build_service.build_trading212_client",
                return_value=MagicMock(),
            ),
        ):
            sched.run_universe_step(attempt=4)

        assert run_step.call_args.kwargs["attempt"] == 4

    def test_refresh_reference_indices_forwards_attempt_to_create_job(self) -> None:
        from app.services.jobs import scheduler as sched

        with (
            patch.object(
                sched._ref_index_jobs, "create_job", return_value=_VALID_JOB_ID
            ) as create,
            patch.object(sched._ref_index_jobs, "update_job"),
            patch.object(
                sched._ref_index_jobs,
                "get_job",
                return_value={"status": "completed"},
            ),
            patch.object(sched, "_heartbeat"),
            patch(
                "app.services.market_data.reference_index_seeder.seed_reference_indices"
            ),
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            sched.refresh_reference_indices("orphan_reclaim", attempt=6)

        assert create.call_args.kwargs["attempt"] == 6


# ---------------------------------------------------------------------------
# End-to-end glue: a stale orphan carries attempt+1 into its re-dispatch
# ---------------------------------------------------------------------------


class TestReclaimEndToEndAttempt:
    def test_stale_orphan_redispatches_with_incremented_attempt(
        self, db_session
    ) -> None:
        """reap_orphans + _redispatch_reclaimed submit attempt+1 to the scheduler.

        Drives the two reclaim collaborators directly (not run_orphan_reaper,
        which commits and would escape the SAVEPOINT test isolation).
        reap_orphans only flushes, so the row is rolled back on teardown.
        """
        from datetime import datetime, timedelta, timezone

        from app.models.jobs.background_job import BackgroundJob
        from app.services.jobs import scheduler as sched

        # Insert a running orphan directly (attempt=1, stale lease).
        stale = datetime.now(timezone.utc) - timedelta(minutes=10)
        orphan_id = uuid.uuid4()
        db_session.add(
            BackgroundJob(
                id=orphan_id,
                job_type="yfinance_fetch",
                status="running",
                attempt=1,
                started_at=datetime.now(timezone.utc),
                last_heartbeat_at=stale,
            )
        )
        db_session.flush()
        repo = BackgroundJobRepository(db_session)

        reaped = repo.reap_orphans("orphan retry")
        assert reaped == [{"job_type": "yfinance_fetch", "attempt": 1}]

        with (
            patch.object(sched.settings, "scheduler_orphan_max_reclaim_attempts", 3),
            patch.object(sched, "_scheduler", MagicMock()) as sch,
        ):
            sched._redispatch_reclaimed(reaped)

        # Orphan failed (slot freed) and re-dispatched at attempt 1 + 1 = 2.
        sch.add_job.assert_called_once()
        assert sch.add_job.call_args.kwargs["kwargs"] == {
            "job_type": "yfinance_fetch",
            "attempt": 2,
        }
        row = repo.get(orphan_id)
        assert row is not None
        assert row.status == "failed"
