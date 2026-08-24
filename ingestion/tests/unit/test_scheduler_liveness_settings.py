"""Settings exposure for liveness reaper (issue #590)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings


class TestLivenessSettings:
    def test_when_defaults_then_heartbeat_cadence_is_thirty_seconds(self) -> None:
        s = Settings()
        assert s.scheduler_heartbeat_cadence_seconds == 30

    def test_when_defaults_then_orphan_timeout_is_five_minutes(self) -> None:
        s = Settings()
        assert s.scheduler_orphan_heartbeat_timeout_seconds == 300

    def test_when_env_overrides_cadence_then_value_is_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCHEDULER_HEARTBEAT_CADENCE_SECONDS", "10")
        s = Settings()
        assert s.scheduler_heartbeat_cadence_seconds == 10

    def test_when_env_overrides_timeout_then_value_is_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS", "120")
        s = Settings()
        assert s.scheduler_orphan_heartbeat_timeout_seconds == 120


class TestOrphanStrategySettings:
    """R3/§5.3 — SCHEDULER_ORPHAN_STRATEGY gates the reclaim behaviour."""

    def test_defaults_to_fail(self) -> None:
        assert Settings().scheduler_orphan_strategy == "fail"

    def test_reclaim_accepted_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SCHEDULER_ORPHAN_STRATEGY", "RECLAIM")
        assert Settings().scheduler_orphan_strategy == "reclaim"

    def test_invalid_strategy_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import ValidationError

        monkeypatch.setenv("SCHEDULER_ORPHAN_STRATEGY", "bogus")
        with pytest.raises(ValidationError):
            Settings()

    def test_max_reclaim_attempts_default_three(self) -> None:
        assert Settings().scheduler_orphan_max_reclaim_attempts == 3


class TestWorkerForwardsTimeout:
    def test_when_worker_reconciles_then_settings_timeout_is_forwarded(self) -> None:
        """app.worker forwards settings.scheduler_orphan_heartbeat_timeout_seconds."""
        from contextlib import contextmanager

        from app import worker as worker_module

        repo_instance = MagicMock()
        repo_instance.reconcile_orphans.return_value = 0
        fake_session = MagicMock()

        @contextmanager
        def fake_get_session():
            yield fake_session

        with (
            patch.object(
                worker_module.database_manager,
                "get_session",
                side_effect=fake_get_session,
            ),
            patch(
                "app.repositories.jobs.background_job_repository.BackgroundJobRepository",
                return_value=repo_instance,
            ),
        ):
            worker_module._reconcile_orphans()

        call = repo_instance.reconcile_orphans.call_args
        # Settings default is 300; the worker must forward it explicitly.
        assert call.kwargs.get("heartbeat_timeout_seconds") == 300


class TestJobServicesAdoptCadence:
    """Module-level BackgroundJobService instances pick up the cadence setting."""

    @pytest.mark.parametrize(
        "attr",
        [
            "_yfinance_jobs",
            "_macro_jobs",
            "_news_fetch_jobs",
            "_summarize_jobs",
            "_calibrate_jobs",
            "_fred_jobs",
            "_ref_index_jobs",
            "_universe_jobs",
        ],
    )
    def test_job_service_cadence_matches_settings(self, attr: str) -> None:
        from app.config import settings
        from app.services.jobs import scheduler

        job_service = getattr(scheduler, attr)
        assert (
            job_service._heartbeat_cadence
            == settings.scheduler_heartbeat_cadence_seconds
        )
