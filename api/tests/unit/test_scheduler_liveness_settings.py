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
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCHEDULER_HEARTBEAT_CADENCE_SECONDS", "10")
        s = Settings()
        assert s.scheduler_heartbeat_cadence_seconds == 10

    def test_when_env_overrides_timeout_then_value_is_applied(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS", "120")
        s = Settings()
        assert s.scheduler_orphan_heartbeat_timeout_seconds == 120


class TestLifespanForwardsTimeout:
    @pytest.mark.asyncio
    async def test_when_lifespan_runs_then_reconcile_called_with_settings_timeout(
        self,
    ) -> None:
        """main.py forwards settings.scheduler_orphan_heartbeat_timeout_seconds."""
        from contextlib import contextmanager

        from fastapi import FastAPI

        from app import main as main_module

        repo_instance = MagicMock()
        repo_instance.reconcile_orphans.return_value = 0
        fake_session = MagicMock()

        @contextmanager
        def fake_get_session():
            yield fake_session

        scheduler_mock = MagicMock()

        # Reset the per-process sentinel so reconcile actually fires.
        main_module._reconciled_this_process = False

        with patch.object(main_module, "init_db"), \
             patch.object(main_module.database_manager, "health_check", return_value=True), \
             patch.object(
                 main_module.database_manager, "get_session",
                 side_effect=fake_get_session,
             ), \
             patch.object(main_module, "create_scheduler", return_value=scheduler_mock), \
             patch(
                 "app.services._benchmark_bootstrap.bootstrap_benchmarks",
                 return_value=None,
             ), \
             patch(
                 "app.repositories.background_job_repository.BackgroundJobRepository",
                 return_value=repo_instance,
             ):
            app = FastAPI()
            async with main_module.lifespan(app):
                pass

        call = repo_instance.reconcile_orphans.call_args
        # Settings default is 300; lifespan must forward it explicitly.
        assert call.kwargs.get("heartbeat_timeout_seconds") == 300


class TestRouteServicesAdoptCadence:
    """Module-level BackgroundJobService instances pick up the cadence setting."""

    def test_yfinance_service_cadence_matches_settings(self) -> None:
        from app.api.v1.yfinance_data import _job_service
        from app.config import settings

        assert (
            _job_service._heartbeat_cadence
            == settings.scheduler_heartbeat_cadence_seconds
        )

    def test_macro_regime_service_cadence_matches_settings(self) -> None:
        from app.api.v1.macro_regime import _job_service
        from app.config import settings

        assert (
            _job_service._heartbeat_cadence
            == settings.scheduler_heartbeat_cadence_seconds
        )

    def test_scheduler_yfinance_jobs_cadence_matches_settings(self) -> None:
        from app.config import settings
        from app.services.scheduler import _yfinance_jobs

        assert (
            _yfinance_jobs._heartbeat_cadence
            == settings.scheduler_heartbeat_cadence_seconds
        )
