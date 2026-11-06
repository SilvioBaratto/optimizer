"""Retention of the ``ticker_news`` article store.

Every daily sweep stores the scraped body of each ticker's articles, so without a
retention rule the table grows without bound. The weekly refetch prunes expired
rows in bounded, individually committed batches under its own heartbeat-wrapped
job slot; a failed prune must never take the weekly refetch down with it.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import CancelledError
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from portopt_db.models import Base
from portopt_db.models.market_data.yfinance_data import TickerNews
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.services.market_data.news_retention_service import run_news_retention

M = "app.services.jobs.scheduler"
_NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
_VALID_JOB_ID = "12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def news_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    """A committing session behind ``database_manager.get_session``.

    The service commits after each batch, which would leak across the shared
    SAVEPOINT fixture, so each test gets a private in-memory engine.
    """
    from app import database as db_module

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    @contextmanager
    def _cm() -> Iterator[Session]:
        yield session

    monkeypatch.setattr(db_module.database_manager, "get_session", _cm)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed(session: Session, published: list[datetime | None]) -> None:
    exchange = Exchange(name="EX")
    session.add(exchange)
    session.flush()
    instrument = Instrument(
        ticker="TSM", short_name="TSM", exchange=exchange, yfinance_ticker="TSM"
    )
    session.add(instrument)
    session.flush()
    for i, when in enumerate(published):
        session.add(
            TickerNews(
                instrument_id=instrument.id,
                news_uuid=f"n-{i}",
                title=f"n-{i}",
                publish_time=when,
                full_content="body " * 50,
            )
        )
    session.commit()


def _count(session: Session) -> int:
    return session.execute(select(func.count()).select_from(TickerNews)).scalar_one()


class TestNewsRetentionSetting:
    """``NEWS_RETENTION_DAYS`` is the operator's knob for the retention window."""

    def test_defaults_to_one_year(self) -> None:
        """A year of articles stays queryable by default."""
        assert Settings().news_retention_days == 365

    def test_env_overrides_the_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The value is read from the environment."""
        monkeypatch.setenv("NEWS_RETENTION_DAYS", "90")
        assert Settings().news_retention_days == 90

    def test_zero_is_accepted_and_means_keep_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero switches pruning off."""
        monkeypatch.setenv("NEWS_RETENTION_DAYS", "0")
        assert Settings().news_retention_days == 0

    def test_a_negative_window_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A negative value would put the cutoff in the future and wipe the table."""
        monkeypatch.setenv("NEWS_RETENTION_DAYS", "-1")
        with pytest.raises(ValueError, match="NEWS_RETENTION_DAYS"):
            Settings()


class TestRunNewsRetention:
    """The service deletes expired articles one committed batch at a time."""

    def test_expired_articles_are_deleted_and_recent_ones_kept(
        self, news_session: Session
    ) -> None:
        """Only rows older than the window go."""
        _seed(
            news_session,
            [
                _NOW - timedelta(days=400),
                _NOW - timedelta(days=366),
                _NOW - timedelta(days=364),
                _NOW - timedelta(hours=2),
            ],
        )

        result = run_news_retention(retention_days=365, now=_NOW)

        assert result["deleted"] == 2
        assert _count(news_session) == 2

    def test_a_large_backlog_is_drained_in_committed_batches(
        self, news_session: Session
    ) -> None:
        """Each batch is its own transaction, so no lock outlives a batch."""
        _seed(news_session, [_NOW - timedelta(days=500 + i) for i in range(5)])
        commits: list[int] = []
        real_commit = news_session.commit

        def _count_commits() -> None:
            commits.append(1)
            real_commit()

        with patch.object(news_session, "commit", side_effect=_count_commits):
            result = run_news_retention(retention_days=365, batch_size=2, now=_NOW)

        assert result["deleted"] == 5
        assert result["batches"] == 3
        assert len(commits) == 3
        assert _count(news_session) == 0

    def test_a_backlog_that_is_an_exact_multiple_still_terminates(
        self, news_session: Session
    ) -> None:
        """The final empty batch is what ends the loop."""
        _seed(news_session, [_NOW - timedelta(days=500 + i) for i in range(4)])

        result = run_news_retention(retention_days=365, batch_size=2, now=_NOW)

        assert result["deleted"] == 4
        assert result["batches"] == 3

    def test_progress_is_reported_after_every_batch(
        self, news_session: Session
    ) -> None:
        """Each batch renews the step's progress watch so it never looks wedged."""
        _seed(news_session, [_NOW - timedelta(days=500 + i) for i in range(5)])
        on_progress = MagicMock()

        run_news_retention(
            retention_days=365, batch_size=2, now=_NOW, on_progress=on_progress
        )

        running = [c for c in on_progress.call_args_list if "status" not in c.kwargs]
        assert len(running) == 3
        final = on_progress.call_args_list[-1].kwargs
        assert final["status"] == "completed"
        assert final["result"]["deleted"] == 5

    def test_a_second_run_is_a_no_op(self, news_session: Session) -> None:
        """Pruning twice deletes nothing the second time."""
        _seed(news_session, [_NOW - timedelta(days=500)])

        assert run_news_retention(retention_days=365, now=_NOW)["deleted"] == 1
        assert run_news_retention(retention_days=365, now=_NOW)["deleted"] == 0

    def test_undated_articles_age_by_their_ingest_time(
        self, news_session: Session
    ) -> None:
        """A row without ``publish_time`` is judged by ``created_at``."""
        _seed(news_session, [None])

        # Ingested "now" in wall-clock time: well inside the window.
        assert run_news_retention(retention_days=365)["deleted"] == 0
        # A window of one day measured from far in the future expires it.
        future = datetime.now(timezone.utc) + timedelta(days=30)
        assert run_news_retention(retention_days=1, now=future)["deleted"] == 1

    def test_zero_days_disables_pruning_without_touching_the_database(self) -> None:
        """No session is opened when retention is switched off."""
        from app import database as db_module

        with patch.object(db_module.database_manager, "get_session") as get_session:
            result = run_news_retention(retention_days=0, now=_NOW)

        get_session.assert_not_called()
        assert result["deleted"] == 0

    def test_the_configured_window_is_used_when_none_is_passed(
        self, news_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``settings.news_retention_days`` drives the cutoff by default."""
        from app.services.market_data import news_retention_service as svc

        monkeypatch.setattr(svc.settings, "news_retention_days", 30)
        _seed(news_session, [_NOW - timedelta(days=60), _NOW - timedelta(days=10)])

        assert run_news_retention(now=_NOW)["deleted"] == 1

    def test_a_database_error_propagates_after_earlier_batches_are_kept(
        self, news_session: Session
    ) -> None:
        """A failure marks the step failed but does not undo committed batches."""
        _seed(news_session, [_NOW - timedelta(days=500 + i) for i in range(4)])
        from portopt_db.repositories.market_data import yfinance_repository as repo

        real = repo.YFinanceRepository.prune_ticker_news
        calls = {"n": 0}

        def _flaky(self: Any, *a: Any, **kw: Any) -> int:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OperationalError("DELETE", {}, Exception("db down"))
            return real(self, *a, **kw)

        with (
            patch.object(repo.YFinanceRepository, "prune_ticker_news", _flaky),
            pytest.raises(OperationalError),
        ):
            run_news_retention(retention_days=365, batch_size=2, now=_NOW)

        assert _count(news_session) == 2

    def test_a_reaped_step_stops_at_the_next_batch(self, news_session: Session) -> None:
        """Cancellation raised by the progress callback ends the loop."""
        _seed(news_session, [_NOW - timedelta(days=500 + i) for i in range(6)])
        on_progress = MagicMock(side_effect=CancelledError("reaped"))

        with pytest.raises(CancelledError):
            run_news_retention(
                retention_days=365, batch_size=2, now=_NOW, on_progress=on_progress
            )

        assert _count(news_session) == 4


def _job_svc() -> MagicMock:
    svc = MagicMock()
    svc.create_job.return_value = _VALID_JOB_ID
    svc.get_job.return_value = {"status": "completed"}
    svc._heartbeat_cadence = 30
    return svc


class TestRunNewsRetentionStep:
    """The step claims its own job slot and heartbeats while it deletes."""

    def test_composes_run_step_with_its_label_slot_and_service(self) -> None:
        """It delegates to ``_run_step`` like every other step."""
        from app.services.jobs.scheduler import (
            _news_retention_jobs,
            run_news_retention_step,
        )

        with (
            patch(f"{M}.settings.news_retention_days", 365),
            patch(f"{M}._run_step", return_value=True) as mock_step,
        ):
            assert run_news_retention_step() is True

        args = mock_step.call_args.args
        assert args[0] == "news_retention"
        assert args[1] is _news_retention_jobs
        assert args[2] is run_news_retention

    def test_is_skipped_without_claiming_a_slot_when_disabled(self) -> None:
        """Zero days is a configuration state, not a failed run."""
        from app.services.jobs.scheduler import run_news_retention_step

        with (
            patch(f"{M}.settings.news_retention_days", 0),
            patch(f"{M}._run_step") as mock_step,
        ):
            assert run_news_retention_step() is False

        mock_step.assert_not_called()

    def test_the_job_type_is_reclaim_dispatchable(self) -> None:
        """Pruning is idempotent, so the reaper may safely re-dispatch it."""
        from app.services.jobs.scheduler import _reclaim_dispatchable

        assert _reclaim_dispatchable("news_retention") is True

    def test_heartbeats_for_the_duration_of_the_step(self) -> None:
        """A long prune must not be reaped for want of a heartbeat."""
        from app.services.jobs import scheduler as sched

        svc = _job_svc()
        svc._heartbeat_cadence = 0.01
        svc._run_heartbeat.side_effect = lambda _j, stop, _c, **_kw: stop.wait(
            timeout=5
        )
        seen: list[list[str]] = []

        def _fake_prune(*, on_progress: Any) -> dict[str, int]:
            seen.append(
                [
                    t.name
                    for t in threading.enumerate()
                    if t.name.startswith("hb:sched:news_retention")
                ]
            )
            return {"deleted": 0}

        with (
            patch.object(sched, "_news_retention_jobs", svc),
            patch.object(sched.settings, "news_retention_days", 365),
            patch(
                "app.services.market_data.news_retention_service.run_news_retention",
                _fake_prune,
            ),
        ):
            assert sched.run_news_retention_step() is True

        assert seen and seen[0], "no news_retention heartbeat thread ran with the step"
        svc._run_heartbeat.assert_called_once()

    def test_a_failing_prune_is_logged_and_recorded_not_raised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The failure lands on the step's own row and never propagates."""
        from app.services.jobs import scheduler as sched

        svc = _job_svc()

        def _boom(*, on_progress: Any) -> None:
            raise OperationalError("DELETE", {}, Exception("db down"))

        with (
            patch.object(sched, "_news_retention_jobs", svc),
            patch.object(sched.settings, "news_retention_days", 365),
            patch(
                "app.services.market_data.news_retention_service.run_news_retention",
                _boom,
            ),
            caplog.at_level("ERROR", logger=sched.logger.name),
        ):
            assert sched.run_news_retention_step() is False

        failed = [
            c
            for c in svc.update_job.call_args_list
            if c.kwargs.get("status") == "failed"
        ]
        assert failed
        assert any("news_retention" in r.getMessage() for r in caplog.records)


class TestWeeklyRefetchPrunesNews:
    """Retention rides on the weekly refetch and cannot break it."""

    def _run(self, *, yfinance_ok: bool = True) -> MagicMock:
        from app.services.jobs import scheduler as sched

        steps = MagicMock()
        steps.run_yfinance_step.return_value = yfinance_ok
        steps.run_macro_step.return_value = True
        steps.run_news_retention_step.return_value = True
        with (
            patch.object(sched, "run_yfinance_step", steps.run_yfinance_step),
            patch.object(sched, "run_macro_step", steps.run_macro_step),
            patch.object(
                sched, "run_news_retention_step", steps.run_news_retention_step
            ),
        ):
            sched.run_weekly_refetch()
        return steps

    def test_the_prune_runs_after_the_sweep_and_macro(self) -> None:
        """Retention comes last so it never competes with the sweep it follows."""
        steps = self._run()

        assert [c[0] for c in steps.mock_calls] == [
            "run_yfinance_step",
            "run_macro_step",
            "run_news_retention_step",
        ]

    def test_the_prune_still_runs_when_the_sweep_fails(self) -> None:
        """Housekeeping does not depend on the sweep succeeding."""
        steps = self._run(yfinance_ok=False)

        steps.run_news_retention_step.assert_called_once()

    def test_a_prune_that_raises_does_not_fail_the_weekly_refetch(self) -> None:
        """Even an unexpected error from the step is contained by the caller."""
        from app.services.jobs import scheduler as sched

        with (
            patch.object(sched, "run_yfinance_step", return_value=True),
            patch.object(sched, "run_macro_step", return_value=True),
            patch.object(
                sched, "run_news_retention_step", side_effect=RuntimeError("boom")
            ),
        ):
            sched.run_weekly_refetch()
