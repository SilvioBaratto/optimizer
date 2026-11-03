"""First-run universe bootstrap in ``run_daily_pipeline``.

A fresh install has an empty ``instruments`` table, and every price step iterates
it, so the first daily run used to report success with zero prices until the next
Saturday universe build. The pipeline now builds the universe first when no active
instrument exists.

Pure-unit: steps are patched, the emptiness probe runs against in-memory SQLite.
"""

from __future__ import annotations

import logging
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy.exc import OperationalError

from tests._fixtures._helpers import add_and_flush

M = "app.services.jobs.scheduler"

_BOOTSTRAP_NOTICE = "first run builds the universe"
_FULL_ORDER = ["universe", "fx", "yfinance", "macro", "fred", "news", "daily_events"]


@pytest.fixture(autouse=True)
def _populated_universe_by_default() -> None:
    """Shadow the suite-wide stub: this module exercises the real probe."""


def _db_error() -> OperationalError:
    return OperationalError("SELECT instruments", {}, Exception("db down"))


def _run_pipeline(
    *,
    empty: bool,
    universe_result: bool | BaseException = True,
    yfinance_result: bool = True,
) -> list[str]:
    """Run the daily pipeline with every step patched; return the step call order."""
    from app.services.jobs import scheduler

    order: list[str] = []

    def _step(name: str, result: bool | BaseException = True):
        def _effect(*_args, **_kwargs):
            order.append(name)
            if isinstance(result, BaseException):
                raise result
            return result

        return _effect

    with (
        patch.object(scheduler, "_universe_is_empty", return_value=empty),
        patch.object(
            scheduler,
            "run_universe_step",
            side_effect=_step("universe", universe_result),
        ),
        patch.object(scheduler, "run_fx_step", side_effect=_step("fx")),
        patch.object(
            scheduler,
            "run_yfinance_step",
            side_effect=_step("yfinance", yfinance_result),
        ),
        patch.object(scheduler, "run_macro_step", side_effect=_step("macro")),
        patch.object(scheduler, "run_fred_step", side_effect=_step("fred")),
        patch.object(scheduler, "run_news_step", side_effect=_step("news")),
        patch.object(
            scheduler, "run_daily_events_step", side_effect=_step("daily_events")
        ),
    ):
        scheduler.run_daily_pipeline()
    return order


class TestEmptyUniverseBootstrap:
    """An empty `instruments` table gets a universe build ahead of the daily steps."""

    def test_builds_the_universe_before_any_other_step(self):
        """The universe step leads, so fx and yfinance see the new instruments."""
        assert _run_pipeline(empty=True) == _FULL_ORDER

    def test_goes_through_the_slotted_universe_step(self):
        """The real ``run_universe_step`` claims its own job slot via ``_run_step``."""
        labels: list[str] = []

        def _effect(label, *_args, **_kwargs):
            labels.append(label)
            return True

        with (
            patch(f"{M}._universe_is_empty", return_value=True),
            patch(f"{M}._run_step", side_effect=_effect),
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            from app.services.jobs.scheduler import run_daily_pipeline

            run_daily_pipeline()

        assert labels[0] == "universe"
        assert labels.index("universe") < labels.index("fx") < labels.index("yfinance")

    def test_announces_the_bootstrap_before_the_price_download(
        self, caplog: pytest.LogCaptureFixture
    ):
        """One INFO line warns the operator before the multi-hour yfinance sweep."""
        from app.services.jobs import scheduler

        seen_at_yfinance: list[str] = []

        def _yfinance(**_kwargs):
            seen_at_yfinance.extend(r.getMessage() for r in caplog.records)
            return True

        with (
            caplog.at_level(logging.INFO, logger=M),
            patch.object(scheduler, "_universe_is_empty", return_value=True),
            patch.object(scheduler, "run_universe_step", return_value=True),
            patch.object(scheduler, "run_fx_step"),
            patch.object(scheduler, "run_yfinance_step", side_effect=_yfinance),
            patch.object(scheduler, "run_macro_step"),
            patch.object(scheduler, "run_fred_step"),
            patch.object(scheduler, "run_news_step"),
            patch.object(scheduler, "run_daily_events_step"),
        ):
            scheduler.run_daily_pipeline()

        notices = [m for m in seen_at_yfinance if _BOOTSTRAP_NOTICE in m]
        assert len(notices) == 1
        assert "many hours" in notices[0]
        assert not [r for r in caplog.records if r.levelno > logging.INFO], (
            "a clean bootstrap must not warn"
        )

    @pytest.mark.parametrize("universe_result", [False, RuntimeError("screener down")])
    def test_a_failed_or_skipped_build_leaves_the_rest_of_the_pipeline_intact(
        self, universe_result, caplog: pytest.LogCaptureFixture
    ):
        """A skipped, failed or raising build warns; every later step still runs."""
        with caplog.at_level(logging.INFO, logger=M):
            order = _run_pipeline(empty=True, universe_result=universe_result)

        assert order == _FULL_ORDER
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("universe bootstrap did not complete" in m for m in warnings)
        assert not [r for r in caplog.records if _BOOTSTRAP_NOTICE in r.getMessage()], (
            "no hours-long download is coming, so no bootstrap notice"
        )

    def test_a_raising_build_is_logged_with_its_traceback(
        self, caplog: pytest.LogCaptureFixture
    ):
        """An exception from the build is swallowed but not hidden."""
        with caplog.at_level(logging.INFO, logger=M):
            _run_pipeline(empty=True, universe_result=RuntimeError("screener down"))

        assert any(
            r.exc_info and "screener down" in str(r.exc_info[1]) for r in caplog.records
        )

    def test_an_interrupt_during_the_build_still_propagates(self):
        """Ctrl-C and SystemExit are not build failures and must not be swallowed."""
        with pytest.raises(KeyboardInterrupt):
            _run_pipeline(empty=True, universe_result=KeyboardInterrupt())


class TestPopulatedUniverse:
    """A populated universe leaves the daily pipeline exactly as it was."""

    def test_does_not_run_the_universe_step(self, caplog: pytest.LogCaptureFixture):
        """No universe build and no bootstrap notice when instruments exist."""
        with caplog.at_level(logging.INFO, logger=M):
            order = _run_pipeline(empty=False)

        assert order == _FULL_ORDER[1:]
        assert not [r for r in caplog.records if _BOOTSTRAP_NOTICE in r.getMessage()]


class TestUniverseIsEmpty:
    """The probe counts only non-delisted rows and never raises."""

    def test_true_on_a_fresh_database(self, patched_session_factory):
        """No rows at all reads as empty."""
        from app.services.jobs.scheduler import _universe_is_empty

        assert _universe_is_empty() is True

    def test_false_once_an_active_instrument_exists(
        self, patched_session_factory, db_session
    ):
        """One live listing is enough to skip the bootstrap."""
        from app.services.jobs.scheduler import _universe_is_empty

        ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
        add_and_flush(
            db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
        )

        assert _universe_is_empty() is False

    def test_true_when_only_delisted_instruments_remain(
        self, patched_session_factory, db_session
    ):
        """A table of delisted rows has no investable universe."""
        from app.services.jobs.scheduler import _universe_is_empty

        ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
        add_and_flush(
            db_session,
            Instrument(
                ticker="OLD",
                short_name="Old",
                exchange=ex,
                delisted_at=date(2020, 1, 1),
            ),
        )

        assert _universe_is_empty() is True

    def test_a_database_fault_reads_as_not_empty_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Unknown is not empty: a blip must not trigger a multi-minute rebuild."""
        from app.services.jobs import scheduler

        with (
            caplog.at_level(logging.WARNING, logger=M),
            patch.object(
                scheduler.database_manager, "get_session", side_effect=_db_error()
            ),
        ):
            assert scheduler._universe_is_empty() is False

        assert any("could not check" in r.getMessage() for r in caplog.records)


class TestCliDailyBootstraps:
    """The CLI `daily` command shares the scheduler's pipeline function."""

    def test_cli_daily_runs_the_same_pipeline(self):
        """`python -m app.cli daily` calls ``run_daily_pipeline``, so it bootstraps too."""
        from typer.testing import CliRunner

        from app.cli import app

        with (
            patch("app.cli.init_db"),
            patch(f"{M}._universe_is_empty", return_value=True),
            patch(f"{M}.run_universe_step", return_value=True) as universe,
            patch(f"{M}.run_fx_step"),
            patch(f"{M}.run_yfinance_step", return_value=True),
            patch(f"{M}.run_macro_step"),
            patch(f"{M}.run_fred_step"),
            patch(f"{M}.run_news_step"),
            patch(f"{M}.run_daily_events_step"),
        ):
            result = CliRunner().invoke(app, ["daily"])

        assert result.exit_code == 0
        universe.assert_called_once()
