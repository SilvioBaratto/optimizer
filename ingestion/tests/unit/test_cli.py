"""Tests for the manual-run CLI (``python -m app.cli``).

Every scheduler step function is patched out, so nothing fetches or touches a
database. What is asserted is the contract the shell drivers depend on:

- each command dispatches to the step the scheduler itself runs (so a manual
  run and a scheduled run take the same job-slot / heartbeat path)
- options reach that step with the right values
- single-step commands exit non-zero when the step did not complete, so
  ``scheduler/fetch.sh`` and CI can gate on the exit code
"""

from __future__ import annotations

import logging
import os
from datetime import date
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()

SCHED = "app.services.jobs.scheduler"


@pytest.fixture(autouse=True)
def _no_db():
    """``_boot`` builds a real engine — stub it out for every command."""
    with patch("app.cli.init_db"):
        yield


class TestSetupCommand:
    """`portopt setup` wires flags to the wizard and maps failures to exit 1."""

    def test_non_interactive_wires_flags(self) -> None:
        with patch("app.setup.wizard.run_setup_noninteractive") as mock_run:
            result = runner.invoke(
                app,
                [
                    "setup",
                    "--non-interactive",
                    "--t212-key",
                    "tk",
                    "--t212-secret",
                    "ts",
                    "--fred-key",
                    "fk",
                ],
            )
        assert result.exit_code == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["t212_key"] == "tk"
        assert kwargs["t212_secret"] == "ts"  # noqa: S105 - test value, not a secret
        assert kwargs["fred_key"] == "fk"

    def test_non_interactive_wires_llm_flags(self) -> None:
        with patch("app.setup.wizard.run_setup_noninteractive") as mock_run:
            result = runner.invoke(
                app,
                [
                    "setup",
                    "--non-interactive",
                    "--llm-provider",
                    "openai",
                    "--llm-model",
                    "gpt-4o",
                    "--llm-base-url",
                    "https://proxy.example/v1",
                    "--llm-key",
                    "sk-x",
                ],
            )
        assert result.exit_code == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["llm_provider"] == "openai"
        assert kwargs["llm_model"] == "gpt-4o"
        assert kwargs["llm_base_url"] == "https://proxy.example/v1"
        assert kwargs["llm_key"] == "sk-x"

    def test_skip_path_install_forwarded(self) -> None:
        with patch("app.setup.wizard.run_setup_noninteractive") as mock_run:
            result = runner.invoke(
                app, ["setup", "--non-interactive", "--skip-path-install"]
            )
        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["skip_path_install"] is True

    def test_non_interactive_skips_path_install_by_default(self) -> None:
        """CI/non-interactive never mutates the User PATH unless explicitly asked;
        `--non-interactive` implies skip_path_install even without the flag."""
        with patch("app.setup.wizard.run_setup_noninteractive") as mock_run:
            result = runner.invoke(app, ["setup", "--non-interactive"])
        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["skip_path_install"] is True

    def test_non_interactive_failure_exits_nonzero(self) -> None:
        from app.setup.wizard import SetupError

        with patch(
            "app.setup.wizard.run_setup_noninteractive",
            side_effect=SetupError("bad key"),
        ):
            result = runner.invoke(
                app,
                [
                    "setup",
                    "--non-interactive",
                    "--fred-key",
                    "x",
                ],
            )
        assert result.exit_code == 1

    def test_wires_skip_validation_and_reconfigure_flags(self) -> None:
        """--skip-validation / --reconfigure forward to the wizard."""
        with patch("app.setup.wizard.run_setup_noninteractive") as mock_run:
            result = runner.invoke(
                app,
                ["setup", "--non-interactive", "--skip-validation", "--reconfigure"],
            )
        assert result.exit_code == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["skip_validation"] is True
        assert kwargs["reconfigure"] is True

    def test_non_interactive_failure_wipes_rendered_plaintext(self) -> None:
        """A failed setup wipes any rendered plaintext secret/env files (AC3)."""
        from app.setup.wizard import SetupError

        with (
            patch(
                "app.setup.wizard.run_setup_noninteractive",
                side_effect=SetupError("bad key"),
            ),
            patch("app.setup.compose_secrets.cleanup") as secrets_cleanup,
            patch("app.setup.compose_env.cleanup") as env_cleanup,
        ):
            result = runner.invoke(
                app, ["setup", "--non-interactive", "--fred-key", "x"]
            )
        assert result.exit_code == 1
        secrets_cleanup.assert_called_once()
        env_cleanup.assert_called_once()

    def test_failed_setup_wipes_the_checkouts_plaintext_not_the_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Rendered plaintext lives under the checkout, so that is what a failed
        setup must wipe — a cwd-relative cleanup misses it when run from elsewhere."""
        from app.setup.wizard import SetupError

        repo = tmp_path / "checkout"
        (repo / "secrets").mkdir(parents=True)
        (repo / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
        (repo / "secrets" / "fred_api_key").write_text("live", encoding="utf-8")
        (repo / ".env.fund").write_text("LLM_PROVIDER=x\n", encoding="utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setenv("OPTIMIZER_REPO", str(repo))
        monkeypatch.chdir(elsewhere)
        with patch(
            "app.setup.wizard.run_setup_noninteractive",
            side_effect=SetupError("bad key"),
        ):
            result = runner.invoke(
                app, ["setup", "--non-interactive", "--fred-key", "x"]
            )
        assert result.exit_code == 1
        assert not (repo / "secrets" / "fred_api_key").exists()
        assert not (repo / ".env.fund").exists()

    def test_success_prints_post_install_note(self) -> None:
        """A successful setup surfaces the passphrase-backup + reopen guidance (AC3)."""
        with patch("app.setup.wizard.run_setup_noninteractive"):
            result = runner.invoke(app, ["setup", "--non-interactive"])
        assert result.exit_code == 0
        assert "PORTOPT_PASSPHRASE" in result.stdout

    def test_interactive_invokes_wizard(self) -> None:
        with (
            patch("app.setup.wizard.run_setup_interactive") as mock_run,
            patch("app.setup.prompts.make_prompter"),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_docker_down_exits_nonzero(self) -> None:
        from app.setup.docker_bootstrap import DockerError

        with (
            patch(
                "app.setup.wizard.run_setup_interactive",
                side_effect=DockerError("daemon down"),
            ),
            patch("app.setup.prompts.make_prompter"),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 1

    def test_corp_ca_generates_bundle_and_sets_ssl_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        bundle = tmp_path / "ca-bundle.pem"
        with (
            patch("app.setup.ca_bundle.generate", return_value=bundle) as mock_gen,
            patch("app.setup.wizard.run_setup_interactive") as mock_run,
            patch("app.setup.prompts.make_prompter"),
        ):
            result = runner.invoke(app, ["setup", "--corp-ca"])
        assert result.exit_code == 0
        mock_gen.assert_called_once()
        mock_run.assert_called_once()
        assert os.environ["SSL_CERT_FILE"] == str(bundle)
        assert os.environ["REQUESTS_CA_BUNDLE"] == str(bundle)

    def test_corp_ca_failure_exits_without_running_the_wizard(self) -> None:
        from app.setup.ca_bundle import CABundleError

        with (
            patch(
                "app.setup.ca_bundle.generate",
                side_effect=CABundleError("no powershell"),
            ),
            patch("app.setup.wizard.run_setup_interactive") as mock_run,
            patch("app.setup.prompts.make_prompter"),
        ):
            result = runner.invoke(app, ["setup", "--corp-ca"])
        assert result.exit_code == 1
        mock_run.assert_not_called()


class TestLifecycleCommands:
    """`portopt start/stop/status` wire to lifecycle and map health to exit codes."""

    def test_start_with_env_passphrase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORTOPT_PASSPHRASE", "pw")
        with patch("app.setup.lifecycle.run_start") as mock_run:
            result = runner.invoke(app, ["start"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with("pw", rebuild=False)

    def test_start_rebuild_flag_threads_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PORTOPT_PASSPHRASE", "pw")
        with patch("app.setup.lifecycle.run_start") as mock_run:
            result = runner.invoke(app, ["start", "--rebuild"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with("pw", rebuild=True)

    def test_start_failure_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORTOPT_PASSPHRASE", "pw")
        from app.setup.secret_store import InvalidPassphraseError

        with patch(
            "app.setup.lifecycle.run_start",
            side_effect=InvalidPassphraseError("wrong"),
        ):
            result = runner.invoke(app, ["start"])
        assert result.exit_code == 1

    def test_stop_invokes_lifecycle(self) -> None:
        with patch("app.setup.lifecycle.run_stop") as mock_run:
            result = runner.invoke(app, ["stop"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(all_services=False)

    def test_stop_all_flag_stops_the_whole_stack(self) -> None:
        with patch("app.setup.lifecycle.run_stop") as mock_run:
            result = runner.invoke(app, ["stop", "--all"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(all_services=True)

    def test_status_all_ok_exits_zero(self) -> None:
        with patch(
            "app.setup.lifecycle.run_status",
            return_value={"docker": True, "db": True, "scheduler": True},
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

    def test_status_degraded_exits_nonzero(self) -> None:
        with patch(
            "app.setup.lifecycle.run_status",
            return_value={"docker": True, "db": False, "scheduler": False},
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 1


class TestSingleStepCommands:
    @pytest.mark.parametrize(
        ("command", "step"),
        [
            ("macro", "run_macro_step"),
            ("news", "run_news_step"),
            ("universe", "run_universe_step"),
            ("market-structure", "run_market_structure_step"),
            ("calendars", "run_calendars_step"),
            ("market-summary", "run_market_summary_step"),
            ("options", "run_options_step"),
            ("daily-events", "run_daily_events_step"),
        ],
    )
    def test_command_invokes_matching_scheduler_step(
        self, command: str, step: str
    ) -> None:
        with patch(f"{SCHED}.{step}", return_value=True) as fn:
            result = runner.invoke(app, [command])

        assert result.exit_code == 0
        fn.assert_called_once()

    @pytest.mark.parametrize(
        ("command", "step"),
        [
            ("macro", "run_macro_step"),
            ("news", "run_news_step"),
            ("universe", "run_universe_step"),
            ("yfinance", "run_yfinance_step"),
            ("fred", "run_fred_step"),
            ("market-structure", "run_market_structure_step"),
            ("calendars", "run_calendars_step"),
            ("market-summary", "run_market_summary_step"),
            ("options", "run_options_step"),
            ("daily-events", "run_daily_events_step"),
        ],
    )
    def test_exits_nonzero_when_step_did_not_complete(
        self, command: str, step: str
    ) -> None:
        """Shell drivers gate on this — a silent 0 on failure would hide a dead fetch."""
        with patch(f"{SCHED}.{step}", return_value=False):
            result = runner.invoke(app, [command])

        assert result.exit_code == 1


class TestYfinanceOptions:
    def test_defaults_to_incremental(self) -> None:
        with patch(f"{SCHED}.run_yfinance_step", return_value=True) as fn:
            runner.invoke(app, ["yfinance"])

        assert fn.call_args.kwargs["mode"] == "incremental"

    def test_full_mode_and_period_forwarded(self) -> None:
        with patch(f"{SCHED}.run_yfinance_step", return_value=True) as fn:
            result = runner.invoke(
                app, ["yfinance", "--mode", "full", "--period", "10y", "--workers", "8"]
            )

        assert result.exit_code == 0
        assert fn.call_args.kwargs == {
            "mode": "full",
            "period": "10y",
            "workers": 8,
        }

    def test_invalid_mode_is_rejected(self) -> None:
        with patch(f"{SCHED}.run_yfinance_step", return_value=True) as fn:
            result = runner.invoke(app, ["yfinance", "--mode", "sideways"])

        assert result.exit_code != 0
        fn.assert_not_called()

    def test_workers_above_ceiling_is_rejected(self) -> None:
        with patch(f"{SCHED}.run_yfinance_step", return_value=True) as fn:
            result = runner.invoke(app, ["yfinance", "--workers", "99"])

        assert result.exit_code != 0
        fn.assert_not_called()


class TestFlagOptions:
    def test_fred_incremental_default_true(self) -> None:
        with patch(f"{SCHED}.run_fred_step", return_value=True) as fn:
            runner.invoke(app, ["fred"])

        assert fn.call_args.kwargs == {"incremental": True}

    def test_fred_no_incremental_forwarded(self) -> None:
        with patch(f"{SCHED}.run_fred_step", return_value=True) as fn:
            runner.invoke(app, ["fred", "--no-incremental"])

        assert fn.call_args.kwargs == {"incremental": False}


class TestCompositeCommands:
    def test_daily_runs_the_daily_pipeline(self) -> None:
        with patch(f"{SCHED}.run_daily_pipeline") as fn:
            result = runner.invoke(app, ["daily"])

        assert result.exit_code == 0
        fn.assert_called_once()

    def test_refetch_all_runs_universe_then_weekly_then_fred(self) -> None:
        """Universe must lead: every later step iterates the instruments it writes."""
        calls: list[str] = []

        with (
            patch(
                f"{SCHED}.run_universe_build",
                side_effect=lambda: calls.append("universe"),
            ),
            patch(
                f"{SCHED}.run_weekly_refetch",
                side_effect=lambda: calls.append("weekly"),
            ),
            patch(
                f"{SCHED}.run_fred_monthly", side_effect=lambda: calls.append("fred")
            ),
        ):
            result = runner.invoke(app, ["refetch-all"])

        assert result.exit_code == 0
        assert calls == ["universe", "weekly", "fred"]


class TestDailyEventsBackfill:
    """`daily-events --backfill START:END` builds one digest per trading day."""

    _CAL = "app.services._shared.trading_calendar.iter_trading_days"
    _BUILD = "app.services.market_data.market_journal_service.run_build_market_journal"

    def test_builds_one_digest_per_trading_day(self) -> None:
        """Each trading day in the range is built with its own ``as_of``."""
        days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        with (
            patch(self._CAL, return_value=days) as cal,
            patch(self._BUILD, return_value={}) as build,
        ):
            result = runner.invoke(
                app, ["daily-events", "--backfill", "2024-01-02:2024-01-04"]
            )

        assert result.exit_code == 0
        cal.assert_called_once()
        assert build.call_count == len(days)
        assert [c.args[0].as_of for c in build.call_args_list] == days

    def test_bad_range_exits_nonzero_without_building(self) -> None:
        """A range with no ``:`` separator is rejected before any build runs."""
        with (
            patch(self._CAL) as cal,
            patch(self._BUILD) as build,
        ):
            result = runner.invoke(app, ["daily-events", "--backfill", "not-a-range"])

        assert result.exit_code == 1
        cal.assert_not_called()
        build.assert_not_called()

    def test_start_after_end_exits_nonzero(self) -> None:
        """START later than END is rejected rather than silently building nothing."""
        with (
            patch(self._CAL) as cal,
            patch(self._BUILD) as build,
        ):
            result = runner.invoke(
                app, ["daily-events", "--backfill", "2024-02-01:2024-01-01"]
            )

        assert result.exit_code == 1
        cal.assert_not_called()
        build.assert_not_called()

    def test_empty_range_builds_nothing_but_succeeds(self) -> None:
        """A range with no trading sessions is a no-op, not a failure."""
        with (
            patch(self._CAL, return_value=[]),
            patch(self._BUILD) as build,
        ):
            result = runner.invoke(
                app, ["daily-events", "--backfill", "2024-01-06:2024-01-07"]
            )

        assert result.exit_code == 0
        build.assert_not_called()

    def test_per_day_failure_exits_nonzero_and_continues(self) -> None:
        """One failing day is logged; the rest still build and the exit is non-zero."""
        days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        with (
            patch(self._CAL, return_value=days),
            patch(self._BUILD, side_effect=[{}, RuntimeError("boom"), {}]) as build,
        ):
            result = runner.invoke(
                app, ["daily-events", "--backfill", "2024-01-02:2024-01-04"]
            )

        assert result.exit_code == 1
        assert build.call_count == 3  # continued past the failing day

    def test_over_cap_range_without_force_exits_nonzero(self) -> None:
        """A range wider than the session cap is refused unless --force is passed."""
        days = [date(2024, 1, 2)] * 1001
        with (
            patch(self._CAL, return_value=days),
            patch(self._BUILD) as build,
        ):
            result = runner.invoke(
                app, ["daily-events", "--backfill", "2000-01-01:2024-01-01"]
            )

        assert result.exit_code == 1
        build.assert_not_called()

    def test_over_cap_range_with_force_builds(self) -> None:
        """--force bypasses the cap and builds every session."""
        days = [date(2024, 1, 2), date(2024, 1, 3)] * 600  # 1200 > cap
        with (
            patch(self._CAL, return_value=days),
            patch(self._BUILD, return_value={}) as build,
        ):
            result = runner.invoke(
                app,
                ["daily-events", "--backfill", "2000-01-01:2024-01-01", "--force"],
            )

        assert result.exit_code == 0
        assert build.call_count == len(days)


class TestCredentialRedactionInLogs:
    """Entry points that configure logging also mask `?api_key=` in log text.

    FRED accepts its key only as a query parameter and urllib3 logs the request
    path at DEBUG, so ``LOG_LEVEL=DEBUG`` would otherwise write the key to the
    terminal, ``docker logs`` and any log shipper.
    """

    @pytest.mark.parametrize(
        "argv", [["setup", "--non-interactive"], ["start"], ["daily"]]
    )
    def test_entry_point_masks_api_key_in_logged_urls(
        self, argv: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        leaked = "FREDSECRETKEY0123456789abcdef"
        with (
            patch("app.setup.wizard.run_setup_noninteractive"),
            patch("app.setup.lifecycle.run_start"),
            patch(f"{SCHED}.run_daily_pipeline"),
            patch("app.services.market_data.yfinance.configure_yfinance"),
            patch.dict(os.environ, {"PORTOPT_PASSPHRASE": "pw"}),
        ):
            result = runner.invoke(app, argv)
        assert result.exit_code == 0

        logging.getLogger("urllib3.connectionpool").warning(
            '"GET /fred/series?series_id=T10Y2Y&api_key=%s HTTP/1.1" 200', leaked
        )

        assert "api_key=***" in caplog.text
        assert leaked not in caplog.text
