"""Tests for the manual-run CLI (``python -m app.cli``).

Every scheduler step function is patched out, so nothing fetches or touches a
database. What is asserted is the contract the shell drivers depend on:

- each command dispatches to the step the scheduler itself runs (so a manual
  run and a scheduled run take the same job-slot / heartbeat path)
- options reach that step with the right values
- single-step commands exit non-zero when the step did not complete, so a
  shell driver or CI can gate on the exit code
- the composite commands (``daily``, ``refetch-all``) exit 0 whatever their
  steps did, so ``scheduler/fetch.sh`` and ``scheduler/refetch_all.sh``, which
  wrap them, are not a failure signal
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import typer.main
from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIFECYCLE_COMMANDS = frozenset({"setup", "start", "stop", "status"})

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

    def test_non_interactive_reads_credentials_from_env(self) -> None:
        """Env vars are a first-class channel so CI keeps keys out of argv.

        A flag value is visible in ``ps``, ``/proc/<pid>/cmdline`` and shell history,
        so the same names the interactive wizard auto-detects must also feed the
        non-interactive path.
        """
        env = {
            "TRADING_212_API_KEY": "env-tk",
            "TRADING_212_SECRET_KEY": "env-ts",
            "FRED_API_KEY": "env-fk",
        }
        with patch("app.setup.wizard.run_setup_noninteractive") as mock_run:
            result = runner.invoke(app, ["setup", "--non-interactive"], env=env)
        assert result.exit_code == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["t212_key"] == "env-tk"
        assert kwargs["t212_secret"] == "env-ts"  # noqa: S105 - test value, not a secret
        assert kwargs["fred_key"] == "env-fk"

    def test_non_interactive_flag_takes_precedence_over_env(self) -> None:
        """An explicit flag wins over an exported env var."""
        with patch("app.setup.wizard.run_setup_noninteractive") as mock_run:
            result = runner.invoke(
                app,
                ["setup", "--non-interactive", "--fred-key", "flag-fk"],
                env={"FRED_API_KEY": "env-fk"},
            )
        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["fred_key"] == "flag-fk"

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

    def test_llm_clear_sentinel_reaches_the_wizard_untouched(self) -> None:
        """``-`` is the "clear the stored value" marker; the CLI must not eat it."""
        with patch("app.setup.wizard.run_setup_noninteractive") as mock_run:
            result = runner.invoke(
                app,
                [
                    "setup",
                    "--non-interactive",
                    "--llm-provider",
                    "ollama",
                    "--llm-base-url",
                    "-",
                    "--llm-model",
                    "none",
                ],
            )
        assert result.exit_code == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["llm_base_url"] == "-"
        assert kwargs["llm_model"] == "none"

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

    @pytest.mark.parametrize(
        "failure_args",
        [
            ("app.setup.wizard.run_setup_noninteractive", ["--non-interactive"]),
            ("app.setup.wizard.run_setup_interactive", []),
        ],
    )
    def test_failed_rerun_keeps_files_the_running_stack_mounts(
        self,
        failure_args: tuple[str, list[str]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """A failed re-run must not delete the secret files `portopt start` rendered.

        The scheduler (restart: unless-stopped) keeps ./secrets and .env.fund
        bind-mounted, and setup renders nothing itself, so removing them on a
        mistyped passphrase leaves Docker to plant empty directories at the next
        restart and the daemon silently loses its API keys.
        """
        from app.setup import compose_env, compose_secrets
        from app.setup.wizard import SetupError

        monkeypatch.chdir(tmp_path)
        rendered = compose_secrets.render(
            {"fred_api_key": "live-key"}, secrets_dir=tmp_path / "secrets"
        )
        env_file = compose_env.render(
            {"llm_provider": "openai"}, path=tmp_path / ".env.fund"
        )
        target, flags = failure_args

        with (
            patch(target, side_effect=SetupError("could not be decrypted")),
            patch("app.setup.prompts.make_prompter"),
        ):
            result = runner.invoke(app, ["setup", *flags])

        assert result.exit_code == 1
        assert all(path.is_file() for path in rendered)
        assert (tmp_path / "secrets" / "fred_api_key").read_text() == "live-key"
        assert env_file.is_file()

    def test_failed_setup_leaves_the_checkouts_rendered_files_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Rendered plaintext lives under the checkout and the running stack mounts
        it, so a failed setup run from another directory must not delete it."""
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
        assert (repo / "secrets" / "fred_api_key").read_text() == "live"
        assert (repo / ".env.fund").is_file()

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
        mock_run.assert_called_once_with(all_services=False, force=False)

    def test_stop_all_flag_stops_the_whole_stack(self) -> None:
        with patch("app.setup.lifecycle.run_stop") as mock_run:
            result = runner.invoke(app, ["stop", "--all"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(all_services=True, force=False)

    def test_stop_force_flag_stops_despite_a_running_round(self) -> None:
        with patch("app.setup.lifecycle.run_stop") as mock_run:
            result = runner.invoke(app, ["stop", "--force"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(all_services=False, force=True)

    def test_stop_without_a_checkout_exits_nonzero(self) -> None:
        from app.setup.lifecycle import LifecycleError

        with patch(
            "app.setup.lifecycle.run_stop", side_effect=LifecycleError("no checkout")
        ):
            result = runner.invoke(app, ["stop"])
        assert result.exit_code == 1
        assert "no checkout" in result.output

    def test_status_without_a_checkout_exits_nonzero(self) -> None:
        from app.setup.lifecycle import LifecycleError

        with patch(
            "app.setup.lifecycle.run_status", side_effect=LifecycleError("no checkout")
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert "no checkout" in result.output

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

    @pytest.mark.parametrize("period", ["5yrs", "5", "0y", "MAX"])
    def test_malformed_period_is_a_usage_error_not_an_empty_sweep(
        self, period: str
    ) -> None:
        """yfinance returns an empty frame for a bad period, which would otherwise
        complete as a zero-row success and exit 0."""
        with patch(f"{SCHED}.run_yfinance_step", return_value=True) as fn:
            result = runner.invoke(
                app, ["yfinance", "--mode", "full", "--period", period]
            )

        assert result.exit_code == 2
        fn.assert_not_called()

    @pytest.mark.parametrize("period", ["max", "ytd", "6mo", "60d"])
    def test_named_and_unit_periods_are_forwarded(self, period: str) -> None:
        with patch(f"{SCHED}.run_yfinance_step", return_value=True) as fn:
            result = runner.invoke(app, ["yfinance", "--period", period])

        assert result.exit_code == 0
        assert fn.call_args.kwargs["period"] == period


class TestFlagOptions:
    def test_fred_incremental_default_true(self) -> None:
        with patch(f"{SCHED}.run_fred_step", return_value=True) as fn:
            runner.invoke(app, ["fred"])

        assert fn.call_args.kwargs == {"incremental": True}

    def test_fred_no_incremental_forwarded(self) -> None:
        with patch(f"{SCHED}.run_fred_step", return_value=True) as fn:
            runner.invoke(app, ["fred", "--no-incremental"])

        assert fn.call_args.kwargs == {"incremental": False}

    def test_fred_incremental_help_describes_the_trailing_window(self) -> None:
        """Incremental re-fetches a window behind the last date; help must say so."""
        result = runner.invoke(app, ["fred", "--help"])

        help_text = " ".join(result.output.split())
        assert "trailing" in help_text
        assert "newer than" not in help_text


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
                f"{SCHED}.run_fred_step",
                side_effect=lambda **kw: calls.append(f"fred:{kw}"),
            ),
        ):
            result = runner.invoke(app, ["refetch-all"])

        assert result.exit_code == 0
        assert calls == ["universe", "weekly", "fred:{'incremental': False}"]

    def test_daily_exits_zero_when_every_step_fails(self) -> None:
        """The wrappers' "do not gate on the exit status" advice rests on this."""
        with (
            patch(f"{SCHED}.run_yfinance_step", return_value=False),
            patch(f"{SCHED}.run_fx_step", return_value=False),
            patch(f"{SCHED}.run_macro_step", return_value=False),
            patch(f"{SCHED}.run_news_step", return_value=False) as news,
            patch(f"{SCHED}.run_daily_events_step", return_value=False),
        ):
            result = runner.invoke(app, ["daily"])

        assert result.exit_code == 0
        news.assert_not_called()

    def test_refetch_all_exits_zero_when_every_step_fails(self) -> None:
        with (
            patch(f"{SCHED}.run_universe_step", return_value=False),
            patch(f"{SCHED}.run_t212_annotate_step", return_value=False),
            patch(f"{SCHED}.run_yfinance_step", return_value=False),
            patch(f"{SCHED}.run_macro_step", return_value=False),
            patch(f"{SCHED}.run_fred_step", return_value=False),
        ):
            result = runner.invoke(app, ["refetch-all"])

        assert result.exit_code == 0


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

    def test_future_end_is_clamped_to_today(self) -> None:
        """A digest for a date that has not happened yet would carry today's data."""
        today = date.today()
        with (
            patch(self._CAL, return_value=[]) as cal,
            patch(self._BUILD) as build,
        ):
            result = runner.invoke(
                app, ["daily-events", "--backfill", "2024-01-02:2999-12-31"]
            )

        assert result.exit_code == 0
        assert cal.call_args.args == (date(2024, 1, 2), today)
        build.assert_not_called()

    def test_start_in_the_future_is_rejected(self) -> None:
        """A range that begins after today has no historical sessions to build."""
        with (
            patch(self._CAL) as cal,
            patch(self._BUILD) as build,
        ):
            result = runner.invoke(
                app, ["daily-events", "--backfill", "2998-01-02:2999-12-31"]
            )

        assert result.exit_code == 1
        cal.assert_not_called()
        build.assert_not_called()

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


def _registered_commands() -> set[str]:
    return set(typer.main.get_command(app).commands)  # type: ignore[attr-defined]


_INVOCATION = re.compile(r"app\.cli ([a-z][a-z-]*)")
_COMMAND_TOKEN = re.compile(r"[a-z][a-z-]*")


def _documented_commands(doc: Path) -> set[str]:
    """Collect the CLI commands a doc advertises, in any of the forms it uses.

    Covers ``python -m app.cli <cmd>`` invocations, ``# also: a | b`` summary
    comments, and the backticked list under "Full CLI surface". A trailing
    parenthetical (``(there is NO ...)``) is prose, not a command list.
    """
    text = doc.read_text(encoding="utf-8")
    found = set(_INVOCATION.findall(text))
    for line in text.splitlines():
        match = re.match(r"#\s*(?:also|lifecycle):\s*(.+)", line)
        if match:
            summary = match.group(1).split("(", 1)[0]
            found |= {
                token.strip()
                for token in summary.split("|")
                if _COMMAND_TOKEN.fullmatch(token.strip())
            }
    surface = re.search(r"Full CLI surface.*?\n\n", text, flags=re.DOTALL)
    if surface:
        found |= {
            token
            for token in re.findall(r"`([^`]+)`", surface.group(0))
            if _COMMAND_TOKEN.fullmatch(token)
        }
    return found


_DOCS = [
    pytest.param(_REPO_ROOT / "README.md", id="README.md"),
    pytest.param(_REPO_ROOT / "CLAUDE.md", id="CLAUDE.md"),
    pytest.param(_REPO_ROOT / "ingestion" / "README.md", id="ingestion-README.md"),
]


class TestCliSurfaceMatchesTooling:
    """Launch configs and docs name only commands the Typer app registers."""

    def test_vscode_launch_configs_run_only_registered_commands(self) -> None:
        raw = (_REPO_ROOT / ".vscode" / "launch.json").read_text(encoding="utf-8")
        configs = json.loads(re.sub(r"(?m)^\s*//.*$", "", raw))["configurations"]
        registered = _registered_commands()

        launched = {
            cfg["name"]: cfg["args"][0]
            for cfg in configs
            if cfg.get("module") == "app.cli"
        }

        assert launched, "launch.json no longer has any app.cli configuration"
        dead = {name: cmd for name, cmd in launched.items() if cmd not in registered}
        assert not dead, f"launch configs run unregistered CLI commands: {dead}"

    @pytest.mark.parametrize("doc", _DOCS)
    def test_docs_advertise_only_registered_commands(self, doc: Path) -> None:
        dead = _documented_commands(doc) - _registered_commands()

        assert not dead, f"{doc.name} advertises unregistered commands: {dead}"

    @pytest.mark.parametrize("doc", _DOCS)
    def test_docs_mention_every_registered_command(self, doc: Path) -> None:
        required = _registered_commands()
        if doc.parent.name == "ingestion":
            # The manual-run list drives `docker compose exec scheduler`; the
            # lifecycle commands run on the host, not inside that container.
            required -= _LIFECYCLE_COMMANDS

        missing = required - _documented_commands(doc)

        assert not missing, f"{doc.name} never mentions CLI commands: {missing}"
