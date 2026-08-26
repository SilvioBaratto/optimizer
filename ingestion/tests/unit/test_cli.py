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
                    "--llm-provider",
                    "openai",
                    "--llm-key",
                    "sk-x",
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
        assert kwargs["llm_provider"] == "openai"
        assert kwargs["llm_key"] == "sk-x"
        assert kwargs["t212_key"] == "tk"
        assert kwargs["t212_secret"] == "ts"  # noqa: S105 - test value, not a secret
        assert kwargs["fred_key"] == "fk"

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
                    "--llm-provider",
                    "openai",
                    "--llm-key",
                    "x",
                ],
            )
        assert result.exit_code == 1

    def test_interactive_invokes_wizard(self) -> None:
        with (
            patch("app.setup.wizard.run_setup_interactive") as mock_run,
            patch("app.setup.prompts.QuestionaryPrompter"),
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
            patch("app.setup.prompts.QuestionaryPrompter"),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 1


class TestSingleStepCommands:
    @pytest.mark.parametrize(
        ("command", "step"),
        [
            ("macro", "run_macro_step"),
            ("news", "run_news_step"),
            ("calibrate", "run_calibrate_step"),
            ("universe", "run_universe_step"),
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
            ("calibrate", "run_calibrate_step"),
            ("universe", "run_universe_step"),
            ("yfinance", "run_yfinance_step"),
            ("fred", "run_fred_step"),
            ("summarize", "run_summarize_step"),
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

    def test_summarize_force_refresh_default_true(self) -> None:
        with patch(f"{SCHED}.run_summarize_step", return_value=True) as fn:
            runner.invoke(app, ["summarize"])

        assert fn.call_args.kwargs == {"force_refresh": True}

    def test_summarize_no_force_refresh_forwarded(self) -> None:
        with patch(f"{SCHED}.run_summarize_step", return_value=True) as fn:
            runner.invoke(app, ["summarize", "--no-force-refresh"])

        assert fn.call_args.kwargs == {"force_refresh": False}


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
