"""Manual ingestion runs: ``python -m app.cli <command>``.

Every command runs the same step function the scheduler runs, through the same
``BackgroundJobService`` lifecycle — so a manual run claims a job slot, writes
heartbeats, lands in ``background_jobs``, and is refused if the scheduler is
already running that step.

    python -m app.cli daily                       # full daily pipeline
    python -m app.cli refetch-all                 # universe + yfinance + macro + fred
    python -m app.cli universe                    # Trading 212 universe rebuild
    python -m app.cli yfinance --mode full --period 5y
    python -m app.cli macro
    python -m app.cli fred
    python -m app.cli news
    python -m app.cli summarize
    python -m app.cli calibrate

Single-step commands exit non-zero when the step did not complete, so shell
drivers can gate on them. The composite commands (``daily``, ``refetch-all``)
apply their own internal gating and always exit 0 — read the logs or
``background_jobs`` for per-step outcomes.
"""

from __future__ import annotations

# Load environment variables FIRST, before any other import reads them.
from dotenv import load_dotenv

load_dotenv()

import logging
import os
from enum import Enum

import typer

from app.config import settings
from app.database import init_db

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Manual triggers for the ingestion pipeline.",
)


class FetchMode(str, Enum):
    """yfinance fetch mode."""

    INCREMENTAL = "incremental"
    FULL = "full"


def _boot() -> None:
    """Configure logging and the DB engine — every command needs both."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    init_db()


def _exit(ok: bool) -> None:
    """Exit non-zero when the step did not complete."""
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def daily() -> None:
    """Run the full daily pipeline: ref-indices, yfinance, macro, news, summarize, calibrate."""
    _boot()
    from app.services.jobs.scheduler import run_daily_pipeline

    run_daily_pipeline()


@app.command(name="refetch-all")
def refetch_all() -> None:
    """Full rebuild: universe, then yfinance + macro (5y), then FRED."""
    _boot()
    from app.services.jobs.scheduler import (
        run_fred_monthly,
        run_universe_build,
        run_weekly_refetch,
    )

    run_universe_build()
    run_weekly_refetch()
    run_fred_monthly()


@app.command()
def universe() -> None:
    """Rebuild the Trading 212 instrument universe (exchanges + instruments)."""
    _boot()
    from app.services.jobs.scheduler import run_universe_step

    _exit(run_universe_step())


@app.command()
def yfinance(
    mode: FetchMode = typer.Option(
        FetchMode.INCREMENTAL,
        help="'incremental' fetches only what is missing; 'full' re-downloads.",
    ),
    period: str = typer.Option("5y", help="Lookback window for mode=full."),
    workers: int = typer.Option(
        settings.yfinance_fetch_workers,
        min=1,
        max=16,
        help="Parallel fetch workers. 1 uses the serial path.",
    ),
) -> None:
    """Fetch prices, fundamentals, holders, and news for every instrument."""
    _boot()
    from app.services.jobs.scheduler import run_yfinance_step

    _exit(run_yfinance_step(mode=mode.value, period=period, workers=workers))


@app.command()
def macro() -> None:
    """Scrape Il Sole 24 Ore + Trading Economics into bond_yields / indicators."""
    _boot()
    from app.services.jobs.scheduler import run_macro_step

    _exit(run_macro_step())


@app.command()
def fred(
    incremental: bool = typer.Option(
        True,
        help="Fetch only observations newer than what is stored.",
    ),
) -> None:
    """Fetch FRED economic series into economic_indicators / fred_observations."""
    _boot()
    from app.services.jobs.scheduler import run_fred_step

    _exit(run_fred_step(incremental=incremental))


@app.command()
def news() -> None:
    """Fetch macro news articles into macro_news."""
    _boot()
    from app.services.jobs.scheduler import run_news_step

    _exit(run_news_step())


@app.command()
def summarize(
    force_refresh: bool = typer.Option(
        True,
        help="Re-summarize countries even when a summary already exists for today.",
    ),
) -> None:
    """LLM-summarize macro news into macro_news_summaries / macro_news_themes."""
    _boot()
    from app.services.jobs.scheduler import run_summarize_step

    _exit(run_summarize_step(force_refresh=force_refresh))


@app.command()
def calibrate() -> None:
    """LLM-calibrate the macro regime (delta / tau) into macro_calibrations."""
    _boot()
    from app.services.jobs.scheduler import run_calibrate_step

    _exit(run_calibrate_step())


@app.command()
def setup(
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Run without prompts (CI); requires the flags below + PORTOPT_PASSPHRASE.",
    ),
    llm_provider: str | None = typer.Option(
        None, "--llm-provider", help="Cloud LLM provider: openai or anthropic."
    ),
    llm_key: str | None = typer.Option(None, "--llm-key", help="Cloud LLM API key."),
    t212_key: str | None = typer.Option(None, "--t212-key", help="Trading212 API key."),
    t212_secret: str | None = typer.Option(
        None, "--t212-secret", help="Trading212 secret key."
    ),
    fred_key: str | None = typer.Option(None, "--fred-key", help="FRED API key."),
) -> None:
    """Install wizard: verify Docker, validate keys live, encrypt secrets, migrate the DB."""
    logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
    from app.setup import docker_bootstrap, wizard
    from app.setup.prompts import QuestionaryPrompter
    from app.setup.validators import ValidationNetworkError

    try:
        if non_interactive:
            wizard.run_setup_noninteractive(
                passphrase=os.getenv("PORTOPT_PASSPHRASE"),
                llm_provider=llm_provider,
                llm_key=llm_key,
                t212_key=t212_key,
                t212_secret=t212_secret,
                fred_key=fred_key,
            )
        else:
            wizard.run_setup_interactive(QuestionaryPrompter())
    except (
        wizard.SetupError,
        docker_bootstrap.DockerError,
        ValidationNetworkError,
    ) as exc:
        typer.echo(f"Setup failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Setup complete. Run `portopt start` to launch the daemon.")


if __name__ == "__main__":
    app()
