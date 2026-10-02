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
    python -m app.cli market-structure             # sector/industry rollups
    python -m app.cli calendars                    # earnings/IPO/splits/economic
    python -m app.cli market-summary               # regional market summaries
    python -m app.cli options                      # full option chains
    python -m app.cli daily-events                  # today's market_journal digest
    python -m app.cli daily-events --backfill 2026-01-01:2026-09-30

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
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def daily() -> None:
    """Run the full daily pipeline: ref-indices, yfinance, macro, news."""
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
    """Rebuild the instrument universe from the yfinance Screener (exchanges + instruments)."""
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


@app.command(name="market-structure")
def market_structure() -> None:
    """Fetch sector/industry rollups across regions into sector_* tables."""
    _boot()
    from app.services.jobs.scheduler import run_market_structure_step

    _exit(run_market_structure_step())


@app.command()
def calendars() -> None:
    """Fetch market-wide earnings/IPO/splits/economic calendars."""
    _boot()
    from app.services.jobs.scheduler import run_calendars_step

    _exit(run_calendars_step())


@app.command(name="market-summary")
def market_summary() -> None:
    """Fetch regional market summaries into market_summaries."""
    _boot()
    from app.services.jobs.scheduler import run_market_summary_step

    _exit(run_market_summary_step())


@app.command()
def options() -> None:
    """Fetch full option chains into options_chain (own staleness gate)."""
    _boot()
    from app.services.jobs.scheduler import run_options_step

    _exit(run_options_step())


@app.command(name="daily-events")
def daily_events(
    backfill: str | None = typer.Option(
        None,
        "--backfill",
        help="Backfill a 'START:END' ISO-date range, one digest per trading day.",
    ),
) -> None:
    """Build the global daily digest (market_journal) for today, or a range."""
    _boot()
    if backfill is None:
        from app.services.jobs.scheduler import run_daily_events_step

        _exit(run_daily_events_step())
        return
    _exit(_run_daily_events_backfill(backfill))


def _run_daily_events_backfill(spec: str) -> bool:
    """Build one digest per trading day across a ``START:END`` ISO-date range.

    Runs the builder directly rather than the slotted ``run_daily_events_step``:
    a historical backfill can span hundreds of sessions, so it skips the per-day
    job slot and heartbeat the single-day path uses. Each day is an idempotent
    upsert, so re-running the range is safe.

    Returns ``True`` when the range parsed and every session built, ``False`` on
    a malformed or reversed range (the caller maps this to a non-zero exit).
    """
    from datetime import date

    from app.schemas.market_data.market_journal import MarketJournalBuildRequest
    from app.services._shared.trading_calendar import iter_trading_days
    from app.services.market_data.market_journal_service import (
        run_build_market_journal,
    )

    try:
        start_str, end_str = spec.split(":", 1)
        start = date.fromisoformat(start_str.strip())
        end = date.fromisoformat(end_str.strip())
    except ValueError:
        typer.echo(
            f"Invalid --backfill range {spec!r}; expected 'START:END' ISO dates "
            "(e.g. 2026-01-01:2026-09-30).",
            err=True,
        )
        return False

    if start > end:
        typer.echo(
            f"Invalid --backfill range {spec!r}: START must not be after END.",
            err=True,
        )
        return False

    days = iter_trading_days(start, end)
    for day in days:
        run_build_market_journal(MarketJournalBuildRequest(as_of=day))
    typer.echo(f"daily-events backfill: built {len(days)} digest(s) for {spec}.")
    return True


@app.command()
def setup(
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Run without prompts (CI); requires the flags below + PORTOPT_PASSPHRASE.",
    ),
    t212_key: str | None = typer.Option(None, "--t212-key", help="Trading212 API key."),
    t212_secret: str | None = typer.Option(
        None, "--t212-secret", help="Trading212 secret key."
    ),
    fred_key: str | None = typer.Option(None, "--fred-key", help="FRED API key."),
    llm_provider: str | None = typer.Option(
        None,
        "--llm-provider",
        help="Fund LLM provider (ollama, openai, anthropic, ...).",
    ),
    llm_model: str | None = typer.Option(
        None, "--llm-model", help="Fund LLM model id."
    ),
    llm_base_url: str | None = typer.Option(
        None,
        "--llm-base-url",
        help="Override the hosted default (local ollama, self-hosted NIM, proxy).",
    ),
    llm_key: str | None = typer.Option(
        None,
        "--llm-key",
        help="LLM provider API key (or the provider's own env var, e.g. OPENAI_API_KEY).",
    ),
    corp_ca: bool = typer.Option(
        False,
        "--corp-ca",
        help="Generate .certs/ca-bundle.pem (certifi + machine roots) for TLS-inspecting proxies.",
    ),
    skip_path_install: bool = typer.Option(
        False,
        "--skip-path-install",
        help="Do not install the `optimizer` launcher onto PATH (CI / manual PATH setup).",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Persist credentials without the live pre-flight checks (offline / CI).",
    ),
    no_launch: bool = typer.Option(
        False,
        "--no-launch",
        help="Configure only; do not bring the stack up at the end of a first setup.",
    ),
    reconfigure: bool = typer.Option(
        False,
        "--reconfigure",
        help="Re-prompt configured sections on a re-run (existing secrets are kept).",
    ),
) -> None:
    """Install wizard: verify Docker, validate keys live, encrypt secrets, migrate the DB."""
    logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
    from app.setup import (
        ca_bundle,
        compose_env,
        compose_secrets,
        docker_bootstrap,
        wizard,
    )
    from app.setup.ca_bundle import CABundleError
    from app.setup.prompts import PromptError, make_prompter
    from app.setup.validators import ValidationNetworkError

    # Non-interactive/CI must never silently mutate the User PATH:
    # `--non-interactive` implies skip unless the operator opts in via interactive setup.
    effective_skip_path_install = skip_path_install or non_interactive
    try:
        if corp_ca:
            # Generate the merged bundle first and point this process's TLS stack at
            # it, so the live key validation below trusts the corporate proxy's root.
            bundle = ca_bundle.generate()
            os.environ["SSL_CERT_FILE"] = str(bundle)
            os.environ["REQUESTS_CA_BUNDLE"] = str(bundle)
            typer.echo(f"Corporate CA bundle written to {bundle}.")
        if non_interactive:
            wizard.run_setup_noninteractive(
                passphrase=os.getenv("PORTOPT_PASSPHRASE"),
                t212_key=t212_key,
                t212_secret=t212_secret,
                fred_key=fred_key,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_key=llm_key,
                skip_path_install=effective_skip_path_install,
                skip_validation=skip_validation,
                reconfigure=reconfigure,
            )
        else:
            wizard.run_setup_interactive(
                make_prompter(),
                skip_path_install=effective_skip_path_install,
                skip_validation=skip_validation,
                reconfigure=reconfigure,
                no_launch=no_launch,
            )
    except (
        wizard.SetupError,
        docker_bootstrap.DockerError,
        ValidationNetworkError,
        PromptError,
        CABundleError,
    ) as exc:
        # All-or-nothing: wipe any plaintext secret/env files a partial run rendered so
        # a failed setup never leaves decrypted credentials on disk.
        compose_secrets.cleanup()
        compose_env.cleanup()
        typer.echo(f"Setup failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_post_install_note()


def _print_post_install_note() -> None:
    typer.echo("Setup complete.")
    typer.echo(
        "  - Back up your PORTOPT_PASSPHRASE — it is never stored and is the only key "
        "to your encrypted secrets."
    )
    typer.echo("  - Adminer: http://localhost:18081   metrics: http://localhost:9000")
    typer.echo(
        "  - Open a new terminal (or run `hash -r`) so the `optimizer` launcher "
        "resolves on PATH, then run `optimizer <portfolio_id>`."
    )


@app.command()
def start() -> None:
    """Decrypt secrets, mount them as compose secrets, and launch the stack."""
    logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
    from app.setup import lifecycle
    from app.setup.docker_bootstrap import DockerError
    from app.setup.prompts import PromptError, make_prompter
    from app.setup.secret_store import SecretStoreError

    passphrase = os.getenv("PORTOPT_PASSPHRASE")
    try:
        if not passphrase:
            passphrase = make_prompter().password("Master passphrase:")
        lifecycle.run_start(passphrase)
    except (
        lifecycle.LifecycleError,
        DockerError,
        SecretStoreError,
        PromptError,
    ) as exc:
        typer.echo(f"Start failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("portopt is running.")


@app.command()
def stop() -> None:
    """Stop the stack and remove the rendered plaintext secret files."""
    from app.setup import lifecycle
    from app.setup.docker_bootstrap import DockerError

    try:
        lifecycle.run_stop()
    except DockerError as exc:
        typer.echo(f"Stop failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("portopt stopped.")


@app.command()
def status() -> None:
    """Report Docker + service health (exit non-zero if anything is down)."""
    from app.setup import lifecycle

    report = lifecycle.run_status()
    for name, ok in report.items():
        typer.echo(f"{name}: {'ok' if ok else 'down'}")
    _exit(all(report.values()))


if __name__ == "__main__":
    app()
