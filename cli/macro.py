"""Macro data command group — maps to /api/v1/macro-data endpoints."""

from __future__ import annotations

import typer

from cli.client import ApiClient
from cli.display import (
    dict_table,
    error_panel,
    list_table,
    progress_loop,
    success_panel,
)

macro_app = typer.Typer(name="macro", help="Fetch and query macroeconomic data.")


def _client(ctx: typer.Context) -> ApiClient:
    return ctx.obj


# ------------------------------------------------------------------
# fetch (bulk)
# ------------------------------------------------------------------


@macro_app.command()
def fetch(
    ctx: typer.Context,
    country: list[str] | None = typer.Option(
        None, help="Country name(s). Repeat for multiple."
    ),
    no_bonds: bool = typer.Option(False, "--no-bonds", help="Skip bond yield scraping"),
) -> None:
    """Start a bulk macro data fetch for all portfolio countries and poll until complete."""
    client = _client(ctx)
    job = client.start_macro_fetch(
        countries=country or None,
        include_bonds=not no_bonds,
    )
    job_id = job["job_id"]
    success_panel(f"Macro fetch started: {job_id}")

    result = progress_loop(lambda: client.get_macro_fetch_status(job_id))

    if result.get("status") == "failed":
        error_panel(f"Fetch failed: {result.get('error', 'unknown')}")
        raise typer.Exit(code=1)

    fetch_result = result.get("result", {})
    if fetch_result:
        dict_table(fetch_result, title="Macro Fetch Result")
    success_panel("Macro fetch completed.")


# ------------------------------------------------------------------
# fetch-status
# ------------------------------------------------------------------


@macro_app.command("fetch-status")
def fetch_status(
    ctx: typer.Context,
    job_id: str = typer.Argument(help="Fetch job UUID"),
) -> None:
    """Check the status of a macro fetch job."""
    data = _client(ctx).get_macro_fetch_status(job_id)
    dict_table(data, title=f"Macro Fetch Job {job_id}")


# ------------------------------------------------------------------
# fetch-country (single sync fetch)
# ------------------------------------------------------------------


@macro_app.command("fetch-country")
def fetch_country(
    ctx: typer.Context,
    country: str = typer.Argument(help="Country name (e.g. USA, Germany, UK)"),
    no_bonds: bool = typer.Option(False, "--no-bonds", help="Skip bond yield scraping"),
) -> None:
    """Synchronously fetch macro data for a single country."""
    data = _client(ctx).fetch_macro_country(country, include_bonds=not no_bonds)
    counts = data.get("counts", {})
    dict_table(
        {"country": data.get("country"), **counts},
        title=f"Macro Fetch Result: {country}",
    )
    errors = data.get("errors", [])
    if errors:
        error_panel("\n".join(errors))


# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------


@macro_app.command()
def summary(
    ctx: typer.Context,
    country: str = typer.Argument(help="Country name"),
) -> None:
    """Show all macro data for a country."""
    data = _client(ctx).get_country_summary(country)

    ei = data.get("economic_indicators", [])
    if ei:
        list_table(
            ei,
            columns=[
                "source",
                "gdp_growth_qq",
                "unemployment",
                "consumer_prices",
                "st_rate",
                "lt_rate",
            ],
            title=f"{country} — Economic Indicators",
        )

    te = data.get("te_indicators", [])
    if te:
        list_table(
            te,
            columns=["indicator_key", "value", "previous", "unit", "reference"],
            title=f"{country} — Trading Economics",
        )

    by = data.get("bond_yields", [])
    if by:
        list_table(
            by,
            columns=[
                "maturity",
                "yield_value",
                "day_change",
                "month_change",
                "year_change",
            ],
            title=f"{country} — Bond Yields",
        )


# ------------------------------------------------------------------
# economic-indicators
# ------------------------------------------------------------------


@macro_app.command("economic-indicators")
def economic_indicators(
    ctx: typer.Context,
    country: str | None = typer.Option(None, help="Filter by country"),
) -> None:
    """List stored economic indicators."""
    rows = _client(ctx).get_economic_indicators(country=country)
    list_table(
        rows,
        columns=[
            "country",
            "source",
            "gdp_growth_qq",
            "unemployment",
            "consumer_prices",
            "st_rate",
            "lt_rate",
        ],
        title="Economic Indicators",
    )


# ------------------------------------------------------------------
# te-indicators
# ------------------------------------------------------------------


@macro_app.command("te-indicators")
def te_indicators(
    ctx: typer.Context,
    country: str | None = typer.Option(None, help="Filter by country"),
) -> None:
    """List stored Trading Economics indicators."""
    rows = _client(ctx).get_te_indicators(country=country)
    list_table(
        rows,
        columns=["country", "indicator_key", "value", "previous", "unit", "reference"],
        title="Trading Economics Indicators",
    )


# ------------------------------------------------------------------
# bond-yields
# ------------------------------------------------------------------


@macro_app.command("bond-yields")
def bond_yields(
    ctx: typer.Context,
    country: str | None = typer.Option(None, help="Filter by country"),
) -> None:
    """List stored bond yields."""
    rows = _client(ctx).get_bond_yields(country=country)
    list_table(
        rows,
        columns=[
            "country",
            "maturity",
            "yield_value",
            "day_change",
            "month_change",
            "year_change",
        ],
        title="Bond Yields",
    )


# ------------------------------------------------------------------
# Macro news
# ------------------------------------------------------------------


@macro_app.command("news-fetch")
def news_fetch(
    ctx: typer.Context,
    max_articles: int = typer.Option(100, help="Max articles to fetch"),
    full_content: bool = typer.Option(
        False, "--full-content", help="Scrape full article content",
    ),
) -> None:
    """Fetch macro-themed news from yfinance and store in DB."""
    client = _client(ctx)
    job = client.start_macro_news_fetch(
        max_articles=max_articles,
        fetch_full_content=full_content,
    )
    job_id = job["job_id"]
    success_panel(f"Macro news fetch started: {job_id}")

    result = progress_loop(lambda: client.get_macro_news_fetch_status(job_id))

    if result.get("status") == "failed":
        error_panel(f"News fetch failed: {result.get('error', 'unknown')}")
        raise typer.Exit(code=1)

    fetch_result = result.get("result", {})
    if fetch_result:
        dict_table(fetch_result, title="Macro News Fetch Result")
    success_panel("Macro news fetch completed.")


@macro_app.command("news")
def news(
    ctx: typer.Context,
    theme: str | None = typer.Option(None, help="Filter by macro theme"),
    limit: int = typer.Option(20, help="Max rows to display"),
) -> None:
    """List stored macro news articles."""
    rows = _client(ctx).get_macro_news(theme=theme, limit=limit)
    list_table(
        rows,
        columns=["title", "publisher", "themes", "publish_time", "source_ticker"],
        title="Macro News",
    )


# ------------------------------------------------------------------
# news summarize
# ------------------------------------------------------------------


@macro_app.command("summarize")
def summarize(
    ctx: typer.Context,
    country: list[str] | None = typer.Option(
        None, help="Country name(s) to summarize. Repeat for multiple. Default: all."
    ),
    force_refresh: bool = typer.Option(
        False, "--force-refresh", help="Bypass daily cache and re-invoke the LLM."
    ),
) -> None:
    """Generate AI news summaries for macro countries via BAML LLM."""
    client = _client(ctx)
    job = client.start_news_summarize(
        countries=country or None,
        force_refresh=force_refresh,
    )
    job_id = job["job_id"]
    success_panel(f"News summarize started: {job_id}")

    result = progress_loop(lambda: client.get_news_summarize_status(job_id))

    if result.get("status") == "failed":
        error_panel(f"Summarize failed: {result.get('error', 'unknown')}")
        raise typer.Exit(code=1)

    fetch_result = result.get("result", {})
    if fetch_result:
        dict_table(fetch_result, title="News Summarize Result")
    success_panel("News summarize completed.")


# ------------------------------------------------------------------
# FRED time-series
# ------------------------------------------------------------------


@macro_app.command("fetch-fred")
def fetch_fred(
    ctx: typer.Context,
    series: list[str] | None = typer.Option(
        None, help="FRED series IDs. Repeat for multiple (default: all configured)."
    ),
    full: bool = typer.Option(
        False, "--full", help="Full history fetch. Default is incremental."
    ),
) -> None:
    """Fetch FRED credit/yield spread time series and store in DB."""
    client = _client(ctx)
    job = client.start_fred_fetch(
        series_ids=series or None,
        incremental=not full,
    )
    job_id = job["job_id"]
    success_panel(f"FRED fetch started: {job_id}")

    result = progress_loop(lambda: client.get_fred_fetch_status(job_id))

    if result.get("status") == "failed":
        error_panel(f"FRED fetch failed: {result.get('error', 'unknown')}")
        raise typer.Exit(code=1)

    fetch_result = result.get("result", {})
    if fetch_result:
        dict_table(fetch_result, title="FRED Fetch Result (series_id: rows_upserted)")
    success_panel("FRED fetch completed.")


@macro_app.command("fred-series")
def fred_series_cmd(
    ctx: typer.Context,
    series_id: str | None = typer.Option(None, help="Filter by series ID"),
    start_date: str | None = typer.Option(None, help="Start date YYYY-MM-DD"),
    end_date: str | None = typer.Option(None, help="End date YYYY-MM-DD"),
    limit: int = typer.Option(20, help="Max rows to display"),
) -> None:
    """List stored FRED observations."""
    rows = _client(ctx).get_fred_observations(
        series_id=series_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    list_table(
        rows,
        columns=["series_id", "date", "value"],
        title="FRED Observations",
    )
