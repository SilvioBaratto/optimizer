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
