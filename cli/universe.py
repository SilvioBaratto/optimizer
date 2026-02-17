"""Universe command group — maps to /api/v1/universe endpoints."""

from __future__ import annotations

from typing import List, Optional

import typer

from cli.client import ApiClient
from cli.display import (
    dict_table,
    error_panel,
    list_table,
    progress_loop,
    success_panel,
)

universe_app = typer.Typer(name="universe", help="Manage the instrument universe.")


def _client(ctx: typer.Context) -> ApiClient:
    return ctx.obj


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

@universe_app.command()
def stats(ctx: typer.Context) -> None:
    """Show exchange and instrument counts."""
    data = _client(ctx).get_stats()
    dict_table(data, title="Universe Stats")


# ------------------------------------------------------------------
# exchanges
# ------------------------------------------------------------------

@universe_app.command()
def exchanges(ctx: typer.Context) -> None:
    """List all exchanges."""
    data = _client(ctx).get_exchanges()
    list_table(data, columns=["id", "name", "t212_id"], title="Exchanges")


# ------------------------------------------------------------------
# instruments
# ------------------------------------------------------------------

@universe_app.command()
def instruments(
    ctx: typer.Context,
    exchange: Optional[str] = typer.Option(None, help="Filter by exchange name"),
    skip: int = typer.Option(0, help="Number of records to skip"),
    limit: int = typer.Option(100, help="Max records to return (1-1000)"),
) -> None:
    """List instruments with optional filtering and pagination."""
    data = _client(ctx).get_instruments(exchange=exchange, skip=skip, limit=limit)
    items = data.get("items", [])
    total = data.get("total", 0)
    list_table(
        items,
        columns=["ticker", "short_name", "yfinance_ticker", "exchange_name"],
        title=f"Instruments ({len(items)} of {total})",
    )


# ------------------------------------------------------------------
# build
# ------------------------------------------------------------------

@universe_app.command()
def build(
    ctx: typer.Context,
    exchange: Optional[List[str]] = typer.Option(
        None, help="Exchange name(s). Repeat for multiple."
    ),
    skip_filters: bool = typer.Option(False, "--skip-filters", help="Skip quality filters"),
    max_workers: int = typer.Option(20, help="Concurrent workers (1-50)"),
) -> None:
    """Start a universe build and poll until complete."""
    client = _client(ctx)
    job = client.build_universe(
        exchanges=exchange or None,
        skip_filters=skip_filters,
        max_workers=max_workers,
    )
    build_id = job["build_id"]
    success_panel(f"Build started: {build_id}")

    result = progress_loop(lambda: client.get_build_status(build_id))

    if result.get("status") == "failed":
        error_panel(f"Build failed: {result.get('error', 'unknown')}")
        raise typer.Exit(code=1)

    build_result = result.get("result", {})
    if build_result:
        dict_table(
            {
                "exchanges_saved": build_result.get("exchanges_saved"),
                "instruments_saved": build_result.get("instruments_saved"),
                "total_processed": build_result.get("total_processed"),
            },
            title="Build Result",
        )
    success_panel("Build completed.")


# ------------------------------------------------------------------
# build-status
# ------------------------------------------------------------------

@universe_app.command("build-status")
def build_status(
    ctx: typer.Context,
    build_id: str = typer.Argument(help="Build job UUID"),
) -> None:
    """Check the status of a universe build."""
    data = _client(ctx).get_build_status(build_id)
    dict_table(data, title=f"Build {build_id}")


# ------------------------------------------------------------------
# cache-stats
# ------------------------------------------------------------------

@universe_app.command("cache-stats")
def cache_stats(ctx: typer.Context) -> None:
    """Show ticker mapping cache statistics."""
    data = _client(ctx).get_cache_stats()
    dict_table(data, title="Cache Stats")


# ------------------------------------------------------------------
# cache-clear
# ------------------------------------------------------------------

@universe_app.command("cache-clear")
def cache_clear(ctx: typer.Context) -> None:
    """Clear the ticker mapping cache."""
    typer.confirm("Are you sure you want to clear the cache?", abort=True)
    _client(ctx).clear_cache()
    success_panel("Cache cleared.")
