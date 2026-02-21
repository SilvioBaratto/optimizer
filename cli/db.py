"""Database management command group — maps to /api/v1/database endpoints."""

from __future__ import annotations

import typer

from cli.client import ApiClient
from cli.display import dict_table, error_panel, list_table, success_panel

db_app = typer.Typer(name="db", help="Database health, status, and table management.")


def _client(ctx: typer.Context) -> ApiClient:
    return ctx.obj


# ------------------------------------------------------------------
# health
# ------------------------------------------------------------------


@db_app.command()
def health(ctx: typer.Context) -> None:
    """Check database connectivity (SELECT 1 health check)."""
    data = _client(ctx).db_health()
    healthy = data.get("healthy", False)

    dict_table(data, title="Database Health")

    if healthy:
        success_panel("Database is healthy.")
    else:
        error_panel("Database is NOT healthy.")
        raise typer.Exit(code=1)


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------


@db_app.command()
def status(ctx: typer.Context) -> None:
    """Show detailed database pool and configuration info."""
    data = _client(ctx).db_status()
    # Flatten nested 'configuration' for display
    config = data.pop("configuration", {})
    pool = data.pop("pool_status", None)

    dict_table(data, title="Database Status")

    if config:
        dict_table(config, title="Configuration")

    if isinstance(pool, dict):
        dict_table(pool, title="Pool Status")
    elif pool is not None:
        dict_table({"pool_status": pool}, title="Pool Status")


# ------------------------------------------------------------------
# tables
# ------------------------------------------------------------------


@db_app.command()
def tables(ctx: typer.Context) -> None:
    """List application tables with row counts."""
    rows = _client(ctx).db_tables()
    list_table(
        rows,
        columns=["table_name", "exists", "row_count"],
        title="Application Tables",
    )


# ------------------------------------------------------------------
# clear
# ------------------------------------------------------------------


@db_app.command()
def clear(
    ctx: typer.Context,
    table: str | None = typer.Argument(
        default=None,
        help="Table name to clear. Omit to clear ALL tables.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Truncate one or all application tables."""
    client = _client(ctx)

    if table:
        if not yes:
            typer.confirm(
                f"This will delete ALL rows from '{table}'. Continue?",
                abort=True,
            )
        result = client.db_clear_table(table, confirm=True)
        dict_table(result, title=f"Clear Table: {table}")
        success_panel(f"Table '{table}' truncated.")
    else:
        if not yes:
            typer.confirm(
                "This will delete ALL rows from ALL application tables. Continue?",
                abort=True,
            )
        result = client.db_clear_all(confirm=True)
        cleared = result.get("cleared", [])
        errors = result.get("errors", [])

        if cleared:
            dict_table(
                {"tables_cleared": len(cleared), "tables": ", ".join(cleared)},
                title="Clear All Tables",
            )
        if errors:
            error_panel("\n".join(errors))

        success_panel(f"Truncated {len(cleared)} table(s).")
