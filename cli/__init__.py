"""Optimizer Platform CLI — Typer app factory and entry point."""

from __future__ import annotations

from typing import Optional

import typer

from cli.client import ApiClient
from cli.db import db_app
from cli.macro import macro_app
from cli.universe import universe_app
from cli.yfinance import yfinance_app

app = typer.Typer(
    name="optimizer",
    help="Optimizer Platform CLI — interact with the API from the terminal.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo("optimizer CLI 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    base_url: str = typer.Option(
        "http://localhost:8000",
        "--base-url",
        envvar="OPTIMIZER_API_URL",
        help="Base URL of the Optimizer API server.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show CLI version and exit.",
    ),
) -> None:
    """Global options applied before any sub-command."""
    ctx.ensure_object(dict)
    ctx.obj = ApiClient(base_url=base_url)


# Register command groups
app.add_typer(db_app)
app.add_typer(macro_app)
app.add_typer(universe_app)
app.add_typer(yfinance_app)


if __name__ == "__main__":
    app()
