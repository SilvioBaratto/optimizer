"""Rich rendering helpers for CLI output (single-responsibility display layer)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


# ------------------------------------------------------------------
# Panels
# ------------------------------------------------------------------


def error_panel(msg: str) -> None:
    """Print a red error panel."""
    console.print(Panel(msg, title="Error", border_style="red"))


def success_panel(msg: str) -> None:
    """Print a green success panel."""
    console.print(Panel(msg, title="Success", border_style="green"))


def warning_panel(msg: str) -> None:
    """Print a yellow warning panel."""
    console.print(Panel(msg, title="Warning", border_style="yellow"))


def info_panel(title: str, body: str) -> None:
    """Print a blue informational panel."""
    console.print(Panel(body, title=title, border_style="blue"))


# ------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------


def dict_table(data: dict[str, Any], title: str = "") -> None:
    """Render a key/value table from a dict."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in data.items():
        table.add_row(str(key), str(value))
    console.print(table)


def list_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    title: str = "",
) -> None:
    """Render a tabular display from a list of dicts.

    Only the keys listed in *columns* are shown, in order.
    """
    if not rows:
        console.print(f"[dim]No data to display for '{title}'.[/dim]")
        return

    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*(str(row.get(col, "")) for col in columns))
    console.print(table)


# ------------------------------------------------------------------
# Live progress polling
# ------------------------------------------------------------------


def progress_loop(
    poll_fn: Callable[[], dict[str, Any]],
    interval: float = 2.0,
) -> dict[str, Any]:
    """Poll *poll_fn* with a live progress bar until the job completes or fails.

    *poll_fn* must return a dict with at least ``status``, ``current``,
    and ``total`` keys.  Returns the final poll result.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )

    with Live(progress, console=console, refresh_per_second=4):
        task_id = progress.add_task("Starting...", total=None)
        while True:
            data = poll_fn()
            status = data.get("status", "unknown")
            current = data.get("current", 0)
            total = data.get("total", 0) or 0

            # Build description from available fields
            desc_parts = [f"[bold]{status}[/bold]"]
            if data.get("current_exchange"):
                desc_parts.append(data["current_exchange"])
            if data.get("current_stock"):
                desc_parts.append(data["current_stock"])
            if data.get("current_ticker"):
                desc_parts.append(data["current_ticker"])
            description = " | ".join(desc_parts)

            progress.update(
                task_id,
                description=description,
                total=total if total > 0 else None,
                completed=current,
            )

            if status in ("completed", "failed"):
                return data

            time.sleep(interval)
