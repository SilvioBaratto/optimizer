"""YFinance data command group — maps to /api/v1/yfinance-data endpoints."""

from __future__ import annotations

from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from cli.client import ApiClient
from cli.display import (
    dict_table,
    error_panel,
    list_table,
    progress_loop,
    success_panel,
    warning_panel,
)

yfinance_app = typer.Typer(name="yfinance", help="Fetch and query yfinance data.")


def _client(ctx: typer.Context) -> ApiClient:
    return ctx.obj


# ------------------------------------------------------------------
# fetch (bulk)
# ------------------------------------------------------------------


def _run_direct_fetch(period: str, mode: str) -> None:
    """Run direct fetch without API using database connection."""
    from cli.direct_fetch import get_direct_fetcher

    warning_panel("API unavailable — running direct fetch mode")

    fetcher = get_direct_fetcher()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task_id = progress.add_task("Initializing...", total=None)

            def on_progress(current: int, total: int, ticker: str) -> None:
                progress.update(
                    task_id,
                    description=f"[bold]Fetching[/bold] | {ticker}",
                    total=total,
                    completed=current,
                )

            result = fetcher.fetch_all(
                period=period,
                mode=mode,
                progress_callback=on_progress,
            )

        if result.get("errors"):
            error_panel(f"Completed with {len(result['errors'])} errors")
            for err in result["errors"][:10]:  # Show first 10 errors
                typer.echo(f"  - {err}")
            if len(result["errors"]) > 10:
                typer.echo(f"  ... and {len(result['errors']) - 10} more errors")

        dict_table(
            {
                "processed": result.get("processed", 0),
                "skipped_categories": result.get("skipped_categories", 0),
                **result.get("counts", {}),
            },
            title="Fetch Result",
        )
        success_panel("Direct fetch completed.")

    finally:
        fetcher.close()


@yfinance_app.command()
def fetch(
    ctx: typer.Context,
    max_workers: int = typer.Option(4, help="Parallel workers (1-20)"),
    period: str = typer.Option("5y", help="Price history period (e.g. 1y, 2y, 5y)"),
    mode: str = typer.Option("incremental", help="Fetch mode: 'incremental' (skip fresh) or 'full'"),
    direct: bool = typer.Option(False, "--direct", "-d", help="Bypass API and fetch directly from database"),
) -> None:
    """Start a bulk yfinance fetch for all instruments and poll until complete."""
    # If --direct flag is set, skip API entirely
    if direct:
        _run_direct_fetch(period=period, mode=mode)
        return

    client = _client(ctx)

    # Check if API is available before attempting
    if not client.is_available():
        _run_direct_fetch(period=period, mode=mode)
        return

    job = client.start_fetch(max_workers=max_workers, period=period, mode=mode)
    job_id = job["job_id"]
    success_panel(f"Fetch started: {job_id}")

    result = progress_loop(lambda: client.get_fetch_status(job_id))

    if result.get("status") == "failed":
        error_panel(f"Fetch failed: {result.get('error', 'unknown')}")
        raise typer.Exit(code=1)

    fetch_result = result.get("result", {})
    if fetch_result:
        dict_table(fetch_result, title="Fetch Result")
    success_panel("Fetch completed.")


# ------------------------------------------------------------------
# fetch-status
# ------------------------------------------------------------------

@yfinance_app.command("fetch-status")
def fetch_status(
    ctx: typer.Context,
    job_id: str = typer.Argument(help="Fetch job UUID"),
) -> None:
    """Check the status of a bulk fetch job."""
    data = _client(ctx).get_fetch_status(job_id)
    dict_table(data, title=f"Fetch Job {job_id}")


# ------------------------------------------------------------------
# fetch-ticker (single sync fetch)
# ------------------------------------------------------------------


def _run_direct_fetch_ticker(ticker: str, period: str, mode: str) -> None:
    """Run direct single ticker fetch without API."""
    from cli.direct_fetch import get_direct_fetcher

    warning_panel("API unavailable — running direct fetch mode")

    fetcher = get_direct_fetcher()

    try:
        data = fetcher.fetch_ticker(ticker, period=period, mode=mode)

        counts = data.get("counts", {})
        skipped = data.get("skipped", [])
        display = {
            "ticker": data.get("ticker"),
            "instrument_id": data.get("instrument_id"),
            **counts,
        }
        if skipped:
            display["skipped"] = ", ".join(skipped)
        dict_table(display, title=f"Fetch Result: {ticker}")

        errors = data.get("errors", [])
        if errors:
            error_panel("\n".join(errors))

    finally:
        fetcher.close()


@yfinance_app.command("fetch-ticker")
def fetch_ticker(
    ctx: typer.Context,
    ticker: str = typer.Argument(help="YFinance ticker symbol (e.g. AAPL, IHG.L)"),
    period: str = typer.Option("5y", help="Price history period"),
    mode: str = typer.Option("incremental", help="Fetch mode: 'incremental' (skip fresh) or 'full'"),
    direct: bool = typer.Option(False, "--direct", "-d", help="Bypass API and fetch directly from database"),
) -> None:
    """Synchronously fetch all yfinance data for a single ticker."""
    # If --direct flag is set, skip API entirely
    if direct:
        _run_direct_fetch_ticker(ticker, period=period, mode=mode)
        return

    client = _client(ctx)

    # Check if API is available before attempting
    if not client.is_available():
        _run_direct_fetch_ticker(ticker, period=period, mode=mode)
        return

    data = client.fetch_ticker(ticker, period=period, mode=mode)
    counts = data.get("counts", {})
    skipped = data.get("skipped", [])
    display = {
        "ticker": data.get("ticker"),
        "instrument_id": data.get("instrument_id"),
        **counts,
    }
    if skipped:
        display["skipped"] = ", ".join(skipped)
    dict_table(display, title=f"Fetch Result: {ticker}")
    errors = data.get("errors", [])
    if errors:
        error_panel("\n".join(errors))


# ------------------------------------------------------------------
# profile
# ------------------------------------------------------------------

@yfinance_app.command()
def profile(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show the stored ticker profile for an instrument."""
    data = _client(ctx).get_profile(instrument_id)
    dict_table(data, title="Ticker Profile")


# ------------------------------------------------------------------
# prices
# ------------------------------------------------------------------

@yfinance_app.command()
def prices(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    limit: int = typer.Option(5000, help="Max rows to return"),
) -> None:
    """Show stored price history (OHLCV) for an instrument."""
    rows = _client(ctx).get_prices(
        instrument_id, start_date=start, end_date=end, limit=limit
    )
    list_table(
        rows,
        columns=["date", "open", "high", "low", "close", "volume"],
        title=f"Prices ({len(rows)} rows)",
    )


# ------------------------------------------------------------------
# financials
# ------------------------------------------------------------------

@yfinance_app.command()
def financials(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
    type_: Optional[str] = typer.Option(
        None,
        "--type",
        help="Statement type: income_statement|balance_sheet|cashflow|earnings",
    ),
    period: Optional[str] = typer.Option(
        None,
        "--period",
        help="Period type: annual|quarterly",
    ),
) -> None:
    """Show stored financial statements for an instrument."""
    rows = _client(ctx).get_financials(
        instrument_id, statement_type=type_, period_type=period
    )
    list_table(
        rows,
        columns=["statement_type", "period_type", "period_date", "line_item", "value"],
        title=f"Financials ({len(rows)} rows)",
    )


# ------------------------------------------------------------------
# dividends
# ------------------------------------------------------------------

@yfinance_app.command()
def dividends(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show stored dividend data for an instrument."""
    rows = _client(ctx).get_dividends(instrument_id)
    list_table(rows, columns=["date", "amount"], title="Dividends")


# ------------------------------------------------------------------
# splits
# ------------------------------------------------------------------

@yfinance_app.command()
def splits(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show stored stock split data for an instrument."""
    rows = _client(ctx).get_splits(instrument_id)
    list_table(rows, columns=["date", "ratio"], title="Stock Splits")


# ------------------------------------------------------------------
# recommendations
# ------------------------------------------------------------------

@yfinance_app.command()
def recommendations(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show stored analyst recommendations for an instrument."""
    rows = _client(ctx).get_recommendations(instrument_id)
    list_table(
        rows,
        columns=["period", "strong_buy", "buy", "hold", "sell", "strong_sell"],
        title="Analyst Recommendations",
    )


# ------------------------------------------------------------------
# price-targets
# ------------------------------------------------------------------

@yfinance_app.command("price-targets")
def price_targets(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show stored analyst price targets for an instrument."""
    data = _client(ctx).get_price_targets(instrument_id)
    dict_table(
        {
            "current": data.get("current"),
            "low": data.get("low"),
            "high": data.get("high"),
            "mean": data.get("mean"),
            "median": data.get("median"),
        },
        title="Price Targets",
    )


# ------------------------------------------------------------------
# holders (institutional)
# ------------------------------------------------------------------

@yfinance_app.command()
def holders(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show stored institutional holders for an instrument."""
    rows = _client(ctx).get_institutional_holders(instrument_id)
    list_table(
        rows,
        columns=["holder_name", "shares", "value", "pct_held"],
        title="Institutional Holders",
    )


# ------------------------------------------------------------------
# fund-holders (mutual fund)
# ------------------------------------------------------------------

@yfinance_app.command("fund-holders")
def fund_holders(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show stored mutual fund holders for an instrument."""
    rows = _client(ctx).get_mutualfund_holders(instrument_id)
    list_table(
        rows,
        columns=["holder_name", "shares", "value", "pct_held"],
        title="Mutual Fund Holders",
    )


# ------------------------------------------------------------------
# insiders
# ------------------------------------------------------------------

@yfinance_app.command()
def insiders(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show stored insider transactions for an instrument."""
    rows = _client(ctx).get_insider_transactions(instrument_id)
    list_table(
        rows,
        columns=["insider_name", "transaction_type", "shares", "value", "start_date"],
        title="Insider Transactions",
    )


# ------------------------------------------------------------------
# news
# ------------------------------------------------------------------

@yfinance_app.command()
def news(
    ctx: typer.Context,
    instrument_id: str = typer.Argument(help="Instrument UUID"),
) -> None:
    """Show stored news articles for an instrument."""
    rows = _client(ctx).get_news(instrument_id)
    list_table(
        rows,
        columns=["title", "publisher", "publish_time"],
        title="News",
    )
