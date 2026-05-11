"""Portfolio inspection utilities — display helpers and data summary."""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.table import Table

from research.reporting._display import (
    dict_table,
    error_panel,
    info_panel,
    warning_panel,
)

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_weights(weights: Any, top_n: int = 20) -> None:
    """Display portfolio weights as a Rich table."""
    import pandas as pd

    if not isinstance(weights, pd.Series) or len(weights) == 0:
        console.print("[dim]No weights to display.[/dim]")
        return

    sorted_weights = weights.sort_values(ascending=False)

    table = Table(
        title=f"Portfolio Weights ({len(sorted_weights)} assets)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="bold")
    table.add_column("Weight", justify="right")
    table.add_column("Weight %", justify="right")

    shown = sorted_weights.head(top_n)
    for ticker, weight in shown.items():
        pct = f"{weight * 100:.2f}%"
        table.add_row(str(ticker), f"{weight:.6f}", pct)

    if len(sorted_weights) > top_n:
        remaining = sorted_weights.iloc[top_n:]
        table.add_row(
            f"... {len(remaining)} more",
            f"{remaining.sum():.6f}",
            f"{remaining.sum() * 100:.2f}%",
            style="dim",
        )

    total = sorted_weights.sum()
    table.add_row(
        "TOTAL",
        f"{total:.6f}",
        f"{total * 100:.2f}%",
        style="bold green",
    )
    console.print(table)


def _display_summary(summary: dict[str, float]) -> None:
    """Display portfolio summary metrics."""
    table = Table(
        title="Portfolio Metrics (in-sample)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    fmt_map = {
        "annualized_mean": ("Annualized Return", "{:.2%}"),
        "mean": ("Mean Return", "{:.6f}"),
        "standard_deviation": ("Volatility", "{:.2%}"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.4f}"),
        "sortino_ratio": ("Sortino Ratio", "{:.4f}"),
        "max_drawdown": ("Max Drawdown", "{:.2%}"),
        "cvar": ("CVaR (95%)", "{:.2%}"),
        "variance": ("Variance", "{:.6f}"),
    }

    for key, (label, fmt) in fmt_map.items():
        val = summary.get(key)
        if val is not None:
            table.add_row(label, fmt.format(val))

    console.print(table)


def _display_backtest(backtest_result: Any) -> None:
    """Display backtest (out-of-sample) summary."""
    if backtest_result is None:
        return

    table = Table(
        title="Backtest (out-of-sample)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    attrs = [
        ("Annualized Return", "annualized_mean", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.4f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.4f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("CVaR (95%)", "cvar", "{:.2%}"),
    ]

    for label, attr, fmt in attrs:
        val = getattr(backtest_result, attr, None)
        if val is not None:
            table.add_row(label, fmt.format(float(val)))

    console.print(table)


# ---------------------------------------------------------------------------
# Data summary
# ---------------------------------------------------------------------------


def data_summary() -> None:
    """Show a summary of available data in the database for optimization."""
    from research.data_assembly import assemble_all
    from research.strategies._runner import _get_db_manager

    console.print("[bold]Checking database data availability...[/bold]")
    try:
        db_manager = _get_db_manager()
    except Exception as exc:
        error_panel(f"Cannot connect to database: {exc}")
        raise SystemExit(1) from exc

    try:
        data = assemble_all(db_manager)
    except Exception as exc:
        error_panel(f"Data assembly failed: {exc}")
        db_manager.close()
        raise SystemExit(1) from exc

    dict_table(data.summary(), title="Data Available for Optimization")

    if data.n_tickers > 0:
        import pandas as pd

        # Date range
        if isinstance(data.prices.index, pd.DatetimeIndex) and len(data.prices) > 0:
            start = data.prices.index.min().date()
            end = data.prices.index.max().date()
            info_panel(
                "Price History",
                f"From {start} to {end}\n"
                f"{data.n_trading_days} trading days "
                f"across {data.n_tickers} tickers",
            )

        # Sector breakdown
        if data.sector_mapping:
            sector_counts: dict[str, int] = {}
            for sector in data.sector_mapping.values():
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            table = Table(
                title="Sector Breakdown",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Sector", style="bold")
            table.add_column("Count", justify="right")
            for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
                table.add_row(sector, str(count))
            console.print(table)

        # Macro data
        if len(data.macro_data) > 0:
            macro_row = data.macro_data.iloc[0]
            macro_info: dict[str, Any] = {}
            if pd.notna(macro_row.get("gdp_growth")):
                macro_info["GDP Growth (QoQ)"] = f"{macro_row['gdp_growth']:.2f}%"
            if pd.notna(macro_row.get("yield_spread")):
                spread = macro_row["yield_spread"]
                macro_info["Yield Spread (10Y-2Y)"] = f"{spread:.2f}%"
            if macro_info:
                dict_table(macro_info, title="Macro Indicators")
    else:
        warning_panel("No data found. POST /api/v1/yfinance-data/fetch to populate.")

    db_manager.close()


def strategies() -> None:
    """List all available optimization strategies."""
    from research.strategies._runner import Strategy

    table = Table(
        title="Available Strategies",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Strategy", style="bold")
    table.add_column("Description")

    descriptions = {
        Strategy.MAX_SHARPE: "Mean-Risk: maximize Sharpe ratio",
        Strategy.MIN_VARIANCE: "Mean-Risk: minimize portfolio variance",
        Strategy.MIN_CVAR: "Mean-Risk: minimize CVaR (95%)",
        Strategy.MAX_UTILITY: "Mean-Risk: maximize utility",
        Strategy.RISK_PARITY: "Risk Budgeting: equal risk contribution",
        Strategy.CVAR_PARITY: "Risk Budgeting: equal CVaR contribution",
        Strategy.HRP: "Hierarchical Risk Parity (no cov inversion)",
        Strategy.HERC: "Hierarchical Equal Risk Contribution",
        Strategy.MAX_DIVERSIFICATION: "Maximum Diversification ratio",
        Strategy.EQUAL_WEIGHT: "Equal Weighted: 1/N baseline",
        Strategy.INVERSE_VOL: "Inverse Volatility weighting",
    }

    for s in Strategy:
        table.add_row(s.value, descriptions.get(s, ""))

    console.print(table)
