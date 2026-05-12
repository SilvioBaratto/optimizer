"""Portfolio inspection utilities — data summary and strategy listing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from research.data._container import DataAssembly

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Display helpers (used by _runner.optimize())
# ---------------------------------------------------------------------------


def _display_weights(weights: Any, top_n: int = 20) -> None:
    import pandas as pd

    if not isinstance(weights, pd.Series) or len(weights) == 0:
        console.print("[dim]No weights to display.[/dim]")
        return

    sorted_w = weights.sort_values(ascending=False)
    table = Table(
        title=f"Portfolio Weights ({len(sorted_w)} assets)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="bold")
    table.add_column("Weight", justify="right")
    table.add_column("Weight %", justify="right")

    for ticker, weight in sorted_w.head(top_n).items():
        table.add_row(str(ticker), f"{weight:.6f}", f"{weight * 100:.2f}%")

    if len(sorted_w) > top_n:
        rest = sorted_w.iloc[top_n:]
        table.add_row(
            f"... {len(rest)} more",
            f"{rest.sum():.6f}",
            f"{rest.sum() * 100:.2f}%",
            style="dim",
        )

    total = sorted_w.sum()
    table.add_row("TOTAL", f"{total:.6f}", f"{total * 100:.2f}%", style="bold green")
    console.print(table)


def _display_summary(summary: dict[str, float]) -> None:
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
    if backtest_result is None:
        return
    table = Table(
        title="Backtest (out-of-sample)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for label, attr, fmt in [
        ("Annualized Return", "annualized_mean", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.4f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.4f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("CVaR (95%)", "cvar", "{:.2%}"),
    ]:
        val = getattr(backtest_result, attr, None)
        if val is not None:
            table.add_row(label, fmt.format(float(val)))
    console.print(table)


# ---------------------------------------------------------------------------
# Inspection API
# ---------------------------------------------------------------------------


def data_summary(data: DataAssembly) -> str:
    """Return a human-readable plain-text summary of assembled data."""
    import pandas as pd

    parts: list[str] = [f"{k}: {v}" for k, v in data.summary().items()]

    if (
        data.n_tickers > 0
        and isinstance(data.prices.index, pd.DatetimeIndex)
        and len(data.prices) > 0
    ):
        start = data.prices.index.min().date()
        end = data.prices.index.max().date()
        parts.append(f"price_range: {start} to {end}")

    return "\n".join(parts)


def strategies() -> dict[str, str]:
    """Return a mapping of strategy value → human-readable description."""
    from research.strategies._runner import Strategy

    return {
        Strategy.MAX_SHARPE.value: "Mean-Risk: maximize Sharpe ratio",
        Strategy.MIN_VARIANCE.value: "Mean-Risk: minimize portfolio variance",
        Strategy.MIN_CVAR.value: "Mean-Risk: minimize CVaR (95%)",
        Strategy.MAX_UTILITY.value: "Mean-Risk: maximize utility",
        Strategy.RISK_PARITY.value: "Risk Budgeting: equal risk contribution",
        Strategy.CVAR_PARITY.value: "Risk Budgeting: equal CVaR contribution",
        Strategy.HRP.value: "Hierarchical Risk Parity (no cov inversion)",
        Strategy.HERC.value: "Hierarchical Equal Risk Contribution",
        Strategy.MAX_DIVERSIFICATION.value: "Maximum Diversification ratio",
        Strategy.EQUAL_WEIGHT.value: "Equal Weighted: 1/N baseline",
        Strategy.INVERSE_VOL.value: "Inverse Volatility weighting",
    }
