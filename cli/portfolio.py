"""Portfolio optimization command group — the glue layer.

Connects the database (API data layer) to the optimizer library
by assembling DataFrames and calling ``run_full_pipeline_with_selection()``.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from cli.display import (
    dict_table,
    error_panel,
    info_panel,
    success_panel,
    warning_panel,
)

# Ensure the api package is importable.
_api_path = Path(__file__).parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

logger = logging.getLogger(__name__)
console = Console()

portfolio_app = typer.Typer(
    name="portfolio",
    help="Portfolio optimization — run the full pipeline from DB to weights.",
)


# ---------------------------------------------------------------------------
# Strategy enum for CLI --strategy option
# ---------------------------------------------------------------------------


class Strategy(str, Enum):
    """Named optimizer strategies exposed via CLI."""

    MAX_SHARPE = "max-sharpe"
    MIN_VARIANCE = "min-variance"
    MIN_CVAR = "min-cvar"
    MAX_UTILITY = "max-utility"
    RISK_PARITY = "risk-parity"
    CVAR_PARITY = "cvar-parity"
    HRP = "hrp"
    HERC = "herc"
    MAX_DIVERSIFICATION = "max-diversification"
    EQUAL_WEIGHT = "equal-weight"
    INVERSE_VOL = "inverse-vol"


def _build_optimizer(strategy: Strategy, rf_daily: float = 0.0) -> Any:
    """Build a skfolio optimizer instance from a named strategy.

    Parameters
    ----------
    strategy : Strategy
        Named optimization strategy.
    rf_daily : float
        Daily risk-free rate (decimal).  Applied to ratio-maximising
        strategies (MAX_SHARPE, MAX_UTILITY).
    """
    from optimizer.optimization import (
        HERCConfig,
        HRPConfig,
        MeanRiskConfig,
        RiskBudgetingConfig,
        build_equal_weighted,
        build_herc,
        build_hrp,
        build_inverse_volatility,
        build_max_diversification,
        build_mean_risk,
        build_risk_budgeting,
    )

    match strategy:
        case Strategy.MAX_SHARPE:
            cfg = MeanRiskConfig.for_max_sharpe()
            return build_mean_risk(dataclasses.replace(cfg, risk_free_rate=rf_daily))
        case Strategy.MIN_VARIANCE:
            return build_mean_risk(MeanRiskConfig.for_min_variance())
        case Strategy.MIN_CVAR:
            return build_mean_risk(MeanRiskConfig.for_min_cvar())
        case Strategy.MAX_UTILITY:
            cfg = MeanRiskConfig.for_max_utility()
            return build_mean_risk(dataclasses.replace(cfg, risk_free_rate=rf_daily))
        case Strategy.RISK_PARITY:
            return build_risk_budgeting(RiskBudgetingConfig.for_risk_parity())
        case Strategy.CVAR_PARITY:
            return build_risk_budgeting(RiskBudgetingConfig.for_cvar_parity())
        case Strategy.HRP:
            return build_hrp(HRPConfig.for_cvar())
        case Strategy.HERC:
            return build_herc(HERCConfig.for_cvar())
        case Strategy.MAX_DIVERSIFICATION:
            return build_max_diversification()
        case Strategy.EQUAL_WEIGHT:
            return build_equal_weighted()
        case Strategy.INVERSE_VOL:
            return build_inverse_volatility()


def _get_db_manager() -> Any:
    """Create and initialize a DatabaseManager."""
    from app.database import DatabaseManager

    db = DatabaseManager()
    db.initialize()
    return db


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
# Commands
# ---------------------------------------------------------------------------


@portfolio_app.command()
def optimize(
    strategy: Strategy = typer.Option(
        Strategy.MAX_SHARPE,
        "--strategy",
        "-s",
        help="Optimization strategy.",
    ),
    backtest: bool = typer.Option(
        False,
        "--backtest",
        "-b",
        help="Run walk-forward backtest (quarterly rolling).",
    ),
    selection: bool = typer.Option(
        True,
        "--selection/--no-selection",
        help="Enable stock pre-selection (universe screening + factor scoring).",
    ),
    macro_country: str = typer.Option(
        "United States",
        "--macro-country",
        help="Country for macro regime data.",
    ),
    top_n: int = typer.Option(
        20,
        "--top-n",
        help="Number of top weights to display.",
    ),
    universe: str = typer.Option(
        "developed",
        "--universe",
        "-u",
        help="Universe: developed, large-cap, broad, small-cap.",
    ),
    sector_tolerance: float = typer.Option(
        0.05,
        "--sector-tolerance",
        help="Max sector weight deviation (0.0-1.0). Use 0.03 for low tracking error.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save weights to CSV file.",
    ),
) -> None:
    """Run the full optimization pipeline: DB → DataFrames → weights.

    Queries all data from the database, applies stock pre-selection
    (universe screening, factor scoring, regime tilts), then runs
    portfolio optimization with the chosen strategy.
    """
    from cli.data_assembly import assemble_all

    # 1. Initialize database
    console.print("[bold]Initializing database connection...[/bold]")
    try:
        db_manager = _get_db_manager()
    except Exception as exc:
        error_panel(f"Cannot connect to database: {exc}")
        raise typer.Exit(code=1)

    # 2. Assemble data
    console.print("[bold]Assembling data from database...[/bold]")
    try:
        data = assemble_all(db_manager, macro_country=macro_country)
    except Exception as exc:
        error_panel(f"Data assembly failed: {exc}")
        db_manager.close()
        raise typer.Exit(code=1)

    dict_table(data.summary(), title="Data Summary")

    if data.n_tickers == 0:
        error_panel(
            "No price data found in database. Run 'python -m cli yfinance fetch' first."
        )
        db_manager.close()
        raise typer.Exit(code=1)

    if data.n_trading_days < 60:
        warning_panel(
            f"Only {data.n_trading_days} trading days available. "
            "Results may be unreliable with fewer than 252 days."
        )

    # 3. Build optimizer (inject risk-free rate from FRED DGS3MO)
    rf_daily = data.risk_free_rate
    if rf_daily > 0:
        rf_ann = rf_daily * 252 * 100
        console.print(f"[bold]Risk-free rate:[/bold] {rf_ann:.2f}% ann. (DGS3MO)")
    console.print(f"[bold]Strategy:[/bold] {strategy.value}")
    optimizer_instance = _build_optimizer(strategy, rf_daily=rf_daily)

    # 4. Configure backtest
    from optimizer.validation import WalkForwardConfig

    cv_config = WalkForwardConfig.for_quarterly_rolling() if backtest else None

    # 5. Run pipeline
    console.print("[bold]Running optimization pipeline...[/bold]")
    try:
        if selection and len(data.fundamentals) > 0:
            from optimizer.factors import (
                CompositeScoringConfig,
                FactorIntegrationConfig,
                SelectionConfig,
            )
            from optimizer.pipeline import run_full_pipeline_with_selection
            from optimizer.universe import InvestabilityScreenConfig

            sel_config = SelectionConfig(sector_tolerance=sector_tolerance)

            # Universe preset
            universe_presets = {
                "developed": InvestabilityScreenConfig.for_developed_markets,
                "large-cap": InvestabilityScreenConfig.for_large_cap,
                "broad": InvestabilityScreenConfig.for_broad_universe,
                "small-cap": InvestabilityScreenConfig.for_small_cap,
            }
            inv_config = universe_presets.get(
                universe, InvestabilityScreenConfig.for_developed_markets
            )()

            # Scoring with coverage gate
            scoring = CompositeScoringConfig.for_ic_weighted_robust()

            # Factor integration (BL alpha bridge)
            integration = FactorIntegrationConfig.for_black_litterman()

            stmts = (
                data.financial_statements
                if len(data.financial_statements) > 0
                else None
            )
            analyst = data.analyst_data if len(data.analyst_data) > 0 else None
            insider = data.insider_data if len(data.insider_data) > 0 else None
            macro = data.macro_data if len(data.macro_data) > 0 else None
            sectors = data.sector_mapping if data.sector_mapping else None

            regime = (
                data.regime_data if len(data.regime_data) > 0 else None
            )

            result = run_full_pipeline_with_selection(
                prices=data.prices,
                optimizer=optimizer_instance,
                fundamentals=data.fundamentals,
                volume_history=data.volumes,
                financial_statements=stmts,
                analyst_data=analyst,
                insider_data=insider,
                macro_data=macro,
                regime_data=regime,
                investability_config=inv_config,
                scoring_config=scoring,
                integration_config=integration,
                sector_mapping=sectors,
                selection_config=sel_config,
                cv_config=cv_config,
            )
        else:
            from optimizer.pipeline import run_full_pipeline

            if not selection:
                console.print("[dim]Stock pre-selection disabled.[/dim]")

            result = run_full_pipeline(
                prices=data.prices,
                optimizer=optimizer_instance,
                sector_mapping=data.sector_mapping if data.sector_mapping else None,
                cv_config=cv_config,
            )
    except Exception as exc:
        error_panel(f"Optimization failed: {exc}")
        logger.exception("Pipeline error")
        db_manager.close()
        raise typer.Exit(code=1)

    # 6. Display results
    console.print()
    _display_weights(result.weights, top_n=top_n)
    console.print()
    _display_summary(result.summary)

    if result.backtest is not None:
        console.print()
        _display_backtest(result.backtest)

    if result.turnover is not None:
        info_panel("Rebalancing", f"Turnover: {result.turnover:.4f}")

    # 7. Save weights to CSV if requested
    if output is not None:
        result.weights.to_csv(output, header=True)
        success_panel(f"Weights saved to {output}")

    success_panel(f"Optimization complete — {len(result.weights)} assets.")
    db_manager.close()


@portfolio_app.command()
def data_summary() -> None:
    """Show a summary of available data in the database for optimization."""
    from cli.data_assembly import assemble_all

    console.print("[bold]Checking database data availability...[/bold]")
    try:
        db_manager = _get_db_manager()
    except Exception as exc:
        error_panel(f"Cannot connect to database: {exc}")
        raise typer.Exit(code=1)

    try:
        data = assemble_all(db_manager)
    except Exception as exc:
        error_panel(f"Data assembly failed: {exc}")
        db_manager.close()
        raise typer.Exit(code=1)

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
        warning_panel("No data found. Run 'python -m cli yfinance fetch' to populate.")

    db_manager.close()


@portfolio_app.command()
def strategies() -> None:
    """List all available optimization strategies."""
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
