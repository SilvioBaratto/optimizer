"""Portfolio optimization glue layer — strategy runner.

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

from rich.console import Console

# Ensure the api package is importable.
_api_path = Path(__file__).parent.parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

logger = logging.getLogger(__name__)
console = Console()


class Strategy(str, Enum):
    """Named optimizer strategies."""

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
    from skfolio.measures import RiskMeasure
    from skfolio.optimization import (
        EqualWeighted,
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
        InverseVolatility,
        MaximumDiversification,
        RiskBudgeting,
    )

    from optimizer.optimization import MeanRiskConfig, build_mean_risk

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
            return RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)
        case Strategy.CVAR_PARITY:
            return RiskBudgeting(risk_measure=RiskMeasure.CVAR)
        case Strategy.HRP:
            return HierarchicalRiskParity(risk_measure=RiskMeasure.CVAR)
        case Strategy.HERC:
            return HierarchicalEqualRiskContribution(risk_measure=RiskMeasure.CVAR)
        case Strategy.MAX_DIVERSIFICATION:
            return MaximumDiversification()
        case Strategy.EQUAL_WEIGHT:
            return EqualWeighted()
        case Strategy.INVERSE_VOL:
            return InverseVolatility()


def _get_db_manager() -> Any:
    """Create and initialize a DatabaseManager."""
    from app.database import DatabaseManager

    db = DatabaseManager()
    db.initialize()
    return db


def optimize(
    strategy: Strategy = Strategy.MAX_SHARPE,
    backtest: bool = False,
    selection: bool = True,
    macro_country: str = "United States",
    top_n: int = 20,
    universe: str = "developed",
    sector_tolerance: float = 0.05,
    output: str | None = None,
) -> None:
    """Run the full optimization pipeline: DB → DataFrames → weights.

    Queries all data from the database, applies stock pre-selection
    (universe screening, factor scoring, regime tilts), then runs
    portfolio optimization with the chosen strategy.
    """
    from research.data_assembly import assemble_all
    from research.reporting._display import (
        dict_table,
        error_panel,
        info_panel,
        success_panel,
        warning_panel,
    )
    from research.strategies._inspect import (
        _display_backtest,
        _display_summary,
        _display_weights,
    )

    # 1. Initialize database
    console.print("[bold]Initializing database connection...[/bold]")
    try:
        db_manager = _get_db_manager()
    except Exception as exc:
        error_panel(f"Cannot connect to database: {exc}")
        raise SystemExit(1) from exc

    # 2. Assemble data
    console.print("[bold]Assembling data from database...[/bold]")
    try:
        data = assemble_all(db_manager, macro_country=macro_country)
    except Exception as exc:
        error_panel(f"Data assembly failed: {exc}")
        db_manager.close()
        raise SystemExit(1) from exc

    dict_table(data.summary(), title="Data Summary")

    if data.n_tickers == 0:
        error_panel(
            "No price data found in database. POST /api/v1/yfinance-data/fetch first."
        )
        db_manager.close()
        raise SystemExit(1)

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
            sectors = dict(data.sector_mapping) if data.sector_mapping else None

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
                sector_mapping=(
                    dict(data.sector_mapping) if data.sector_mapping else None
                ),
                cv_config=cv_config,
            )
    except Exception as exc:
        error_panel(f"Optimization failed: {exc}")
        logger.exception("Pipeline error")
        db_manager.close()
        raise SystemExit(1) from exc

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
