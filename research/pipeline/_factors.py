"""Steps 3-5 — Factor history, IS validation, OOS validation, coverage gate.

Extracted from ``stock_selection_pipeline.py`` lines 81, 168, 586–821.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from optimizer.exceptions import FactorCoverageError
from optimizer.factors import (
    FactorOOSConfig,
    FactorValidationConfig,
    run_factor_oos_validation,
    run_factor_validation,
)
from optimizer.factors._config import (
    FactorConstructionConfig,
    StandardizationConfig,
)
from research.factors._history import build_factor_scores_history
from research.factors._validator import validate_factors

console = Console()
logger = logging.getLogger(__name__)

# Cycle-2 §4.3 spec: NW lag=4, BH alpha=0.10, |t|>=1.645 (two-sided p<0.10).
_IS_VALIDATION_CONFIG = FactorValidationConfig(
    newey_west_lags=4,
    fdr_alpha=0.10,
    t_stat_threshold=1.645,
)

# Fix #239: index-based parameters, not calendar months.
OOS_CONFIG = FactorOOSConfig(train_periods=8, val_periods=4, step_periods=2)

# Cycle-2 §6.1: portfolio size constrained to [25, 50] selected stocks.
N_SELECTED_MIN: int = 15
N_SELECTED_MAX: int = 30

# 11 top-level sectors using Yahoo Finance naming (matches assembly.sector_mapping).
# GICS equivalents: Materials→Basic Materials, Consumer Discretionary→Consumer Cyclical,
# Consumer Staples→Consumer Defensive, Health Care→Healthcare,
# Financials→Financial Services, Information Technology→Technology.
_GICS_SECTORS: tuple[str, ...] = (
    "Energy",
    "Basic Materials",
    "Industrials",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Healthcare",
    "Financial Services",
    "Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
)

# Re-export needed by downstream pipeline steps
REBALANCE_FREQ: int = 63
MIN_SUCCESS_FRACTION: float = 0.5


# ---------------------------------------------------------------------------
# Cycle-2 §6.1 helpers
# ---------------------------------------------------------------------------


def _validate_n_selected(n_selected: int) -> None:
    """Reject ``n_selected`` outside the Cycle-2 §6.1 [25, 50] band."""
    if not N_SELECTED_MIN <= n_selected <= N_SELECTED_MAX:
        raise ValueError(
            f"n_selected={n_selected} outside Cycle-2 spec range "
            f"[{N_SELECTED_MIN}, {N_SELECTED_MAX}]."
        )


def _missing_gics_sectors(
    weights: pd.Series, sector_mapping: dict[str, str]
) -> list[str]:
    """Return GICS Level-1 sectors absent from the selected weights."""
    present = {sector_mapping.get(str(t), "") for t in weights.index}
    return [s for s in _GICS_SECTORS if s not in present]


# ---------------------------------------------------------------------------
# Step 3 — Factor scores history
# ---------------------------------------------------------------------------


def build_history(
    assembly: Any,
    investable: pd.Index,
    rebalance_freq: int = REBALANCE_FREQ,
    market_prices: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, Any]:
    """Build rolling PIT factor scores and forward-return history."""
    console.print(Panel("[bold]Step 3[/bold] — Building factor history", style="blue"))

    inv_cols = list(investable)
    investable_prices: pd.DataFrame = assembly.prices.loc[:, inv_cols]
    investable_volumes: pd.DataFrame = assembly.volumes.loc[:, inv_cols]
    investable_fundamentals: pd.DataFrame = assembly.fundamentals.loc[
        assembly.fundamentals.index.isin(inv_cols)
    ]

    factor_scores_dict, returns_history, health = build_factor_scores_history(
        investable_prices=investable_prices,
        investable_volumes=investable_volumes,
        investable_fundamentals=investable_fundamentals,
        assembly=assembly,
        factor_config=FactorConstructionConfig(),
        std_config=StandardizationConfig(),
        sector_mapping=assembly.sector_mapping,
        rebalance_freq=rebalance_freq,
        fundamental_history=assembly.fundamental_history,
        min_success_fraction=MIN_SUCCESS_FRACTION,
        market_prices=market_prices,
    )

    console.print(
        f"  {health.succeeded_dates}/{health.total_dates} rebalancing dates succeeded "
        f"({len(factor_scores_dict)} factors)"
    )
    if health.failed_dates:
        console.print(
            f"  [yellow]Warnings:[/yellow] {health.failed_dates} dates skipped"
        )
    return factor_scores_dict, returns_history, health


# ---------------------------------------------------------------------------
# Step 4 — In-sample factor validation
# ---------------------------------------------------------------------------


def validate_is(
    factor_scores_dict: dict[str, pd.DataFrame],
    returns_history: pd.DataFrame,
) -> Any:
    """Run in-sample factor validation (IC, VIF, quintile spreads)."""
    console.print(Panel("[bold]Step 4[/bold] — IS factor validation", style="blue"))

    last_date = sorted(next(iter(factor_scores_dict.values())).index)[-1]
    standardized_snapshot = pd.DataFrame(
        {
            k: v.loc[last_date]
            for k, v in factor_scores_dict.items()
            if last_date in v.index
        }
    )

    report = run_factor_validation(
        factor_scores_history=factor_scores_dict,
        returns_history=returns_history,
        config=_IS_VALIDATION_CONFIG,
    )
    vif_report = validate_factors(
        factor_scores_history=factor_scores_dict,
        returns_history=returns_history,
        standardized=standardized_snapshot,
    )

    table = Table(
        title="In-Sample Factor IC", show_header=True, header_style="bold cyan"
    )
    table.add_column("Factor", style="dim", width=26)
    table.add_column("Mean IC", justify="right")
    table.add_column("t-stat (NW)", justify="right")
    table.add_column("Significant", justify="center")

    for ic_res in sorted(report.ic_results, key=lambda r: -abs(r.mean_ic)):
        sig = "[green]✓[/green]" if ic_res.significant else "[red]✗[/red]"
        table.add_row(
            ic_res.factor_name,
            f"{ic_res.mean_ic:.4f}",
            f"{ic_res.t_stat:.2f}",
            sig,
        )
    console.print(table)

    if vif_report.vif_scores is not None:
        high_vif = vif_report.vif_scores[vif_report.vif_scores > 5.0]
        if not high_vif.empty:
            names: list[str] = [str(n) for n in high_vif.index]
            console.print(
                f"  [yellow]High VIF factors (>5):[/yellow] {', '.join(names)}"
            )

    return report


# ---------------------------------------------------------------------------
# Step 5 — Out-of-sample validation
# ---------------------------------------------------------------------------


def validate_oos(
    factor_scores_dict: dict[str, pd.DataFrame],
    returns_history: pd.DataFrame,
) -> Any:
    """Run rolling walk-forward OOS factor validation.

    Fix issue #239: parameters are index-based, not calendar months.
    OOS_CONFIG uses train_periods=8 (~2 years quarterly), val_periods=4 (~1 yr),
    step_periods=2 — yielding ≥6 folds on 5-year history.
    """
    console.print(Panel("[bold]Step 5[/bold] — Out-of-sample validation", style="blue"))

    stacked_factors = [
        df.stack().rename(name) for name, df in factor_scores_dict.items()
    ]
    scores_mi = pd.concat(stacked_factors, axis=1)
    scores_mi.index.names = ["date", "ticker"]

    returns_mi = returns_history.stack().to_frame("return")
    returns_mi.index.names = ["date", "ticker"]

    oos_result = run_factor_oos_validation(
        scores=scores_mi,
        returns=returns_mi,
        config=OOS_CONFIG,
    )

    if oos_result.n_folds == 0:
        raise RuntimeError(
            f"OOS validation produced 0 folds "
            f"(train_periods={OOS_CONFIG.train_periods}, "
            f"val_periods={OOS_CONFIG.val_periods}, "
            f"step_periods={OOS_CONFIG.step_periods}). "
            "Increase history or reduce train_periods."
        )

    table = Table(
        title=f"OOS Factor Validation ({oos_result.n_folds} folds)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Factor", style="dim", width=26)
    table.add_column("OOS Mean IC", justify="right")
    table.add_column("OOS ICIR", justify="right")

    sorted_factors = oos_result.mean_oos_ic.sort_values(ascending=False)
    for factor_name, mean_ic in sorted_factors.items():
        icir = oos_result.mean_oos_icir.get(factor_name, float("nan"))
        color = "green" if mean_ic > 0 else "red"
        ic_str = f"[{color}]{mean_ic:.4f}[/{color}]"
        table.add_row(str(factor_name), ic_str, f"{icir:.3f}")
    console.print(table)

    return oos_result


# ---------------------------------------------------------------------------
# Step 5b — Coverage gate (Cycle-2 §4.4)
# ---------------------------------------------------------------------------


def _check_factor_coverage(
    is_report: Any,
    oos_result: Any,
    *,
    min_factors: int = 2,
) -> None:
    """Abort when fewer than ``min_factors`` pass IS BH AND OOS ICIR>0."""
    is_sig = set(is_report.significant_factors)
    icir = oos_result.mean_oos_icir
    oos_pos = {f for f in icir.index if icir[f] > 0}
    passing = is_sig & oos_pos
    if len(passing) >= min_factors:
        return
    is_only = sorted(is_sig - oos_pos)
    oos_only = sorted(oos_pos - is_sig)
    raise FactorCoverageError(
        f"Factor coverage gate failed: {len(passing)}/{min_factors} factors "
        "satisfy NW p<0.10 AND OOS ICIR>0. "
        f"Passing: {sorted(passing)}. "
        f"IS-only: {is_only}. OOS-only: {oos_only}."
    )
