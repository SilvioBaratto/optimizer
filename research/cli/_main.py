"""Thin orchestrator for the research pipeline — calls pipeline step modules.

Extracted from ``stock_selection_pipeline.py`` lines 1752-1954.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from app.database import DatabaseManager
from rich.console import Console
from rich.panel import Panel

from optimizer.exceptions import FactorCoverageError
from research.optimization._rebalance import _decide_rebalance
from research.pipeline._checklist import _apply_terminal_gate
from research.pipeline._factors import (
    N_SELECTED_MAX,
    N_SELECTED_MIN,
    _check_factor_coverage,
    _missing_gics_sectors,
    _validate_n_selected,
    build_history,
    validate_is,
    validate_oos,
)
from research.pipeline._load import (
    N_SELECTED,
    REBALANCE_FREQ,
    _materialise_clean_returns,
    _read_last_review_date,
    _write_last_review_date,
    load_data,
)
from research.pipeline._optimize import (
    compute_weighted_cost_bps,
    optimize_portfolio,
)
from research.pipeline._regime import classify_and_tilt
from research.pipeline._report import (
    _persist_research_snapshot,
    _render_research_report,
    report_performance,
)
from research.pipeline._screen import screen_investable

console = Console()
logger = logging.getLogger(__name__)


def main(
    rebalance_freq: int = REBALANCE_FREQ,
    n_selected: int = N_SELECTED,
    cost_bps: float = 10.0,
    *,
    tax_rate: float = 0.26,
    base_currency: str = "EUR",
    robust: bool = False,
    persist: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
    seed: int = 42,
    output_dir: Path = Path("research/output"),
) -> None:
    """Run the full stock selection pipeline."""
    console.print(
        Panel(
            "[bold cyan]Stock Selection Pipeline[/bold cyan]\n"
            "Factor-based stock selection with macro regime overlay",
            style="bold",
        )
    )

    # Issue #531: enforce Cycle-2 §6.1 portfolio size constraint upfront.
    _validate_n_selected(n_selected)

    pass_count = 0
    checklist_rules: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float]] = {}

    try:
        # 1. Load
        db_manager = DatabaseManager()
        db_manager.initialize()
        assembly, country_map, db_manager = load_data(
            db_manager, base_currency=base_currency
        )

        # 1b. Optional date slicing (Cycle 5 §13)
        if start_date is not None or end_date is not None:
            assembly.prices = assembly.prices.loc[start_date:end_date]
            console.print(
                f"  Sliced prices to "
                f"[cyan]{start_date or assembly.prices.index[0].date()}[/cyan] "
                f"… [cyan]{end_date or assembly.prices.index[-1].date()}[/cyan]"
            )

        # 2. Screen
        investable = screen_investable(assembly)

        # 2.5. Materialise clean linear-return panel (Cycle 1 §3 hand-off).
        clean_returns = _materialise_clean_returns(assembly, investable)
        logger.info(
            "Cycle 1 clean_returns: %d days x %d tickers, no NaN/inf.",
            len(clean_returns),
            clean_returns.shape[1],
        )

        # 3. Build factor history
        factor_scores_dict, returns_history, health = build_history(
            assembly, investable, rebalance_freq=rebalance_freq
        )
        logger.info(
            "Factor history health: %s/%s dates succeeded",
            health.succeeded_dates,
            health.total_dates,
        )

        # 4. IS validation
        is_report = validate_is(factor_scores_dict, returns_history)

        # 5. OOS validation
        oos_result = validate_oos(factor_scores_dict, returns_history)

        # validate_oos raises when n_folds == 0, so per_fold_ic is always defined.
        ic_history: pd.DataFrame = oos_result.per_fold_ic

        # 5b. Coverage gate — abort if fewer than 4 factors pass IS BH AND OOS ICIR>0
        _check_factor_coverage(is_report, oos_result)

        # 6. Regime + tilts (regime_data passed to pipeline internally).
        # db_manager forwarded so the rule-based regime is cached to
        # macro_calibrations.regime_classification (issue #530).
        current_regime, tilts = classify_and_tilt(
            assembly, db_manager=db_manager
        )
        logger.info("Regime: %s, tilts: %s", current_regime.value, tilts)

        # 7. Optimize
        result = optimize_portfolio(
            assembly=assembly,
            investable=investable,
            ic_history=ic_history,
            n_selected=n_selected,
            cost_bps=cost_bps,
            country_map=country_map,
            robust=robust,
            seed=seed,
        )

        # 7b. Cycle-3 §11 hybrid rebalance decision.
        current_date = pd.Timestamp.today().normalize()
        last_review_date = _read_last_review_date(current_date)
        prev_weights = getattr(assembly, "previous_weights", None)
        target_arr = result.weights.to_numpy(dtype=float)
        prev_arr = (
            np.asarray(prev_weights, dtype=float)
            if prev_weights is not None
            else None
        )
        result.rebalance_decision = _decide_rebalance(
            prev_weights=prev_arr,
            target_weights=target_arr,
            current_date=current_date,
            last_review_date=last_review_date,
        )
        decision, reason = result.rebalance_decision
        if decision or reason == "cold_start":
            _write_last_review_date(current_date)

        # Cycle-2 §6.1 sanity: weight count in [25, 50] + sector coverage.
        n_weights = len(result.weights)
        if not N_SELECTED_MIN <= n_weights <= N_SELECTED_MAX:
            console.print(
                f"  [yellow]Warning: weight count {n_weights} outside "
                f"[{N_SELECTED_MIN}, {N_SELECTED_MAX}].[/yellow]"
            )
        missing_sectors = _missing_gics_sectors(
            result.weights, assembly.sector_mapping
        )
        if missing_sectors:
            console.print(
                "  [yellow]Sectors absent from selection: "
                f"{', '.join(missing_sectors)}[/yellow]"
            )

        # 7c. Cycle 4 §9.1: portfolio-weighted measured cost in bps.
        cost_bps_actual = compute_weighted_cost_bps(
            result.weights, country_map
        )
        logger.info(
            "Measured cost: %.2f bps (weighted by country)", cost_bps_actual
        )

        # 8. Report
        pass_count, checklist_rules, metrics, chart_paths = report_performance(
            result,
            assembly,
            country_map,
            cost_bps_actual=cost_bps_actual,
            cost_bps=cost_bps,
            tax_rate=tax_rate,
            validation_report=is_report,
            oos_per_fold_ic=oos_result.per_fold_ic,
            output_dir=output_dir,
        )

        # 8b. Render report.md (always, before the terminal gate so a
        # FAIL still emits a diagnostic artefact).
        _render_research_report(
            output_dir=output_dir,
            assembly_hash=assembly.assembly_hash,
            current_regime=current_regime,
            tilts=tilts,
            validation_report=is_report,
            oos_per_fold_ic=oos_result.per_fold_ic,
            result=result,
            country_map=country_map,
            checklist_rules=checklist_rules,
            metrics=metrics,
            chart_paths=chart_paths,
        )

        # 8c. Optional DB persistence — only on 17/17 PASS.
        if persist and pass_count == len(checklist_rules) and checklist_rules:
            _persist_research_snapshot(
                result=result,
                assembly=assembly,
                metrics=metrics,
                cost_bps=cost_bps,
            )

    except FactorCoverageError as exc:
        console.print(f"[bold red]Factor coverage error:[/bold red] {exc}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise SystemExit(0) from None

    # Cycle 4 §10 terminal gate: 17/17 → exit 0 + weights.csv; else exit 1.
    if pass_count == len(checklist_rules) and checklist_rules:
        console.print(
            Panel("[bold green]Pipeline complete[/bold green]", style="green")
        )
    _apply_terminal_gate(
        rules=checklist_rules,
        weights=result.weights,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    from research.cli._parser import build_parser

    args = build_parser().parse_args()
    main(
        rebalance_freq=args.rebalance_freq,
        n_selected=args.n_selected,
        cost_bps=args.cost_bps,
        tax_rate=args.tax_rate,
        base_currency=args.base_currency,
        robust=args.robust,
        persist=args.persist,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
    )
