"""Step 7 — Portfolio optimization.

Extracted from ``stock_selection_pipeline.py`` lines 132–141, 183, 897–1005.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from optimizer.factors._config import (
    CompositeScoringConfig,
    RegimeTiltConfig,
    SelectionConfig,
    SelectionMethod,
)
from optimizer.pipeline import run_full_pipeline_with_selection
from optimizer.validation import WalkForwardConfig
from research.data._container import DataAssembly
from research.optimization._config import _make_builder, _make_opt_config
from research.optimization._rebalance import _hockey_stick_warn
from research.optimization._retighten import _TOP_N, _solve_with_retighten

console = Console()
logger = logging.getLogger(__name__)

# Cycle 4 §9.1: per-country round-trip transaction-cost components in bps
# (stamp duty / quoted spread / FX conversion).  Country keys mirror the
# raw `ticker_profiles.country` strings used by `_REGION_MAP`.
COUNTRY_COSTS_BPS: dict[str, dict[str, float]] = {
    "United Kingdom": {"stamp": 50.0, "spread": 8.0, "fx": 0.0},  # = 58 bps
    "France": {"stamp": 30.0, "spread": 6.0, "fx": 12.0},  # = 48 bps
    "Italy": {"stamp": 10.0, "spread": 8.0, "fx": 12.0},  # = 30 bps
    "United States": {"stamp": 0.0, "spread": 3.0, "fx": 12.0},  # = 15 bps
}
_DEFAULT_COSTS: dict[str, float] = {"stamp": 0.0, "spread": 6.0, "fx": 12.0}  # 18 bps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_weighted_cost_bps(
    weights: pd.Series, country_map: dict[str, str]
) -> float:
    """Portfolio-weighted total round-trip cost in bps (Cycle 4 §9.1)."""
    clean = weights.dropna()
    if clean.empty:
        return 0.0
    totals = clean.index.map(
        lambda t: sum(
            COUNTRY_COSTS_BPS.get(country_map.get(t, ""), _DEFAULT_COSTS).values()
        )
    )
    return float((clean.to_numpy() * np.asarray(totals, dtype=float)).sum())


# ---------------------------------------------------------------------------
# Step 7 — Portfolio optimization
# ---------------------------------------------------------------------------


def optimize_portfolio(
    assembly: DataAssembly,
    investable: pd.Index,
    ic_history: pd.DataFrame | None,
    n_selected: int = 50,
    cost_bps: float = 10.0,
    *,
    country_map: dict[str, str] | None = None,
    previous_weights: np.ndarray | None = None,
    robust: bool = False,
    uncertainty_level: float = 0.95,
    seed: int | None = None,
) -> Any:
    """Run factor-based stock selection + Cycle-3 §7.1 hard-constrained MeanRisk."""
    console.print(Panel("[bold]Step 7[/bold] — Portfolio optimization", style="blue"))

    # Cycle-3 §8: enable sklearn metadata routing (idempotent; required
    # before any `set_fit_request` call downstream, e.g. BenchmarkTracker.y).
    import sklearn

    sklearn.set_config(enable_metadata_routing=True)

    # Fix #238 + Issue #531: IC-weighted scoring with Cycle-2 EWM half-life=4
    scoring_config = CompositeScoringConfig.for_ic_weighted(ic_decay_halflife=4)

    investable_cols = [t for t in investable if t in assembly.prices.columns]
    investable_prices: pd.DataFrame = assembly.prices.loc[:, investable_cols]
    investable_volumes: pd.DataFrame = assembly.volumes.loc[
        :, [c for c in investable_cols if c in assembly.volumes.columns]
    ]

    opt_config = _make_opt_config(
        n_survivors=len(investable_cols),
        target_count=n_selected,
        cost_bps=cost_bps,
    )
    builder = _make_builder(
        sector_mapping=dict(assembly.sector_mapping),
        country_map=country_map,
        previous_weights=previous_weights,
        robust=robust,
        uncertainty_level=uncertainty_level,
    )
    optimizer_instance = builder(opt_config)

    retighten_trace: list[dict[str, Any]] = []
    if len(investable_cols) > _TOP_N:
        from skfolio.preprocessing import prices_to_returns

        full_returns = prices_to_returns(investable_prices)
        optimizer_instance, retighten_trace = _solve_with_retighten(
            optimizer_instance,
            full_returns,
            config=opt_config,
            builder=builder,
        )

    # Issue #529: enable regime tilts.  Orchestrator owns classification +
    # tilt application (single source of truth); main()'s `classify_and_tilt`
    # is logging-only and does NOT govern the actual optimisation path.
    regime_config = RegimeTiltConfig(enable=True)

    result = run_full_pipeline_with_selection(
        prices=investable_prices,
        optimizer=optimizer_instance,
        fundamentals=assembly.fundamentals.loc[
            assembly.fundamentals.index.isin(investable_cols)
        ],
        volume_history=investable_volumes,
        analyst_data=assembly.analyst_data,
        insider_data=assembly.insider_data,
        macro_data=assembly.macro_data,
        regime_data=assembly.regime_data,
        sector_mapping=dict(assembly.sector_mapping),
        scoring_config=scoring_config,  # Fix #238: explicitly IC-weighted
        selection_config=SelectionConfig(
            target_count=n_selected,
            method=SelectionMethod.FIXED_COUNT,
            buffer_fraction=0.05,
            sector_balance=True,
            max_per_sector=8,
        ),
        regime_config=regime_config,  # Issue #529: enable regime tilts
        cv_config=WalkForwardConfig(  # Cycle-3 §8: 3-year train, quarterly test
            train_size=252 * 3,
            test_size=63,
            expend_train=False,
            purged_size=5,
        ),
        ic_history=ic_history,
        risk_free_rate=assembly.risk_free_rate,  # Fix #246: proper rf
        delisting_returns=dict(assembly.delisting_returns),
        currency_map=dict(assembly.currency_map),
        fx_rates=assembly.fx_rates,
        cost_bps=cost_bps,
    )

    is_sharpe = result.summary.get("sharpe_ratio", float("nan"))
    net_sharpe = result.net_sharpe_ratio
    msg = (
        f"  Portfolio: [cyan]{len(result.weights)}[/cyan] tickers, "
        f"IS Sharpe = [cyan]{is_sharpe:.3f}[/cyan]"
    )
    if net_sharpe is not None:
        msg += f", Net Sharpe = [cyan]{net_sharpe:.3f}[/cyan]"
    console.print(msg)
    result.retighten_trace = retighten_trace
    _hockey_stick_warn(getattr(result, "net_returns", None))
    return result
