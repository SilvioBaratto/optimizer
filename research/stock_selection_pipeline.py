"""Stock selection research pipeline.

End-to-end script that builds a factor-based stock selection portfolio:

  1. Load data from the database
  2. Screen investable universe
  3. Build rolling factor score history  (point-in-time, no look-ahead)
  4. Validate factors in-sample  (IC, VIF)
  5. Validate factors out-of-sample  (rolling walk-forward IC)
  6. Classify macro regime  (composite: PMI + 2s10s + HY OAS)
  7. Apply regime-conditional group tilts
  8. Compute IC-weighted composite score
  9. Optimize portfolio with factor exposure constraints
 10. Report performance vs benchmark (with proper risk-free rate)

Fixed bugs from research-audit:
  - #237: Composite regime path unreachable → fixed via assemble_regime_data()
  - #238: IC-weighted scoring silently overridden → scoring_config threaded through
  - #239: OOS produced 0 folds with quarterly data → periods corrected
  - #246: Sharpe computed without risk-free rate → uses FRED DGS3MO series

Usage::

    python research/stock_selection_pipeline.py
    python research/stock_selection_pipeline.py --n-selected 50 --rebalance-freq 21
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Path setup — api/ must be importable when running the script directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_API_DIR = _PROJECT_ROOT / "api"
for _p in [str(_PROJECT_ROOT), str(_API_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.database import DatabaseManager  # noqa: E402

from cli.data_assembly import (  # noqa: E402
    DataAssembly,
    assemble_all,
    assemble_regime_data,
)
from optimizer.exceptions import FactorCoverageError  # noqa: E402
from optimizer.factors import (  # noqa: E402
    FactorOOSConfig,
    FactorValidationConfig,
    MacroRegime,
    apply_regime_tilts,
    classify_regime,
    run_factor_oos_validation,
    run_factor_validation,
)
from optimizer.factors._config import (  # noqa: E402
    CompositeScoringConfig,
    FactorConstructionConfig,
    FactorGroupType,
    RegimeTiltConfig,
    SelectionConfig,
    SelectionMethod,
    StandardizationConfig,
)

# Cycle-2 §4.3 spec: NW lag=4, BH alpha=0.10, |t|>=1.645 (two-sided p<0.10).
_IS_VALIDATION_CONFIG = FactorValidationConfig(
    newey_west_lags=4,
    fdr_alpha=0.10,
    t_stat_threshold=1.645,
)
from optimizer.optimization import build_mean_risk  # noqa: E402
from optimizer.optimization._config import MeanRiskConfig  # noqa: E402
from optimizer.pipeline import run_full_pipeline_with_selection  # noqa: E402
from optimizer.universe import InvestabilityScreenConfig, screen_universe  # noqa: E402
from optimizer.validation import WalkForwardConfig  # noqa: E402
from research._backtest_plots import generate_backtest_plots  # noqa: E402
from research._factors import (  # noqa: E402
    build_factor_scores_history,
    validate_factors,
)
from research._preflight import run_db_preflight as _run_db_preflight  # noqa: E402
from research._preprocessing import (  # noqa: E402
    apply_fx_to_prices,
    build_return_preprocessing_pipeline,
)

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via command-line args or environment)
# ---------------------------------------------------------------------------

REBALANCE_FREQ: int = 63    # quarterly  (~63 trading days)
N_SELECTED: int = 50        # target portfolio size after factor ranking
TOP_N_DISPLAY: int = 25     # tickers shown in the selection table
MIN_SUCCESS_FRACTION: float = 0.5

# Fix #239: parameters are index-based, not calendar months.
# With quarterly rebalancing (~20 dates for 5 years):
#   train=8 ≈ 2 years, val=4 ≈ 1 year → (20-8)//2 = 6 folds
OOS_CONFIG = FactorOOSConfig(train_periods=8, val_periods=4, step_periods=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _annualized_return(r: pd.Series) -> float:
    """Compound annualized return from daily returns."""
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def _sharpe(
    returns: pd.Series,
    rf_series: pd.Series | None = None,
) -> float:
    """Annualized Sharpe ratio with time-varying risk-free rate.

    Fix issue #246: previous version used rf=0 (return-to-vol ratio), which
    systematically overstated Sharpe for low-volatility strategies by up to
    40-50% during 2022-2024 when Fed Funds exceeded 5%.  This version uses
    contemporaneous daily FRED DGS3MO as the risk-free benchmark.
    """
    if returns.empty:
        return float("nan")

    if rf_series is not None and not rf_series.empty:
        # Annual rate → daily; forward-fill to trading calendar
        daily_rf = (
            rf_series.reindex(returns.index, method="ffill").fillna(0.0) / 252.0
        )
        excess = returns - daily_rf
    else:
        excess = returns

    ann_excess = _annualized_return(excess)
    std_val = cast(float, excess.std(ddof=1))
    vol = std_val * np.sqrt(252.0)
    return ann_excess / vol if vol > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# Step 1 — Data assembly
# ---------------------------------------------------------------------------


def _fetch_benchmark_returns(
    start: pd.Timestamp,
    end: pd.Timestamp,
    ticker: str = "SPY",
) -> pd.Series:
    """Download daily benchmark returns from yfinance."""
    import yfinance as yf
    from skfolio.preprocessing import prices_to_returns

    data: pd.DataFrame = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if data is None or data.empty:
        logger.warning("Could not download benchmark %s", ticker)
        return pd.Series(dtype=float, name=ticker)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close_series: pd.Series = close  # type: ignore[assignment]
    ret_df = prices_to_returns(close_series.to_frame(ticker))
    ret_series: pd.Series = ret_df.iloc[:, 0]
    ret_series.name = ticker
    return ret_series


def _build_country_map(db_manager: DatabaseManager) -> dict[str, str]:
    """Build ticker → country mapping from ticker_profiles."""
    from sqlalchemy import text

    with db_manager.get_session() as session:
        rows = session.execute(
            text(
                "SELECT i.yfinance_ticker, tp.country "
                "FROM instruments i "
                "LEFT JOIN ticker_profiles tp ON tp.instrument_id = i.id "
                "WHERE tp.country IS NOT NULL"
            )
        ).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}


_MIN_ASSEMBLY_TICKERS: int = 2_000
_MIN_ASSEMBLY_TRADING_DAYS: int = 1_260


def _assert_assembly_size(assembly: DataAssembly) -> None:
    """Enforce minimum tickers and trading-day count post-assembly (issue #522)."""
    if assembly.n_tickers < _MIN_ASSEMBLY_TICKERS:
        raise RuntimeError(
            f"DataAssembly: n_tickers={assembly.n_tickers} below floor "
            f"{_MIN_ASSEMBLY_TICKERS}."
        )
    if assembly.n_trading_days < _MIN_ASSEMBLY_TRADING_DAYS:
        raise RuntimeError(
            f"DataAssembly: n_trading_days={assembly.n_trading_days} below floor "
            f"{_MIN_ASSEMBLY_TRADING_DAYS}."
        )


def _materialise_clean_returns(
    assembly: DataAssembly,
    investable: pd.Index,
) -> pd.DataFrame:
    """Slice EUR prices to investable, convert to linear returns, preprocess.

    Cycle 1 hand-off (§3): the cleaned linear-return panel is the single
    artefact downstream cycles consume.
    """
    from skfolio.preprocessing import prices_to_returns

    cols = list(investable)
    investable_prices = assembly.prices.loc[:, cols]
    returns = cast(pd.DataFrame, prices_to_returns(investable_prices))
    pipeline = build_return_preprocessing_pipeline(assembly.sector_mapping)
    clean: pd.DataFrame = pipeline.fit_transform(returns)
    arr = clean.to_numpy()
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise RuntimeError("clean_returns contains NaN or inf after preprocessing.")
    console.print(
        f"  Clean returns: [cyan]{len(clean)}[/cyan] days x "
        f"[cyan]{clean.shape[1]}[/cyan] tickers, "
        f"range [cyan]{clean.index[0].date()}[/cyan] -> "
        f"[cyan]{clean.index[-1].date()}[/cyan]"
    )
    return clean


def load_data() -> tuple[DataAssembly, dict[str, str], DatabaseManager]:
    """Assemble all data from the database.

    Returns
    -------
    tuple[DataAssembly, dict[str, str], DatabaseManager]
        (assembly, country_map, db_manager) where ``assembly.prices`` is
        FX-converted to EUR base currency, ``country_map`` is
        ticker → country, and ``db_manager`` is the initialised handle so
        downstream steps can persist results back to the DB (issue #530).
    """
    console.print(Panel("[bold]Step 1[/bold] — Loading data", style="blue"))
    db_manager = DatabaseManager()
    db_manager.initialize()
    _run_db_preflight(db_manager)
    assembly = assemble_all(db_manager, include_delisted=True)
    assembly.prices = apply_fx_to_prices(
        assembly.prices,
        assembly.currency_map,
        assembly.fx_rates,
        base_currency="EUR",
    )
    _assert_assembly_size(assembly)
    country_map = _build_country_map(db_manager)
    console.print(
        f"  Loaded [cyan]{assembly.n_tickers}[/cyan] tickers, "
        f"[cyan]{assembly.n_trading_days}[/cyan] trading days "
        f"(EUR base, hash=[cyan]{assembly.assembly_hash}[/cyan])"
    )
    return assembly, country_map, db_manager


# ---------------------------------------------------------------------------
# Step 2 — Investability screening
# ---------------------------------------------------------------------------

_UNIVERSE_FLOOR: int = 200
_UNIVERSE_BAND: tuple[int, int] = (300, 1500)


def _assert_universe_size(passing: pd.Index) -> None:
    """Enforce universe-size floor and warn on out-of-band counts (issue #520)."""
    n = len(passing)
    if n < _UNIVERSE_FLOOR:
        raise RuntimeError(
            f"Investable universe has only {n} tickers "
            f"(floor: {_UNIVERSE_FLOOR}). Check `InvestabilityScreenConfig` "
            f"thresholds and DB price/volume coverage."
        )
    low, high = _UNIVERSE_BAND
    if not (low <= n <= high):
        logger.warning(
            "Investable universe size %d is outside expected band [%d, %d].",
            n, low, high,
        )


def screen_investable(assembly: DataAssembly) -> pd.Index:
    """Apply investability screens and return the passing ticker index."""
    console.print(Panel("[bold]Step 2[/bold] — Screening universe", style="blue"))
    config = InvestabilityScreenConfig.for_developed_markets()
    passing = screen_universe(
        fundamentals=assembly.fundamentals,
        price_history=assembly.prices,
        volume_history=assembly.volumes,
        financial_statements=assembly.financial_statements,
        config=config,
    )
    console.print(f"  {len(passing)} tickers pass investability screens")
    _assert_universe_size(passing)
    return passing


# ---------------------------------------------------------------------------
# Step 3 — Factor scores history
# ---------------------------------------------------------------------------


def build_history(
    assembly: DataAssembly,
    investable: pd.Index,
    rebalance_freq: int = REBALANCE_FREQ,
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
        fundamental_history=assembly.fundamental_history,  # PIT — no look-ahead (#273)
        min_success_fraction=MIN_SUCCESS_FRACTION,
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

    # Use the last cross-section for VIF (standardized snapshot)
    last_date = sorted(
        next(iter(factor_scores_dict.values())).index
    )[-1]
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
                f"  [yellow]High VIF factors (>5):[/yellow] "
                f"{', '.join(names)}"
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

    # Reshape to MultiIndex (date, ticker) × factors
    stacked_factors = [
        df.stack().rename(name)
        for name, df in factor_scores_dict.items()
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
# Cycle-2 §6.1 spec helpers — selection size + sector coverage (issue #531)
# ---------------------------------------------------------------------------

# Cycle-2 §6.1: portfolio size constrained to [25, 50] selected stocks.
N_SELECTED_MIN: int = 25
N_SELECTED_MAX: int = 50

# GICS Level-1 sectors (11). Used for the sector-coverage warning only.
_GICS_SECTORS: tuple[str, ...] = (
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
)


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
# Step 5b — Coverage gate (Cycle-2 §4.4)
# ---------------------------------------------------------------------------


def _check_factor_coverage(
    is_report: Any,
    oos_result: Any,
    *,
    min_factors: int = 4,
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


# ---------------------------------------------------------------------------
# Step 6 — Macro regime classification
# ---------------------------------------------------------------------------


def classify_and_tilt(
    assembly: DataAssembly,
    db_manager: DatabaseManager | None = None,
) -> tuple[MacroRegime, dict[FactorGroupType, float]]:
    """Classify macro regime and compute regime-conditional group tilts.

    Fix issue #237: the previous code called classify_regime(assembly.macro_data)
    which only contains gdp_growth + yield_spread, making the composite path
    (requiring pmi, spread_2s10s, hy_oas) unreachable.

    Issue #529: this function is **logging-only** for tilt application.  The
    orchestrator inside ``run_full_pipeline_with_selection`` owns the actual
    classification + tilt application against publication-lag-shifted macro
    data; the regime printed here is the unlagged snapshot for inspection.

    Issue #530: when ``db_manager`` is supplied, the rule-based regime is
    cached into ``macro_calibrations.regime_classification`` for the ``US``
    country row so downstream services can read it without re-classifying.
    """
    console.print(Panel("[bold]Step 6[/bold] — Regime classification", style="blue"))

    regime_data = assemble_regime_data(
        macro_data=assembly.macro_data,
        fred_data=assembly.fred_data,
        te_observations=assembly.te_observations,
        sentiment_data=assembly.sentiment_data,
    )

    regime = classify_regime(regime_data)
    tilt_config = RegimeTiltConfig(enable=True)
    tilts = apply_regime_tilts(
        group_weights=dict.fromkeys(FactorGroupType, 1.0),
        regime=regime,
        config=tilt_config,
    )

    regime_colors = {
        MacroRegime.EXPANSION: "[green]",
        MacroRegime.SLOWDOWN: "[yellow]",
        MacroRegime.RECESSION: "[red]",
        MacroRegime.RECOVERY: "[cyan]",
    }
    color = regime_colors.get(regime, "")
    console.print(f"  Macro regime: {color}{regime.value.upper()}[/]")

    if db_manager is not None:
        _cache_regime_classification(db_manager, regime)

    return regime, tilts


def _cache_regime_classification(
    db_manager: DatabaseManager, regime: MacroRegime
) -> None:
    """Persist the rule-based regime to ``macro_calibrations`` for country='US'."""
    from app.repositories.macro_regime_repository import MacroRegimeRepository

    with db_manager.get_session() as session:
        MacroRegimeRepository(session).upsert_regime_classification(
            country="US", regime=regime.value
        )
        session.commit()


# ---------------------------------------------------------------------------
# Step 7 — Portfolio optimization
# ---------------------------------------------------------------------------


def optimize_portfolio(
    assembly: DataAssembly,
    investable: pd.Index,
    ic_history: pd.DataFrame | None,
    n_selected: int = N_SELECTED,
    cost_bps: float = 10.0,
) -> Any:
    """Run factor-based stock selection and portfolio optimization.

    Fix issue #238: scoring_config is explicitly passed as ic_weighted.

    Checklist constraints (from portfolio_checklist.md):
      - max_weights=0.10  → single-stock cap 10%
      - max_sector_weight=0.20 → sector diversification
      - L2 regularisation (0.05) → push toward equal-weight, naturally
        keeping positions above ~2% for 20-40 stock universes
    """
    console.print(Panel("[bold]Step 7[/bold] — Portfolio optimization", style="blue"))

    # Fix #238 + Issue #531: IC-weighted scoring with Cycle-2 EWM half-life=4
    scoring_config = CompositeScoringConfig.for_ic_weighted(ic_decay_halflife=4)

    investable_cols = [t for t in investable if t in assembly.prices.columns]
    investable_prices: pd.DataFrame = assembly.prices.loc[:, investable_cols]
    investable_volumes: pd.DataFrame = assembly.volumes.loc[
        :, [c for c in investable_cols if c in assembly.volumes.columns]
    ]

    # Checklist-compliant optimizer: sector + position caps.
    # L2 regularisation (0.05) pushes toward equal-weight, naturally keeping
    # positions above ~2% for 20-40 stock universes.
    # Hard constraints: max_weights=0.10 (single-stock cap) and
    # max_sector_weight=0.20 (sector diversification).
    # Note: min_weights stays at 0.0 to avoid infeasibility in walk-forward
    # CV splits where fewer stocks are available.
    opt_config = MeanRiskConfig.for_max_sharpe_sector_constrained(
        max_sector_weight=0.20,
    )
    optimizer_instance = build_mean_risk(
        opt_config, sector_mapping=assembly.sector_mapping
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
        sector_mapping=assembly.sector_mapping,
        scoring_config=scoring_config,  # Fix #238: explicitly IC-weighted
        selection_config=SelectionConfig(
            target_count=n_selected,
            method=SelectionMethod.FIXED_COUNT,
            buffer_fraction=0.05,
            sector_balance=True,
            max_per_sector=8,
        ),
        regime_config=regime_config,  # Issue #529: enable regime tilts
        cv_config=WalkForwardConfig(),  # walk-forward backtest
        ic_history=ic_history,
        risk_free_rate=assembly.risk_free_rate,  # Fix #246: proper rf
        delisting_returns=assembly.delisting_returns,
        currency_map=assembly.currency_map,
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
    return result


# ---------------------------------------------------------------------------
# Diversification analysis
# ---------------------------------------------------------------------------


def _print_diversification(
    all_weights: list[tuple[str, float]],
    sector_mapping: dict[str, str],
    country_map: dict[str, str],
) -> None:
    """Print sector, country, and concentration breakdown tables."""
    from collections import defaultdict

    n = len(all_weights)

    # --- Sector breakdown ---
    sector_w: dict[str, float] = defaultdict(float)
    for ticker, w in all_weights:
        sector_w[sector_mapping.get(ticker, "Unknown")] += w

    sector_table = Table(
        title="Sector Allocation",
        show_header=True,
        header_style="bold cyan",
    )
    sector_table.add_column("Sector", style="dim")
    sector_table.add_column("Weight", justify="right")
    sector_table.add_column("# Tickers", justify="right")

    sector_counts: dict[str, int] = defaultdict(int)
    for ticker, _ in all_weights:
        sector_counts[sector_mapping.get(ticker, "Unknown")] += 1

    for sector, w in sorted(sector_w.items(), key=lambda x: -x[1]):
        sector_table.add_row(sector, f"{w:.2%}", str(sector_counts[sector]))
    console.print(sector_table)

    # --- Country breakdown ---
    country_w: dict[str, float] = defaultdict(float)
    country_counts: dict[str, int] = defaultdict(int)
    for ticker, w in all_weights:
        country = country_map.get(ticker, "Unknown")
        country_w[country] += w
        country_counts[country] += 1

    country_table = Table(
        title="Country Allocation",
        show_header=True,
        header_style="bold cyan",
    )
    country_table.add_column("Country", style="dim")
    country_table.add_column("Weight", justify="right")
    country_table.add_column("# Tickers", justify="right")

    for country, w in sorted(country_w.items(), key=lambda x: -x[1]):
        country_table.add_row(country, f"{w:.2%}", str(country_counts[country]))
    console.print(country_table)

    # --- Concentration metrics ---
    sorted_w = sorted((w for _, w in all_weights), reverse=True)
    top1 = sorted_w[0] if sorted_w else 0.0
    top5 = sum(sorted_w[:5])
    top10 = sum(sorted_w[:10])
    hhi = sum(w**2 for w in sorted_w)
    eff_n = 1.0 / hhi if hhi > 0 else 0.0

    conc_table = Table(
        title="Concentration Metrics",
        show_header=True,
        header_style="bold cyan",
    )
    conc_table.add_column("Metric", style="dim")
    conc_table.add_column("Value", justify="right")

    conc_table.add_row("Total positions", str(n))
    conc_table.add_row("Top-1 weight", f"{top1:.2%}")
    conc_table.add_row("Top-5 weight", f"{top5:.2%}")
    conc_table.add_row("Top-10 weight", f"{top10:.2%}")
    conc_table.add_row("HHI", f"{hhi:.4f}")
    conc_table.add_row(
        "Effective N (1/HHI)", f"{eff_n:.1f}"
    )
    conc_table.add_row("Sectors", str(len(sector_w)))
    conc_table.add_row("Countries", str(len(country_w)))
    console.print(conc_table)


# ---------------------------------------------------------------------------
# Checklist validation
# ---------------------------------------------------------------------------


def _validate_checklist(
    all_weights: list[tuple[str, float]],
    sector_mapping: dict[str, str],
    country_map: dict[str, str],
    metrics: dict[str, dict[str, float]],
) -> None:
    """Validate portfolio against portfolio_checklist.md criteria."""
    from collections import defaultdict

    sorted_w = sorted((w for _, w in all_weights), reverse=True)
    n = len(sorted_w)

    # Sector weights
    sector_w: dict[str, float] = defaultdict(float)
    for ticker, w in all_weights:
        sector_w[sector_mapping.get(ticker, "Unknown")] += w
    max_sector = max(sector_w.values()) if sector_w else 0.0
    max_sector_name = max(sector_w, key=sector_w.get) if sector_w else "N/A"  # type: ignore[arg-type]

    # Country weights
    country_w: dict[str, float] = defaultdict(float)
    for ticker, w in all_weights:
        country_w[country_map.get(ticker, "Unknown")] += w

    # Region weights (approximate)
    region_map = {
        "United States": "Americas", "Canada": "Americas",
        "Colombia": "Americas", "Argentina": "Americas",
        "Brazil": "Americas", "Mexico": "Americas", "Chile": "Americas",
        "United Kingdom": "Europe", "France": "Europe",
        "Germany": "Europe", "Netherlands": "Europe",
        "Switzerland": "Europe", "Ireland": "Europe",
        "Italy": "Europe", "Spain": "Europe", "Monaco": "Europe",
        "Luxembourg": "Europe", "Belgium": "Europe",
        "Norway": "Europe", "Sweden": "Europe", "Denmark": "Europe",
        "Finland": "Europe", "Austria": "Europe", "Portugal": "Europe",
        "China": "Asia-Pacific", "Taiwan": "Asia-Pacific",
        "Japan": "Asia-Pacific", "South Korea": "Asia-Pacific",
        "Indonesia": "Asia-Pacific", "India": "Asia-Pacific",
        "Singapore": "Asia-Pacific", "Australia": "Asia-Pacific",
        "Hong Kong": "Asia-Pacific",
        "Israel": "Middle East & Africa", "Turkey": "Middle East & Africa",
        "South Africa": "Middle East & Africa",
    }
    region_w: dict[str, float] = defaultdict(float)
    for country, w in country_w.items():
        region_w[region_map.get(country, "Other")] += w
    max_region = max(region_w.values()) if region_w else 0.0

    # Concentration
    top4 = sum(sorted_w[:4])
    hhi = sum(w**2 for w in sorted_w)

    # Tech / Healthcare sector weights
    tech_w = sector_w.get("Technology", 0.0)
    health_w = sector_w.get("Healthcare", 0.0)
    min_pos = sorted_w[-1] if sorted_w else 0.0

    # Performance metrics (from portfolio if available)
    pm = metrics.get("Portfolio", {})
    sharpe = pm.get("Sharpe (rf)", float("nan"))
    max_dd = pm.get("Max Drawdown", float("nan"))

    # Build checklist table
    table = Table(
        title="Portfolio Checklist Validation",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rule", style="dim", width=42)
    table.add_column("Target", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Status", justify="center")

    def _row(
        rule: str, target: str, actual: str, *, ok: bool
    ) -> None:
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(rule, target, actual, status)

    _row(
        "No single sector > 15%",
        "≤ 15%",
        f"{max_sector:.1%} ({max_sector_name})",
        ok=max_sector <= 0.15,
    )
    _row(
        "No single region > 60-70%",
        "≤ 60%",
        f"{max_region:.1%}",
        ok=max_region <= 0.60,
    )
    _row("Top-4 holdings < 30%", "< 30%", f"{top4:.1%}", ok=top4 < 0.30)
    _row(
        "Single-stock cap ≤ 10%",
        "≤ 10%",
        f"{sorted_w[0]:.1%}" if sorted_w else "N/A",
        ok=sorted_w[0] <= 0.10 if sorted_w else False,
    )
    _row(
        "Min position ≥ 2%",
        "≥ 2%",
        f"{min_pos:.1%}",
        ok=min_pos >= 0.02,
    )
    _row(
        "Healthcare exposure ≥ 8%",
        "≥ 8%",
        f"{health_w:.1%}",
        ok=health_w >= 0.08,
    )
    _row(
        "Technology exposure ≥ 10%",
        "≥ 10%",
        f"{tech_w:.1%}",
        ok=tech_w >= 0.10,
    )
    _row("HHI < 0.12", "< 0.12", f"{hhi:.4f}", ok=hhi < 0.12)
    _row(
        "Sharpe > 1.0",
        "> 1.0",
        f"{sharpe:.3f}" if not np.isnan(sharpe) else "N/A",
        ok=sharpe > 1.0 if not np.isnan(sharpe) else False,
    )
    _row(
        "Max drawdown < benchmark (-22%)",
        "> -22%",
        f"{max_dd:.1%}" if not np.isnan(max_dd) else "N/A",
        ok=max_dd > -0.22 if not np.isnan(max_dd) else False,
    )
    _row(
        "Total positions ≥ 20",
        "≥ 20",
        str(n),
        ok=n >= 20,
    )
    _row(
        "Countries ≥ 5",
        "≥ 5",
        str(len(country_w)),
        ok=len(country_w) >= 5,
    )

    console.print(table)

    # Summary
    pass_count = sum(
        1
        for ok in [
            max_sector <= 0.15,
            max_region <= 0.60,
            top4 < 0.30,
            (sorted_w[0] <= 0.10 if sorted_w else False),
            min_pos >= 0.02,
            health_w >= 0.08,
            tech_w >= 0.10,
            hhi < 0.12,
            (sharpe > 1.0 if not np.isnan(sharpe) else False),
            (max_dd > -0.22 if not np.isnan(max_dd) else False),
            n >= 20,
            len(country_w) >= 5,
        ]
        if ok
    )
    total = 12
    color = "green" if pass_count == total else "yellow" if pass_count >= 9 else "red"
    console.print(
        f"  [{color}]Checklist: {pass_count}/{total} passed[/{color}]"
    )


# ---------------------------------------------------------------------------
# Step 8 — Benchmark comparison
# ---------------------------------------------------------------------------


def report_performance(
    result: Any,
    assembly: DataAssembly,
    country_map: dict[str, str],
) -> None:
    """Print portfolio performance, weights, and diversification breakdown.

    Fix issue #246: Sharpe ratio now uses FRED DGS3MO series for rf, not rf=0.
    """
    console.print(Panel("[bold]Step 8[/bold] — Performance report", style="blue"))

    rf_series = assembly.risk_free_rate_series  # pd.Series from FRED DGS3MO

    portfolio_returns: pd.Series | None = None
    benchmark_returns: pd.Series | None = None

    # Extract backtest returns from result.backtest (MultiPeriodPortfolio)
    # or result.net_returns (transaction-cost adjusted series)
    if result.net_returns is not None and not result.net_returns.empty:
        net_rets: pd.Series = result.net_returns.copy()
        net_rets.name = "Portfolio (net)"
        portfolio_returns = net_rets
    elif result.backtest is not None:
        with contextlib.suppress(AttributeError):
            portfolio_returns = pd.Series(
                result.backtest.returns,
                name="Portfolio",
            )

    # Fetch SPY benchmark aligned to backtest period
    if portfolio_returns is not None and not portfolio_returns.empty:
        console.print("  Downloading SPY benchmark...")
        benchmark_returns = _fetch_benchmark_returns(
            start=portfolio_returns.index[0],
            end=portfolio_returns.index[-1],
        )
        # Align to portfolio trading dates
        if benchmark_returns is not None and not benchmark_returns.empty:
            common_idx = portfolio_returns.index.intersection(
                benchmark_returns.index
            )
            benchmark_returns = benchmark_returns.loc[common_idx]

    # Compute metrics for portfolio and benchmark
    metrics: dict[str, dict[str, float]] = {}
    series_pairs: list[tuple[str, pd.Series | None]] = [
        ("Portfolio", portfolio_returns),
        ("SPY (benchmark)", benchmark_returns),
    ]
    for label, rets in series_pairs:
        if rets is None or rets.empty:
            continue
        cumulative = (1.0 + rets).cumprod()
        drawdowns: pd.Series = cumulative / cumulative.cummax() - 1.0
        max_dd = float(drawdowns.min()) if len(rets) > 1 else float("nan")
        metrics[label] = {
            "Ann. Return": _annualized_return(rets),
            "Ann. Vol": cast(float, rets.std()) * np.sqrt(252.0),
            "Sharpe (rf)": _sharpe(rets, rf_series),
            "Max Drawdown": max_dd,
        }

    if not metrics:
        console.print("  [yellow]No backtest results available[/yellow]")
        return

    table = Table(
        title="Performance Metrics",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Strategy", style="dim")
    table.add_column("Ann. Return", justify="right")
    table.add_column("Ann. Vol", justify="right")
    table.add_column("Sharpe (rf-adj)", justify="right")
    table.add_column("Max Drawdown", justify="right")

    for label, m in metrics.items():
        table.add_row(
            label,
            f"{m['Ann. Return']:.2%}",
            f"{m['Ann. Vol']:.2%}",
            f"{m['Sharpe (rf)']:.3f}",
            f"{m['Max Drawdown']:.2%}",
        )
    console.print(table)

    # Print top weights with country
    weights_table = Table(
        title=f"Top {min(TOP_N_DISPLAY, len(result.weights))} Positions",
        show_header=True,
        header_style="bold cyan",
    )
    weights_table.add_column("Ticker", style="dim")
    weights_table.add_column("Weight", justify="right")
    weights_table.add_column("Sector", justify="left")
    weights_table.add_column("Country", justify="left")

    all_weights: list[tuple[str, float]] = sorted(
        result.weights.items(), key=lambda x: -x[1]
    )
    for ticker, w in all_weights[:TOP_N_DISPLAY]:
        sector = assembly.sector_mapping.get(ticker, "Unknown")
        country = country_map.get(ticker, "Unknown")
        weights_table.add_row(ticker, f"{w:.2%}", sector, country)
    console.print(weights_table)

    # --- Diversification analysis ---
    _print_diversification(all_weights, assembly.sector_mapping, country_map)

    # --- Checklist validation ---
    _validate_checklist(all_weights, assembly.sector_mapping, country_map, metrics)

    # --- Backtest charts ---
    if portfolio_returns is not None and not portfolio_returns.empty:
        from pathlib import Path

        chart_paths = generate_backtest_plots(
            portfolio_returns=portfolio_returns,
            weight_history=result.weight_history,
            sector_mapping=assembly.sector_mapping,
            benchmark_returns=benchmark_returns,
            rf_series=rf_series,
            output_dir=Path("research/output"),
        )
        console.print(
            f"\n  [cyan]Saved {len(chart_paths)} charts:[/cyan]"
        )
        for p in chart_paths:
            console.print(f"    {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    rebalance_freq: int = REBALANCE_FREQ,
    n_selected: int = N_SELECTED,
    cost_bps: float = 10.0,
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

    try:
        # 1. Load
        assembly, country_map, db_manager = load_data()

        # 2. Screen
        investable = screen_investable(assembly)

        # 2.5. Materialise clean linear-return panel (Cycle 1 §3 hand-off).
        # Cycle 2 will consume `clean_returns` for factor construction.
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
        current_regime, tilts = classify_and_tilt(assembly, db_manager=db_manager)
        logger.info("Regime: %s, tilts: %s", current_regime.value, tilts)

        # 7. Optimize
        result = optimize_portfolio(
            assembly=assembly,
            investable=investable,
            ic_history=ic_history,
            n_selected=n_selected,
            cost_bps=cost_bps,
        )

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

        # 8. Report
        report_performance(result, assembly, country_map)

    except FactorCoverageError as exc:
        console.print(f"[bold red]Factor coverage error:[/bold red] {exc}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise SystemExit(0) from None

    console.print(Panel("[bold green]Pipeline complete[/bold green]", style="green"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stock selection research pipeline")
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=REBALANCE_FREQ,
        help=f"Rebalancing frequency in trading days (default: {REBALANCE_FREQ})",
    )
    parser.add_argument(
        "--n-selected",
        type=int,
        default=N_SELECTED,
        help=f"Number of stocks to select (default: {N_SELECTED})",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points (default: 10)",
    )
    args = parser.parse_args()
    main(
        rebalance_freq=args.rebalance_freq,
        n_selected=args.n_selected,
        cost_bps=args.cost_bps,
    )
