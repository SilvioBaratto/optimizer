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
from datetime import date
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
from optimizer.optimization import build_region_linear_constraints  # noqa: E402
from research.data_assembly import (  # noqa: E402
    DataAssembly,
    assemble_all,
    assemble_regime_data,
)

# Cycle-2 §4.3 spec: NW lag=4, BH alpha=0.10, |t|>=1.645 (two-sided p<0.10).
_IS_VALIDATION_CONFIG = FactorValidationConfig(
    newey_west_lags=4,
    fdr_alpha=0.10,
    t_stat_threshold=1.645,
)
from optimizer.pipeline import run_full_pipeline_with_selection  # noqa: E402
from optimizer.universe import InvestabilityScreenConfig, screen_universe  # noqa: E402
from optimizer.validation import WalkForwardConfig  # noqa: E402
from research._backtest_plots import generate_backtest_plots  # noqa: E402
from research._factors import (  # noqa: E402
    build_factor_scores_history,
    validate_factors,
)
from research._optimization import (  # noqa: E402
    _REGION_MAP,
    _TOP_N,
    _decide_rebalance,
    _hockey_stick_warn,
    _make_builder,
    _make_opt_config,
    _solve_with_retighten,
)
from research._persistence import _diff_from_default, persist_research_run  # noqa: E402
from research._preflight import run_db_preflight as _run_db_preflight  # noqa: E402
from research._preprocessing import (  # noqa: E402
    apply_fx_to_prices,
    build_return_preprocessing_pipeline,
)
from research._report import compute_binding_constraints, render_report  # noqa: E402
from research._returns import compute_after_tax_returns  # noqa: E402

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via command-line args or environment)
# ---------------------------------------------------------------------------

REBALANCE_FREQ: int = 63  # quarterly  (~63 trading days)
N_SELECTED: int = 50  # target portfolio size after factor ranking
TOP_N_DISPLAY: int = 25  # tickers shown in the selection table
MIN_SUCCESS_FRACTION: float = 0.5

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

# Cycle-3 §11: hybrid rebalance review-date persistence (file-backed; DB
# write-back is Cycle 5).
_LAST_REVIEW_DATE_FILE = (
    Path(__file__).resolve().parent / "output" / "last_review_date.txt"
)


def _read_last_review_date(today: pd.Timestamp) -> pd.Timestamp:
    """Read ``last_review_date`` from disk, falling back to the prior quarter-end."""
    if _LAST_REVIEW_DATE_FILE.exists():
        text = _LAST_REVIEW_DATE_FILE.read_text().strip()
        if text:
            return pd.Timestamp(text).normalize()
    return (today.to_period("Q") - 1).end_time.normalize()


def _write_last_review_date(date: pd.Timestamp) -> None:
    """Persist ``date`` (ISO ``YYYY-MM-DD``) for the next pipeline run."""
    _LAST_REVIEW_DATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LAST_REVIEW_DATE_FILE.write_text(date.date().isoformat() + "\n")


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


def compute_weighted_cost_bps(weights: pd.Series, country_map: dict[str, str]) -> float:
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
        daily_rf = rf_series.reindex(returns.index, method="ffill").fillna(0.0) / 252.0
        excess = returns - daily_rf
    else:
        excess = returns

    ann_excess = _annualized_return(excess)
    std_val = cast(float, excess.std(ddof=1))
    vol = std_val * np.sqrt(252.0)
    return ann_excess / vol if vol > 0.0 else float("nan")


def _daily_rf(returns: pd.Series, rf_series: pd.Series | None) -> pd.Series:
    """Forward-filled daily risk-free rate aligned to ``returns`` index."""
    if rf_series is None or rf_series.empty:
        return pd.Series(0.0, index=returns.index)
    return rf_series.reindex(returns.index, method="ffill").fillna(0.0) / 252.0


def _sortino(returns: pd.Series, rf_series: pd.Series | None = None) -> float:
    """Annualised Sortino: excess return / downside vol (Cycle 4 §9.3)."""
    if returns.empty:
        return 0.0
    daily_rf = _daily_rf(returns, rf_series)
    excess = returns - daily_rf
    downside = excess[excess < 0.0]
    if downside.empty:
        return 0.0
    downside_vol = float(downside.std(ddof=1)) * np.sqrt(252.0)
    if downside_vol <= 0.0:
        return 0.0
    return _annualized_return(excess) / downside_vol


def _downside_vol(returns: pd.Series, rf_series: pd.Series | None = None) -> float:
    """Annualised std of below-rf returns (Cycle 4 §9.3)."""
    if returns.empty:
        return 0.0
    daily_rf = _daily_rf(returns, rf_series)
    downside = (returns - daily_rf)[(returns - daily_rf) < 0.0]
    if downside.empty:
        return 0.0
    return float(downside.std(ddof=1)) * np.sqrt(252.0)


def _information_ratio(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Annualised IR = mean(active) / std(active) × √252 (Cycle 4 §9.3)."""
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) == 0:
        return float("nan")
    active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    std_val = float(active.std(ddof=1))
    if std_val <= 1e-12:
        return 0.0
    return float(active.mean()) / std_val * np.sqrt(252.0)


_METRICS_KEY_MAP: dict[str, str] = {
    "Ann. Return": "ann_return",
    "Ann. Vol": "ann_vol",
    "Sharpe (rf)": "sharpe",
    "Sortino": "sortino",
    "Info Ratio": "info_ratio",
    "Downside Vol": "downside_vol",
    "Max Drawdown": "max_drawdown",
}


def _to_json_safe(value: float) -> float | None:
    """Cast numpy scalars to float; replace NaN with None for strict JSON."""
    if value is None:
        return None
    f = float(value)
    return None if np.isnan(f) else f


def _project_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Convert display-key metrics dict to JSON-safe schema dict."""
    return {
        json_key: _to_json_safe(metrics.get(display_key, float("nan")))
        for display_key, json_key in _METRICS_KEY_MAP.items()
    }


def write_metrics_json(
    metrics_by_label: dict[str, dict[str, float]], output_dir: Path
) -> Path:
    """Persist Cycle 4 §9.3 metrics block to ``metrics.json`` (Cycle 5 input)."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "rf_assumption": "FRED DGS3MO daily forward-fill ÷ 252",
    }
    if "Portfolio" in metrics_by_label:
        payload["net_of_cost"] = _project_metrics(metrics_by_label["Portfolio"])
    if "Portfolio (after-tax)" in metrics_by_label:
        payload["after_tax"] = _project_metrics(
            metrics_by_label["Portfolio (after-tax)"]
        )
    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return out_path


def _project_rule_for_json(rule: dict[str, Any]) -> dict[str, Any]:
    """Convert checklist rule dict to JSON-safe form (NaN floats → null)."""
    measured = rule.get("measured")
    if isinstance(measured, float) and np.isnan(measured):
        measured = None
    return {
        "rule": rule["rule"],
        "pass": bool(rule["pass"]),
        "measured": measured,
        "target": rule["target"],
    }


def write_checklist_json(
    *,
    rules: list[dict[str, Any]],
    gross_metrics: dict[str, float] | None,
    net_metrics: dict[str, float] | None,
    after_tax_metrics: dict[str, float] | None,
    output_dir: Path,
) -> Path:
    """Persist Cycle 4 §10 checklist results to ``checklist.json``."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in rules if r.get("pass"))
    payload: dict[str, Any] = {
        "rules": [_project_rule_for_json(r) for r in rules],
        "summary": {"passed": passed, "total": len(rules)},
        "breakdown": {
            "gross": _project_metrics(gross_metrics or {}),
            "net_of_cost": _project_metrics(net_metrics or {}),
            "after_tax": _project_metrics(after_tax_metrics or {}),
        },
    }
    out_path = output_dir / "checklist.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return out_path


def write_weights_csv(weights: pd.Series, output_dir: Path) -> Path:
    """Persist final portfolio weights to ``weights.csv`` sorted desc."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_w = weights.sort_values(ascending=False)
    df = pd.DataFrame(
        {"ticker": list(sorted_w.index), "weight": sorted_w.to_numpy(dtype=float)}
    )
    out_path = output_dir / "weights.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _render_failure_table(failed_rules: list[dict[str, Any]]) -> None:
    """Print the measured-vs-target table for failing checklist rules only."""
    table = Table(
        title="Failing Rules",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Rule", style="dim", width=42)
    table.add_column("Target", justify="right")
    table.add_column("Measured", justify="right")
    for r in failed_rules:
        table.add_row(r["rule"], r["target"], str(r["measured"]))
    console.print(table)


_CHECKLIST_TOTAL = 17


def _apply_terminal_gate(
    *,
    rules: list[dict[str, Any]],
    weights: pd.Series,
    output_dir: Path,
) -> None:
    """Cycle 4 §10 terminal gate: 17/17 → exit 0 + weights.csv; else exit 1."""
    assert len(rules) == _CHECKLIST_TOTAL, (  # noqa: S101  -- invariant guard
        f"checklist must have exactly {_CHECKLIST_TOTAL} rules, got {len(rules)}"
    )
    pass_count = sum(1 for r in rules if r.get("pass"))
    total = len(rules)
    if pass_count == total:
        console.print(f"  [green]Checklist: {pass_count}/{total} PASS[/green]")
        weights_path = write_weights_csv(weights, output_dir)
        console.print(f"  [cyan]Saved weights:[/cyan] {weights_path}")
        raise SystemExit(0)
    failed = [r for r in rules if not r.get("pass")]
    _render_failure_table(failed)
    console.print(
        f"  [red]Checklist: {pass_count}/{total} — "
        f"{total - pass_count} rule(s) failed[/red]"
    )
    raise SystemExit(1)


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


def load_data(
    *,
    base_currency: str = "EUR",
) -> tuple[DataAssembly, dict[str, str], DatabaseManager]:
    """Assemble all data from the database.

    Returns
    -------
    tuple[DataAssembly, dict[str, str], DatabaseManager]
        (assembly, country_map, db_manager) where ``assembly.prices`` is
        FX-converted to ``base_currency`` (default EUR), ``country_map`` is
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
        base_currency=base_currency,
    )
    _assert_assembly_size(assembly)
    country_map = _build_country_map(db_manager)
    console.print(
        f"  Loaded [cyan]{assembly.n_tickers}[/cyan] tickers, "
        f"[cyan]{assembly.n_trading_days}[/cyan] trading days "
        f"({base_currency} base, hash=[cyan]{assembly.assembly_hash}[/cyan])"
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
            n,
            low,
            high,
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

    # Reshape to MultiIndex (date, ticker) × factors
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
    from app.repositories.macro.macro_regime_repository import MacroRegimeRepository

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
        sector_mapping=assembly.sector_mapping,
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
        cv_config=WalkForwardConfig(  # Cycle-3 §8: 3-year train, quarterly test
            train_size=252 * 3,
            test_size=63,
            expend_train=False,
            purged_size=5,
        ),
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
    result.retighten_trace = retighten_trace
    _hockey_stick_warn(getattr(result, "net_returns", None))
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
    conc_table.add_row("Effective N (1/HHI)", f"{eff_n:.1f}")
    conc_table.add_row("Sectors", str(len(sector_w)))
    conc_table.add_row("Countries", str(len(country_w)))
    console.print(conc_table)


# ---------------------------------------------------------------------------
# Checklist validation
# ---------------------------------------------------------------------------


def _rule(rule: str, *, ok: bool, measured: str | float, target: str) -> dict[str, Any]:
    """Build a single checklist rule result dict."""
    return {"rule": rule, "pass": bool(ok), "measured": measured, "target": target}


def _sector_weights(
    all_weights: list[tuple[str, float]], sector_mapping: dict[str, str]
) -> dict[str, float]:
    from collections import defaultdict

    out: dict[str, float] = defaultdict(float)
    for ticker, w in all_weights:
        out[sector_mapping.get(ticker, "Unknown")] += w
    return dict(out)


def _country_weights(
    all_weights: list[tuple[str, float]], country_map: dict[str, str]
) -> dict[str, float]:
    from collections import defaultdict

    out: dict[str, float] = defaultdict(float)
    for ticker, w in all_weights:
        out[country_map.get(ticker, "Unknown")] += w
    return dict(out)


def _sector_lookup(sector_w: dict[str, float], *names: str) -> float:
    """Sum sector weights across alternative spellings."""
    return sum(sector_w.get(n, 0.0) for n in names)


def _eval_metric_threshold(
    metrics: dict[str, dict[str, float]],
    label: str,
    key: str,
    rule: str,
    target: str,
    *,
    pass_pred: Any,
    fmt: str = "{:.3f}",
) -> dict[str, Any]:
    """Evaluate a metric-bound rule with NaN → pass=False, measured='N/A'."""
    value = metrics.get(label, {}).get(key, float("nan"))
    if isinstance(value, float) and np.isnan(value):
        return _rule(rule, ok=False, measured="N/A", target=target)
    return _rule(rule, ok=pass_pred(value), measured=fmt.format(value), target=target)


def _validate_checklist(
    all_weights: list[tuple[str, float]],
    sector_mapping: dict[str, str],
    country_map: dict[str, str],
    metrics: dict[str, dict[str, float]],
    *,
    benchmark_returns: pd.Series | None,
    net_returns: pd.Series | None,
    after_tax_returns: pd.Series | None,
    cost_bps_actual: float | None,
    currency_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Evaluate the 17 §10 portfolio checklist rules.

    Returns a list of ``{"rule", "pass", "measured", "target"}`` dicts in
    deterministic order.  Rules 12-15 evaluate the after-tax series
    (``metrics["Portfolio (after-tax)"]``).  Currency-hedge advisory is
    logged only — not a rule entry.
    """
    rules: list[dict[str, Any]] = []
    sorted_w = sorted((w for _, w in all_weights), reverse=True)
    sector_w = _sector_weights(all_weights, sector_mapping)
    country_w = _country_weights(all_weights, country_map)
    region_w: dict[str, float] = {}
    for country, w in country_w.items():
        region_w[_REGION_MAP.get(country, "Other")] = (
            region_w.get(_REGION_MAP.get(country, "Other"), 0.0) + w
        )
    label_at = "Portfolio (after-tax)"

    # Rule 1 — region ≤ 60%
    max_region = max(region_w.values()) if region_w else 0.0
    rules.append(
        _rule(
            "No single region > 60%",
            ok=max_region <= 0.60,
            measured=f"{max_region:.1%}",
            target="≤ 60%",
        )
    )
    # Rule 2 — sector ≤ 15%
    max_sector = max(sector_w.values()) if sector_w else 0.0
    max_sector_name = max(sector_w, key=lambda k: sector_w[k]) if sector_w else "N/A"
    rules.append(
        _rule(
            "No single sector > 15%",
            ok=max_sector <= 0.15,
            measured=f"{max_sector:.1%} ({max_sector_name})",
            target="≤ 15%",
        )
    )
    # Rule 3 — HHI < 0.12
    hhi = sum(w**2 for w in sorted_w)
    rules.append(
        _rule("HHI < 0.12", ok=hhi < 0.12, measured=f"{hhi:.4f}", target="< 0.12")
    )
    # Rule 4 — Top-4 < 30%
    top4 = sum(sorted_w[:4])
    rules.append(
        _rule(
            "Top-4 holdings < 30%",
            ok=top4 < 0.30,
            measured=f"{top4:.1%}",
            target="< 30%",
        )
    )
    # Rule 5 — Health Care ≥ 8%
    health_w = _sector_lookup(sector_w, "Health Care", "Healthcare")
    rules.append(
        _rule(
            "Health Care exposure ≥ 8%",
            ok=health_w >= 0.08,
            measured=f"{health_w:.1%}",
            target="≥ 8%",
        )
    )
    # Rule 6 — Information Technology ≥ 10%
    tech_w = _sector_lookup(sector_w, "Information Technology", "Technology")
    rules.append(
        _rule(
            "Information Technology exposure ≥ 10%",
            ok=tech_w >= 0.10,
            measured=f"{tech_w:.1%}",
            target="≥ 10%",
        )
    )
    # Rule 7 — all 11 GICS Level-1 sectors present
    present = {s for s in sector_w if sector_w.get(s, 0.0) > 0.0}
    missing = [s for s in _GICS_SECTORS if s not in present]
    rules.append(
        _rule(
            "All 11 GICS sectors present",
            ok=len(missing) == 0,
            measured=f"{11 - len(missing)}/11 ({', '.join(missing) or 'all'})",
            target="11/11",
        )
    )
    # Rule 8 — Single-stock cap ≤ 10%
    max_w = sorted_w[0] if sorted_w else 0.0
    rules.append(
        _rule(
            "Single-stock cap ≤ 10%",
            ok=max_w <= 0.10,
            measured=f"{max_w:.1%}",
            target="≤ 10%",
        )
    )
    # Rule 9 — Min position ≥ 2%
    min_w = sorted_w[-1] if sorted_w else 0.0
    rules.append(
        _rule(
            "Min position ≥ 2%",
            ok=min_w >= 0.02,
            measured=f"{min_w:.1%}",
            target="≥ 2%",
        )
    )
    # Rule 10 — Max drawdown > -22%
    rules.append(
        _eval_metric_threshold(
            metrics,
            label_at,
            "Max Drawdown",
            "Max drawdown > -22%",
            "> -22%",
            pass_pred=lambda v: v > -0.22,
            fmt="{:.1%}",
        )
    )
    # Rule 11 — Vol ≤ benchmark vol
    p_vol = metrics.get(label_at, {}).get("Ann. Vol", float("nan"))
    b_vol = metrics.get("SPY (benchmark)", {}).get("Ann. Vol", float("nan"))
    if np.isnan(p_vol) or np.isnan(b_vol):
        rules.append(
            _rule(
                "Vol ≤ benchmark vol",
                ok=False,
                measured="N/A",
                target="≤ benchmark",
            )
        )
    else:
        rules.append(
            _rule(
                "Vol ≤ benchmark vol",
                ok=p_vol <= b_vol,
                measured=f"{p_vol:.1%} vs {b_vol:.1%}",
                target="≤ benchmark",
            )
        )
    # Rule 12 — Sharpe ∈ (1.0, 2.0)
    rules.append(
        _eval_metric_threshold(
            metrics,
            label_at,
            "Sharpe (rf)",
            "Sharpe ∈ (1.0, 2.0)",
            "∈ (1.0, 2.0)",
            pass_pred=lambda v: 1.0 < v < 2.0,
        )
    )
    # Rule 13 — Sortino > 1.5
    rules.append(
        _eval_metric_threshold(
            metrics,
            label_at,
            "Sortino",
            "Sortino > 1.5",
            "> 1.5",
            pass_pred=lambda v: v > 1.5,
        )
    )
    # Rule 14 — IR > 0.5
    rules.append(
        _eval_metric_threshold(
            metrics,
            label_at,
            "Info Ratio",
            "Info Ratio > 0.5",
            "> 0.5",
            pass_pred=lambda v: v > 0.5,
        )
    )
    # Rule 15 — Downside vol < 75% × total vol
    if np.isnan(p_vol):
        rules.append(
            _rule(
                "Downside vol < 75% x total vol",
                ok=False,
                measured="N/A",
                target="< 75% total",
            )
        )
    else:
        d_vol = metrics.get(label_at, {}).get("Downside Vol", float("nan"))
        if np.isnan(d_vol):
            rules.append(
                _rule(
                    "Downside vol < 75% x total vol",
                    ok=False,
                    measured="N/A",
                    target="< 75% total",
                )
            )
        else:
            rules.append(
                _rule(
                    "Downside vol < 75% x total vol",
                    ok=d_vol < 0.75 * p_vol,
                    measured=f"{d_vol:.1%} vs 75% x {p_vol:.1%} = {0.75 * p_vol:.1%}",
                    target="< 75% total",
                )
            )
    # Rule 16 — Total cost ≤ 100 bps
    if cost_bps_actual is None or (
        isinstance(cost_bps_actual, float) and np.isnan(cost_bps_actual)
    ):
        rules.append(
            _rule(
                "Total cost ≤ 100 bps",
                ok=False,
                measured="N/A",
                target="≤ 100 bps",
            )
        )
    else:
        rules.append(
            _rule(
                "Total cost ≤ 100 bps",
                ok=cost_bps_actual <= 100.0,
                measured=f"{cost_bps_actual:.1f} bps",
                target="≤ 100 bps",
            )
        )
    # Rule 17 — OOS span ≥ 8 years
    if net_returns is None or net_returns.empty:
        rules.append(
            _rule(
                "OOS span ≥ 8 years",
                ok=False,
                measured="N/A",
                target="≥ 8 yrs",
            )
        )
    else:
        years = (net_returns.index[-1] - net_returns.index[0]).days / 365.25
        rules.append(
            _rule(
                "OOS span ≥ 8 years",
                ok=years >= 8.0,
                measured=f"{years:.2f} yrs",
                target="≥ 8 yrs",
            )
        )

    # Currency-hedge advisory (log-only, no rule entry)
    fx_w: dict[str, float] = {}
    for ticker, w in all_weights:
        ccy = currency_map.get(ticker)
        if ccy and ccy != "EUR":
            fx_w[ccy] = fx_w.get(ccy, 0.0) + w
    for ccy, w in fx_w.items():
        if w > 0.30:
            logger.warning(
                "Currency exposure %s = %.1f%% > 30%% — consider hedging.",
                ccy,
                w * 100.0,
            )

    return rules


def _render_checklist_table(rules: list[dict[str, Any]]) -> None:
    """Render a Rich Table summary of checklist rules to stdout."""
    table = Table(
        title="Portfolio Checklist Validation",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rule", style="dim", width=42)
    table.add_column("Target", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Status", justify="center")
    pass_count = 0
    for r in rules:
        status = "[green]PASS[/green]" if r["pass"] else "[red]FAIL[/red]"
        table.add_row(r["rule"], r["target"], str(r["measured"]), status)
        pass_count += int(bool(r["pass"]))
    console.print(table)
    total = len(rules)
    color = "green" if pass_count == total else "yellow" if pass_count >= 13 else "red"
    console.print(f"  [{color}]Checklist: {pass_count}/{total} passed[/{color}]")


# ---------------------------------------------------------------------------
# Step 8 — Benchmark comparison
# ---------------------------------------------------------------------------


def report_performance(
    result: Any,
    assembly: DataAssembly,
    country_map: dict[str, str],
    cost_bps_actual: float | None = None,
    cost_bps: float = 10.0,
    *,
    tax_rate: float = 0.26,
    validation_report: Any | None = None,
    oos_per_fold_ic: pd.DataFrame | None = None,
    output_dir: Path = Path("research/output"),
) -> tuple[int, list[dict[str, Any]], dict[str, dict[str, float]], list[Path]]:
    """Print portfolio performance, weights, and diversification breakdown.

    Fix issue #246: Sharpe ratio now uses FRED DGS3MO series for rf, not rf=0.
    Cycle 4 §9.1: ``cost_bps_actual`` is the portfolio-weighted measured cost
    in bps (Σᵢ wᵢ × COUNTRY_COSTS_BPS[country]) consumed by §10 checklist.
    Cycle 4 §9.2: derives ``after_tax_returns`` from ``result.gross_returns``
    and ``result.weight_history`` using ``tax_rate`` (Italian default 0.26).
    Cycle 5 §13: emits up to six PNGs when ``validation_report`` and
    ``oos_per_fold_ic`` are supplied (factor IC + country charts).

    Returns ``(pass_count, rules, metrics, chart_paths)`` so :func:`main`
    can apply the §10 terminal gate, render ``report.md``, and persist.
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

    # Cycle 4 §9.1/§9.2 — gross + after-tax return series.
    gross_returns: pd.Series | None = result.gross_returns
    if gross_returns is not None and not gross_returns.empty:
        gross_returns = gross_returns.rename("Portfolio (gross)")
    after_tax_returns = compute_after_tax_returns(
        result.gross_returns,
        result.weight_history,
        assembly.prices,
        cost_bps=cost_bps,
        tax_rate=tax_rate,
    )
    if after_tax_returns is not None:
        after_tax_returns = after_tax_returns.rename("Portfolio (after-tax)")

    # Fetch SPY benchmark aligned to backtest period
    if portfolio_returns is not None and not portfolio_returns.empty:
        console.print("  Downloading SPY benchmark...")
        benchmark_returns = _fetch_benchmark_returns(
            start=portfolio_returns.index[0],
            end=portfolio_returns.index[-1],
        )
        # Align to portfolio trading dates
        if benchmark_returns is not None and not benchmark_returns.empty:
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            benchmark_returns = benchmark_returns.loc[common_idx]

    # Compute metrics for portfolio and benchmark
    metrics: dict[str, dict[str, float]] = {}
    series_pairs: list[tuple[str, pd.Series | None]] = [
        ("Portfolio (gross)", gross_returns),
        ("Portfolio", portfolio_returns),
        ("Portfolio (after-tax)", after_tax_returns),
        ("SPY (benchmark)", benchmark_returns),
    ]
    for label, rets in series_pairs:
        if rets is None or rets.empty:
            continue
        cumulative = (1.0 + rets).cumprod()
        drawdowns: pd.Series = cumulative / cumulative.cummax() - 1.0
        max_dd = float(drawdowns.min()) if len(rets) > 1 else float("nan")
        ir = (
            _information_ratio(rets, benchmark_returns)
            if benchmark_returns is not None and not benchmark_returns.empty
            else float("nan")
        )
        metrics[label] = {
            "Ann. Return": _annualized_return(rets),
            "Ann. Vol": cast(float, rets.std()) * np.sqrt(252.0),
            "Sharpe (rf)": _sharpe(rets, rf_series),
            "Sortino": _sortino(rets, rf_series),
            "Info Ratio": ir,
            "Downside Vol": _downside_vol(rets, rf_series),
            "Max Drawdown": max_dd,
        }

    if not metrics:
        console.print("  [yellow]No backtest results available[/yellow]")
        return 0, [], {}, []

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

    # --- Checklist validation (Cycle 4 §10) ---
    checklist_rules = _validate_checklist(
        all_weights,
        assembly.sector_mapping,
        country_map,
        metrics,
        benchmark_returns=benchmark_returns,
        net_returns=result.net_returns,
        after_tax_returns=after_tax_returns,
        cost_bps_actual=cost_bps_actual,
        currency_map=assembly.currency_map,
    )
    _render_checklist_table(checklist_rules)

    # --- metrics.json (Cycle 4 §9.3 → Cycle 5 report.md input) ---
    metrics_path = write_metrics_json(metrics, output_dir)
    console.print(f"  [cyan]Saved metrics:[/cyan] {metrics_path}")

    # --- checklist.json (Cycle 4 §10 → Cycle 5 report.md input) ---
    checklist_path = write_checklist_json(
        rules=checklist_rules,
        gross_metrics=metrics.get("Portfolio (gross)"),
        net_metrics=metrics.get("Portfolio"),
        after_tax_metrics=metrics.get("Portfolio (after-tax)"),
        output_dir=output_dir,
    )
    console.print(f"  [cyan]Saved checklist:[/cyan] {checklist_path}")

    # --- Backtest charts ---
    chart_paths: list[Path] = []
    if portfolio_returns is not None and not portfolio_returns.empty:
        chart_paths = generate_backtest_plots(
            portfolio_returns=portfolio_returns,
            weight_history=result.weight_history,
            sector_mapping=assembly.sector_mapping,
            benchmark_returns=benchmark_returns,
            rf_series=rf_series,
            country_map=country_map,
            validation_report=validation_report,
            oos_per_fold_ic=oos_per_fold_ic,
            output_dir=output_dir,
        )
        console.print(f"\n  [cyan]Saved {len(chart_paths)} charts:[/cyan]")
        for p in chart_paths:
            console.print(f"    {p}")

    pass_count = sum(1 for r in checklist_rules if r.get("pass"))
    return pass_count, checklist_rules, metrics, chart_paths


# ---------------------------------------------------------------------------
# Cycle 5 §13: report.md + DB persistence helpers
# ---------------------------------------------------------------------------


def _render_research_report(
    *,
    output_dir: Path,
    assembly_hash: str,
    current_regime: Any,
    tilts: dict[str, float],
    validation_report: Any,
    oos_per_fold_ic: pd.DataFrame,
    result: Any,
    country_map: dict[str, str],
    checklist_rules: list[dict[str, Any]],
    metrics: dict[str, dict[str, float]],
    chart_paths: list[Path],
) -> Path:
    """Render ``report.md`` from the run artefacts."""
    # Best-effort optimizer-diff: derive from result if present, else empty.
    optimizer_diff: dict[str, Any] = {}
    opt_cfg = getattr(result, "opt_config", None)
    if opt_cfg is not None:
        optimizer_diff = _diff_from_default(opt_cfg)
    if country_map:
        region_groups, region_rows = build_region_linear_constraints(
            country_map,
            _REGION_MAP,
            max_region_weight=0.60,
        )
    else:
        region_groups, region_rows = {}, []
    binding = _build_binding_constraints(
        weights=result.weights,
        groups=region_groups,
        labels=region_rows,
    )
    return render_report(
        output_dir=output_dir,
        regime=getattr(current_regime, "value", str(current_regime)),
        tilts={str(k): float(v) for k, v in tilts.items()},
        validation_report=validation_report,
        oos_per_fold_ic=oos_per_fold_ic,
        optimizer_diff=optimizer_diff,
        binding_constraints=binding,
        retighten_trace=getattr(result, "retighten_trace", []),
        rebalance_decision=result.rebalance_decision,
        checklist_rows=checklist_rules,
        metrics=metrics,
        chart_paths=chart_paths,
        assembly_hash=assembly_hash,
    )


def _build_binding_constraints(
    *,
    weights: pd.Series,
    groups: dict[str, str],
    labels: list[str],
) -> list[str]:
    """Compute binding region rows for the report.

    Each row of ``A`` sums weights belonging to one region; ``b`` carries the
    region cap parsed from ``labels`` (``"<region> <= <cap>"``).
    """
    if not labels or weights is None or len(weights) == 0:
        return []
    region_order = [row.split(" <= ")[0] for row in labels]
    caps = np.array([float(row.split(" <= ")[1]) for row in labels])
    weight_arr = np.asarray(weights.values, dtype=float)
    tickers = list(weights.index)
    a_mat = np.zeros((len(region_order), len(tickers)), dtype=float)
    for j, ticker in enumerate(tickers):
        region = groups.get(ticker)
        if region in region_order:
            a_mat[region_order.index(region), j] = 1.0
    return compute_binding_constraints(a_mat, caps, weight_arr, labels=labels)


def _persist_research_snapshot(
    *,
    result: Any,
    assembly: DataAssembly,
    metrics: dict[str, dict[str, float]],
    cost_bps: float,
) -> None:
    """Persist the final research run when 17/17 PASS and ``--persist`` set."""
    opt_cfg = getattr(result, "opt_config", None) or _make_opt_config(
        n_survivors=len(result.weights),
        target_count=len(result.weights),
        cost_bps=cost_bps,
    )
    persist_research_run(
        snapshot_date=pd.Timestamp.today().date(),
        weights=result.weights.to_dict(),
        metrics=metrics,
        optimizer_cfg=opt_cfg,
        sector_mapping=assembly.sector_mapping,
        turnover=getattr(result, "turnover", None),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    try:
        # 1. Load
        assembly, country_map, db_manager = load_data(base_currency=base_currency)

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
            np.asarray(prev_weights, dtype=float) if prev_weights is not None else None
        )
        result.rebalance_decision = _decide_rebalance(
            prev_weights=prev_arr,
            target_weights=target_arr,
            current_date=current_date,
            last_review_date=last_review_date,
        )
        # Persist the new review baseline when a rebalance occurred OR on
        # cold-start to seed subsequent runs.  No DB write — Cycle 5 owns that.
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
        missing_sectors = _missing_gics_sectors(result.weights, assembly.sector_mapping)
        if missing_sectors:
            console.print(
                "  [yellow]Sectors absent from selection: "
                f"{', '.join(missing_sectors)}[/yellow]"
            )

        # 7c. Cycle 4 §9.1: portfolio-weighted measured cost in bps for the
        # §10 checklist + metrics output.
        cost_bps_actual = compute_weighted_cost_bps(result.weights, country_map)
        logger.info("Measured cost: %.2f bps (weighted by country)", cost_bps_actual)

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
    from research._cli import build_parser

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
