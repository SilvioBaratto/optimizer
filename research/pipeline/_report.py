"""Step 8 — Performance reporting and research report rendering.

Extracted from ``stock_selection_pipeline.py`` lines 1013–1090, 1446–1745.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from optimizer.optimization import build_region_linear_constraints
from research.data._container import DataAssembly
from research.optimization._config import _REGION_MAP, _make_opt_config
from research.persistence import _diff_from_default, persist_research_run
from research.pipeline._checklist import (
    _render_checklist_table,
    _validate_checklist,
    write_checklist_json,
    write_metrics_json,
)
from research.pipeline._metrics import (
    TOP_N_DISPLAY,
    _annualized_return,
    _downside_vol,
    _fetch_benchmark_returns,
    _information_ratio,
    _sharpe,
    _sortino,
)
from research.reporting._plots import generate_backtest_plots
from research.reporting._report import compute_binding_constraints, render_report
from research.returns._tax import compute_after_tax_returns

console = Console()
logger = logging.getLogger(__name__)


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
# Step 8 — Performance report
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
    sector_bands: dict[str, tuple[float, float]] | None = None,
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
    _print_diversification(all_weights, dict(assembly.sector_mapping), country_map)

    # --- Checklist validation (Cycle 4 §10) ---
    checklist_rules = _validate_checklist(
        all_weights,
        dict(assembly.sector_mapping),
        country_map,
        metrics,
        benchmark_returns=benchmark_returns,
        net_returns=result.net_returns,
        after_tax_returns=after_tax_returns,
        cost_bps_actual=cost_bps_actual,
        currency_map=dict(assembly.currency_map),
        sector_bands=sector_bands,
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
            sector_mapping=dict(assembly.sector_mapping),
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
# Report.md rendering
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


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


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
        sector_mapping=dict(assembly.sector_mapping),
        turnover=getattr(result, "turnover", None),
    )
