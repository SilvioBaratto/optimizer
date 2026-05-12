"""Backtest visualisation for the stock selection pipeline.

Generates up to six PNG charts from a walk-forward backtest result:

  1. Cumulative returns (portfolio vs benchmark)
  2. Underwater / drawdown chart
  3. Rolling 12-month Sharpe ratio
  4. Sector allocation over time (stacked area)
  5. Country allocation over time (stacked area)
  6. Per-factor IS / OOS mean IC bars

All charts use matplotlib (not plotly) for static file output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from optimizer.factors import FactorValidationReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("research/output")
ROLLING_WINDOW = 252  # ~12 months for rolling metrics
FIG_WIDTH = 12
FIG_HEIGHT = 5
DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Cumulative return chart: portfolio vs benchmark."""
    out = _ensure_output_dir(output_dir)

    cum_port = (1.0 + portfolio_returns).cumprod()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(cum_port.index, cum_port.values, label="Portfolio", linewidth=1.5)

    if benchmark_returns is not None and not benchmark_returns.empty:
        cum_bench = (1.0 + benchmark_returns).cumprod()
        ax.plot(
            cum_bench.index,
            cum_bench.values,
            label="Benchmark",
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
        )

    ax.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(cum_port.index[0], cum_port.index[-1])

    path = out / "cumulative_returns.png"
    _save(fig, path)
    return path


def plot_drawdowns(
    portfolio_returns: pd.Series,
    *,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Underwater / drawdown chart."""
    out = _ensure_output_dir(output_dir)

    cum = (1.0 + portfolio_returns).cumprod()
    drawdowns = cum / cum.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.fill_between(
        drawdowns.index,
        drawdowns.values,
        0,
        color="crimson",
        alpha=0.4,
        label="Drawdown",
    )
    ax.plot(drawdowns.index, drawdowns.values, color="crimson", linewidth=0.8)

    ax.set_title("Underwater Chart", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(drawdowns.index[0], drawdowns.index[-1])

    path = out / "drawdowns.png"
    _save(fig, path)
    return path


def plot_rolling_sharpe(
    portfolio_returns: pd.Series,
    *,
    rf_series: pd.Series | None = None,
    window: int = ROLLING_WINDOW,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Rolling annualized Sharpe ratio."""
    out = _ensure_output_dir(output_dir)

    if rf_series is not None and not rf_series.empty:
        daily_rf = (
            rf_series.reindex(portfolio_returns.index, method="ffill").fillna(0.0)
            / 252.0
        )
        excess = portfolio_returns - daily_rf
    else:
        excess = portfolio_returns

    rolling_mean = excess.rolling(window).mean() * 252.0
    rolling_std = excess.rolling(window).std(ddof=1) * np.sqrt(252.0)
    rolling_sharpe = rolling_mean / rolling_std

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.2, color="teal")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1.0")
    ax.axhline(y=0.0, color="gray", linestyle="-", alpha=0.3)

    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    valid = rolling_sharpe.dropna()
    if not valid.empty:
        ax.set_xlim(valid.index[0], valid.index[-1])

    path = out / "rolling_sharpe.png"
    _save(fig, path)
    return path


def plot_sector_weights_over_time(
    weight_history: pd.DataFrame | None,
    sector_mapping: dict[str, str],
    *,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Stacked area chart of sector allocation over rebalancing dates."""
    if weight_history is None or weight_history.empty:
        return None

    out = _ensure_output_dir(output_dir)

    # Aggregate weights by sector per rebalancing date
    rows: list[dict[str, float]] = []
    for _, row in weight_history.iterrows():
        sector_totals: dict[str, float] = {}
        for ticker, w in row.items():
            if w > 0:
                sector = sector_mapping.get(str(ticker), "Unknown")
                sector_totals[sector] = sector_totals.get(sector, 0.0) + w
        rows.append(sector_totals)

    sector_df = pd.DataFrame(rows, index=weight_history.index).fillna(0.0)
    # Sort by average weight
    order = sector_df.mean().sort_values(ascending=False).index
    sector_df = sector_df[order]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.stackplot(
        sector_df.index,
        [sector_df[col].values for col in sector_df.columns],
        labels=sector_df.columns,
        alpha=0.8,
    )
    ax.set_title("Sector Allocation Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(sector_df.index[0], sector_df.index[-1])
    ax.set_ylim(0, 1)

    path = out / "sector_allocation.png"
    _save(fig, path)
    return path


def plot_country_allocation(
    weight_history: pd.DataFrame | None,
    country_map: dict[str, str],
    *,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Stacked area chart of country allocation over rebalancing dates."""
    if weight_history is None or weight_history.empty:
        return None

    out = _ensure_output_dir(output_dir)
    country_df = _aggregate_by_group(weight_history, country_map)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.stackplot(
        country_df.index,
        [country_df[col].values for col in country_df.columns],
        labels=country_df.columns,
        alpha=0.8,
    )
    ax.set_title("Country Allocation Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(country_df.index[0], country_df.index[-1])
    ax.set_ylim(0, 1)

    path = out / "country_allocation.png"
    _save(fig, path)
    return path


def plot_factor_ic(
    validation_report: FactorValidationReport,
    oos_per_fold_ic: pd.DataFrame,
    *,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Grouped bar chart: per-factor IS mean IC vs OOS mean IC, sorted by OOS."""
    out = _ensure_output_dir(output_dir)

    is_ic = pd.Series({r.factor_name: r.mean_ic for r in validation_report.ic_results})
    oos_ic = oos_per_fold_ic.mean(axis=0)
    factors = sorted(set(is_ic.index) & set(oos_ic.index), key=lambda f: -oos_ic[f])
    is_vals = [is_ic[f] for f in factors]
    oos_vals = [oos_ic[f] for f in factors]

    x = np.arange(len(factors))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.bar(
        x - bar_width / 2,
        is_vals,
        bar_width,
        label="IS Mean IC",
        color="steelblue",
    )
    ax.bar(
        x + bar_width / 2,
        oos_vals,
        bar_width,
        label="OOS Mean IC",
        color="darkorange",
    )
    ax.axhline(y=0.0, color="gray", linestyle="-", alpha=0.4)

    ax.set_title("Factor Information Coefficient", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean IC")
    ax.set_xticks(x)
    ax.set_xticklabels(factors, rotation=30, ha="right")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    path = out / "factor_ic.png"
    _save(fig, path)
    return path


def _aggregate_by_group(
    weight_history: pd.DataFrame,
    group_map: dict[str, str],
) -> pd.DataFrame:
    """Sum positive weights by group label per row, sorted by mean weight desc."""
    rows: list[dict[str, float]] = []
    for _, row in weight_history.iterrows():
        totals: dict[str, float] = {}
        for ticker, w in row.items():
            if w > 0:
                group = group_map.get(str(ticker), "Unknown")
                totals[group] = totals.get(group, 0.0) + w
        rows.append(totals)
    df = pd.DataFrame(rows, index=weight_history.index).fillna(0.0)
    return df[df.mean().sort_values(ascending=False).index]


# ---------------------------------------------------------------------------
# Public API — generate all charts
# ---------------------------------------------------------------------------


def generate_backtest_plots(
    portfolio_returns: pd.Series,
    weight_history: pd.DataFrame | None,
    sector_mapping: dict[str, str],
    *,
    benchmark_returns: pd.Series | None = None,
    rf_series: pd.Series | None = None,
    country_map: dict[str, str] | None = None,
    validation_report: FactorValidationReport | None = None,
    oos_per_fold_ic: pd.DataFrame | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """Generate all backtest charts and return the list of saved file paths."""
    paths: list[Path] = [
        plot_cumulative_returns(
            portfolio_returns,
            benchmark_returns=benchmark_returns,
            output_dir=output_dir,
        ),
        plot_drawdowns(portfolio_returns, output_dir=output_dir),
        plot_rolling_sharpe(
            portfolio_returns,
            rf_series=rf_series,
            output_dir=output_dir,
        ),
    ]

    sector_path = plot_sector_weights_over_time(
        weight_history, sector_mapping, output_dir=output_dir
    )
    if sector_path is not None:
        paths.append(sector_path)

    if country_map is not None:
        country_path = plot_country_allocation(
            weight_history, country_map, output_dir=output_dir
        )
        if country_path is not None:
            paths.append(country_path)

    if validation_report is not None and oos_per_fold_ic is not None:
        paths.append(
            plot_factor_ic(
                validation_report,
                oos_per_fold_ic,
                output_dir=output_dir,
            )
        )

    return paths
