"""Reporting modules — markdown rendering, chart generation, and CLI display."""

from research.reporting._display import (
    dict_table,
    error_panel,
    info_panel,
    list_table,
    progress_loop,
    success_panel,
    warning_panel,
)
from research.reporting._plots import (
    generate_backtest_plots,
    plot_country_allocation,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_factor_ic,
    plot_rolling_sharpe,
    plot_sector_weights_over_time,
)
from research.reporting._report import compute_binding_constraints, render_report

__all__ = [
    "compute_binding_constraints",
    "dict_table",
    "error_panel",
    "generate_backtest_plots",
    "info_panel",
    "list_table",
    "plot_country_allocation",
    "plot_cumulative_returns",
    "plot_drawdowns",
    "plot_factor_ic",
    "plot_rolling_sharpe",
    "plot_sector_weights_over_time",
    "progress_loop",
    "render_report",
    "success_panel",
    "warning_panel",
]
