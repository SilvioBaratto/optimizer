"""Pipeline step modules — one module per step.

Re-exports all public symbols from the 8 step modules so callers can
``from research.pipeline import load_data, optimize_portfolio, ...``.
"""

from optimizer.factors._regime import classify_regime
from optimizer.pipeline import run_full_pipeline_with_selection
from optimizer.universe import screen_universe
from research.data._orchestrator import assemble_all
from research.optimization._config import _REGION_MAP
from research.optimization._rebalance import _decide_rebalance
from research.optimization._retighten import _solve_with_retighten
from research.pipeline._checklist import (
    _CHECKLIST_TOTAL,
    _apply_terminal_gate,
    _country_weights,
    _eval_metric_threshold,
    _project_rule_for_json,
    _render_checklist_table,
    _render_failure_table,
    _rule,
    _sector_lookup,
    _sector_weights,
    _validate_checklist,
    write_checklist_json,
    write_metrics_json,
    write_weights_csv,
)
from research.pipeline._factors import (
    _GICS_SECTORS,
    _IS_VALIDATION_CONFIG,
    MIN_SUCCESS_FRACTION,
    N_SELECTED_MAX,
    N_SELECTED_MIN,
    OOS_CONFIG,
    REBALANCE_FREQ,
    _check_factor_coverage,
    _missing_gics_sectors,
    _validate_n_selected,
    build_history,
    validate_is,
    validate_oos,
)
from research.pipeline._load import (
    _LAST_REVIEW_DATE_FILE,
    N_SELECTED,
    _assert_assembly_size,
    _materialise_clean_returns,
    _read_last_review_date,
    _write_last_review_date,
    load_data,
)
from research.pipeline._metrics import (
    _METRICS_KEY_MAP,
    TOP_N_DISPLAY,
    _annualized_return,
    _build_country_map,
    _daily_rf,
    _downside_vol,
    _fetch_benchmark_returns,
    _information_ratio,
    _project_metrics,
    _sharpe,
    _sortino,
    _to_json_safe,
)
from research.pipeline._optimize import (
    _DEFAULT_COSTS,
    COUNTRY_COSTS_BPS,
    compute_weighted_cost_bps,
    optimize_portfolio,
)
from research.pipeline._regime import (
    _cache_regime_classification,
    classify_and_tilt,
)
from research.pipeline._report import (
    _build_binding_constraints,
    _persist_research_snapshot,
    _print_diversification,
    _render_research_report,
    report_performance,
)
from research.pipeline._screen import (
    _UNIVERSE_BAND,
    _UNIVERSE_FLOOR,
    _assert_universe_size,
    screen_investable,
)

__all__ = [  # noqa: RUF022
    # _checklist.py
    "_CHECKLIST_TOTAL",
    "_apply_terminal_gate",
    "_country_weights",
    "_eval_metric_threshold",
    "_project_rule_for_json",
    "_render_checklist_table",
    "_render_failure_table",
    "_rule",
    "_sector_lookup",
    "_sector_weights",
    "_validate_checklist",
    "write_checklist_json",
    "write_metrics_json",
    "write_weights_csv",
    # _factors.py
    "_GICS_SECTORS",
    "_IS_VALIDATION_CONFIG",
    "_check_factor_coverage",
    "_missing_gics_sectors",
    "_validate_n_selected",
    "MIN_SUCCESS_FRACTION",
    "N_SELECTED_MAX",
    "N_SELECTED_MIN",
    "OOS_CONFIG",
    "REBALANCE_FREQ",
    "build_history",
    "validate_is",
    "validate_oos",
    # _load.py
    "N_SELECTED",
    "_LAST_REVIEW_DATE_FILE",
    "_assert_assembly_size",
    "_materialise_clean_returns",
    "_read_last_review_date",
    "_write_last_review_date",
    "load_data",
    # _metrics.py
    "TOP_N_DISPLAY",
    "_METRICS_KEY_MAP",
    "_annualized_return",
    "_build_country_map",
    "_daily_rf",
    "_downside_vol",
    "_fetch_benchmark_returns",
    "_information_ratio",
    "_project_metrics",
    "_sharpe",
    "_sortino",
    "_to_json_safe",
    # _optimize.py
    "COUNTRY_COSTS_BPS",
    "_DEFAULT_COSTS",
    "_decide_rebalance",
    "_solve_with_retighten",
    "compute_weighted_cost_bps",
    "optimize_portfolio",
    "run_full_pipeline_with_selection",
    # _regime.py
    "_cache_regime_classification",
    "classify_and_tilt",
    "classify_regime",
    # _report.py
    "_REGION_MAP",
    "_build_binding_constraints",
    "_persist_research_snapshot",
    "_print_diversification",
    "_render_research_report",
    "report_performance",
    # _screen.py
    "_UNIVERSE_BAND",
    "_UNIVERSE_FLOOR",
    "_assert_universe_size",
    "screen_investable",
    # compat shims (still referenced by callers)
    "assemble_all",
    "screen_universe",
]
