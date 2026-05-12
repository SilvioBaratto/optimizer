"""Optimization module — config construction, retighten loop, and rebalancing.

Convention — intentional private exports
-----------------------------------------
Unlike ``research.data``, which exposes only a stable public API (zero
underscore-prefixed symbols), this sub-package intentionally includes
private (underscore-prefixed) symbols in ``__all__``.  The reason is
test-patching: internal constants (``_REGION_MAP``, ``_SECTOR_FLOORS``,
``_TOP4_THRESHOLD``, etc.) and helpers (``_make_builder``, ``_solve_with_retighten``,
etc.) must be reachable for ``unittest.mock.patch`` calls such as::

    patch("research.optimization._config._REGION_MAP", {...})

Removing them from ``__all__`` would break test isolation without providing
any meaningful encapsulation benefit, since this module is not a shipped
public library API.
"""

from research.optimization._config import (
    _MAX_REGION_WEIGHT,
    _REGION_MAP,
    _SECTOR_FLOORS,
    _SOLVER_PARAMS,
    _make_builder,
    _make_opt_config,
    _resolve_min_weights,
    _select_optimizer,
    build_research_optimizer,
)
from research.optimization._rebalance import (
    _HOCKEY_STICK_NEG_THRESHOLD,
    _HOCKEY_STICK_POS_THRESHOLD,
    _decide_rebalance,
    _hockey_stick_warn,
)
from research.optimization._retighten import (
    _SHRINK_FACTOR,
    _TOP4_THRESHOLD,
    _TOP_N,
    _TRADING_DAYS_PER_YEAR,
    _annualized_sharpe,
    _retighten_error_message,
    _solve_with_retighten,
    _split_into_subperiods,
    _top4_weight,
)

__all__ = [
    "_HOCKEY_STICK_NEG_THRESHOLD",
    "_HOCKEY_STICK_POS_THRESHOLD",
    "_MAX_REGION_WEIGHT",
    "_REGION_MAP",
    "_SECTOR_FLOORS",
    "_SHRINK_FACTOR",
    "_SOLVER_PARAMS",
    "_TOP4_THRESHOLD",
    "_TOP_N",
    "_TRADING_DAYS_PER_YEAR",
    "_annualized_sharpe",
    "_decide_rebalance",
    "_hockey_stick_warn",
    "_make_builder",
    "_make_opt_config",
    "_resolve_min_weights",
    "_retighten_error_message",
    "_select_optimizer",
    "_solve_with_retighten",
    "_split_into_subperiods",
    "_top4_weight",
    "build_research_optimizer",
]
