"""Deterministic ``@tool`` backbone: pure DB/optimizer wrappers
that return the ``{ok,data}`` / ``{ok:false,error}`` envelope and never raise."""

from __future__ import annotations

from fund.tools.macro import get_macro_series
from fund.tools.moments import estimate_moments
from fund.tools.optimize import optimize_portfolio
from fund.tools.orders import place_orders
from fund.tools.prices import get_prices
from fund.tools.risk import backtest, risk_check
from fund.tools.universe import universe_filter

# NOTE: fund.tools.events.get_recent_events and fund.tools.history.
# get_portfolio_history exist (Phase-4 T8) but are intentionally NOT exported here
# yet. `TOOLS_BY_AGENT` is a strict bijection over __all__ (test_toolsets), so a
# tool enters __all__ only together with its role binding — both land in T9.

__all__ = [
    "backtest",
    "estimate_moments",
    "get_macro_series",
    "get_prices",
    "optimize_portfolio",
    "place_orders",
    "risk_check",
    "universe_filter",
]
