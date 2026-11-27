"""Deterministic ``@tool`` backbone: pure DB/optimizer wrappers
that return the ``{ok,data}`` / ``{ok:false,error}`` envelope and never raise.

``optimize_portfolio`` stays in :mod:`fund.tools.optimize` for the readouts that
replay it, but no agent binds it: a run's weights come from ``compute_candidates``.
"""

from __future__ import annotations

from fund.tools.candidates import compute_candidates
from fund.tools.events import get_recent_events
from fund.tools.history import get_portfolio_history
from fund.tools.macro import get_macro_series
from fund.tools.moments import estimate_moments
from fund.tools.orders import place_orders
from fund.tools.prices import get_prices
from fund.tools.regime import get_regime_stats
from fund.tools.risk import backtest, risk_check
from fund.tools.universe import list_universe, universe_filter

__all__ = [
    "backtest",
    "compute_candidates",
    "estimate_moments",
    "get_macro_series",
    "get_portfolio_history",
    "get_prices",
    "get_recent_events",
    "get_regime_stats",
    "list_universe",
    "place_orders",
    "risk_check",
    "universe_filter",
]
