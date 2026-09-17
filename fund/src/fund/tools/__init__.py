"""Deterministic ``@tool`` backbone (SPEC Fase 3): pure DB/optimizer wrappers
that return the ``{ok,data}`` / ``{ok:false,error}`` envelope and never raise."""

from __future__ import annotations

from fund.tools.macro import get_macro_series
from fund.tools.moments import estimate_moments
from fund.tools.optimize import optimize_portfolio
from fund.tools.orders import place_orders
from fund.tools.prices import get_prices
from fund.tools.risk import backtest, risk_check
from fund.tools.universe import universe_filter

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
