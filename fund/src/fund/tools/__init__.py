"""Deterministic ``@tool`` backbone (SPEC Fase 3): pure DB/optimizer wrappers
that return the ``{ok,data}`` / ``{ok:false,error}`` envelope and never raise."""

from __future__ import annotations

from fund.tools.moments import estimate_moments
from fund.tools.optimize import optimize_portfolio
from fund.tools.prices import get_prices
from fund.tools.universe import universe_filter

__all__ = ["estimate_moments", "get_prices", "optimize_portfolio", "universe_filter"]
