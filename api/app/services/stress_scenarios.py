"""LLM-driven stress scenario design for forward-looking tail risk events.

Architecture:
  1. Call the BAML ``DesignStressScenarios`` function with portfolio tickers
     and a macro context string supplied by the caller.
  2. Validate and clamp scenario fields to acceptance criteria.
  3. Convert each scenario to ``sample_args`` format accepted by
     ``build_synthetic_data(conditioning={...})``.

Usage::

    from app.services.stress_scenarios import (
        generate_stress_scenarios,
        scenario_to_synthetic_data_args,
    )

    scenarios = generate_stress_scenarios(
        n_scenarios=3,
        current_portfolio={"SPY": 0.4, "QQQ": 0.3, "GLD": 0.2, "TLT": 0.1},
        macro_context="US CPI at 4.2%, Fed on hold ...",
    )
    for s in scenarios:
        args = scenario_to_synthetic_data_args(s)
        # args == {"conditioning": {"SPY": -0.15, "GLD": 0.08}}
"""

from __future__ import annotations

import logging

from baml_client import b
from baml_client.types import StressScenario

logger = logging.getLogger(__name__)

# Hard bounds enforced after LLM generation
_PROB_MIN: float = 1e-4
_PROB_MAX: float = 1.0 - 1e-4
_SHOCK_MIN: float = -0.9999
_SHOCK_MAX: float = 0.9999


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _clamp_scenario(scenario: StressScenario) -> StressScenario:
    """Return a copy of *scenario* with all fields clamped to valid ranges."""
    clamped_prob = max(_PROB_MIN, min(_PROB_MAX, scenario.probability))
    clamped_shocks = {
        ticker: max(_SHOCK_MIN, min(_SHOCK_MAX, shock))
        for ticker, shock in scenario.shocks.items()
    }
    return StressScenario(
        name=scenario.name,
        description=scenario.description,
        shocks=clamped_shocks,
        probability=clamped_prob,
        horizon_days=max(1, scenario.horizon_days),
    )


def _ensure_market_drawdown(scenarios: list[StressScenario]) -> list[StressScenario]:
    """Verify at least one scenario has a broad market drawdown (>10% on average).

    If none qualifies, the first scenario is patched to enforce the criterion.
    This preserves the LLM narrative but adjusts equity shocks downward.
    """
    for s in scenarios:
        equity_shocks = [v for v in s.shocks.values() if v < 0]
        if equity_shocks and abs(sum(equity_shocks) / len(equity_shocks)) >= 0.10:
            return scenarios

    # No drawdown scenario found — patch the first one
    if not scenarios:
        return scenarios

    first = scenarios[0]
    patched_shocks = {
        ticker: min(shock, -0.10) for ticker, shock in first.shocks.items()
    }
    patched = StressScenario(
        name=first.name,
        description=first.description,
        shocks=patched_shocks,
        probability=first.probability,
        horizon_days=first.horizon_days,
    )
    return [patched] + scenarios[1:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_stress_scenarios(
    n_scenarios: int,
    current_portfolio: dict[str, float],
    macro_context: str,
) -> list[StressScenario]:
    """Generate forward-looking stress scenarios via LLM.

    Calls the BAML ``DesignStressScenarios`` function, then validates and
    clamps each scenario to the acceptance criteria.

    Args:
        n_scenarios: Number of distinct scenarios to request from the LLM.
            Must be ≥ 1.
        current_portfolio: Dict mapping asset ticker to portfolio weight.
            Tickers are passed to the LLM to ensure all shocks are populated.
        macro_context: Free-text description of current market conditions
            (inflation, rates, geopolitics, …) used as LLM context.

    Returns:
        List of validated :class:`StressScenario` objects.  May be shorter
        than *n_scenarios* if the LLM returns fewer.

    Raises:
        ValueError: If *n_scenarios* < 1 or *current_portfolio* is empty.
        RuntimeError: If the BAML call fails.
    """
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}")
    if not current_portfolio:
        raise ValueError("current_portfolio must not be empty")

    tickers = list(current_portfolio.keys())

    try:
        raw_scenarios = b.DesignStressScenarios(
            tickers=tickers,
            macro_context=macro_context,
            n_scenarios=n_scenarios,
        )
    except Exception as exc:
        logger.error("BAML DesignStressScenarios failed: %s", exc)
        raise RuntimeError(f"LLM stress scenario generation failed: {exc}") from exc

    validated = [_clamp_scenario(s) for s in raw_scenarios]
    validated = _ensure_market_drawdown(validated)

    logger.info(
        "Generated %d stress scenario(s) for portfolio of %d assets",
        len(validated),
        len(tickers),
    )
    return validated


def scenario_to_synthetic_data_args(scenario: StressScenario) -> dict[str, object]:
    """Convert a StressScenario to sample_args for build_synthetic_data().

    The returned dict can be passed directly as the ``sample_args`` kwarg
    to ``build_synthetic_data()``, which expects::

        {"conditioning": {ticker: return_value, ...}}

    Args:
        scenario: A validated stress scenario.

    Returns:
        Dict with key ``"conditioning"`` mapping tickers to shock values.
    """
    return {"conditioning": dict(scenario.shocks)}
