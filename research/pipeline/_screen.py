"""Step 2 — Investability screening.

Extracted from ``stock_selection_pipeline.py`` lines 543–583.
"""

from __future__ import annotations

import logging

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from optimizer.universe._config import InvestabilityScreenConfig
from optimizer.universe._factory import screen_universe
from research.data._container import DataAssembly

console = Console()
logger = logging.getLogger(__name__)

_UNIVERSE_FLOOR: int = 200
_UNIVERSE_BAND: tuple[int, int] = (300, 1500)


def _assert_universe_size(passing: pd.Index) -> None:
    """Enforce universe-size floor and warn on out-of-band counts (issue #520)."""
    n = len(passing)
    if n < _UNIVERSE_FLOOR:
        raise RuntimeError(
            f"Investable universe has only {n} tickers "
            f"(floor: {_UNIVERSE_FLOOR}). Check `InvestabilityScreenConfig` "
            f"thresholds and DB price/volume coverage."
        )
    low, high = _UNIVERSE_BAND
    if not (low <= n <= high):
        logger.warning(
            "Investable universe size %d is outside expected band [%d, %d].",
            n,
            low,
            high,
        )


def screen_investable(assembly: DataAssembly) -> pd.Index:
    """Apply investability screens and return the passing ticker index."""
    console.print(Panel("[bold]Step 2[/bold] — Screening universe", style="blue"))
    config = InvestabilityScreenConfig.for_developed_markets()
    passing = screen_universe(
        fundamentals=assembly.fundamentals,
        price_history=assembly.prices,
        volume_history=assembly.volumes,
        financial_statements=assembly.financial_statements,
        config=config,
    )
    console.print(f"  {len(passing)} tickers pass investability screens")
    _assert_universe_size(passing)
    return passing
