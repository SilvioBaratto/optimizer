"""Step 6 — Macro regime classification.

Extracted from ``stock_selection_pipeline.py`` lines 827–889.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel

from optimizer.factors._config import (
    FactorGroupType,
    MacroRegime,
    RegimeTiltConfig,
)
from optimizer.factors._regime import apply_regime_tilts, classify_regime
from research.data._regime import assemble_regime_data

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 6 — Macro regime classification
# ---------------------------------------------------------------------------


def classify_and_tilt(
    assembly: Any,
    db_manager: Any = None,
) -> tuple[MacroRegime, dict[FactorGroupType, float]]:
    """Classify macro regime and compute regime-conditional group tilts.

    Fix issue #237: the previous code called classify_regime(assembly.macro_data)
    which only contains gdp_growth + yield_spread, making the composite path
    (requiring pmi, spread_2s10s, hy_oas) unreachable.

    Issue #529: this function is **logging-only** for tilt application.  The
    orchestrator inside ``run_full_pipeline_with_selection`` owns the actual
    classification + tilt application against publication-lag-shifted macro
    data; the regime printed here is the unlagged snapshot for inspection.

    Issue #530: when ``db_manager`` is supplied, the rule-based regime is
    cached into ``macro_calibrations.regime_classification`` for the ``US``
    country row so downstream services can read it without re-classifying.
    """
    console.print(Panel("[bold]Step 6[/bold] — Regime classification", style="blue"))

    regime_data = assemble_regime_data(
        macro_data=assembly.macro_data,
        fred_data=assembly.fred_data,
        te_observations=assembly.te_observations,
        sentiment_data=assembly.sentiment_data,
    )

    regime = classify_regime(regime_data)
    tilt_config = RegimeTiltConfig(enable=True)
    tilts = apply_regime_tilts(
        group_weights=dict.fromkeys(FactorGroupType, 1.0),
        regime=regime,
        config=tilt_config,
    )

    regime_colors = {
        MacroRegime.EXPANSION: "[green]",
        MacroRegime.SLOWDOWN: "[yellow]",
        MacroRegime.RECESSION: "[red]",
        MacroRegime.RECOVERY: "[cyan]",
    }
    color = regime_colors.get(regime, "")
    console.print(f"  Macro regime: {color}{regime.value.upper()}[/]")

    if db_manager is not None:
        _cache_regime_classification(db_manager, regime)

    return regime, tilts


def _cache_regime_classification(
    db_manager: Any, regime: MacroRegime
) -> None:
    """Persist the rule-based regime to ``macro_calibrations`` for country='US'."""
    from app.repositories.macro.macro_regime_repository import MacroRegimeRepository

    with db_manager.get_session() as session:
        MacroRegimeRepository(session).upsert_regime_classification(
            country="US", regime=regime.value
        )
        session.commit()
