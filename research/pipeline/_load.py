"""Step 1 — Data loading from the database.

Extracted from ``stock_selection_pipeline.py`` lines 46–540.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from research.data._container import DataAssembly
from research.data._orchestrator import assemble_all
from research.pipeline._metrics import (
    _build_country_map,
)
from research.preflight._checks import (
    run_db_preflight as _run_db_preflight,
)
from research.returns._preprocessing import (
    apply_fx_to_prices,
    build_return_preprocessing_pipeline,
)

console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via command-line args or environment)
# ---------------------------------------------------------------------------

REBALANCE_FREQ: int = 63
N_SELECTED: int = 20
MIN_SUCCESS_FRACTION: float = 0.5

# Cycle-3 §11: hybrid rebalance review-date persistence
_LAST_REVIEW_DATE_FILE = (
    Path(__file__).resolve().parent.parent / "output" / "last_review_date.txt"
)

_MIN_ASSEMBLY_TICKERS: int = 2_000
_MIN_ASSEMBLY_TRADING_DAYS: int = 1_260


# ---------------------------------------------------------------------------
# Review-date persistence
# ---------------------------------------------------------------------------


def _read_last_review_date(today: pd.Timestamp) -> pd.Timestamp:
    """Read ``last_review_date`` from disk, falling back to the prior quarter-end."""
    if _LAST_REVIEW_DATE_FILE.exists():
        text = _LAST_REVIEW_DATE_FILE.read_text().strip()
        if text:
            return pd.Timestamp(text).normalize()
    return (today.to_period("Q") - 1).end_time.normalize()


def _write_last_review_date(date: pd.Timestamp) -> None:
    """Persist ``date`` (ISO ``YYYY-MM-DD``) for the next pipeline run."""
    _LAST_REVIEW_DATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LAST_REVIEW_DATE_FILE.write_text(date.date().isoformat() + "\n")


# ---------------------------------------------------------------------------
# Assembly validation
# ---------------------------------------------------------------------------


def _assert_assembly_size(assembly: DataAssembly) -> None:
    """Enforce minimum tickers and trading-day count post-assembly (issue #522)."""
    if assembly.n_tickers < _MIN_ASSEMBLY_TICKERS:
        raise RuntimeError(
            f"DataAssembly: n_tickers={assembly.n_tickers} below floor "
            f"{_MIN_ASSEMBLY_TICKERS}."
        )
    if assembly.n_trading_days < _MIN_ASSEMBLY_TRADING_DAYS:
        raise RuntimeError(
            f"DataAssembly: n_trading_days={assembly.n_trading_days} below floor "
            f"{_MIN_ASSEMBLY_TRADING_DAYS}."
        )


def _materialise_clean_returns(
    assembly: DataAssembly,
    investable: pd.Index,
) -> pd.DataFrame:
    """Slice EUR prices to investable, convert to linear returns, preprocess.

    Cycle 1 hand-off (§3): the cleaned linear-return panel is the single
    artefact downstream cycles consume.
    """
    from skfolio.preprocessing import prices_to_returns

    cols = list(investable)
    investable_prices = assembly.prices.loc[:, cols]
    returns = cast(pd.DataFrame, prices_to_returns(investable_prices))
    pipeline = build_return_preprocessing_pipeline(dict(assembly.sector_mapping))
    clean: pd.DataFrame = pipeline.fit_transform(returns)
    arr = clean.to_numpy()
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise RuntimeError("clean_returns contains NaN or inf after preprocessing.")
    console.print(
        f"  Clean returns: [cyan]{len(clean)}[/cyan] days x "
        f"[cyan]{clean.shape[1]}[/cyan] tickers, "
        f"range [cyan]{clean.index[0].date()}[/cyan] -> "
        f"[cyan]{clean.index[-1].date()}[/cyan]"
    )
    return clean


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------


def load_data(
    db_manager: Any,
    *,
    base_currency: str = "EUR",
) -> tuple[DataAssembly, dict[str, str], Any]:
    """Assemble all data from the database.

    Args:
        db_manager: Initialised ``DatabaseManager`` instance (injected by caller
            to avoid api-layer import in pipeline module — issue #653).

    Returns
    -------
    tuple[DataAssembly, dict[str, str], Any]
        (assembly, country_map, db_manager) where ``assembly.prices`` is
        FX-converted to ``base_currency`` (default EUR), ``country_map`` is
        ticker → country, and ``db_manager`` is the initialised handle so
        downstream steps can persist results back to the DB (issue #530).
    """
    console.print(Panel("[bold]Step 1[/bold] — Loading data", style="blue"))
    _run_db_preflight(db_manager)
    assembly = assemble_all(db_manager, include_delisted=True)
    assembly = dataclasses.replace(
        assembly,
        prices=apply_fx_to_prices(
            assembly.prices,
            dict(assembly.currency_map),
            assembly.fx_rates,
            base_currency=base_currency,
        ),
    )
    _assert_assembly_size(assembly)
    country_map = _build_country_map(db_manager)
    console.print(
        f"  Loaded [cyan]{assembly.n_tickers}[/cyan] tickers, "
        f"[cyan]{assembly.n_trading_days}[/cyan] trading days "
        f"({base_currency} base, hash=[cyan]{assembly.assembly_hash}[/cyan])"
    )
    return assembly, country_map, db_manager
