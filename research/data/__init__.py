"""Data assembly layer — DB → DataFrames glue for the optimizer pipeline."""

from research.data._container import DataAssembly
from research.data._currency import (
    CURRENCY_DEDUP_PRIORITY,
    MINOR_CURRENCY_DIVISORS,
    build_currency_map,
    currency_dedup_rank,
    normalize_fundamentals,
    normalize_prices,
)
from research.data._helpers import REGION_MAP
from research.data._macro import FRED_SERIES_IDS
from research.data._orchestrator import assemble_all

__all__ = [
    "CURRENCY_DEDUP_PRIORITY",
    "FRED_SERIES_IDS",
    "MINOR_CURRENCY_DIVISORS",
    "REGION_MAP",
    "DataAssembly",
    "assemble_all",
    "build_currency_map",
    "currency_dedup_rank",
    "normalize_fundamentals",
    "normalize_prices",
]
