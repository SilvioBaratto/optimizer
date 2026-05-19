"""Shared test fixtures for the optimizer test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add api directory to path so tests can import from app
api_dir = Path(__file__).parent.parent / "api"
if str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic returns: 20 assets, 200 obs, seed 42."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.001, scale=0.02, size=(200, 20))
    tickers = [f"TICK_{i:02d}" for i in range(20)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.bdate_range("2023-01-01", periods=200, freq="B"),
    )


@pytest.fixture()
def prices_df(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic prices derived from returns_df."""
    return (1 + returns_df).cumprod() * 100


@pytest.fixture()
def sector_mapping(returns_df: pd.DataFrame) -> dict[str, str]:
    """Sector mapping for returns_df tickers."""
    sectors = ["Technology", "Financials", "Healthcare", "Energy"]
    return {col: sectors[i % len(sectors)] for i, col in enumerate(returns_df.columns)}


@pytest.fixture()
def benchmark_returns(returns_df: pd.DataFrame) -> pd.Series:
    """Equal-weight benchmark from returns_df."""
    return returns_df.mean(axis=1).rename("benchmark")
