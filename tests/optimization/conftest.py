"""Shared fixtures for optimization tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.moments._hmm import HMMResult

# Module-scoped constants for the 10-asset fixture
_N_ASSETS = 10
_N_OBS = 252
_TICKERS = [f"A{i:02d}" for i in range(_N_ASSETS)]
_DATES = pd.date_range("2020-01-02", periods=_N_OBS, freq="B")


@pytest.fixture(scope="module")
def returns_10a_252() -> pd.DataFrame:
    """Module-scoped: 10 assets, 252 obs, seed 42."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0005, scale=0.01, size=(_N_OBS, _N_ASSETS))
    return pd.DataFrame(data, index=_DATES, columns=_TICKERS)


def make_hmm_result(
    n_assets: int,
    n_obs: int,
    tickers: list[str],
    n_states: int = 2,
    last_probs: list[float] | None = None,
    returns: pd.DataFrame | None = None,
) -> HMMResult:
    """Build a synthetic HMMResult without fitting.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    n_obs : int
        Number of observations.
    tickers : list[str]
        Asset ticker names.
    n_states : int
        Number of HMM states.
    last_probs : list[float] or None
        Override the last row of filtered probabilities.
    returns : pd.DataFrame or None
        If provided, use its index for the filtered_probs DatetimeIndex.
    """
    dates = (
        returns.index
        if returns is not None
        else pd.date_range("2020-01-01", periods=n_obs, freq="B")
    )

    rng = np.random.default_rng(42)
    raw = rng.random((n_obs, n_states))
    filtered = raw / raw.sum(axis=1, keepdims=True)

    if last_probs is not None:
        filtered[-1] = last_probs

    filtered_df = pd.DataFrame(filtered, index=dates, columns=list(range(n_states)))
    transition = np.full((n_states, n_states), 1.0 / n_states)
    means = pd.DataFrame(np.zeros((n_states, n_assets)), columns=tickers)
    covs = np.stack([np.eye(n_assets) * 1e-4] * n_states)

    return HMMResult(
        transition_matrix=transition,
        regime_means=means,
        regime_covariances=covs,
        filtered_probs=filtered_df,
        smoothed_probs=filtered_df,
        log_likelihood=-100.0,
    )
