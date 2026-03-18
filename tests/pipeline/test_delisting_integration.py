"""Tests for survivorship-bias fix in run_full_pipeline (issue #274)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import EqualWeighted

from optimizer.pipeline import run_full_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(
    n_active: int = 5,
    n_delisted: int = 2,
    n_days: int = 200,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build a price DataFrame with active and delisted tickers.

    Delisted tickers have trailing NaN after 75% of the series.
    Returns (prices_df, delisting_returns_dict).
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    active = [f"LIVE_{i:02d}" for i in range(n_active)]
    dead = [f"DEAD_{i:02d}" for i in range(n_delisted)]

    # Active: full price history
    active_rets = rng.normal(0.0004, 0.02, size=(n_days, n_active))
    active_prices = np.cumprod(1 + active_rets, axis=0) * 100

    # Delisted: price history for first 75%, then NaN
    cutoff = int(n_days * 0.75)
    dead_rets = rng.normal(0.0004, 0.02, size=(cutoff, n_delisted))
    dead_prices = np.cumprod(1 + dead_rets, axis=0) * 100

    data: dict[str, np.ndarray] = {}
    for i, t in enumerate(active):
        data[t] = active_prices[:, i]
    for i, t in enumerate(dead):
        series = np.full(n_days, np.nan)
        series[:cutoff] = dead_prices[:, i]
        data[t] = series

    df = pd.DataFrame(data, index=dates)
    dl_returns = dict.fromkeys(dead, -0.3)
    return df, dl_returns


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDelistingInPipeline:
    """Verify the returns-space delisting correction in run_full_pipeline."""

    def test_pipeline_runs_with_delisting_returns(self) -> None:
        """End-to-end: pipeline completes and weights sum to 1."""
        prices, dl_rets = _make_prices()
        result = run_full_pipeline(
            prices=prices,
            optimizer=EqualWeighted(),
            delisting_returns=dl_rets,
        )
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_delisting_returns_applied_via_helper(self) -> None:
        """apply_delisting_returns is called with the correct ticker subset."""
        prices, dl_rets = _make_prices()
        with patch(
            "optimizer.pipeline._orchestrator._apply_delisting_rets",
            wraps=__import__(
                "optimizer.preprocessing._delisting",
                fromlist=["apply_delisting_returns"],
            ).apply_delisting_returns,
        ) as mock_apply:
            run_full_pipeline(
                prices=prices,
                optimizer=EqualWeighted(),
                delisting_returns=dl_rets,
            )
            mock_apply.assert_called_once()
            call_dict = mock_apply.call_args[0][1]
            # Only tickers present in the prices columns should be passed
            assert set(call_dict.keys()) <= set(prices.columns)
            assert set(call_dict.keys()) == {"DEAD_00", "DEAD_01"}

    def test_empty_delisting_returns_is_noop(self) -> None:
        """An empty dict does not crash and skips the apply call."""
        prices, _ = _make_prices()
        with patch(
            "optimizer.pipeline._orchestrator._apply_delisting_rets",
        ) as mock_apply:
            result = run_full_pipeline(
                prices=prices,
                optimizer=EqualWeighted(),
                delisting_returns={},
            )
            mock_apply.assert_not_called()
            assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_nonexistent_tickers_silently_ignored(self) -> None:
        """Tickers not in the DataFrame are filtered out, no error raised."""
        prices, _ = _make_prices()
        result = run_full_pipeline(
            prices=prices,
            optimizer=EqualWeighted(),
            delisting_returns={"NONEXISTENT": -0.50},
        )
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_none_delisting_returns_is_noop(self) -> None:
        """Default None value works as before (backward compat)."""
        prices, _ = _make_prices()
        result = run_full_pipeline(
            prices=prices,
            optimizer=EqualWeighted(),
        )
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_last_valid_return_replaced(self) -> None:
        """The last valid return of a delisted ticker is overwritten."""
        prices, dl_rets = _make_prices(n_active=3, n_delisted=1)
        from skfolio.preprocessing import prices_to_returns

        from optimizer.preprocessing._delisting import apply_delisting_returns

        returns = prices_to_returns(prices)
        dead_col = returns["DEAD_00"].dropna()
        last_idx = dead_col.index[-1]

        # Apply the correction
        present = {t: r for t, r in dl_rets.items() if t in returns.columns}
        corrected = apply_delisting_returns(returns, present)

        assert corrected.at[last_idx, "DEAD_00"] == pytest.approx(-0.30)
