"""Tests for the pipeline-composable investability-screen selector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from optimizer.exceptions import DataError
from optimizer.universe import (
    InvestabilityScreenConfig,
    InvestabilityScreenSelector,
    build_investability_screen,
)


@pytest.fixture()
def screening_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "MSFT", "GOOG", "TINY", "NEW"]
    dates = pd.bdate_range("2023-01-01", periods=300)
    prices = np.abs(100.0 + rng.normal(0, 1, (300, 5)).cumsum(axis=0))
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    # NEW: recent IPO (only 50 days of history)
    price_df.loc[price_df.index[:-50], "NEW"] = np.nan

    vol = rng.integers(1_000_000, 5_000_000, size=(300, 5))
    volume_df = pd.DataFrame(vol, index=dates, columns=tickers)
    volume_df.loc[price_df["NEW"].isna(), "NEW"] = 0
    # TINY: sparse trading
    tiny_mask = rng.random(300) < 0.20
    volume_df.loc[tiny_mask, "TINY"] = 0

    fundamentals = pd.DataFrame(
        {
            "market_cap": [2e9, 1.5e9, 3e9, 50e6, 500e6],
            "current_price": [150.0, 300.0, 100.0, 0.5, 25.0],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return {
        "fundamentals": fundamentals,
        "price_history": price_df,
        "volume_history": volume_df,
    }


def _returns_from(price_history: pd.DataFrame) -> pd.DataFrame:
    """Linear returns with tickers as columns (skfolio input convention)."""
    return price_history.pct_change().iloc[1:]


class TestInvestabilityScreenSelector:
    def test_selects_investable_columns(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        selector = build_investability_screen(**screening_data)
        selector.fit(X)

        kept = set(selector.investable_universe_)
        assert "AAPL" in kept
        assert "MSFT" in kept
        assert "GOOG" in kept
        # TINY fails mcap/price; NEW fails listing age
        assert "TINY" not in kept
        assert "NEW" not in kept

    def test_transform_returns_subset_preserving_order(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        selector = build_investability_screen(**screening_data)
        selector.set_output(transform="pandas")
        selector.fit(X)
        out = selector.transform(X)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["AAPL", "MSFT", "GOOG"]
        # Column order follows X, not screen output ordering.
        assert list(out.columns) == [c for c in X.columns if c in out.columns]

    def test_get_support_mask_matches_columns(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        selector = build_investability_screen(**screening_data).fit(X)
        mask = selector.get_support()
        assert mask.dtype == bool
        assert mask.tolist() == [True, True, True, False, False]
        assert list(selector.get_feature_names_out()) == ["AAPL", "MSFT", "GOOG"]

    def test_composes_in_sklearn_pipeline(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        pipe = Pipeline([("screen", build_investability_screen(**screening_data))])
        pipe.set_output(transform="pandas")
        out = pipe.fit_transform(X)
        assert list(out.columns) == ["AAPL", "MSFT", "GOOG"]

    def test_get_params_exposes_config(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        cfg = InvestabilityScreenConfig.for_large_cap()
        selector = build_investability_screen(**screening_data, config=cfg)
        params = selector.get_params()
        assert params["config"] is cfg
        assert "fundamentals" in params

    def test_config_none_uses_defaults(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        selector = InvestabilityScreenSelector(config=None, **screening_data)
        selector.fit(X)
        assert isinstance(selector.investable_universe_, pd.Index)

    def test_missing_screening_data_raises(self) -> None:
        X = pd.DataFrame({"A": [0.01, -0.01], "B": [0.02, 0.0]})
        selector = InvestabilityScreenSelector()
        with pytest.raises(DataError, match="requires fundamentals"):
            selector.fit(X)

    def test_array_input_without_feature_names_raises(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"]).to_numpy()
        selector = build_investability_screen(**screening_data)
        with pytest.raises(DataError, match="ticker column names"):
            selector.fit(X)

    def test_allows_nan_returns(self, screening_data: dict[str, pd.DataFrame]) -> None:
        X = _returns_from(screening_data["price_history"])
        # NEW column legitimately carries NaN (pre-IPO) — must not raise.
        assert X["NEW"].isna().any()
        selector = build_investability_screen(**screening_data).fit(X)
        assert "NEW" not in set(selector.investable_universe_)

    def test_current_members_hysteresis(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        first = build_investability_screen(**screening_data).fit(X)
        second = build_investability_screen(
            **screening_data, current_members=first.investable_universe_
        ).fit(X)
        assert set(first.investable_universe_) == set(second.investable_universe_)
