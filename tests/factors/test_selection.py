"""Tests for stock selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    SelectionConfig,
    SelectionMethod,
    apply_sector_balance,
    select_fixed_count,
    select_quantile,
    select_stocks,
)


@pytest.fixture()
def scores() -> pd.Series:
    """Composite scores for 50 tickers."""
    rng = np.random.default_rng(42)
    tickers = [f"T{i:03d}" for i in range(50)]
    return pd.Series(rng.normal(0, 1, 50), index=tickers)


@pytest.fixture()
def sector_labels() -> pd.Series:
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]
    return pd.Series(
        [sectors[i % 5] for i in range(50)],
        index=[f"T{i:03d}" for i in range(50)],
    )


class TestSelectFixedCount:
    def test_selects_correct_count(self, scores: pd.Series) -> None:
        result = select_fixed_count(scores, target_count=10)
        assert len(result) == 10

    def test_top_scored_selected(self, scores: pd.Series) -> None:
        result = select_fixed_count(scores, target_count=5)
        top_5 = scores.nlargest(5).index
        assert set(result) == set(top_5)

    def test_buffer_retains_members(self, scores: pd.Series) -> None:
        # First pass: select top 10
        first = select_fixed_count(scores, target_count=10)
        # Modify a member's score to just below cutoff (rank 11-12)
        modified_scores = scores.copy()
        sorted_idx = scores.sort_values(ascending=False).index
        # Member at position 9 (last in top 10) gets slightly lower score
        modified_scores[sorted_idx[9]] = scores[sorted_idx[11]]

        # With buffer, the old member should be retained
        result = select_fixed_count(
            modified_scores, target_count=10, buffer_fraction=0.2, current_members=first
        )
        assert sorted_idx[9] in result

    def test_fewer_than_target(self) -> None:
        scores = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        result = select_fixed_count(scores, target_count=10)
        assert len(result) == 3


class TestSelectQuantile:
    def test_selects_above_threshold(self, scores: pd.Series) -> None:
        result = select_quantile(scores, target_quantile=0.8)
        threshold = scores.quantile(0.8)
        assert all(scores[t] >= threshold for t in result)

    def test_hysteresis_retains(self, scores: pd.Series) -> None:
        first = select_quantile(scores, target_quantile=0.8)
        # With exit at 0.7, members between 0.7 and 0.8 quantile are retained
        result = select_quantile(
            scores,
            target_quantile=0.8,
            exit_quantile=0.7,
            current_members=first,
        )
        assert len(result) >= len(first)

    def test_empty_scores(self) -> None:
        result = select_quantile(pd.Series(dtype=float), target_quantile=0.8)
        assert len(result) == 0


class TestApplySectorBalance:
    def test_basic_rebalancing(
        self, scores: pd.Series, sector_labels: pd.Series
    ) -> None:
        # Select top 20
        selected = scores.nlargest(20).index
        result = apply_sector_balance(
            selected,
            scores,
            sector_labels,
            parent_universe=scores.index,
            tolerance=0.05,
        )
        assert isinstance(result, pd.Index)
        assert len(result) > 0


class TestSelectStocks:
    def test_fixed_count(self, scores: pd.Series) -> None:
        config = SelectionConfig(
            method=SelectionMethod.FIXED_COUNT,
            target_count=10,
            sector_balance=False,
        )
        result = select_stocks(scores, config=config)
        assert len(result) == 10

    def test_quantile(self, scores: pd.Series) -> None:
        config = SelectionConfig(
            method=SelectionMethod.QUANTILE,
            target_quantile=0.8,
            sector_balance=False,
        )
        result = select_stocks(scores, config=config)
        assert len(result) > 0

    def test_with_sector_balance(
        self, scores: pd.Series, sector_labels: pd.Series
    ) -> None:
        config = SelectionConfig(
            target_count=20,
            sector_balance=True,
        )
        result = select_stocks(
            scores,
            config=config,
            sector_labels=sector_labels,
            parent_universe=scores.index,
        )
        assert isinstance(result, pd.Index)
        assert len(result) > 0

    def test_default_config(self, scores: pd.Series) -> None:
        # Default selects top 100 but we only have 50
        result = select_stocks(scores)
        assert len(result) == 50
