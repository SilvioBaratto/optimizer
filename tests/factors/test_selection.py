"""Tests for stock selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    SelectionConfig,
    SelectionMethod,
    apply_sector_balance,
    compute_selection_turnover,
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
        # Count must never exceed target_count
        assert len(result) == 10

    def test_buffer_never_exceeds_target_count(self, scores: pd.Series) -> None:
        """Buffer zone must not inflate the returned set beyond target_count."""
        first = select_fixed_count(scores, target_count=30)
        # Demote several top-30 members into the buffer zone
        modified = scores.copy()
        sorted_idx = scores.sort_values(ascending=False).index
        for i in [27, 28, 29]:
            modified[sorted_idx[i]] = scores[sorted_idx[31]]

        result = select_fixed_count(
            modified, target_count=30, buffer_fraction=0.1, current_members=first
        )
        assert len(result) == 30

    def test_buffer_evicts_lowest_ranked_direct(self, scores: pd.Series) -> None:
        """Retained buffer member displaces the lowest-ranked direct entrant."""
        sorted_idx = scores.sort_values(ascending=False).index
        current = pd.Index(list(sorted_idx[:10]))
        # Demote rank-9 ticker to just below the buffer cutoff position
        modified = scores.copy()
        delta = abs(scores[sorted_idx[10]]) * 0.01
        modified[sorted_idx[9]] = scores[sorted_idx[10]] - delta
        result = select_fixed_count(
            modified, target_count=10, buffer_fraction=0.2, current_members=current
        )
        assert len(result) == 10
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

    def test_convergence_multi_sector_cascade(self) -> None:
        """Single-pass leaves a violation; convergence loop fixes it (#280).

        Universe: X=50%, Y=30%, Z=20% (100 stocks).
        Initial selection: 25 X, 5 Y, 0 Z = 30 selected.

        Single-pass (old code, n_selected=30 fixed):
          X: max_n=16, remove 9 → 16 X.  Y: add 3.  Z: add 4.
          Final = 16X + 8Y + 4Z = 28.
          At n=28: X max_n=round(15.4)=15, but X count=16 → VIOLATION.

        Convergence loop catches the cascade and trims X to 15.
        """
        x_tickers = [f"X{i}" for i in range(50)]
        y_tickers = [f"Y{i}" for i in range(30)]
        z_tickers = [f"Z{i}" for i in range(20)]
        all_tickers = x_tickers + y_tickers + z_tickers

        sector_labels = pd.Series(
            ["X"] * 50 + ["Y"] * 30 + ["Z"] * 20,
            index=all_tickers,
        )
        parent_universe = pd.Index(all_tickers)

        scores = pd.Series(
            [float(100 - i) for i in range(100)],
            index=all_tickers,
        )

        # Initial selection: 25 X, 5 Y, 0 Z
        initial_selected = pd.Index(x_tickers[:25] + y_tickers[:5])

        result = apply_sector_balance(
            initial_selected,
            scores,
            sector_labels,
            parent_universe=parent_universe,
            tolerance=0.05,
        )

        # Verify discrete count constraints at FINAL n_selected
        result_sectors = sector_labels.reindex(result).dropna()
        n = len(result)
        target_weights = (
            sector_labels.reindex(parent_universe).dropna().value_counts(normalize=True)
        )
        for sector, tw in target_weights.items():
            count = int((result_sectors == sector).sum())
            min_n = max(0, round((tw - 0.05) * n))
            max_n = round((tw + 0.05) * n)
            assert min_n <= count <= max_n, (
                f"Sector {sector}: count={count}, expected [{min_n}, {max_n}] at n={n}"
            )

    def test_no_shrinkage_when_overweight_sector_trimmed(self) -> None:
        """Trimming overweight sector must not shrink below target_count.

        Regression test for #300.

        Universe: Tech=50%, Finance=30%, Energy=20% (20 stocks).
        Initial: Tech=14, Finance=5, Energy=3 → 22 selected.
        With tolerance=0.1, Tech max_n = round((0.5+0.1)*22)=13 → 1 removed.
        Bug: n_selected then becomes 21, Finance min_n drops from 5→4, so
        Finance is not topped up, leaving 21 stocks instead of 22.
        Fix: n_target is snapshotted at 22 for the whole inner loop; Finance
        min_n stays at 5 and the loop converges with the correct count.
        """
        tech = [f"TECH{i}" for i in range(10)]
        finance = [f"FIN{i}" for i in range(6)]
        energy = [f"ENE{i}" for i in range(4)]
        all_tickers = tech + finance + energy

        sector_labels = pd.Series(
            ["Tech"] * 10 + ["Finance"] * 6 + ["Energy"] * 4,
            index=all_tickers,
        )
        parent_universe = pd.Index(all_tickers)
        scores = pd.Series(
            [float(100 - i) for i in range(20)],
            index=all_tickers,
        )

        # Initial: 7 Tech, 3 Finance, 2 Energy = 12 selected; Tech is overweight
        initial_selected = pd.Index(tech[:7] + finance[:3] + energy[:2])

        result = apply_sector_balance(
            initial_selected,
            scores,
            sector_labels,
            parent_universe=parent_universe,
            tolerance=0.1,
        )

        # Verify that every sector count is within tolerance of n_target
        n = len(result)
        result_sectors = sector_labels.reindex(result).dropna()
        target_weights = (
            sector_labels.reindex(parent_universe).dropna().value_counts(normalize=True)
        )
        for sector, tw in target_weights.items():
            count = int((result_sectors == sector).sum())
            min_n = max(0, round((tw - 0.1) * n))
            max_n = round((tw + 0.1) * n)
            assert min_n <= count <= max_n, (
                f"Sector {sector}: count={count} not in [{min_n}, {max_n}] at n={n}"
            )

    def test_already_balanced_single_iteration(self) -> None:
        """A perfectly balanced selection converges with zero changes."""
        tickers = [f"S{i}" for i in range(9)]
        sector_labels = pd.Series(
            ["X", "X", "X", "Y", "Y", "Y", "Z", "Z", "Z"], index=tickers
        )
        scores = pd.Series([3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0], index=tickers)
        selected = pd.Index(["S0", "S3", "S6"])
        result = apply_sector_balance(
            selected,
            scores,
            sector_labels,
            parent_universe=pd.Index(tickers),
            tolerance=0.05,
        )
        assert set(result) == {"S0", "S3", "S6"}


class TestApplySectorBalanceMaxPerSector:
    """Hard sector cap (issue #525)."""

    def _build_dominant_sector_inputs(
        self,
    ) -> tuple[pd.Series, pd.Series, pd.Index, pd.Index]:
        # 15 Tech + 5 each of Finance/Health/Energy = 30 candidates
        tech = [f"T{i:02d}" for i in range(15)]
        fin = [f"F{i:02d}" for i in range(5)]
        hlt = [f"H{i:02d}" for i in range(5)]
        eng = [f"E{i:02d}" for i in range(5)]
        all_tickers = tech + fin + hlt + eng
        sectors = ["Tech"] * 15 + ["Finance"] * 5 + ["Health"] * 5 + ["Energy"] * 5
        sector_labels = pd.Series(sectors, index=all_tickers)
        # Tech ranked T00 (best) → T14 (worst); other sectors ranked similarly
        scores = pd.Series(
            [float(100 - i) for i in range(15)]
            + [float(50 - i) for i in range(5)]
            + [float(45 - i) for i in range(5)]
            + [float(40 - i) for i in range(5)],
            index=all_tickers,
        )
        # Initially select all 15 Tech + 5 Finance + 5 Health + 5 Energy
        initial = pd.Index(all_tickers)
        return scores, sector_labels, pd.Index(all_tickers), initial

    def test_when_max_per_sector_zero_baseline_unchanged(self) -> None:
        scores, sector_labels, parent, initial = self._build_dominant_sector_inputs()
        baseline = apply_sector_balance(
            initial, scores, sector_labels, parent_universe=parent, tolerance=0.05
        )
        with_zero = apply_sector_balance(
            initial,
            scores,
            sector_labels,
            parent_universe=parent,
            tolerance=0.05,
            max_per_sector=0,
        )
        assert set(baseline) == set(with_zero)

    def test_when_max_per_sector_caps_dominant_sector(self) -> None:
        scores, sector_labels, parent, initial = self._build_dominant_sector_inputs()
        result = apply_sector_balance(
            initial,
            scores,
            sector_labels,
            parent_universe=parent,
            tolerance=0.20,
            max_per_sector=8,
        )
        result_sectors = sector_labels.reindex(result).dropna()
        assert int((result_sectors == "Tech").sum()) <= 8

    def test_when_max_per_sector_evicts_lowest_scoring_first(self) -> None:
        scores, sector_labels, parent, initial = self._build_dominant_sector_inputs()
        result = apply_sector_balance(
            initial,
            scores,
            sector_labels,
            parent_universe=parent,
            tolerance=0.20,
            max_per_sector=8,
        )
        # Top-8 Tech (T00..T07) survive; T08..T14 evicted
        retained_tech = {t for t in result if t.startswith("T")}
        for keeper in [f"T{i:02d}" for i in range(8)]:
            assert keeper in retained_tech
        for evictee in [f"T{i:02d}" for i in range(8, 15)]:
            assert evictee not in retained_tech


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

    def test_fixed_count_with_buffer_respects_target(self, scores: pd.Series) -> None:
        """select_stocks returns exactly target_count when buffer retains extras."""
        config = SelectionConfig(
            method=SelectionMethod.FIXED_COUNT,
            target_count=10,
            buffer_fraction=0.2,
            sector_balance=False,
        )
        first = select_stocks(scores, config=config)
        # Demote the last two direct members into the buffer zone
        modified = scores.copy()
        sorted_idx = scores.sort_values(ascending=False).index
        for i in [8, 9]:
            modified[sorted_idx[i]] = scores[sorted_idx[11]]
        result = select_stocks(modified, config=config, current_members=first)
        assert len(result) == 10

    def test_when_config_max_per_sector_set_select_stocks_enforces_cap(self) -> None:
        tech = [f"T{i:02d}" for i in range(15)]
        fin = [f"F{i:02d}" for i in range(10)]
        all_tickers = tech + fin
        sector_labels = pd.Series(["Tech"] * 15 + ["Finance"] * 10, index=all_tickers)
        scores = pd.Series(
            [float(100 - i) for i in range(15)] + [float(50 - i) for i in range(10)],
            index=all_tickers,
        )
        config = SelectionConfig(
            method=SelectionMethod.FIXED_COUNT,
            target_count=20,
            sector_balance=True,
            sector_tolerance=0.30,
            max_per_sector=8,
        )
        result = select_stocks(
            scores,
            config=config,
            sector_labels=sector_labels,
            parent_universe=pd.Index(all_tickers),
        )
        result_sectors = sector_labels.reindex(result).dropna()
        assert int((result_sectors == "Tech").sum()) <= 8


class TestComputeSelectionTurnover:
    """Tests for compute_selection_turnover (issue #82)."""

    def test_no_change_zero_turnover(self) -> None:
        idx = pd.Index(["A", "B", "C"])
        assert compute_selection_turnover(idx, idx, idx) == 0.0

    def test_complete_replacement(self) -> None:
        current = pd.Index(["A", "B"])
        new = pd.Index(["C", "D"])
        universe = pd.Index(["A", "B", "C", "D"])
        assert compute_selection_turnover(current, new, universe) == 1.0

    def test_partial_overlap(self) -> None:
        current = pd.Index(["A", "B", "C"])
        new = pd.Index(["B", "C", "D"])
        universe = pd.Index(["A", "B", "C", "D", "E"])
        # added = {D}, removed = {A} → 2 / 5 = 0.4
        assert compute_selection_turnover(current, new, universe) == pytest.approx(0.4)

    def test_empty_universe_returns_zero(self) -> None:
        assert (
            compute_selection_turnover(pd.Index(["A"]), pd.Index(["B"]), pd.Index([]))
            == 0.0
        )

    def test_return_turnover_flag(self, scores: pd.Series) -> None:
        config = SelectionConfig(
            method=SelectionMethod.FIXED_COUNT,
            target_count=10,
            sector_balance=False,
        )
        result = select_stocks(scores, config=config, return_turnover=True)
        assert isinstance(result, tuple)
        selected, turnover = result
        assert isinstance(selected, pd.Index)
        assert isinstance(turnover, float)
        assert len(selected) == 10
