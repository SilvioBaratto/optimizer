"""Tests for the moving-average and momentum signals.

Golden values were worked out by hand or with plain-Python loops that share no
code with the module under test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.strategies import (
    MovingAverageKind,
    cross_sectional_momentum_weights,
    momentum_scores,
    moving_average,
    moving_average_crossover_positions,
)

_NAN = float("nan")


def _frame(**columns: Sequence[float]) -> pd.DataFrame:
    """Return a price frame with one column per keyword, on business days."""
    length = len(next(iter(columns.values())))
    index = pd.bdate_range("2024-01-01", periods=length)
    return pd.DataFrame(columns, index=index, dtype=float)


# Hand-made path: fast=2 and slow=3 averages cross on rows 3 and 6 and again on row 9.
_CROSS_PATH = [10, 10, 10, 11, 12, 11, 9, 8, 9, 11]
# An exponential-average path: fast=2 and slow=3 cross down on row 4 and up on row 7.
_EMA_PATH = [10, 12, 14, 12, 10, 8, 10, 12]
# Four assets, lookback=3 and skip=1: the leader is B on row 3, C on row 5, A on row 7.
_PANEL = {
    "A": [100, 101, 102, 103, 104, 108, 112, 113, 114],
    "B": [100, 102, 104, 106, 105, 104, 103, 102, 101],
    "C": [100, 100, 101, 103, 108, 109, 110, 111, 112],
    "D": [100, 99, 98, 97, 96, 95, 94, 93, 92],
}


def _growth_panel(rates: list[float], rows: int) -> pd.DataFrame:
    """Return prices that grow at a constant rate per column, from 100."""
    steps = np.arange(rows)[:, None]
    values = 100.0 * (1.0 + np.asarray(rates)[None, :]) ** steps
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2024-01-01", periods=rows),
        columns=[f"a{i}" for i in range(len(rates))],
    )


class TestMovingAverage:
    """``moving_average`` for the simple and the exponential kind."""

    def test_sma_matches_the_hand_computed_series(self) -> None:
        """The 3-row mean of 1..6 is 2, 3, 4, 5 after two warm-up rows."""
        result = moving_average(_frame(x=[1, 2, 3, 4, 5, 6]), 3, MovingAverageKind.SMA)
        expected = [_NAN, _NAN, 2.0, 3.0, 4.0, 5.0]
        np.testing.assert_allclose(result["x"].to_numpy(), expected)

    def test_ema_matches_the_hand_computed_series(self) -> None:
        """With span 3 the weight is 0.5: 1, 1.5, 2.25, 3.125, 4.0625 on 1..5."""
        result = moving_average(_frame(x=[1, 2, 3, 4, 5]), 3, MovingAverageKind.EMA)
        expected = [_NAN, _NAN, 2.25, 3.125, 4.0625]
        np.testing.assert_allclose(result["x"].to_numpy(), expected)

    def test_a_window_of_one_returns_the_prices(self) -> None:
        """An average over one row, or with weight one, is the price itself."""
        prices = _frame(x=[3, 1, 4, 1, 5])
        for kind in MovingAverageKind:
            pd.testing.assert_frame_equal(moving_average(prices, 1, kind), prices)

    def test_columns_are_averaged_independently(self) -> None:
        """Each asset has its own average and the labels are kept."""
        prices = _frame(x=[1, 2, 3, 4], y=[4, 3, 2, 1])
        result = moving_average(prices, 2)
        np.testing.assert_allclose(result["x"].to_numpy(), [_NAN, 1.5, 2.5, 3.5])
        np.testing.assert_allclose(result["y"].to_numpy(), [_NAN, 3.5, 2.5, 1.5])
        assert result.index.equals(prices.index)
        assert list(result.columns) == ["x", "y"]

    def test_a_missing_price_poisons_the_windows_that_contain_it(self) -> None:
        """A window with a gap has no average rather than a partial one."""
        result = moving_average(_frame(x=[1, 2, _NAN, 4, 5]), 2)
        np.testing.assert_allclose(result["x"].to_numpy(), [_NAN, 1.5, _NAN, _NAN, 4.5])

    def test_the_kind_may_be_given_as_text(self) -> None:
        """A kind named as a string selects the same average."""
        prices = _frame(x=[1, 2, 3, 4, 5])
        pd.testing.assert_frame_equal(
            moving_average(prices, 3, "ema"),  # type: ignore[arg-type]
            moving_average(prices, 3, MovingAverageKind.EMA),
        )

    def test_an_unknown_kind_raises(self) -> None:
        """A misspelt kind must not fall back to the simple average."""
        with pytest.raises(ConfigurationError, match="kind"):
            moving_average(_frame(x=[1, 2, 3]), 2, "wma")  # type: ignore[arg-type]

    @pytest.mark.parametrize("window", [0, -3])
    def test_a_window_below_one_raises(self, window: int) -> None:
        """An average needs at least one row."""
        with pytest.raises(ConfigurationError, match="window"):
            moving_average(_frame(x=[1, 2, 3]), window)

    @pytest.mark.parametrize("kind", list(MovingAverageKind))
    def test_later_prices_do_not_change_earlier_averages(
        self, kind: MovingAverageKind
    ) -> None:
        """An average at a date uses only data up to that date."""
        prices = _frame(x=[5, 6, 7, 8, 9, 10, 11])
        changed = prices.copy()
        changed.iloc[4:] *= 3.0
        before = moving_average(prices, 3, kind).iloc[:4]
        after = moving_average(changed, 3, kind).iloc[:4]
        pd.testing.assert_frame_equal(before, after)


class TestCrossoverPositions:
    """``moving_average_crossover_positions`` on hand-made price paths."""

    def test_simple_average_cross_dates_long_only(self) -> None:
        """Long on rows 3-5 and 9, flat otherwise; the tie on row 2 is flat."""
        # Worked by hand: the fast average first exceeds the slow one on row 3
        # (10.5 against 10.33), falls below it on row 6 (10 against 10.67), and
        # rises above it again on row 9 (10 against 9.33). On row 2 both are 10.
        positions = moving_average_crossover_positions(
            _frame(x=_CROSS_PATH), fast=2, slow=3
        )
        expected = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
        np.testing.assert_array_equal(positions["x"].to_numpy(), expected)

    def test_simple_average_cross_dates_with_shorting(self) -> None:
        """The same path goes short on rows 6-8 when shorting is allowed."""
        positions = moving_average_crossover_positions(
            _frame(x=_CROSS_PATH), fast=2, slow=3, allow_short=True
        )
        expected = [0, 0, 0, 1, 1, 1, -1, -1, -1, 1]
        np.testing.assert_array_equal(positions["x"].to_numpy(), expected)

    def test_exponential_average_cross_dates(self) -> None:
        """Hand-run exponential averages cross down on row 4 and up on row 7."""
        # Worked by hand with weights 2/3 (span 2) and 1/2 (span 3): on row 2 the
        # fast average is 13.11 against 12.5, on row 4 it is 10.79 against 11.125,
        # and on row 7 it is 11.21 against 10.89.
        prices = _frame(x=_EMA_PATH)
        kind = MovingAverageKind.EMA
        long_only = moving_average_crossover_positions(prices, 2, 3, kind)
        np.testing.assert_array_equal(
            long_only["x"].to_numpy(), [0, 0, 1, 1, 0, 0, 0, 1]
        )
        short = moving_average_crossover_positions(prices, 2, 3, kind, True)
        np.testing.assert_array_equal(
            short["x"].to_numpy(), [0, 0, 1, 1, -1, -1, -1, 1]
        )

    def test_no_position_before_the_slow_average_exists(self) -> None:
        """Rows without a slow average are flat, even when shorting is allowed."""
        positions = moving_average_crossover_positions(
            _frame(x=[5, 4, 3, 2, 1]), fast=1, slow=4, allow_short=True
        )
        np.testing.assert_array_equal(positions["x"].to_numpy(), [0, 0, 0, -1, -1])

    def test_a_series_shorter_than_the_slow_window_is_flat(self) -> None:
        """With no slow average at all the positions are all zero, not NaN."""
        positions = moving_average_crossover_positions(
            _frame(x=[1, 2, 3]), fast=2, slow=10
        )
        assert positions.to_numpy().tolist() == [[0.0], [0.0], [0.0]]

    def test_a_flat_price_has_no_signal(self) -> None:
        """Equal averages give no position: a tie is not a crossover."""
        positions = moving_average_crossover_positions(
            _frame(x=[7, 7, 7, 7, 7, 7]), 2, 4, allow_short=True
        )
        assert (positions.to_numpy() == 0.0).all()

    def test_positions_are_nan_free_and_keep_the_labels(self) -> None:
        """Gaps in one asset do not leak NaN into the positions."""
        prices = _frame(x=_CROSS_PATH, y=[_NAN, _NAN, 5, 6, 7, 6, 5, 4, 5, 6])
        positions = moving_average_crossover_positions(prices, 2, 3)
        assert not positions.isna().any().any()
        assert positions.index.equals(prices.index)
        assert list(positions.columns) == ["x", "y"]
        assert positions["y"].iloc[:4].tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_assets_are_independent(self) -> None:
        """Adding a column does not change another column's positions."""
        alone = moving_average_crossover_positions(_frame(x=_CROSS_PATH), 2, 3)
        both = moving_average_crossover_positions(
            _frame(x=_CROSS_PATH, y=_CROSS_PATH[::-1]), 2, 3
        )
        pd.testing.assert_series_equal(alone["x"], both["x"])

    def test_later_prices_do_not_change_earlier_positions(self) -> None:
        """A position at a date is decided from data up to that date."""
        prices = _frame(x=_CROSS_PATH)
        changed = prices.copy()
        changed.iloc[7:, 0] = [50.0, 1.0, 90.0]
        before = moving_average_crossover_positions(prices, 2, 3, allow_short=True)
        after = moving_average_crossover_positions(changed, 2, 3, allow_short=True)
        pd.testing.assert_frame_equal(before.iloc[:7], after.iloc[:7])

    @pytest.mark.parametrize(("fast", "slow"), [(3, 3), (5, 3), (0, 3), (-1, 3)])
    def test_the_fast_window_must_be_positive_and_below_the_slow_one(
        self, fast: int, slow: int
    ) -> None:
        """A crossover needs 1 <= fast < slow."""
        with pytest.raises(ConfigurationError, match="fast"):
            moving_average_crossover_positions(_frame(x=_CROSS_PATH), fast, slow)


class TestMomentumScores:
    """``momentum_scores``: total return from t-lookback to t-skip."""

    def test_matches_the_hand_computed_scores(self) -> None:
        """Score on row t is price[t-skip] / price[t-lookback] - 1."""
        scores = momentum_scores(
            _frame(x=[100, 120, 90, 135, 150, 120]), lookback=4, skip=1
        )
        # Worked by hand: row 4 is 135 over 100, row 5 is 150 over 120.
        np.testing.assert_allclose(
            scores["x"].to_numpy(), [_NAN, _NAN, _NAN, _NAN, 0.35, 0.25]
        )

    def test_a_zero_skip_uses_the_latest_price(self) -> None:
        """With skip 0 the score is the plain lookback return."""
        scores = momentum_scores(
            _frame(x=[100, 120, 90, 135, 150, 120]), lookback=2, skip=0
        )
        # Worked by hand: 90/100, 135/120, 150/90 and 120/135, each less one.
        expected = [_NAN, _NAN, -0.1, 0.125, 2.0 / 3.0, -1.0 / 9.0]
        np.testing.assert_allclose(scores["x"].to_numpy(), expected)

    def test_the_classic_twelve_minus_one_defaults(self) -> None:
        """The defaults look back 252 rows and skip the last 21."""
        prices = _frame(x=[100.0 + row for row in range(300)])
        scores = momentum_scores(prices)
        assert scores["x"].iloc[:252].isna().all()
        # Row 260 compares the price on row 239 (339) with the one on row 8 (108).
        assert scores["x"].iloc[260] == pytest.approx(339.0 / 108.0 - 1.0)

    def test_an_asset_without_early_history_has_no_early_score(self) -> None:
        """A late-listed asset is unranked until both prices exist."""
        prices = _frame(x=[10, 11, 12, 13, 14], y=[_NAN, _NAN, 20, 22, 24])
        scores = momentum_scores(prices, lookback=2, skip=0)
        assert scores["y"].iloc[:4].isna().all()
        assert scores["y"].iloc[4] == pytest.approx(24.0 / 20.0 - 1.0)

    def test_later_prices_do_not_change_earlier_scores(self) -> None:
        """A score at a date uses only data up to that date."""
        prices = _frame(x=[100, 120, 90, 135, 150, 120, 130])
        changed = prices.copy()
        changed.iloc[5:] *= 10.0
        before = momentum_scores(prices, 2, 0).iloc[:5]
        after = momentum_scores(changed, 2, 0).iloc[:5]
        pd.testing.assert_frame_equal(before, after)

    @pytest.mark.parametrize(
        ("lookback", "skip"), [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 15)]
    )
    def test_invalid_windows_raise(self, lookback: int, skip: int) -> None:
        """Lookback must be positive and skip in [0, lookback)."""
        with pytest.raises(ConfigurationError, match=r"lookback|skip"):
            momentum_scores(_frame(x=[1, 2, 3]), lookback, skip)


class TestCrossSectionalMomentumWeights:
    """``cross_sectional_momentum_weights`` rebalancing and leg construction."""

    @staticmethod
    def _weights(**overrides: float | bool) -> pd.DataFrame:
        """Run the strategy on the four-asset panel with small windows."""
        settings: dict[str, float | bool] = {
            "lookback": 3,
            "skip": 1,
            "top_fraction": 0.25,
            "rebalance_every": 2,
        }
        settings.update(overrides)
        return cross_sectional_momentum_weights(
            _frame(**_PANEL),
            **settings,  # type: ignore[arg-type]
        )

    def test_long_only_picks_the_known_leader_and_holds_it(self) -> None:
        """B, then C, then A lead on rows 3, 5 and 7 and are held in between."""
        # Worked by hand: on row 3 the scores are 0.02, 0.04, 0.01 and -0.02 for
        # A to D; on row 5 C leads with 0.069; on row 7 A leads with 0.077.
        weights = self._weights()
        expected = {"A": 0, "B": 0, "C": 0, "D": 0}
        leaders = ["", "", "", "B", "B", "C", "C", "A", "A"]
        for row, leader in enumerate(leaders):
            held = dict(expected)
            if leader:
                held[leader] = 1
            assert weights.iloc[row].to_dict() == held, f"row {row}"

    def test_long_only_weights_sum_to_one_once_scores_exist(self) -> None:
        """The book is fully invested from the first rebalance and flat before."""
        sums = self._weights().sum(axis=1).to_numpy()
        np.testing.assert_allclose(sums, [0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_long_short_is_dollar_neutral_with_unit_legs(self) -> None:
        """The leader is held long and the laggard (D) short, +1 and -1."""
        weights = self._weights(long_only=False)
        np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 0.0, atol=1e-12)
        assert weights.iloc[3].to_dict() == {"A": 0, "B": 1, "C": 0, "D": -1}
        assert weights.iloc[6].to_dict() == {"A": 0, "B": 0, "C": 1, "D": -1}
        assert weights.iloc[8].to_dict() == {"A": 1, "B": 0, "C": 0, "D": -1}
        assert (weights.iloc[:3] == 0.0).all().all()

    def test_targets_change_only_on_rebalance_dates(self) -> None:
        """Between rebalances every row repeats the previous one."""
        weights = self._weights(rebalance_every=3)
        # The rebalances fall on rows 3 and 6, so each block of three repeats.
        for start in (3, 6):
            block = weights.iloc[start : start + 3]
            assert (block == block.iloc[0]).all().all()
        assert not np.array_equal(
            weights.iloc[3].to_numpy(), weights.iloc[6].to_numpy()
        )

    def test_equal_weights_inside_each_leg(self) -> None:
        """Two names per leg get 0.5 each: the two fastest and the two slowest."""
        rates = [-0.02, -0.01, -0.005, 0.0, 0.002, 0.004, 0.01, 0.02]
        weights = cross_sectional_momentum_weights(
            _growth_panel(rates, 10),
            lookback=3,
            skip=1,
            top_fraction=0.25,
            rebalance_every=1,
            long_only=False,
        )
        last = weights.iloc[-1].to_numpy()
        np.testing.assert_allclose(last, [-0.5, -0.5, 0, 0, 0, 0, 0.5, 0.5])

    def test_long_only_leg_size_follows_the_fraction(self) -> None:
        """A quarter of eight assets is two names at 0.5 each."""
        rates = [-0.02, -0.01, -0.005, 0.0, 0.002, 0.004, 0.01, 0.02]
        weights = cross_sectional_momentum_weights(
            _growth_panel(rates, 10), 3, 1, 0.25, 1, True
        )
        np.testing.assert_allclose(
            weights.iloc[-1].to_numpy(), [0, 0, 0, 0, 0, 0, 0.5, 0.5]
        )

    def test_at_least_one_name_is_always_held(self) -> None:
        """A fraction too small for one name still takes the single best one."""
        weights = cross_sectional_momentum_weights(
            _growth_panel([0.01, 0.02, 0.03], 8), 3, 1, 0.2, 1
        )
        np.testing.assert_allclose(weights.iloc[-1].to_numpy(), [0, 0, 1])

    def test_a_fraction_that_lands_on_an_integer_is_not_rounded_down(self) -> None:
        """29% of 100 assets is 29 names although 0.29 * 100 is 28.999999999999996."""
        rates = np.linspace(-0.01, 0.01, 100).tolist()
        weights = cross_sectional_momentum_weights(
            _growth_panel(rates, 6), 3, 1, 0.29, 1
        )
        last = weights.iloc[-1].to_numpy()
        assert (last > 0).sum() == 29
        np.testing.assert_allclose(last[last > 0], 1.0 / 29.0)
        assert (last[71:] > 0).all()

    def test_ties_are_broken_by_column_order(self) -> None:
        """With equal scores the earlier columns rank higher."""
        flat = pd.DataFrame(
            100.0,
            index=pd.bdate_range("2024-01-01", periods=5),
            columns=["a", "b", "c", "d"],
        )
        long_short = cross_sectional_momentum_weights(
            flat, 2, 0, 0.5, 1, long_only=False
        )
        np.testing.assert_allclose(
            long_short.iloc[-1].to_numpy(), [0.5, 0.5, -0.5, -0.5]
        )
        long_only = cross_sectional_momentum_weights(flat, 2, 0, 0.5, 1)
        np.testing.assert_allclose(long_only.iloc[-1].to_numpy(), [0.5, 0.5, 0, 0])

    def test_a_fraction_of_one_holds_every_ranked_asset_equally(self) -> None:
        """The whole universe at equal weight."""
        weights = self._weights(top_fraction=1.0)
        np.testing.assert_allclose(weights.iloc[-1].to_numpy(), [0.25] * 4)

    def test_only_assets_with_a_score_are_ranked(self) -> None:
        """A late-listed asset joins once it has a score; before, it is skipped."""
        prices = _frame(a=[10, 11, 12, 13, 14, 15], b=[_NAN, _NAN, _NAN, 5, 6, 8])
        long_only = cross_sectional_momentum_weights(prices, 2, 0, 0.5, 1)
        # On row 2 only a has a score and takes the whole book. On row 5 both are
        # ranked and b (8 over 5) beats a (15 over 13), so b is the single name.
        assert long_only.iloc[2].to_dict() == {"a": 1.0, "b": 0.0}
        assert long_only.iloc[5].to_dict() == {"a": 0.0, "b": 1.0}

    def test_a_long_short_book_needs_two_ranked_assets(self) -> None:
        """With a single ranked asset the legs cannot be formed: stay flat."""
        prices = _frame(a=[10, 11, 12, 13, 14, 15], b=[_NAN, _NAN, _NAN, 5, 6, 8])
        weights = cross_sectional_momentum_weights(
            prices, 2, 0, 0.5, 1, long_only=False
        )
        assert (weights.iloc[:5] == 0.0).all().all()
        assert weights.iloc[5].to_dict() == {"a": -1.0, "b": 1.0}

    def test_no_weights_before_the_first_score(self) -> None:
        """A series shorter than the lookback is flat everywhere, with NaN-free rows."""
        weights = cross_sectional_momentum_weights(_frame(**_PANEL), 20, 1, 0.5, 2)
        assert (weights == 0.0).all().all()
        assert weights.shape == (9, 4)

    def test_weights_keep_the_price_labels(self) -> None:
        """The frame shares the prices' index and columns."""
        prices = _frame(**_PANEL)
        weights = self._weights()
        assert weights.index.equals(prices.index)
        assert list(weights.columns) == list(prices.columns)

    def test_weights_are_decided_from_data_up_to_the_rebalance_date(self) -> None:
        """Rewriting prices after row 5 leaves rows 0-5 untouched."""
        prices = _frame(**_PANEL)
        changed = prices.copy()
        changed.iloc[6:] = changed.iloc[6:] * [3.0, 0.2, 0.5, 9.0]
        before = cross_sectional_momentum_weights(prices, 3, 1, 0.25, 2)
        after = cross_sectional_momentum_weights(changed, 3, 1, 0.25, 2)
        pd.testing.assert_frame_equal(before.iloc[:6], after.iloc[:6])

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5, float("nan")])
    def test_a_fraction_outside_zero_one_raises(self, fraction: float) -> None:
        """The leg fraction must lie in (0, 1]."""
        with pytest.raises(ConfigurationError, match="top_fraction"):
            self._weights(top_fraction=fraction)

    def test_a_long_short_fraction_above_one_half_raises(self) -> None:
        """Above one half the long and short legs would overlap."""
        with pytest.raises(ConfigurationError, match="top_fraction"):
            self._weights(top_fraction=0.6, long_only=False)

    def test_a_rebalance_interval_below_one_raises(self) -> None:
        """Rebalancing every zero rows is meaningless."""
        with pytest.raises(ConfigurationError, match="rebalance_every"):
            self._weights(rebalance_every=0)

    def test_invalid_momentum_windows_raise(self) -> None:
        """The lookback and skip rules of the scores apply here too."""
        with pytest.raises(ConfigurationError, match="skip"):
            self._weights(lookback=3, skip=3)


def _broken_prices(kind: str) -> object:
    """Return a price input that is invalid in the way ``kind`` names."""
    prices = _frame(a=[10, 11, 12, 13], b=[20, 21, 22, 23])
    zero, negative, infinite = prices.copy(), prices.copy(), prices.copy()
    zero.iloc[1, 0] = 0.0
    negative.iloc[1, 0] = -1.0
    infinite.iloc[1, 0] = np.inf
    builders = {
        "series": lambda: prices["a"],
        "empty_rows": lambda: prices.iloc[:0],
        "empty_columns": lambda: prices.iloc[:, :0],
        "duplicate_index": lambda: prices.set_axis(prices.index[[0, 0, 1, 2]], axis=0),
        "descending_index": lambda: prices.iloc[::-1],
        "duplicate_columns": lambda: prices.set_axis(["a", "a"], axis=1),
        "text": lambda: prices.astype(str),
        "zero_price": lambda: zero,
        "negative_price": lambda: negative,
        "infinite_price": lambda: infinite,
    }
    return builders[kind]()


class TestPriceValidation:
    """Every signal rejects prices that cannot be real."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda prices: moving_average(prices, 2),
            lambda prices: moving_average_crossover_positions(prices, 1, 2),
            lambda prices: momentum_scores(prices, 2, 0),
            lambda prices: cross_sectional_momentum_weights(prices, 2, 0, 0.5, 1),
        ],
        ids=["moving_average", "crossover", "momentum", "cross_sectional"],
    )
    @pytest.mark.parametrize(
        "bad",
        [
            "series",
            "empty_rows",
            "empty_columns",
            "duplicate_index",
            "descending_index",
            "duplicate_columns",
            "text",
            "zero_price",
            "negative_price",
            "infinite_price",
        ],
    )
    def test_bad_prices_raise_a_data_error(self, call: object, bad: str) -> None:
        """A wrong container, shape, index, dtype or price value is a DataError."""
        with pytest.raises(DataError):
            call(_broken_prices(bad))  # type: ignore[operator]

    def test_missing_prices_are_allowed(self) -> None:
        """A NaN price is a gap, not an error: it only blanks the signal."""
        prices = _frame(a=[10, _NAN, 12, 13])
        average = moving_average(prices, 2)
        np.testing.assert_allclose(average["a"].to_numpy(), [_NAN, _NAN, _NAN, 12.5])
