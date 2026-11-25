"""Tests for the z-score trading rules.

The expected position paths are worked out by hand, step by step, from the rule
stated in each docstring; none of them is read off the function under test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.statarb import zscore_positions

_NAN = math.nan


def _z(values: list[float]) -> pd.Series:
    """Return a z-score Series on a business-day index."""
    return pd.Series(
        values, index=pd.date_range("2022-01-03", periods=len(values), freq="B")
    )


def _positions(
    values: list[float],
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float | None = None,
) -> list[int]:
    """Return the positions for a z path as a plain list."""
    return zscore_positions(_z(values), entry=entry, exit=exit, stop=stop).tolist()


class TestZscorePositionsRules:
    """Entry, hold, exit, stop and missing values on a hand-made path."""

    def test_entry_hold_through_and_exit_without_a_stop(self) -> None:
        """Short above +2, long below -2, hold until ``|z| < 0.5``.

        By hand: the first two readings are flat, 2.5 opens a short, 3.0, 1.0 and
        0.6 keep it, 0.4 closes it, 0.1 and -1.0 stay flat, -2.5 opens a long,
        -3.0 and -4.5 keep it (no stop is set) and -0.2 closes it.
        """
        path = [0.0, 1.0, 2.5, 3.0, 1.0, 0.6, 0.4, 0.1, -1.0, -2.5, -3.0, -4.5, -0.2]
        assert _positions(path) == [0, 0, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 0]

    def test_stop_out_then_flat_until_the_spread_reverts(self) -> None:
        """After a stop-out nothing is entered until ``|z| < exit`` again.

        By hand with a stop at 4: -2.5 opens a long, -3.0 keeps it, -4.5 stops it
        out and locks, -3.0 stays flat although it is below -2, -0.2 unlocks, 2.2
        opens a short, 5.0 stops it out, 4.5 and 1.0 stay locked, 0.3 unlocks and
        2.1 opens a short again.
        """
        path = [-2.5, -3.0, -4.5, -3.0, -0.2, 2.2, 5.0, 4.5, 1.0, 0.3, 2.1]
        assert _positions(path, stop=4.0) == [1, 1, 0, 0, 0, -1, 0, 0, 0, 0, -1]

    def test_a_first_reading_beyond_the_stop_is_not_entered(self) -> None:
        """Entering and being stopped out on the same bar would be a round trip.

        By hand with a stop at 4: 4.5 is above both levels so the series stays
        flat and locked, 3.0 is still locked, 0.2 unlocks and -2.2 opens a long.
        """
        path = [0.0, 4.5, 3.0, 0.2, -2.2]
        assert _positions(path, stop=4.0) == [0, 0, 0, 0, 1]

    def test_missing_scores_keep_the_previous_position(self) -> None:
        """A NaN neither opens, closes nor changes the lock."""
        path = [_NAN, 2.5, _NAN, _NAN, 1.0, _NAN, 0.1, _NAN, -2.5, _NAN]
        assert _positions(path) == [0, -1, -1, -1, -1, -1, 0, 0, 1, 1]

    def test_missing_scores_keep_a_stop_out_locked(self) -> None:
        """A NaN in the middle of a lock does not release it."""
        path = [2.5, 5.0, _NAN, 3.0, 0.2, 2.5]
        assert _positions(path, stop=4.0) == [-1, 0, 0, 0, 0, -1]

    def test_a_position_is_closed_before_it_is_reversed(self) -> None:
        """A jump to the other side that skips the exit band stays in the trade."""
        # The docstring of zscore_positions promises that only the exit band or the
        # stop closes a position, so a jump across zero does not flip it.
        path = [2.5, -2.5, 0.2, -2.5]
        assert _positions(path) == [-1, -1, 0, 1]

    def test_levels_are_strict_inequalities(self) -> None:
        """Exactly at the entry, exit or stop level nothing happens.

        By hand with a stop at 4: 2.0 equals the entry level and does not enter,
        2.5 opens a short, 0.5 equals the exit level and keeps it, 4.0 equals the
        stop and keeps it, 4.1 stops it out.
        """
        assert _positions([2.0, 2.5, 0.5, 4.0, 4.1], stop=4.0) == [0, -1, -1, -1, 0]

    def test_a_zero_exit_level_never_closes_a_position(self) -> None:
        """``|z| < 0`` is impossible, so only the stop can close the trade."""
        assert _positions([2.5, 0.0, -3.0, 5.0], exit=0.0, stop=4.0) == [-1, -1, -1, 0]

    def test_custom_levels(self) -> None:
        """Entry 1, exit 0.25 on a short path."""
        assert _positions([0.9, 1.1, 0.5, 0.2, -1.2], entry=1.0, exit=0.25) == [
            0,
            -1,
            -1,
            0,
            1,
        ]


class TestZscorePositionsProperties:
    """Structural guarantees that hold for any input."""

    def test_the_output_keeps_the_index_and_uses_only_three_values(self) -> None:
        """Positions are integers in {-1, 0, 1} on the input's index."""
        z = _z(list(np.random.default_rng(0).normal(scale=2.5, size=400)))
        positions = zscore_positions(z, stop=5.0)
        assert positions.index.equals(z.index)
        assert set(positions.unique()) <= {-1, 0, 1}
        assert positions.dtype.kind == "i"

    def test_the_series_starts_flat(self) -> None:
        """Even a first reading beyond the entry level cannot trade before it exists."""
        assert zscore_positions(_z([_NAN, _NAN, 0.1])).tolist() == [0, 0, 0]

    def test_a_position_never_depends_on_later_scores(self) -> None:
        """The positions of a prefix equal the prefix of the positions."""
        values = list(np.random.default_rng(1).normal(scale=2.0, size=300))
        values[40] = _NAN
        full = zscore_positions(_z(values), stop=4.0)
        for cut in (10, 77, 150, 299):
            prefix = zscore_positions(_z(values[:cut]), stop=4.0)
            assert prefix.tolist() == full.iloc[:cut].tolist()

    def test_the_input_is_not_modified(self) -> None:
        """The z-score Series is left as it was."""
        z = _z([0.0, 2.5, _NAN, 0.1])
        before = z.copy()
        zscore_positions(z)
        pd.testing.assert_series_equal(z, before)

    def test_an_empty_series_gives_an_empty_result(self) -> None:
        """No scores, no positions."""
        assert zscore_positions(pd.Series(dtype=float)).empty


class TestZscorePositionsValidation:
    """Every rejected setting of ``zscore_positions``."""

    @pytest.mark.parametrize(
        ("entry", "exit_level"),
        [(2.0, 2.0), (2.0, 2.5), (2.0, -0.1), (0.0, 0.0)],
    )
    def test_exit_must_lie_in_zero_to_entry(
        self, entry: float, exit_level: float
    ) -> None:
        """``0 <= exit < entry`` is required."""
        with pytest.raises(ConfigurationError, match="exit"):
            zscore_positions(_z([0.0]), entry=entry, exit=exit_level)

    @pytest.mark.parametrize("stop", [2.0, 1.0, 0.0])
    def test_stop_must_exceed_entry(self, stop: float) -> None:
        """A stop at or below the entry level would close every trade at once."""
        with pytest.raises(ConfigurationError, match="stop"):
            zscore_positions(_z([0.0]), entry=2.0, exit=0.5, stop=stop)

    def test_non_series_input_raises(self) -> None:
        """The index of the z-scores is the index of the positions."""
        with pytest.raises(DataError, match="Series"):
            zscore_positions([0.0, 3.0])  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_levels_must_be_finite(self, bad: float) -> None:
        """A NaN or infinite level would make every comparison false."""
        with pytest.raises(ConfigurationError, match="finite"):
            zscore_positions(_z([0.0]), entry=bad)
        with pytest.raises(ConfigurationError, match="finite"):
            zscore_positions(_z([0.0]), stop=bad)
