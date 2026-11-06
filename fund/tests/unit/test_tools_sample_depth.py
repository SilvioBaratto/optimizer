"""The common-sample depth rule shared by the allocator and risk tools.

A young listing used to shrink a multi-year panel to its own few bars without the
tools noticing: ``optimize_portfolio`` fitted on about a month of data, ``backtest``
measured the drawdown ceiling over the same month, and ``universe_filter`` dropped
every asset that began even one bar after the oldest. The panels here are synthetic
and injected through ``load_price_frame``, so no database is involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fund.tools import moments as moments_mod
from fund.tools import optimize as optimize_mod
from fund.tools import risk as risk_mod
from fund.tools import universe as universe_mod

_ASOF = "2026-10-30"
_BARS = 1250


def _path(rng: np.random.Generator, n: int, vol: float = 0.01) -> np.ndarray:
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0003, vol, n))


def _series(rng: np.random.Generator, index: pd.DatetimeIndex, n: int, vol=0.01):
    return pd.Series(_path(rng, n, vol), index=index[-n:])


@pytest.fixture
def index() -> pd.DatetimeIndex:
    """About five years of business days ending near the decision bar."""
    return pd.bdate_range("2021-11-05", periods=_BARS)


def _install(monkeypatch, module, frame: pd.DataFrame) -> None:
    def load(session, asof, universe, *, base_currency=None, **kwargs):
        """Serve the synthetic panel the way ``load_price_frame`` serves the DB."""
        columns = [t for t in universe if t in frame.columns]
        missing = [t for t in universe if t not in frame.columns]
        return frame[columns], missing

    monkeypatch.setattr(module, "load_price_frame", load)


def _panel_with_listing(index, young_bars: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "A": _series(rng, index, _BARS),
            "B": _series(rng, index, _BARS, 0.004),
            "NEW": _series(rng, index, young_bars, 0.008),
        }
    )


class TestOptimizeRefusesACollapsedSample:
    """``optimize_portfolio`` must not fit on the youngest listing's few bars."""

    def test_a_listing_with_a_month_of_bars_in_a_five_year_panel_is_refused(
        self, monkeypatch, index
    ) -> None:
        """The error names the young asset instead of fitting on ~24 rows."""
        _install(monkeypatch, optimize_mod, _panel_with_listing(index, 25))

        result = optimize_mod.optimize_portfolio(None, _ASOF, ["A", "B", "NEW"])

        assert result["ok"] is False
        assert "NEW" in result["error"]

    def test_a_listing_below_half_the_median_depth_is_refused(
        self, monkeypatch, index
    ) -> None:
        """A listing past the absolute floor still cuts the sample by more than half."""
        _install(monkeypatch, optimize_mod, _panel_with_listing(index, 500))

        result = optimize_mod.optimize_portfolio(None, _ASOF, ["A", "B", "NEW"])

        assert result["ok"] is False
        assert "NEW" in result["error"]

    def test_a_listing_with_most_of_the_median_depth_is_fitted(
        self, monkeypatch, index
    ) -> None:
        """Only a collapsed sample is refused; a modestly younger listing is fine."""
        _install(monkeypatch, optimize_mod, _panel_with_listing(index, 900))

        result = optimize_mod.optimize_portfolio(None, _ASOF, ["A", "B", "NEW"])

        assert result["ok"] is True
        assert result["data"]["n_observations"] == 899


class TestBacktestRefusesACollapsedSample:
    """``backtest`` must not measure a drawdown over the youngest listing's bars."""

    def test_a_young_listing_cannot_hide_a_deep_drawdown(
        self, monkeypatch, index
    ) -> None:
        """Moving 5% into a month-old listing must not turn a failing book green."""
        _install(monkeypatch, risk_mod, _panel_with_listing(index, 25))

        result = risk_mod.backtest(
            None,
            _ASOF,
            {"A": 0.9, "B": 0.05, "NEW": 0.05},
            window={"train_size": 756},
            budget=1.0,
        )

        assert result["ok"] is False
        assert "NEW" in result["error"]


class TestMomentsAndOptimizerShareTheRule:
    """The depth floor is one rule, so the PM is never shown a different sample."""

    def test_estimate_moments_drops_what_optimize_would_refuse(
        self, monkeypatch, index
    ) -> None:
        """The tool that drops a young name and the tool that refuses it agree."""
        _install(monkeypatch, moments_mod, _panel_with_listing(index, 500))

        result = moments_mod.estimate_moments(None, _ASOF, ["A", "B", "NEW"])

        assert result["ok"] is True
        assert result["data"]["dropped"] == ["NEW"]


class TestUniverseFilterKeepsLateStarters:
    """``universe_filter`` drops only genuinely short histories."""

    def _frame(self, index) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "A": _series(rng, index, _BARS),
                "B": _series(rng, index, _BARS, 0.012),
                "C": _series(rng, index, _BARS - 1, 0.009),
                "D": _series(rng, index, 1000, 0.011),
            }
        )

    def test_assets_starting_a_bar_or_a_year_late_survive(
        self, monkeypatch, index
    ) -> None:
        """Four years against five is not a short history."""
        _install(monkeypatch, universe_mod, self._frame(index))

        result = universe_mod.universe_filter(None, _ASOF, ["A", "B", "C", "D"])

        assert result["data"] == ["A", "B", "C", "D"]

    def test_a_month_old_listing_is_dropped_and_the_rest_survive(
        self, monkeypatch, index
    ) -> None:
        """The short history is screened out without costing the others."""
        rng = np.random.default_rng(5)
        frame = self._frame(index)
        frame["NEW"] = _series(rng, index, 25, 0.008)
        _install(monkeypatch, universe_mod, frame)

        result = universe_mod.universe_filter(None, _ASOF, ["A", "B", "C", "D", "NEW"])

        assert result["data"] == ["A", "B", "C", "D"]

    def test_a_foreign_listing_whose_fx_series_starts_later_survives(
        self, monkeypatch, index
    ) -> None:
        """FX conversion leaves leading NaNs on a column; a few of them are not a
        short history."""
        frame = self._frame(index)
        frame.loc[frame.index[:3], "B"] = np.nan
        _install(monkeypatch, universe_mod, frame)

        result = universe_mod.universe_filter(None, _ASOF, ["A", "B", "C", "D"])

        assert result["data"] == ["A", "B", "C", "D"]

    def test_an_asset_that_stopped_quoting_is_still_dropped(
        self, monkeypatch, index
    ) -> None:
        """Trailing silence is not forgiven; only a late start is."""
        frame = self._frame(index)
        frame.loc[frame.index[-30:], "C"] = np.nan
        _install(monkeypatch, universe_mod, frame)

        result = universe_mod.universe_filter(None, _ASOF, ["A", "B", "C", "D"])

        assert result["data"] == ["A", "B", "D"]


class TestDroppedColumnsAreReported:
    """A ticker the returns step silently drops must not vanish from the result."""

    def test_an_all_nan_column_is_reported_missing_by_optimize(
        self, monkeypatch, index
    ) -> None:
        """FX conversion can blank a whole column; the caller must be told."""
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {
                "A": _series(rng, index, _BARS),
                "B": _series(rng, index, _BARS, 0.004),
                "GONE": pd.Series(np.nan, index=index),
            }
        )
        _install(monkeypatch, optimize_mod, frame)

        result = optimize_mod.optimize_portfolio(None, _ASOF, ["A", "B", "GONE"])

        assert result["ok"] is True
        assert "GONE" in result["data"]["missing"]
        assert result["data"]["assets"] == ["A", "B"]

    def test_an_all_nan_column_is_reported_missing_by_estimate_moments(
        self, monkeypatch, index
    ) -> None:
        """The moments tool lists it with the unpriced tickers, not in ``dropped``."""
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {
                "A": _series(rng, index, _BARS),
                "B": _series(rng, index, _BARS, 0.004),
                "GONE": pd.Series(np.nan, index=index),
            }
        )
        _install(monkeypatch, moments_mod, frame)

        result = moments_mod.estimate_moments(None, _ASOF, ["A", "B", "GONE"])

        assert result["ok"] is True
        assert result["data"]["missing"] == ["GONE"]
        assert result["data"]["assets"] == ["A", "B"]

    def test_a_weighted_all_nan_column_is_an_error_for_backtest(
        self, monkeypatch, index
    ) -> None:
        """Dropping a weighted name would evaluate its share as idle cash."""
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {
                "A": _series(rng, index, _BARS),
                "GONE": pd.Series(np.nan, index=index),
            }
        )
        _install(monkeypatch, risk_mod, frame)

        result = risk_mod.backtest(None, _ASOF, {"A": 0.5, "GONE": 0.5}, budget=1.0)

        assert result["ok"] is False
        assert "GONE" in result["error"]
