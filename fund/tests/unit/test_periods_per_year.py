"""Periods per year is derived from the return index, once, for every annualisation.

A union calendar across venues carries about 260 bars a year (holidays of one venue
are forward-filled zero returns on another), so a fixed 252 understated the annual
mean and volatility and compounded the risk-free rate to a different frequency than
the returns it is compared with. The panels here are synthetic and injected through
``load_price_frame``, so no database is involved.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from fund.tools import moments as moments_mod
from fund.tools import optimize as optimize_mod
from fund.tools import risk as risk_mod
from fund.tools._base import (
    DEFAULT_PERIODS_PER_YEAR,
    MAX_PERIODS_PER_YEAR,
    MIN_PERIODS_PER_YEAR,
    annualized_metrics,
    per_period_rate,
    periods_per_year,
)

_ASOF = "2026-10-30"
_BARS = 1250


class TestPeriodsPerYearHelper:
    """``periods_per_year`` counts the bars a calendar year of the index holds."""

    def test_business_day_calendar_is_about_261_bars_a_year(self) -> None:
        """A weekday-only union calendar yields 261, not the fixed 252."""
        index = pd.bdate_range("2021-11-05", periods=_BARS)

        assert periods_per_year(index) == 261

    def test_an_exchange_calendar_with_holidays_is_below_the_union_calendar(
        self,
    ) -> None:
        """A single exchange's holidays pull the figure toward 252."""
        us = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        index = pd.date_range("2021-11-05", periods=_BARS, freq=us)

        derived = periods_per_year(index)

        assert 248 <= derived <= 253
        assert derived < 261

    def test_every_calendar_day_is_365_bars_a_year(self) -> None:
        """A seven-day-a-week series (crypto, test fixtures) is 365."""
        index = pd.date_range("2024-01-01", periods=80, freq="D")

        assert periods_per_year(index) == 365

    def test_the_result_is_a_plain_int(self) -> None:
        """The figure serialises into a tool reply, so it is an ``int``."""
        derived = periods_per_year(pd.bdate_range("2024-01-01", periods=300))

        assert type(derived) is int

    def test_a_coarse_calendar_is_held_to_the_lower_band_edge(self) -> None:
        """A weekly index is outside the daily band and clamps to its floor."""
        weekly = pd.date_range("2020-01-05", periods=200, freq="W")

        assert periods_per_year(weekly) == MIN_PERIODS_PER_YEAR

    def test_an_intraday_calendar_is_held_to_the_upper_band_edge(self) -> None:
        """An hourly index clamps to the ceiling."""
        hourly = pd.date_range("2024-01-01", periods=2000, freq="h")

        assert periods_per_year(hourly) == MAX_PERIODS_PER_YEAR

    @pytest.mark.parametrize(
        "index",
        [
            pd.RangeIndex(300),
            pd.DatetimeIndex([]),
            pd.DatetimeIndex(["2024-01-02"]),
            pd.DatetimeIndex(["2024-01-02", "2024-01-02"]),
        ],
        ids=["not-dated", "empty", "single-bar", "zero-span"],
    )
    def test_an_index_with_no_calendar_falls_back_to_the_default(self, index) -> None:
        """Without two distinct dates there is nothing to derive from."""
        assert periods_per_year(index) == DEFAULT_PERIODS_PER_YEAR

    def test_a_short_sample_stays_inside_the_band(self) -> None:
        """A handful of bars cannot push the figure out of the sane band."""
        index = pd.bdate_range("2024-01-02", periods=4)

        assert MIN_PERIODS_PER_YEAR <= periods_per_year(index) <= MAX_PERIODS_PER_YEAR


class TestPerPeriodRate:
    """``per_period_rate`` compounds an annual rate down to the return frequency."""

    def test_compounding_the_per_period_rate_recovers_the_annual_rate(self) -> None:
        """A year of per-period rates compounds back to the annual rate."""
        daily = per_period_rate(0.04, 261)

        assert (1.0 + daily) ** 261 == pytest.approx(1.04)

    def test_zero_stays_zero(self) -> None:
        """No risk-free rate stays no risk-free rate."""
        assert per_period_rate(0.0, 261) == 0.0

    def test_a_total_loss_rate_is_refused(self) -> None:
        """A rate of -100% has no real per-period root."""
        with pytest.raises(ValueError, match="annual"):
            per_period_rate(-1.0, 261)


class TestAnnualizedMetrics:
    """``annualized_metrics`` scales by the periods it is given."""

    class _Portfolio:
        """The three per-bar figures the helper reads."""

        mean = 0.001
        standard_deviation = 0.01
        sharpe_ratio = 0.1

    def test_scaling_follows_the_supplied_periods(self) -> None:
        """Mean scales by the periods, volatility and Sharpe by their root."""
        metrics = annualized_metrics(self._Portfolio(), 261)

        assert metrics["annualized_mean"] == pytest.approx(0.001 * 261)
        assert metrics["annualized_standard_deviation"] == pytest.approx(
            0.01 * math.sqrt(261)
        )
        assert metrics["annualized_sharpe_ratio"] == pytest.approx(0.1 * math.sqrt(261))


def _install(monkeypatch, module, frame: pd.DataFrame) -> None:
    """Serve ``frame`` from ``module.load_price_frame`` the way the DB would."""

    def load(session, asof, universe, *, base_currency=None, **kwargs):
        """Split the requested tickers into those the panel holds and the rest."""
        columns = [t for t in universe if t in frame.columns]
        missing = [t for t in universe if t not in frame.columns]
        return frame[columns], missing

    monkeypatch.setattr(module, "load_price_frame", load)


@pytest.fixture
def union_panel() -> pd.DataFrame:
    """Five years of a union calendar (261 bars a year) for three assets."""
    index = pd.bdate_range("2021-11-05", periods=_BARS)
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            name: 100.0 * np.cumprod(1.0 + rng.normal(0.0004, vol, _BARS))
            for name, vol in (("A", 0.01), ("B", 0.006), ("C", 0.012))
        },
        index=index,
    )


class TestToolsUseTheDerivedFigure:
    """Moments, optimizer and backtest all scale by the one derived figure."""

    def test_estimate_moments_reports_the_derived_periods(
        self, monkeypatch, union_panel
    ) -> None:
        """The PM is told 261, the figure that annualises ``mu`` and ``cov``."""
        _install(monkeypatch, moments_mod, union_panel)

        result = moments_mod.estimate_moments(None, _ASOF, ["A", "B", "C"])

        assert result["ok"] is True
        assert result["data"]["periods_per_year"] == 261

    def test_optimize_annualises_with_the_derived_periods(
        self, monkeypatch, union_panel
    ) -> None:
        """The in-sample metrics are the per-bar figures scaled by 261."""
        _install(monkeypatch, optimize_mod, union_panel)

        result = optimize_mod.optimize_portfolio(None, _ASOF, ["A", "B", "C"])

        assert result["ok"] is True
        data = result["data"]
        assert data["periods_per_year"] == 261
        returns = union_panel.pct_change().dropna()
        weights = np.array([data["weights"][c] for c in returns.columns])
        daily = returns.to_numpy() @ weights
        assert data["metrics"]["annualized_mean"] == pytest.approx(
            daily.mean() * 261, rel=1e-6
        )
        assert data["metrics"]["annualized_standard_deviation"] == pytest.approx(
            daily.std(ddof=1) * math.sqrt(261), rel=1e-6
        )

    def test_optimize_compounds_the_annual_risk_free_rate_with_the_same_periods(
        self, monkeypatch, union_panel
    ) -> None:
        """The risk-free rate and the returns it is compared with share a frequency."""
        _install(monkeypatch, optimize_mod, union_panel)

        result = optimize_mod.optimize_portfolio(
            None,
            _ASOF,
            ["A", "B", "C"],
            constraints={"annual_risk_free_rate": 0.04},
        )

        assert result["ok"] is True
        applied = result["data"]["risk_free_rate"]
        assert applied == pytest.approx(1.04 ** (1 / 261) - 1)
        assert (1.0 + applied) ** result["data"]["periods_per_year"] == pytest.approx(
            1.04
        )

    def test_backtest_annualises_with_the_derived_periods(
        self, monkeypatch, union_panel
    ) -> None:
        """The walk-forward metrics are scaled by the panel's own bars a year."""
        _install(monkeypatch, risk_mod, union_panel)
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = risk_mod.backtest(None, _ASOF, weights, budget=1.0)

        assert result["ok"] is True
        data = result["data"]
        assert data["periods_per_year"] == 261
        returns = union_panel.pct_change().dropna()
        daily = pd.Series(
            returns.to_numpy() @ np.array([weights[c] for c in returns.columns]),
            index=returns.index,
        )
        evaluated = daily.loc[data["window_start"] : data["window_end"]]
        assert len(evaluated) == data["n_observations"]
        assert data["metrics"]["annualized_mean"] == pytest.approx(
            evaluated.mean() * 261, rel=1e-6
        )
