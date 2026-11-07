"""The Performance strip annualises with the same bars-per-year rule as the tools.

A NAV marked on every weekday holds about 261 bars a year, not 252. When the tools
scale moments and risk by the calendar but the strip kept a fixed 252, one book showed
two different volatilities depending on where it was read.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from fund.performance import NavPoint, portfolio_metrics
from fund.periods import periods_per_year

_PID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _weekday_points(n: int = 300) -> tuple[list[NavPoint], int]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-01", periods=n)
    navs = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, n))
    points = [
        NavPoint(date=d.date(), nav=float(v), benchmark=None)
        for d, v in zip(dates, navs, strict=True)
    ]
    return points, periods_per_year(dates)


def test_default_annualisation_is_derived_from_the_nav_calendar() -> None:
    """With no explicit factor the strip uses the bars the series holds per year."""
    points, derived = _weekday_points()
    assert derived == 261

    default = portfolio_metrics(_PID, points)
    explicit = portfolio_metrics(_PID, points, annualization_factor=float(derived))

    assert default.volatility == pytest.approx(explicit.volatility)
    assert default.sharpe == pytest.approx(explicit.sharpe)


def test_the_derived_factor_differs_from_the_old_fixed_252() -> None:
    """The weekday calendar must not silently fall back to 252."""
    points, _ = _weekday_points()

    default = portfolio_metrics(_PID, points)
    fixed = portfolio_metrics(_PID, points, annualization_factor=252.0)

    assert default.volatility != pytest.approx(fixed.volatility)


def test_an_explicit_factor_still_wins() -> None:
    """A caller that passes the factor keeps full control of the scale."""
    points, _ = _weekday_points()

    one = portfolio_metrics(_PID, points, annualization_factor=100.0)
    two = portfolio_metrics(_PID, points, annualization_factor=400.0)

    assert two.volatility == pytest.approx(one.volatility * 2.0)


def test_a_short_series_without_a_calendar_falls_back_to_the_default() -> None:
    """One or two points carry no calendar; the strip still renders."""
    points, _ = _weekday_points(2)

    summary = portfolio_metrics(_PID, points)

    assert summary.volatility is None or summary.volatility >= 0.0
