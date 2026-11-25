"""Tests for the security market line.

The golden numbers come from assets built to lie exactly on, or at a known distance
from, the line, and from closed forms written out in this file with plain numpy.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.regression import SmlResult, capm_regression, security_market_line

_YEAR = 252
_BETAS = [0.5, 1.0, 1.5, 2.0]
_COLUMNS = ["low", "mid", "high", "levered"]


def _market(n_obs: int = 250) -> pd.Series:
    """Return a seeded daily market return series."""
    rng = np.random.default_rng(13)
    dates = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    return pd.Series(rng.normal(0.0004, 0.01, n_obs), index=dates, name="market")


def _assets(
    market: pd.Series,
    alphas: list[float] | None = None,
    risk_free: float = 0.0,
    noise: float = 0.0,
) -> pd.DataFrame:
    """Return assets with the given betas and per-period alphas, plus optional noise."""
    rng = np.random.default_rng(29)
    offsets = alphas or [0.0] * len(_BETAS)
    excess = market.to_numpy() - risk_free
    columns = {
        name: risk_free + alpha + beta * excess + noise * rng.normal(size=len(market))
        for name, beta, alpha in zip(_COLUMNS, _BETAS, offsets, strict=True)
    }
    return pd.DataFrame(columns, index=market.index)


def _slope_and_intercept(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Return the closed-form slope and intercept of y on x."""
    slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    return slope, float(y.mean() - slope * x.mean())


class TestAssetsOnTheLine:
    """Assets whose return is exactly beta times the market's."""

    def test_the_betas_are_recovered_and_no_asset_has_alpha(self) -> None:
        """Every alpha against the line is zero."""
        market = _market()
        result = security_market_line(_assets(market), market)
        assert result.table["beta"].to_numpy() == pytest.approx(_BETAS, rel=1e-9)
        assert result.table["alpha_vs_sml"].to_numpy() == pytest.approx(0.0, abs=1e-10)

    def test_the_slope_is_the_market_premium_and_the_intercept_is_zero(self) -> None:
        """The cross-sectional line is the textbook security market line."""
        market = _market()
        result = security_market_line(_assets(market), market)
        premium = float(market.mean()) * _YEAR
        assert result.market_premium == pytest.approx(premium, rel=1e-9)
        assert result.slope == pytest.approx(premium, rel=1e-9)
        assert result.intercept == pytest.approx(0.0, abs=1e-10)

    def test_the_table_layout(self) -> None:
        """One row per asset and the four documented columns, in order."""
        market = _market()
        result = security_market_line(_assets(market), market)
        assert list(result.table.index) == _COLUMNS
        assert list(result.table.columns) == [
            "beta",
            "annualized_mean_excess_return",
            "capm_expected_excess_return",
            "alpha_vs_sml",
        ]

    def test_the_expected_return_is_beta_times_the_market_premium(self) -> None:
        """CAPM expects each asset to earn its beta times the premium."""
        market = _market()
        result = security_market_line(_assets(market), market)
        expected = result.table["beta"] * result.market_premium
        assert result.table["capm_expected_excess_return"].to_numpy() == pytest.approx(
            expected.to_numpy(), rel=1e-12
        )
        assert result.table["annualized_mean_excess_return"].to_numpy() == (
            pytest.approx(
                _assets(market).mean().to_numpy() * _YEAR,
                rel=1e-9,
            )
        )


class TestAssetsOffTheLine:
    """Assets with a known alpha sit that far above or below the line."""

    def test_alpha_against_the_line_is_the_annualised_alpha(self) -> None:
        """Adding a daily alpha moves an asset up by alpha times periods per year."""
        market = _market()
        alphas = [0.0001, -0.0002, 0.0003, 0.0]
        result = security_market_line(_assets(market, alphas=alphas), market)
        assert result.table["alpha_vs_sml"].to_numpy() == pytest.approx(
            np.array(alphas) * _YEAR, rel=1e-9
        )

    def test_the_line_is_the_closed_form_regression_across_assets(self) -> None:
        """Slope and intercept equal the cross-sectional OLS of return on beta."""
        market = _market()
        assets = _assets(market, alphas=[0.0001, -0.0002, 0.0003, 0.0])
        result = security_market_line(assets, market)
        slope, intercept = _slope_and_intercept(
            result.table["annualized_mean_excess_return"].to_numpy(),
            result.table["beta"].to_numpy(),
        )
        assert result.slope == pytest.approx(slope, rel=1e-9)
        assert result.intercept == pytest.approx(intercept, rel=1e-9)

    def test_each_row_agrees_with_the_time_series_capm_regression(self) -> None:
        """Alpha against the line is Jensen's alpha, annualised."""
        market = _market()
        assets = _assets(market, alphas=[0.0001, -0.0002, 0.0003, 0.0], noise=0.003)
        result = security_market_line(assets, market, risk_free_rate=0.0001)
        for name in _COLUMNS:
            capm = capm_regression(assets[name], market, risk_free_rate=0.0001)
            row = result.table.loc[name]
            assert row["beta"] == pytest.approx(capm.beta, rel=1e-9)
            assert row["alpha_vs_sml"] == pytest.approx(capm.annualized_alpha, rel=1e-9)


class TestRiskFreeRate:
    """Excess returns are measured over the risk-free rate."""

    def test_the_premium_is_the_annualised_mean_excess_market_return(self) -> None:
        """The market premium subtracts the per-period rate before annualising."""
        market = _market()
        rate = 0.0002
        result = security_market_line(
            _assets(market, risk_free=rate), market, risk_free_rate=rate
        )
        assert result.market_premium == pytest.approx(
            (float(market.mean()) - rate) * _YEAR, rel=1e-9
        )
        assert result.table["alpha_vs_sml"].to_numpy() == pytest.approx(0.0, abs=1e-9)
        assert result.slope == pytest.approx(result.market_premium, rel=1e-9)

    def test_a_series_rate_is_subtracted_period_by_period(self) -> None:
        """A time-varying rate gives the same line as the equivalent excess data."""
        market = _market()
        rng = np.random.default_rng(2)
        rate = pd.Series(rng.uniform(0.0, 0.0004, len(market)), index=market.index)
        assets = _assets(market, noise=0.002)
        result = security_market_line(assets, market, risk_free_rate=rate)
        excess_market = (market - rate).to_numpy()
        betas = [
            _slope_and_intercept((assets[name] - rate).to_numpy(), excess_market)[0]
            for name in _COLUMNS
        ]
        assert result.table["beta"].to_numpy() == pytest.approx(betas, rel=1e-9)
        assert result.market_premium == pytest.approx(
            excess_market.mean() * _YEAR, rel=1e-9
        )


class TestSmlValidation:
    """Every bad input raises the documented error."""

    def test_fewer_than_three_assets_raise(self) -> None:
        """A line through two points has no residual to judge."""
        market = _market()
        with pytest.raises(DataError, match="at least 3 assets"):
            security_market_line(_assets(market).iloc[:, :2], market)

    def test_assets_with_the_same_beta_cannot_define_a_slope(self) -> None:
        """The cross-section needs betas that differ."""
        market = _market()
        same = pd.DataFrame(dict.fromkeys("abc", market))
        with pytest.raises(DataError, match="collinear"):
            security_market_line(same, market)

    def test_assets_must_be_a_data_frame(self) -> None:
        """A single series is not a panel of assets."""
        market = _market()
        with pytest.raises(DataError, match="DataFrame"):
            security_market_line(market, market)  # type: ignore[arg-type]

    def test_duplicate_asset_names_raise(self) -> None:
        """The table is indexed by asset name."""
        market = _market()
        assets = _assets(market)
        assets.columns = pd.Index(["a", "a", "b", "c"])
        with pytest.raises(DataError, match="unique"):
            security_market_line(assets, market)

    def test_misaligned_indexes_raise(self) -> None:
        """Assets and market must share their dates."""
        market = _market()
        with pytest.raises(DataError, match="index"):
            security_market_line(_assets(market), market.shift(freq="D"))

    def test_missing_asset_returns_raise(self) -> None:
        """NaN is rejected rather than skipped."""
        market = _market()
        assets = _assets(market)
        assets.iloc[5, 1] = np.nan
        with pytest.raises(DataError, match="finite"):
            security_market_line(assets, market)

    def test_a_constant_market_raises(self) -> None:
        """Beta is undefined against a market that never moves."""
        market = _market()
        flat = pd.Series(0.001, index=market.index)
        with pytest.raises(DataError, match="constant"):
            security_market_line(_assets(market), flat)

    @pytest.mark.parametrize("periods", [0, 2.5])
    def test_a_bad_periods_per_year_raises(self, periods: float) -> None:
        """Annualisation needs a positive whole number of periods."""
        market = _market()
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            security_market_line(
                _assets(market),
                market,
                periods_per_year=periods,  # type: ignore[arg-type]
            )

    def test_a_bad_risk_free_rate_raises(self) -> None:
        """The scalar rate has to be a finite number."""
        market = _market()
        with pytest.raises(ConfigurationError, match="risk_free_rate"):
            security_market_line(_assets(market), market, risk_free_rate=float("nan"))

    def test_the_result_is_immutable_and_exported(self) -> None:
        """A shared result cannot be changed behind a caller's back."""
        market = _market()
        result = security_market_line(_assets(market), market)
        assert isinstance(result, SmlResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.slope = 0.0  # type: ignore[misc]
