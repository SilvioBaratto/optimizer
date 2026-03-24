"""Tests for factor construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorConstructionConfig,
    FactorType,
    align_to_pit,
    compute_all_factors,
    compute_factor,
)


@pytest.fixture()
def fundamentals() -> pd.DataFrame:
    """Synthetic fundamentals for 10 tickers."""
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(10)]
    return pd.DataFrame(
        {
            "market_cap": rng.uniform(1e9, 50e9, 10),
            "book_value": rng.uniform(1e8, 10e9, 10),
            "net_income": rng.uniform(-1e8, 5e9, 10),
            "total_equity": rng.uniform(1e8, 10e9, 10),
            "total_revenue": rng.uniform(1e9, 50e9, 10),
            "total_assets": rng.uniform(5e9, 100e9, 10),
            "gross_profit": rng.uniform(5e8, 20e9, 10),
            "operating_income": rng.uniform(-5e8, 10e9, 10),
            "operating_cashflow": rng.uniform(-1e9, 10e9, 10),
            "ebitda": rng.uniform(1e8, 10e9, 10),
            "enterprise_value": rng.uniform(2e9, 80e9, 10),
            "asset_growth": rng.uniform(-0.1, 0.3, 10),
            "dividend_yield": rng.uniform(0, 0.05, 10),
            "current_price": rng.uniform(10, 500, 10),
        },
        index=pd.Index(tickers, name="ticker"),
    )


@pytest.fixture()
def price_history() -> pd.DataFrame:
    """300 days of synthetic prices for 10 tickers."""
    rng = np.random.default_rng(42)
    n_days, n_tickers = 300, 10
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    returns = rng.normal(0.0005, 0.02, (n_days, n_tickers))
    prices = 100 * np.exp(returns.cumsum(axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture()
def volume_history(price_history: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.integers(100_000, 5_000_000, size=price_history.shape),
        index=price_history.index,
        columns=price_history.columns,
    )


class TestIndividualFactors:
    def test_book_to_price(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.BOOK_TO_PRICE, fundamentals, price_history)
        assert isinstance(result, pd.Series)
        assert len(result) == len(fundamentals)
        assert result.notna().any()

    def test_earnings_yield(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.EARNINGS_YIELD, fundamentals, price_history)
        assert len(result) == len(fundamentals)

    def test_gross_profitability(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(
            FactorType.GROSS_PROFITABILITY, fundamentals, price_history
        )
        assert len(result) == len(fundamentals)

    def test_roe(self, fundamentals: pd.DataFrame, price_history: pd.DataFrame) -> None:
        result = compute_factor(FactorType.ROE, fundamentals, price_history)
        assert len(result) == len(fundamentals)

    def test_asset_growth(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.ASSET_GROWTH, fundamentals, price_history)
        # Raw growth value returned (no sign flip — direction in standardization)
        assert (result == fundamentals["asset_growth"]).all()

    def test_momentum(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.MOMENTUM_12_1, fundamentals, price_history)
        assert isinstance(result, pd.Series)
        assert len(result) == price_history.shape[1]

    def test_volatility(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.VOLATILITY, fundamentals, price_history)
        # Raw annualized vol returned (no sign flip — direction in standardization)
        assert (result >= 0).all()

    def test_beta(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.BETA, fundamentals, price_history)
        assert isinstance(result, pd.Series)
        assert len(result) == price_history.shape[1]

    def test_amihud(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        result = compute_factor(
            FactorType.AMIHUD_ILLIQUIDITY,
            fundamentals,
            price_history,
            volume_history=volume_history,
        )
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()

    def test_dividend_yield(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(FactorType.DIVIDEND_YIELD, fundamentals, price_history)
        assert len(result) == len(fundamentals)

    def test_recommendation_change_none(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(
            FactorType.RECOMMENDATION_CHANGE, fundamentals, price_history
        )
        assert len(result) == 0  # No analyst data

    def test_net_insider_buying_none(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        result = compute_factor(
            FactorType.NET_INSIDER_BUYING, fundamentals, price_history
        )
        assert len(result) == 0  # No insider data


class TestComputeAllFactors:
    def test_core_factors(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        result = compute_all_factors(fundamentals, price_history, volume_history)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(fundamentals)
        assert result.shape[1] > 0

    def test_all_factors(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
        volume_history: pd.DataFrame,
    ) -> None:
        config = FactorConstructionConfig.for_all_factors()
        result = compute_all_factors(
            fundamentals, price_history, volume_history, config=config
        )
        assert isinstance(result, pd.DataFrame)
        # Should have columns for computable factors
        assert result.shape[1] >= 8

    def test_custom_lookbacks(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> None:
        config = FactorConstructionConfig(
            factors=(FactorType.MOMENTUM_12_1,),
            momentum_lookback=126,
            momentum_skip=10,
        )
        result = compute_all_factors(fundamentals, price_history, config=config)
        assert "momentum_12_1" in result.columns


# ---------------------------------------------------------------------------
# Point-in-time alignment tests
# ---------------------------------------------------------------------------


def _make_annual_data(
    tickers: list[str],
    period_dates: list[str],
    values: list[float],
) -> pd.DataFrame:
    """Build a minimal time-series fundamentals DataFrame."""
    rows = []
    for ticker, date, val in zip(tickers, period_dates, values, strict=True):
        rows.append({"ticker": ticker, "period_date": date, "earnings": val})
    return pd.DataFrame(rows)


class TestAlignToPit:
    """Acceptance criteria: correct lag applied per source type."""

    def test_annual_data_unavailable_before_90_days(self) -> None:
        """Annual Dec 31 data is NOT available before March 31 (90 days)."""
        data = _make_annual_data(
            tickers=["AAPL"],
            period_dates=["2023-12-31"],
            values=[5.0],
        )
        # 89 days after Dec 31 → not yet available
        as_of_date = pd.Timestamp("2023-12-31") + pd.Timedelta(days=89)
        result = align_to_pit(data, "period_date", as_of_date, lag_days=90)
        assert result.empty

    def test_annual_data_available_from_march_31(self) -> None:
        """Annual Dec 31 data IS available on March 31 (exactly 90 days)."""
        data = _make_annual_data(
            tickers=["AAPL"],
            period_dates=["2023-12-31"],
            values=[5.0],
        )
        # Exactly 90 days after Dec 31 = March 31
        as_of_date = pd.Timestamp("2023-12-31") + pd.Timedelta(days=90)
        result = align_to_pit(data, "period_date", as_of_date, lag_days=90)
        assert len(result) == 1
        assert result.loc["AAPL", "earnings"] == 5.0

    def test_analyst_data_unavailable_before_5_days(self) -> None:
        """Analyst revision on Monday is NOT available before Friday (5 days)."""
        monday = pd.Timestamp("2024-01-08")  # a Monday
        data = _make_annual_data(
            tickers=["GOOG"],
            period_dates=[str(monday.date())],
            values=[3.0],
        )
        # 4 days later (Friday) → not yet available
        as_of_friday = monday + pd.Timedelta(days=4)
        result = align_to_pit(data, "period_date", as_of_friday, lag_days=5)
        assert result.empty

    def test_analyst_data_available_after_5_days(self) -> None:
        """Analyst revision on Monday IS available 5 days later."""
        monday = pd.Timestamp("2024-01-08")
        data = _make_annual_data(
            tickers=["GOOG"],
            period_dates=[str(monday.date())],
            values=[3.0],
        )
        as_of = monday + pd.Timedelta(days=5)
        result = align_to_pit(data, "period_date", as_of, lag_days=5)
        assert len(result) == 1
        assert result.loc["GOOG", "earnings"] == 3.0

    def test_most_recent_record_returned_per_ticker(self) -> None:
        """When multiple records exist, the most recent available is returned."""
        data = pd.DataFrame(
            [
                {"ticker": "AAPL", "period_date": "2023-03-31", "earnings": 1.0},
                {"ticker": "AAPL", "period_date": "2023-06-30", "earnings": 2.0},
                {"ticker": "AAPL", "period_date": "2023-09-30", "earnings": 3.0},
            ]
        )
        # as_of = 2023-11-15, lag = 45 days → cutoff = 2023-10-01
        # Sep 30 + 45 = Nov 14 ≤ Nov 15: available
        as_of = pd.Timestamp("2023-11-15")
        result = align_to_pit(data, "period_date", as_of, lag_days=45)
        assert len(result) == 1
        assert result.loc["AAPL", "earnings"] == 3.0  # most recent

    def test_multiple_tickers_independent(self) -> None:
        """Each ticker gets its own most-recent available record."""
        data = pd.DataFrame(
            [
                {"ticker": "AAPL", "period_date": "2023-06-30", "earnings": 2.0},
                {"ticker": "AAPL", "period_date": "2023-09-30", "earnings": 3.0},
                {"ticker": "MSFT", "period_date": "2023-06-30", "earnings": 4.0},
                # MSFT Q3 not yet available (released too recently)
                {"ticker": "MSFT", "period_date": "2023-10-15", "earnings": 5.0},
            ]
        )
        # as_of = Nov 15; lag = 45; cutoff = Oct 1
        # AAPL Sep 30 available; MSFT Oct 15 not (Oct 15 + 45 > Nov 15)
        as_of = pd.Timestamp("2023-11-15")
        result = align_to_pit(data, "period_date", as_of, lag_days=45)
        assert result.loc["AAPL", "earnings"] == 3.0
        assert result.loc["MSFT", "earnings"] == 4.0  # falls back to Q2

    def test_empty_result_when_all_too_recent(self) -> None:
        """Returns empty DataFrame when no records pass the cutoff."""
        data = _make_annual_data(
            tickers=["AAPL"],
            period_dates=["2023-12-31"],
            values=[5.0],
        )
        # Only 10 days after Dec 31 → not available with 90-day lag
        as_of = pd.Timestamp("2024-01-10")
        result = align_to_pit(data, "period_date", as_of, lag_days=90)
        assert result.empty

    def test_accepts_string_as_of_date(self) -> None:
        """align_to_pit accepts a string for as_of_date."""
        data = _make_annual_data(
            tickers=["AAPL"],
            period_dates=["2023-12-31"],
            values=[5.0],
        )
        result = align_to_pit(data, "period_date", "2024-04-01", lag_days=90)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# GBX → GBP value factor currency correctness (issue #289)
# ---------------------------------------------------------------------------


class TestValueFactorCurrencyCorrectness:
    """Regression for issue #289: value factors must NOT be 100× deflated.

    After assemble_fundamentals() normalises GBX market data to GBP via
    normalize_fundamentals(), both numerators (financial statement GBP) and
    denominators (market_cap/EV in GBP post-normalisation) are in the same
    unit.  These tests pin the correct ratio values and explicitly document
    the pre-fix 100× deflation so the bug cannot silently re-emerge.
    """

    @pytest.fixture()
    def lse_fundamentals_normalised(self) -> pd.DataFrame:
        """LSE stock fundamentals after GBX → GBP normalisation.

        market_cap and enterprise_value are in GBP (divided by 100).
        Financial statement values (net_income, operating_cashflow,
        total_revenue, ebitda, book_value) are already in GBP from
        their respective sources.
        """
        return pd.DataFrame(
            {
                "market_cap": [1_000_000_000.0],  # 1 B GBP
                "enterprise_value": [1_200_000_000.0],  # 1.2 B GBP
                "net_income": [80_000_000.0],  # 80 M GBP
                "book_value": [400_000_000.0],  # 400 M GBP
                "operating_cashflow": [120_000_000.0],  # 120 M GBP
                "total_revenue": [800_000_000.0],  # 800 M GBP
                "ebitda": [200_000_000.0],  # 200 M GBP
                "total_equity": [400_000_000.0],  # fallback for book_to_price
            },
            index=pd.Index(["BARC.L"], name="ticker"),
        )

    @pytest.fixture()
    def lse_fundamentals_deflated(self) -> pd.DataFrame:
        """Pre-normalisation (buggy) DataFrame: market_cap still in GBX.

        market_cap is 100× too large (GBX not yet divided by 100).
        Numerators are in GBP from financial statements.
        This is the state before PR #286 fix.
        """
        return pd.DataFrame(
            {
                "market_cap": [100_000_000_000.0],  # 100 B GBX (not divided)
                "enterprise_value": [120_000_000_000.0],  # 120 B GBX
                "net_income": [80_000_000.0],  # 80 M GBP (correct)
                "book_value": [400_000_000.0],  # 400 M GBP (correct)
                "operating_cashflow": [120_000_000.0],  # 120 M GBP (correct)
                "total_revenue": [800_000_000.0],  # 800 M GBP (correct)
                "ebitda": [200_000_000.0],  # 200 M GBP (correct)
            },
            index=pd.Index(["BARC.L"], name="ticker"),
        )

    @pytest.fixture()
    def dummy_prices(self) -> pd.DataFrame:
        """Minimal price history (value factors don't use it)."""
        dates = pd.bdate_range("2023-01-01", periods=5)
        return pd.DataFrame(
            {"BARC.L": [100.0, 101.0, 99.0, 102.0, 100.5]},
            index=dates,
        )

    def test_book_to_price_normalised(
        self,
        lse_fundamentals_normalised: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        result = compute_factor(
            FactorType.BOOK_TO_PRICE, lse_fundamentals_normalised, dummy_prices
        )
        # 400M / 1000M = 0.40
        assert result["BARC.L"] == pytest.approx(0.40, rel=1e-6)

    def test_earnings_yield_normalised(
        self,
        lse_fundamentals_normalised: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        result = compute_factor(
            FactorType.EARNINGS_YIELD, lse_fundamentals_normalised, dummy_prices
        )
        # 80M / 1000M = 0.08
        assert result["BARC.L"] == pytest.approx(0.08, rel=1e-6)

    def test_cash_flow_yield_normalised(
        self,
        lse_fundamentals_normalised: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        result = compute_factor(
            FactorType.CASH_FLOW_YIELD, lse_fundamentals_normalised, dummy_prices
        )
        # 120M / 1000M = 0.12
        assert result["BARC.L"] == pytest.approx(0.12, rel=1e-6)

    def test_sales_to_price_normalised(
        self,
        lse_fundamentals_normalised: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        result = compute_factor(
            FactorType.SALES_TO_PRICE, lse_fundamentals_normalised, dummy_prices
        )
        # 800M / 1000M = 0.80
        assert result["BARC.L"] == pytest.approx(0.80, rel=1e-6)

    def test_ebitda_to_ev_normalised(
        self,
        lse_fundamentals_normalised: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        result = compute_factor(
            FactorType.EBITDA_TO_EV, lse_fundamentals_normalised, dummy_prices
        )
        # 200M / 1200M ≈ 0.1667
        assert result["BARC.L"] == pytest.approx(200_000_000 / 1_200_000_000, rel=1e-6)

    def test_book_to_price_deflated_100x(
        self,
        lse_fundamentals_deflated: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        """Pre-fix (GBX market_cap) produces 100× deflated ratio."""
        result = compute_factor(
            FactorType.BOOK_TO_PRICE, lse_fundamentals_deflated, dummy_prices
        )
        # 400M GBP / 100B GBX = 0.004 (100× smaller than correct 0.40)
        assert result["BARC.L"] == pytest.approx(0.004, rel=1e-6)

    def test_earnings_yield_deflated_100x(
        self,
        lse_fundamentals_deflated: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        """Pre-fix earnings_yield is 100× too small."""
        result = compute_factor(
            FactorType.EARNINGS_YIELD, lse_fundamentals_deflated, dummy_prices
        )
        assert result["BARC.L"] == pytest.approx(0.0008, rel=1e-6)

    def test_normalised_ratio_is_100x_larger_than_deflated(
        self,
        lse_fundamentals_normalised: pd.DataFrame,
        lse_fundamentals_deflated: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        """The normalised:deflated ratio must be exactly 100 for value factors."""
        for factor_type in (
            FactorType.BOOK_TO_PRICE,
            FactorType.EARNINGS_YIELD,
            FactorType.CASH_FLOW_YIELD,
            FactorType.SALES_TO_PRICE,
        ):
            normalised = compute_factor(
                factor_type, lse_fundamentals_normalised, dummy_prices
            )
            deflated = compute_factor(
                factor_type, lse_fundamentals_deflated, dummy_prices
            )
            ratio = normalised["BARC.L"] / deflated["BARC.L"]
            assert ratio == pytest.approx(100.0, rel=1e-6), (
                f"{factor_type}: expected 100x ratio, got {ratio:.4f}"
            )

    def test_ebitda_to_ev_normalised_vs_deflated(
        self,
        lse_fundamentals_normalised: pd.DataFrame,
        lse_fundamentals_deflated: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        """EBITDA/EV is also 100× deflated when EV is in GBX."""
        norm = compute_factor(
            FactorType.EBITDA_TO_EV, lse_fundamentals_normalised, dummy_prices
        )
        defl = compute_factor(
            FactorType.EBITDA_TO_EV, lse_fundamentals_deflated, dummy_prices
        )
        assert norm["BARC.L"] / defl["BARC.L"] == pytest.approx(100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# asset_growth currency invariance (issue #290)
# ---------------------------------------------------------------------------


class TestAssetGrowthCurrencyInvariance:
    """Regression for issue #290: asset_growth must be currency-invariant.

    asset_growth = (current_total_assets - prior_total_assets) / abs(prior)
    is a dimensionless ratio.  Multiplying all total_assets values for a
    ticker by any constant k > 0 (e.g. k=100 for GBX->GBP confusion)
    leaves the ratio unchanged.
    """

    @pytest.fixture()
    def fundamentals_with_growth(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"asset_growth": [0.12]},
            index=pd.Index(["BARC.L"], name="ticker"),
        )

    @pytest.fixture()
    def dummy_prices(self) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-01", periods=5)
        return pd.DataFrame(
            {"BARC.L": [100.0, 101.0, 99.0, 102.0, 100.5]},
            index=dates,
        )

    def test_asset_growth_factor_uses_precomputed_ratio(
        self,
        fundamentals_with_growth: pd.DataFrame,
        dummy_prices: pd.DataFrame,
    ) -> None:
        """ASSET_GROWTH factor is raw fundamentals['asset_growth'] (no sign flip)."""
        result = compute_factor(
            FactorType.ASSET_GROWTH, fundamentals_with_growth, dummy_prices
        )
        # Raw value returned: no negation (direction applied in standardization)
        assert result["BARC.L"] == pytest.approx(0.12, rel=1e-9)

    def test_asset_growth_ratio_cancels_currency_scale(self) -> None:
        """Direct proof: (k*a1 - k*a0) / abs(k*a0) == (a1-a0) / abs(a0)."""
        a0, a1, k = 1_000_000_000.0, 1_120_000_000.0, 100.0
        growth_major = (a1 - a0) / abs(a0)
        growth_minor = (k * a1 - k * a0) / abs(k * a0)
        assert growth_major == pytest.approx(growth_minor, rel=1e-12)

    def test_asset_growth_identical_across_scales(self) -> None:
        """asset_growth computed from scaled total_assets is identical."""
        a0_gbp, a1_gbp = 5_000_000_000.0, 5_600_000_000.0
        a0_gbx, a1_gbx = a0_gbp * 100, a1_gbp * 100  # GBX scale
        a0_zac, a1_zac = a0_gbp * 100, a1_gbp * 100  # ZAC scale

        growth_gbp = (a1_gbp - a0_gbp) / abs(a0_gbp)
        growth_gbx = (a1_gbx - a0_gbx) / abs(a0_gbx)
        growth_zac = (a1_zac - a0_zac) / abs(a0_zac)

        assert growth_gbp == pytest.approx(growth_gbx, rel=1e-12)
        assert growth_gbp == pytest.approx(growth_zac, rel=1e-12)
        assert growth_gbp == pytest.approx(0.12, rel=1e-9)


# ---------------------------------------------------------------------------
# Beta market proxy threading (issue #294)
# ---------------------------------------------------------------------------


class TestBetaMarketProxy:
    """Regression for issue #294: beta uses mixed-currency market proxy.

    When ``price_history`` contains stocks from multiple currency zones,
    the default equal-weight cross-sectional mean is contaminated.
    Passing an explicit ``market_returns`` series bypasses this.
    """

    def test_beta_with_explicit_market_returns(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> None:
        """compute_factor passes market_returns through to _compute_beta."""
        mkt = price_history.pct_change().dropna().mean(axis=1)
        result = compute_factor(
            FactorType.BETA,
            fundamentals,
            price_history,
            market_returns=mkt,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == price_history.shape[1]

    def test_beta_external_proxy_differs_from_equal_weight(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> None:
        """A non-equal-weight proxy produces different betas."""
        # Single-stock proxy: just the first column's returns
        single_stock_proxy = price_history.iloc[:, 0].pct_change().dropna()
        result_proxy = compute_factor(
            FactorType.BETA,
            fundamentals,
            price_history,
            market_returns=single_stock_proxy,
        )
        result_default = compute_factor(
            FactorType.BETA,
            fundamentals,
            price_history,
        )
        # They should differ because the proxy differs from equal weight
        assert not result_proxy.equals(result_default)

    def test_equal_weight_proxy_matches_default(
        self,
        fundamentals: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> None:
        """Passing the equal-weight mean explicitly matches the default."""
        returns = price_history.pct_change().dropna()
        # _compute_beta uses tail.mean(axis=1) when market_returns is None.
        # We replicate that here: the lookback default is 252, and we have
        # 300 days of prices -> 299 return rows.
        tail = returns.iloc[-252:]
        mkt = tail.mean(axis=1)

        config = FactorConstructionConfig(factors=(FactorType.BETA,))
        with_proxy = compute_all_factors(
            fundamentals, price_history, config=config, market_returns=mkt
        )
        without_proxy = compute_all_factors(fundamentals, price_history, config=config)
        pd.testing.assert_series_equal(
            with_proxy["beta"].round(10),
            without_proxy["beta"].round(10),
        )

    def test_mixed_currency_proxy_distorts_beta(self) -> None:
        """Adding uncorrelated 'GBX noise' stocks shifts beta estimates.

        Constructs 10 correlated USD stocks plus 5 noise stocks
        simulating uncorrelated GBX micro-caps.  The endogenous
        equal-weight proxy is contaminated by the noise.
        """
        rng = np.random.default_rng(0)
        n_days = 300
        dates = pd.bdate_range("2023-01-01", periods=n_days)

        # 10 correlated stocks ("USD universe")
        common = rng.normal(0, 0.01, n_days)
        usd_returns = pd.DataFrame(
            {f"USD{i}": common + rng.normal(0, 0.005, n_days) for i in range(10)},
            index=dates,
        )

        # 5 noise stocks (simulating GBX micro-caps, low correlation)
        noise_returns = pd.DataFrame(
            {f"GBX{i}": rng.normal(0, 0.02, n_days) for i in range(5)},
            index=dates,
        )

        all_returns = pd.concat([usd_returns, noise_returns], axis=1)
        all_prices = (1 + all_returns).cumprod() * 100

        usd_cols = [c for c in all_prices.columns if c.startswith("USD")]
        usd_prices = all_prices[usd_cols]

        # Clean proxy: equal-weight of USD stocks only
        clean_proxy = usd_returns.mean(axis=1)

        from optimizer.factors._construction import _compute_beta

        beta_clean = _compute_beta(usd_prices, market_returns=clean_proxy)
        beta_polluted = _compute_beta(all_prices)  # endogenous, includes GBX

        # Beta for USD stocks should differ when proxy is contaminated
        beta_clean_vals = beta_clean.reindex(usd_cols).values
        beta_polluted_vals = beta_polluted.reindex(usd_cols).values
        assert not np.allclose(beta_polluted_vals, beta_clean_vals, atol=0.05)


# ---------------------------------------------------------------------------
# Issue #299: sign-flip removed from construction (direction in standardization)
# ---------------------------------------------------------------------------


class TestFactorDirectionConventions:
    """Verify raw construction values are in natural units (no sign flip).

    The direction multiplier (-1 for lower-is-better factors) is applied
    in standardize_factor() via FACTOR_DIRECTION, not in construction.
    """

    def test_volatility_is_non_negative(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        """Annualized volatility must be non-negative in natural units."""
        result = compute_factor(FactorType.VOLATILITY, fundamentals, price_history)
        assert (result >= 0).all(), "volatility must be positive before direction flip"

    def test_volatility_is_annualized(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        """Annualized vol should be in [0.01, 2.0] for typical equity data."""
        result = compute_factor(FactorType.VOLATILITY, fundamentals, price_history)
        valid = result.dropna()
        assert valid.between(0.01, 2.0).all(), (
            f"annualized vol out of expected range:\n{valid.describe()}"
        )

    def test_beta_positive_mean_for_correlated_stocks(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        """Stocks generated with a common drift should have positive mean beta."""
        result = compute_factor(FactorType.BETA, fundamentals, price_history)
        assert result.mean() > 0, (
            f"expected mean beta > 0 for correlated stocks, got {result.mean():.4f}"
        )

    def test_asset_growth_matches_raw_fundamentals(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        """ASSET_GROWTH must equal fundamentals['asset_growth'] exactly."""
        result = compute_factor(FactorType.ASSET_GROWTH, fundamentals, price_history)
        expected = fundamentals["asset_growth"].reindex(result.index)
        pd.testing.assert_series_equal(
            result.rename(None),
            expected.rename(None).astype(float),
        )

    def test_beta_not_negated(
        self, fundamentals: pd.DataFrame, price_history: pd.DataFrame
    ) -> None:
        """Raw beta from _compute_beta must not be negated by compute_factor."""
        from optimizer.factors._construction import _compute_beta

        raw_beta = _compute_beta(price_history)
        api_beta = compute_factor(FactorType.BETA, fundamentals, price_history)

        common = raw_beta.index.intersection(api_beta.index)
        pd.testing.assert_series_equal(
            api_beta.reindex(common).round(10),
            raw_beta.reindex(common).round(10),
        )
