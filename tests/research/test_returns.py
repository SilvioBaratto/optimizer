"""Tests for research/_returns.py — Cycle 4 §9.2 after-tax returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research._returns import compute_after_tax_returns


@pytest.fixture()
def daily_dates() -> pd.DatetimeIndex:
    """20 business days for a backtest series."""
    return pd.bdate_range("2024-01-01", periods=20)


@pytest.fixture()
def gross(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Constant 0% gross return, so portfolio_value stays at 1.0."""
    return pd.Series(0.0, index=daily_dates, name="returns")


@pytest.fixture()
def prices(daily_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Linearly rising prices for assets A and B."""
    return pd.DataFrame(
        {
            "A": np.linspace(100.0, 119.0, len(daily_dates)),
            "B": np.linspace(50.0, 69.0, len(daily_dates)),
        },
        index=daily_dates,
    )


class TestComputeAfterTaxReturns:
    """Cycle 4 §9.2 — after-tax returns from weight_history realised PnL."""

    def test_when_weight_history_is_none_then_none_returned(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        assert compute_after_tax_returns(gross, None, prices) is None

    def test_when_gross_returns_is_none_then_none_returned(
        self, prices: pd.DataFrame
    ) -> None:
        wh = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex(["2024-01-02"]))
        assert compute_after_tax_returns(None, wh, prices) is None

    def test_when_cold_start_only_then_no_tax_first_period(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        rb_date = gross.index[0]
        wh = pd.DataFrame({"A": [0.6], "B": [0.4]}, index=pd.DatetimeIndex([rb_date]))
        result = compute_after_tax_returns(gross, wh, prices, cost_bps=0.0)
        assert result is not None
        # No tax + no cost → identical to gross
        pd.testing.assert_series_equal(result, gross)

    def test_when_cold_start_with_cost_then_only_txn_cost_deducted(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        rb_date = gross.index[0]
        wh = pd.DataFrame({"A": [0.6], "B": [0.4]}, index=pd.DatetimeIndex([rb_date]))
        # turnover = (|0.6| + |0.4|) / 2 = 0.5; cost = 0.5 * 10/1e4 = 5e-4
        result = compute_after_tax_returns(gross, wh, prices, cost_bps=10.0)
        assert result is not None
        assert result.at[rb_date] == pytest.approx(-5e-4)

    def test_when_two_rebalances_positive_pnl_then_tax_deducted(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        """Hand-computed 2-period, 2-asset case.

        - t1 (cold start): A=0.6, B=0.4, no prior → no tax
        - t2: A=0.4, B=0.4 → sell 0.2 of A
          prices at t1: A=100, B=50; prices at t2: A=110, B=60
          realised_pnl = 0.2 * (110 - 100) + 0 * (60 - 50) = 2.0
          portfolio_value at t2 = 1.0 (gross is 0)
          tax = 0.26 * 2.0 / 1.0 = 0.52
        """
        t1 = pd.Timestamp("2024-01-01")
        t2 = pd.Timestamp("2024-01-15")
        idx = pd.bdate_range(t1, t2)
        gross_local = pd.Series(0.0, index=idx, name="returns")
        prices_local = pd.DataFrame(
            {
                "A": np.linspace(100.0, 110.0, len(idx)),
                "B": np.linspace(50.0, 60.0, len(idx)),
            },
            index=idx,
        )
        wh = pd.DataFrame(
            {"A": [0.6, 0.4], "B": [0.4, 0.4]},
            index=pd.DatetimeIndex([idx[0], idx[-1]]),
        )
        result = compute_after_tax_returns(
            gross_local, wh, prices_local, cost_bps=0.0, tax_rate=0.26
        )
        assert result is not None
        # t1: cold start → no tax, no cost → 0.0
        assert result.at[idx[0]] == pytest.approx(0.0)
        # t2: realised_pnl = 0.2 * (110 - 100) = 2.0; tax = 0.26 * 2.0 / 1.0 = 0.52
        assert result.at[idx[-1]] == pytest.approx(-0.52)

    def test_when_negative_realised_pnl_then_no_tax_deducted(
        self, gross: pd.Series
    ) -> None:
        t1 = pd.Timestamp("2024-01-01")
        t2 = pd.Timestamp("2024-01-15")
        idx = pd.bdate_range(t1, t2)
        gross_local = pd.Series(0.0, index=idx, name="returns")
        # Prices fall: A 100 → 90
        prices_local = pd.DataFrame(
            {
                "A": np.linspace(100.0, 90.0, len(idx)),
                "B": np.linspace(50.0, 50.0, len(idx)),
            },
            index=idx,
        )
        wh = pd.DataFrame(
            {"A": [0.6, 0.4], "B": [0.4, 0.6]},
            index=pd.DatetimeIndex([idx[0], idx[-1]]),
        )
        result = compute_after_tax_returns(
            gross_local, wh, prices_local, cost_bps=0.0, tax_rate=0.26
        )
        assert result is not None
        # Sell A by 0.2; price diff -10 → realised_pnl = -2.0 (negative, NOT taxed)
        assert result.at[idx[-1]] == pytest.approx(0.0)

    def test_when_zero_portfolio_value_then_no_tax_deducted(self) -> None:
        """portfolio_value <= 0 → skip tax (no division by zero)."""
        t1 = pd.Timestamp("2024-01-01")
        t2 = pd.Timestamp("2024-01-15")
        idx = pd.bdate_range(t1, t2)
        # Total wipeout: -100% on first day → cumulative = 0
        gross_local = pd.Series(
            [-1.0] + [0.0] * (len(idx) - 1), index=idx, name="returns"
        )
        prices_local = pd.DataFrame(
            {
                "A": np.linspace(100.0, 110.0, len(idx)),
                "B": np.linspace(50.0, 60.0, len(idx)),
            },
            index=idx,
        )
        wh = pd.DataFrame(
            {"A": [0.6, 0.4], "B": [0.4, 0.6]},
            index=pd.DatetimeIndex([idx[0], idx[-1]]),
        )
        result = compute_after_tax_returns(
            gross_local, wh, prices_local, cost_bps=0.0, tax_rate=0.26
        )
        assert result is not None
        # No tax even with positive PnL because pv = 0
        assert np.isfinite(result.at[idx[-1]])
        assert result.at[idx[-1]] == pytest.approx(0.0)

    def test_when_after_tax_le_net_le_gross(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        rb_dates = [gross.index[0], gross.index[10]]
        wh = pd.DataFrame(
            {"A": [0.5, 0.3], "B": [0.5, 0.7]},
            index=pd.DatetimeIndex(rb_dates),
        )
        result = compute_after_tax_returns(
            gross, wh, prices, cost_bps=10.0, tax_rate=0.26
        )
        assert result is not None
        # at_tax ≤ gross at every date (cost + tax ≥ 0)
        diff = gross - result
        assert (diff >= -1e-12).all()

    def test_when_default_tax_rate_then_026(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        """Default tax rate is 0.26 (Italy capital-gains)."""
        import inspect

        sig = inspect.signature(compute_after_tax_returns)
        assert sig.parameters["tax_rate"].default == 0.26


class TestComputeAfterTaxReturnsCoverage:
    """Issue #555 — gap coverage on top of TestComputeAfterTaxReturns.

    Closes coverage gaps:
    zero turnover, all-loss multi-rebalance, mixed gain/loss only-gains-taxed,
    ``tax_rate=0.0`` no-op, empty input, output index preservation.
    """

    def test_when_zero_turnover_then_after_tax_equals_gross(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        """Constant weight_history across rebalances → no sells → no tax."""
        rb_dates = [gross.index[0], gross.index[10]]
        wh = pd.DataFrame(
            {"A": [0.6, 0.6], "B": [0.4, 0.4]},
            index=pd.DatetimeIndex(rb_dates),
        )
        result = compute_after_tax_returns(
            gross,
            wh,
            prices,
            cost_bps=0.0,
            tax_rate=0.26,
        )
        assert result is not None
        pd.testing.assert_series_equal(result, gross)

    def test_when_all_loss_multi_rebalance_then_after_tax_equals_gross(
        self,
    ) -> None:
        """Three rebalances with strictly falling prices → no tax anywhere."""
        idx = pd.bdate_range("2024-01-01", periods=20)
        gross_local = pd.Series(0.0, index=idx, name="returns")
        prices_local = pd.DataFrame(
            {
                "A": np.linspace(100.0, 80.0, len(idx)),
                "B": np.linspace(50.0, 40.0, len(idx)),
            },
            index=idx,
        )
        rb_dates = [idx[0], idx[7], idx[14]]
        wh = pd.DataFrame(
            {"A": [0.6, 0.4, 0.3], "B": [0.4, 0.6, 0.7]},
            index=pd.DatetimeIndex(rb_dates),
        )
        result = compute_after_tax_returns(
            gross_local,
            wh,
            prices_local,
            cost_bps=0.0,
            tax_rate=0.26,
        )
        assert result is not None
        pd.testing.assert_series_equal(result, gross_local)

    def test_when_mixed_gain_loss_then_only_gains_taxed(self) -> None:
        """Two rebalances: first leg = gain (taxed), second leg = loss (not taxed)."""
        idx = pd.bdate_range("2024-01-01", periods=21)
        gross_local = pd.Series(0.0, index=idx, name="returns")
        # Prices rise to t=10 (gain leg), fall after (loss leg).
        a_prices = np.concatenate(
            [np.linspace(100.0, 120.0, 11), np.linspace(120.0, 105.0, 10)]
        )
        b_prices = np.concatenate(
            [np.linspace(50.0, 60.0, 11), np.linspace(60.0, 55.0, 10)]
        )
        prices_local = pd.DataFrame(
            {"A": a_prices, "B": b_prices},
            index=idx,
        )
        rb_dates = [idx[0], idx[10], idx[20]]
        wh = pd.DataFrame(
            {"A": [0.6, 0.4, 0.5], "B": [0.4, 0.6, 0.5]},
            index=pd.DatetimeIndex(rb_dates),
        )
        result = compute_after_tax_returns(
            gross_local,
            wh,
            prices_local,
            cost_bps=0.0,
            tax_rate=0.26,
        )
        assert result is not None
        # t0: cold start → no tax, no cost.
        assert result.at[rb_dates[0]] == pytest.approx(0.0)
        # t1 gain leg: sell 0.2 of A at 120 vs 100 → realised = 0.2 * 20 = 4.0;
        # tax = 0.26 * 4.0 / 1.0 = 1.04 (drag).
        assert result.at[rb_dates[1]] == pytest.approx(-1.04)
        # t2 loss leg: sell 0.1 of B at 55 vs 60 → realised = 0.1 * (-5) = -0.5
        # → no tax. cost_bps=0 → result = 0.
        assert result.at[rb_dates[2]] == pytest.approx(0.0)

    def test_when_tax_rate_zero_then_after_tax_equals_gross_minus_cost(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        """``tax_rate=0`` removes the tax leg entirely; cost_bps=0 → identity."""
        rb_dates = [gross.index[0], gross.index[10]]
        wh = pd.DataFrame(
            {"A": [0.6, 0.4], "B": [0.4, 0.6]},
            index=pd.DatetimeIndex(rb_dates),
        )
        result = compute_after_tax_returns(
            gross,
            wh,
            prices,
            cost_bps=0.0,
            tax_rate=0.0,
        )
        assert result is not None
        pd.testing.assert_series_equal(result, gross)

    def test_when_empty_gross_returns_then_empty_series_returned(
        self, prices: pd.DataFrame
    ) -> None:
        """Empty gross series + non-empty weight_history → empty output."""
        empty_gross = pd.Series(
            dtype=float,
            index=pd.DatetimeIndex([]),
            name="returns",
        )
        wh = pd.DataFrame(
            {"A": [0.5], "B": [0.5]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )
        result = compute_after_tax_returns(empty_gross, wh, prices)
        assert result is not None
        assert result.empty

    def test_when_called_then_output_index_matches_input(
        self, gross: pd.Series, prices: pd.DataFrame
    ) -> None:
        """Output index identical to ``gross_returns`` — no reindex."""
        rb_dates = [gross.index[0], gross.index[10]]
        wh = pd.DataFrame(
            {"A": [0.6, 0.4], "B": [0.4, 0.6]},
            index=pd.DatetimeIndex(rb_dates),
        )
        result = compute_after_tax_returns(
            gross,
            wh,
            prices,
            cost_bps=0.0,
            tax_rate=0.26,
        )
        assert result is not None
        pd.testing.assert_index_equal(result.index, gross.index)
