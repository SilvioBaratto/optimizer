"""Tests for FX return decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.fx._decomposition import FxReturnDecomposition, decompose_fx_returns


class TestDecomposeReturns:
    """Tests for decompose_fx_returns()."""

    def _make_fixtures(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
        """Build local prices, base prices, FX rates, and currency map."""
        dates = pd.bdate_range("2024-01-02", periods=11)
        rng = np.random.default_rng(42)

        local_prices = pd.DataFrame(
            {
                "LLOY.L": 50.0 + rng.standard_normal(11).cumsum(),
                "ORA.PA": 90.0 + rng.standard_normal(11).cumsum(),
                "SPY": 480.0 + rng.standard_normal(11).cumsum(),
            },
            index=dates,
        )

        fx_rates = pd.DataFrame(
            {
                "GBP": np.linspace(1.15, 1.17, 11),
                "USD": np.linspace(0.91, 0.93, 11),
            },
            index=dates,
        )

        currency_map = {"LLOY.L": "GBP", "ORA.PA": "EUR", "SPY": "USD"}

        # Build base prices by multiplying local × FX
        base_prices = local_prices.copy()
        base_prices["LLOY.L"] = local_prices["LLOY.L"] * fx_rates["GBP"]
        base_prices["SPY"] = local_prices["SPY"] * fx_rates["USD"]
        # ORA.PA is EUR (base) — unchanged

        return local_prices, base_prices, fx_rates, currency_map

    def test_decomposition_returns_correct_types(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        assert isinstance(result, FxReturnDecomposition)
        assert isinstance(result.total_returns, pd.DataFrame)
        assert isinstance(result.local_returns, pd.DataFrame)
        assert isinstance(result.fx_returns, pd.DataFrame)
        assert isinstance(result.cross_terms, pd.DataFrame)

    def test_base_currency_fx_returns_zero(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        # EUR ticker should have zero FX returns
        np.testing.assert_array_equal(
            result.fx_returns["ORA.PA"].values,
            np.zeros(len(result.fx_returns)),
        )

    def test_foreign_ticker_has_nonzero_fx(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        # GBP ticker should have non-zero FX returns
        assert result.fx_returns["LLOY.L"].abs().sum() > 0
        assert result.fx_returns["SPY"].abs().sum() > 0

    def test_algebraic_identity(self) -> None:
        """Verify r_total == r_local + r_fx + r_local * r_fx EXACTLY.

        Because base_price = local_price * rate, we have
        (1 + r_total) = (1 + r_local)(1 + r_fx), so the identity is exact
        (up to floating-point) — NOT merely approximate.  A previous
        implementation zeroed the first-row FX return, breaking this.
        """
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        reconstructed = result.local_returns + result.fx_returns + result.cross_terms

        diff = (result.total_returns - reconstructed).abs()
        assert diff.max().max() < 1e-10, (
            f"Max decomposition error: {diff.max().max():.3e}"
        )

    def test_first_row_fx_return_not_dropped(self) -> None:
        """Regression: the first return date must carry the real FX return.

        A large rate jump on the first return date must be reflected in
        ``fx_returns`` (previously it was silently set to 0.0, breaking
        the identity by the full magnitude of that jump).
        """
        dates = pd.bdate_range("2024-01-02", periods=6)
        local = pd.DataFrame({"LLOY.L": [50, 51, 52, 53, 54, 55.0]}, index=dates)
        rate = pd.Series([1.10, 1.30, 1.31, 1.32, 1.33, 1.34], index=dates)
        fx = pd.DataFrame({"GBP": rate})
        base = local.copy()
        base["LLOY.L"] = local["LLOY.L"] * rate
        cmap = {"LLOY.L": "GBP"}

        result = decompose_fx_returns(local, base, fx, cmap, "EUR")

        assert result.fx_returns["LLOY.L"].iloc[0] == pytest.approx(1.30 / 1.10 - 1.0)
        recon = result.local_returns + result.fx_returns + result.cross_terms
        assert (result.total_returns - recon).abs().max().max() < 1e-10

    def test_case_insensitive_currency_columns(self) -> None:
        """FX columns quoted in lower-case must still match the currency map."""
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        fx_lower = fx_rates.rename(columns=str.lower)
        result = decompose_fx_returns(local_prices, base_prices, fx_lower, cmap, "EUR")

        assert result.fx_returns["LLOY.L"].abs().sum() > 0
        assert result.fx_returns["SPY"].abs().sum() > 0

    def test_hedged_returns_full_hedge_equals_local(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        hedged = result.hedged_returns(hedge_ratio=1.0)
        pd.testing.assert_frame_equal(hedged, result.local_returns)

    def test_hedged_returns_zero_hedge_equals_total(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        unhedged = result.hedged_returns(hedge_ratio=0.0)
        # r_local + r_fx + r_cross == r_total (exact identity)
        pd.testing.assert_frame_equal(
            unhedged, result.total_returns, check_exact=False, atol=1e-10
        )

    def test_hedged_returns_partial(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        half = result.hedged_returns(hedge_ratio=0.5)
        expected = result.local_returns + 0.5 * (result.fx_returns + result.cross_terms)
        pd.testing.assert_frame_equal(half, expected)

    def test_hedged_returns_rejects_non_finite(self) -> None:
        from optimizer.exceptions import DataError

        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        with pytest.raises(DataError, match="finite"):
            result.hedged_returns(hedge_ratio=float("nan"))
        with pytest.raises(DataError, match="finite"):
            result.hedged_returns(hedge_ratio=float("inf"))

    def test_cumulative_contributions(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        summary = result.cumulative_contributions()
        assert list(summary.columns) == ["local", "fx", "cross", "total"]
        assert set(summary.index) == set(local_prices.columns)
        # Base-currency ticker has no FX contribution.
        assert summary.loc["ORA.PA", "fx"] == 0.0
        # total column matches compounded total_returns.
        expected_total = (1.0 + result.total_returns).prod() - 1.0
        pd.testing.assert_series_equal(
            summary["total"], expected_total, check_names=False
        )

    def test_shapes_consistent(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        # All return DataFrames should have same shape
        assert result.total_returns.shape == result.local_returns.shape
        assert result.total_returns.shape == result.fx_returns.shape
        assert result.total_returns.shape == result.cross_terms.shape

        # Should be 1 fewer row than prices (pct_change drops first)
        assert result.total_returns.shape[0] == len(local_prices) - 1

    def test_metadata_stored(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(local_prices, base_prices, fx_rates, cmap, "EUR")

        assert result.currency_map == cmap
        assert result.base_currency == "EUR"


class TestDecomposeMinorUnits:
    """Minor-unit tickers must resolve to their major FX column.

    Regression: ``ZAc`` upper-cases to ``ZAC`` which does not match the ``ZAR``
    rate column, so the FX return was silently dropped and the identity broke
    for Johannesburg / Tel Aviv listings.
    """

    def test_zac_ticker_fx_return_not_dropped(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=6)
        # Local prices quoted in cents (ZAc).
        local = pd.DataFrame(
            {"NPN.JO": [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0]}, index=dates
        )
        rate = pd.Series([0.050, 0.051, 0.052, 0.051, 0.053, 0.052], index=dates)
        fx = pd.DataFrame({"ZAR": rate})
        # base_price = (local / 100) * ZAR-rate  (what FxPriceConverter yields).
        base = (local / 100.0).multiply(rate, axis=0)
        cmap = {"NPN.JO": "ZAc"}

        result = decompose_fx_returns(local, base, fx, cmap, "EUR")

        # FX return must be picked up from the ZAR column (non-zero).
        assert result.fx_returns["NPN.JO"].abs().sum() > 0
        # Exact algebraic identity holds (scale cancels in returns).
        recon = result.local_returns + result.fx_returns + result.cross_terms
        assert (result.total_returns - recon).abs().max().max() < 1e-10

    def test_gbp_base_pence_ticker_zero_fx(self) -> None:
        """A pence ticker with GBP base has major == base -> zero FX return."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        local = pd.DataFrame(
            {"LLOY.L": [500.0, 510.0, 520.0, 530.0, 540.0]}, index=dates
        )
        base = local / 100.0  # rescaled to pounds, no FX (base is GBP)
        cmap = {"LLOY.L": "GBp"}

        result = decompose_fx_returns(
            local, base, pd.DataFrame(index=dates), cmap, "GBP"
        )

        np.testing.assert_array_equal(
            result.fx_returns["LLOY.L"].values, np.zeros(len(result.fx_returns))
        )
