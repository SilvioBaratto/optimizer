"""Tests for FX return decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
        result = decompose_fx_returns(
            local_prices, base_prices, fx_rates, cmap, "EUR"
        )

        assert isinstance(result, FxReturnDecomposition)
        assert isinstance(result.total_returns, pd.DataFrame)
        assert isinstance(result.local_returns, pd.DataFrame)
        assert isinstance(result.fx_returns, pd.DataFrame)
        assert isinstance(result.cross_terms, pd.DataFrame)

    def test_base_currency_fx_returns_zero(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(
            local_prices, base_prices, fx_rates, cmap, "EUR"
        )

        # EUR ticker should have zero FX returns
        np.testing.assert_array_equal(
            result.fx_returns["ORA.PA"].values,
            np.zeros(len(result.fx_returns)),
        )

    def test_foreign_ticker_has_nonzero_fx(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(
            local_prices, base_prices, fx_rates, cmap, "EUR"
        )

        # GBP ticker should have non-zero FX returns
        assert result.fx_returns["LLOY.L"].abs().sum() > 0
        assert result.fx_returns["SPY"].abs().sum() > 0

    def test_algebraic_identity(self) -> None:
        """Verify r_total ≈ r_local + r_fx + r_local * r_fx."""
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(
            local_prices, base_prices, fx_rates, cmap, "EUR"
        )

        reconstructed = (
            result.local_returns + result.fx_returns + result.cross_terms
        )

        # The identity should hold closely (not exactly due to
        # discrete compounding and alignment)
        diff = (result.total_returns - reconstructed).abs()
        # Allow tolerance — the identity is approximate because
        # pct_change on base_prices = pct_change(local * fx) includes
        # the cross term implicitly
        assert diff.max().max() < 0.01, (
            f"Max decomposition error: {diff.max().max():.6f}"
        )

    def test_shapes_consistent(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(
            local_prices, base_prices, fx_rates, cmap, "EUR"
        )

        # All return DataFrames should have same shape
        assert result.total_returns.shape == result.local_returns.shape
        assert result.total_returns.shape == result.fx_returns.shape
        assert result.total_returns.shape == result.cross_terms.shape

        # Should be 1 fewer row than prices (pct_change drops first)
        assert result.total_returns.shape[0] == len(local_prices) - 1

    def test_metadata_stored(self) -> None:
        local_prices, base_prices, fx_rates, cmap = self._make_fixtures()
        result = decompose_fx_returns(
            local_prices, base_prices, fx_rates, cmap, "EUR"
        )

        assert result.currency_map == cmap
        assert result.base_currency == "EUR"
