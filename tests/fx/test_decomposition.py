"""Tests for FX return decomposition."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from optimizer.fx import FxPriceConverter
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


class TestDecomposeDecimalPrices:
    """DB ``Numeric`` columns arrive as ``Decimal`` (object dtype)."""

    def test_decimal_local_prices_pair_with_converter_output(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=6)
        local = pd.DataFrame(
            {
                "LLOY.L": [Decimal(v) for v in ("50", "51", "52", "53", "54", "55")],
                "SPY": [Decimal(v) for v in ("480", "482", "481", "485", "487", "490")],
            },
            index=dates,
        )
        assert local["SPY"].dtype == object
        fx = pd.DataFrame(
            {
                "GBP": [1.10, 1.11, 1.12, 1.11, 1.13, 1.14],
                "USD": [0.91, 0.92, 0.92, 0.93, 0.92, 0.93],
            },
            index=dates,
        )
        cmap = {"LLOY.L": "GBP", "SPY": "USD"}
        converter = FxPriceConverter(
            base_currency="EUR", currency_map=cmap, fx_rates=fx
        )
        base = converter.fit_transform(local)

        result = decompose_fx_returns(local, base, converter.fx_aligned_, cmap, "EUR")

        recon = result.local_returns + result.fx_returns + result.cross_terms
        assert (result.total_returns - recon).abs().max().max() < 1e-10
        assert result.local_returns["SPY"].iloc[0] == pytest.approx(482 / 480 - 1.0)


class TestDecomposeUnknownFx:
    """An unknown FX rate must not be booked as a 0% FX move."""

    @staticmethod
    def _frames(
        rate_values: list[float],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
        dates = pd.bdate_range("2024-01-02", periods=len(rate_values))
        rng = np.random.default_rng(7)
        local = pd.DataFrame(
            {"A": 100.0 + rng.standard_normal(len(dates)).cumsum()}, index=dates
        )
        rate = pd.Series(rate_values, index=dates)
        base = local.multiply(rate, axis=0)
        return local, base, pd.DataFrame({"USD": rate}), {"A": "USD"}

    @pytest.mark.parametrize(
        "rate_values",
        [
            pytest.param(
                [np.nan] * 4 + [0.90, 0.91, 0.92, 0.91, 0.93, 0.94],
                id="rates-start-late",
            ),
            pytest.param(
                [0.90, 0.91, 0.92, np.nan, np.nan, 0.93, 0.92, 0.94, 0.95, 0.94],
                id="gap-beyond-fill-limit",
            ),
        ],
    )
    def test_unhedged_returns_match_total_where_fx_unknown(
        self, rate_values: list[float]
    ) -> None:
        local, base, fx, cmap = self._frames(rate_values)
        result = decompose_fx_returns(local, base, fx, cmap, "EUR")

        unhedged = result.hedged_returns(hedge_ratio=0.0)
        pd.testing.assert_frame_equal(
            unhedged, result.total_returns, check_exact=False, atol=1e-10
        )

    @pytest.mark.parametrize(
        "rate_values",
        [
            pytest.param(
                [np.nan] * 4 + [0.90, 0.91, 0.92, 0.91, 0.93, 0.94],
                id="rates-start-late",
            ),
            pytest.param(
                [0.90, 0.91, 0.92, np.nan, np.nan, 0.93, 0.92, 0.94, 0.95, 0.94],
                id="gap-beyond-fill-limit",
            ),
        ],
    )
    def test_cumulative_components_reconcile_where_fx_unknown(
        self, rate_values: list[float]
    ) -> None:
        local, base, fx, cmap = self._frames(rate_values)
        summary = decompose_fx_returns(
            local, base, fx, cmap, "EUR"
        ).cumulative_contributions()

        row = summary.loc["A"]
        assert (1.0 + row["local"]) * (1.0 + row["fx"]) - 1.0 == pytest.approx(
            row["total"], abs=1e-10
        )


class TestDecomposeUnionCalendarGaps:
    """A NaN price (market holiday on a union calendar) must not drop its move."""

    @staticmethod
    def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
        dates = pd.bdate_range("2024-01-02", periods=6)
        local = pd.DataFrame(
            {
                "A": [100.0, 101.0, np.nan, 103.0, 104.0, 104.0],
                "B": [50.0, 51.0, np.nan, 49.0, 52.0, 53.0],
            },
            index=dates,
        )
        rate = pd.Series([0.90, 0.91, 0.93, 0.92, 0.94, 0.95], index=dates)
        base = local.copy()
        base["A"] = local["A"] * rate
        return local, base, pd.DataFrame({"USD": rate}), {"A": "USD", "B": "EUR"}

    def test_gap_move_lands_on_first_observed_date(self) -> None:
        local, base, fx, cmap = self._frames()
        result = decompose_fx_returns(local, base, fx, cmap, "EUR")
        dates = local.index

        assert np.isnan(result.local_returns.loc[dates[2], "A"])
        assert result.local_returns.loc[dates[3], "A"] == pytest.approx(103 / 101 - 1)
        assert result.fx_returns.loc[dates[3], "A"] == pytest.approx(0.92 / 0.91 - 1)

    def test_cumulative_return_covers_full_price_move(self) -> None:
        local, base, fx, cmap = self._frames()
        summary = decompose_fx_returns(
            local, base, fx, cmap, "EUR"
        ).cumulative_contributions()

        assert summary.loc["A", "local"] == pytest.approx(104 / 100 - 1)
        assert summary.loc["A", "fx"] == pytest.approx(0.95 / 0.90 - 1)
        assert summary.loc["A", "total"] == pytest.approx(
            (104 * 0.95) / (100 * 0.90) - 1
        )
        assert summary.loc["B", "total"] == pytest.approx(53 / 50 - 1)

    def test_identity_holds_across_gap(self) -> None:
        local, base, fx, cmap = self._frames()
        result = decompose_fx_returns(local, base, fx, cmap, "EUR")

        recon = result.local_returns + result.fx_returns + result.cross_terms
        pd.testing.assert_frame_equal(
            result.total_returns, recon, check_exact=False, atol=1e-10
        )
        assert result.total_returns["A"].notna().sum() == 4


class TestCumulativeContributionsNoData:
    """A ticker with no valid returns has an unknown, not a 0%, return."""

    def test_all_nan_returns_stay_nan(self) -> None:
        dates = pd.bdate_range("2024-01-03", periods=4)
        nan_col = pd.Series(np.nan, index=dates)
        valid = pd.Series([0.01, -0.02, 0.03, 0.0], index=dates)
        result = FxReturnDecomposition(
            total_returns=pd.DataFrame({"A": nan_col, "B": valid}),
            local_returns=pd.DataFrame({"A": nan_col, "B": valid}),
            fx_returns=pd.DataFrame({"A": nan_col, "B": 0.0}),
            cross_terms=pd.DataFrame({"A": nan_col, "B": 0.0}),
            currency_map={"A": "USD", "B": "EUR"},
            base_currency="EUR",
        )

        summary = result.cumulative_contributions()

        assert summary.loc["A"].isna().all()
        assert summary.loc["B", "total"] == pytest.approx(
            (1.01 * 0.98 * 1.03 * 1.0) - 1.0
        )

    def test_ticker_without_prices_is_nan_end_to_end(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        local = pd.DataFrame(
            {"NEW": np.nan, "SPY": [480.0, 482.0, 481.0, 485.0, 487.0]}, index=dates
        )
        rate = pd.Series([0.91, 0.92, 0.92, 0.93, 0.92], index=dates)
        base = local.multiply(rate, axis=0)
        cmap = {"NEW": "USD", "SPY": "USD"}

        summary = decompose_fx_returns(
            local, base, pd.DataFrame({"USD": rate}), cmap, "EUR"
        ).cumulative_contributions()

        assert summary.loc["NEW"].isna().all()
        assert summary.loc["SPY", "total"] == pytest.approx(
            (487 * 0.92) / (480 * 0.91) - 1.0
        )
