"""Tests for FxPriceConverter."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.fx._converter import FxPriceConverter


@pytest.fixture()
def price_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=10)


@pytest.fixture()
def local_prices(price_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """3 tickers: GBP, EUR (base), USD."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "LLOY.L": 50.0 + rng.standard_normal(10).cumsum(),
            "ORA.PA": 90.0 + rng.standard_normal(10).cumsum(),
            "SPY": 480.0 + rng.standard_normal(10).cumsum(),
        },
        index=price_dates,
    )


@pytest.fixture()
def currency_map() -> dict[str, str]:
    return {"LLOY.L": "GBP", "ORA.PA": "EUR", "SPY": "USD"}


@pytest.fixture()
def fx_rates(price_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """EUR-base rates: GBP→EUR ≈1.16, USD→EUR ≈0.92."""
    return pd.DataFrame(
        {
            "GBP": np.linspace(1.15, 1.17, 10),
            "USD": np.linspace(0.91, 0.93, 10),
        },
        index=price_dates,
    )


class TestFxPriceConverterFit:
    """Tests for FxPriceConverter.fit()."""

    def test_fit_identifies_foreign_tickers(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_rates,
        )
        converter.fit(local_prices)

        assert "LLOY.L" in converter.foreign_tickers_
        assert "SPY" in converter.foreign_tickers_
        assert "ORA.PA" not in converter.foreign_tickers_

    def test_fit_no_missing_currencies(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_rates,
        )
        converter.fit(local_prices)
        assert converter.missing_currencies_ == set()

    def test_fit_missing_currency_warning(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
    ) -> None:
        incomplete_fx = pd.DataFrame(
            {"GBP": np.ones(10)},
            index=local_prices.index,
        )
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=incomplete_fx,
        )
        converter.fit(local_prices)
        assert "USD" in converter.missing_currencies_

    def test_fit_missing_currency_raises_when_required(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
    ) -> None:
        incomplete_fx = pd.DataFrame(
            {"GBP": np.ones(10)},
            index=local_prices.index,
        )
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=incomplete_fx,
            require_full_coverage=True,
        )
        with pytest.raises(DataError, match="Missing FX rates"):
            converter.fit(local_prices)

    def test_fit_rejects_non_dataframe(self) -> None:
        converter = FxPriceConverter()
        with pytest.raises(DataError, match="DataFrame"):
            converter.fit(np.array([[1, 2], [3, 4]]))


class TestFxPriceConverterTransform:
    """Tests for FxPriceConverter.transform()."""

    def test_base_currency_unchanged(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_rates,
        )
        converter.fit(local_prices)
        result = converter.transform(local_prices)

        # EUR ticker should be unchanged
        pd.testing.assert_series_equal(result["ORA.PA"], local_prices["ORA.PA"])

    def test_foreign_tickers_converted(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_rates,
        )
        converter.fit(local_prices)
        result = converter.transform(local_prices)

        # GBP ticker: converted = local × GBP/EUR rate
        expected_lloy = local_prices["LLOY.L"] * fx_rates["GBP"]
        pd.testing.assert_series_equal(
            result["LLOY.L"], expected_lloy, check_names=False
        )

        # USD ticker: converted = local × USD/EUR rate
        expected_spy = local_prices["SPY"] * fx_rates["USD"]
        pd.testing.assert_series_equal(result["SPY"], expected_spy, check_names=False)

    def test_missing_currency_skipped(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
    ) -> None:
        # Only provide GBP rates — USD should be skipped (left unchanged)
        incomplete_fx = pd.DataFrame(
            {"GBP": np.linspace(1.15, 1.17, 10)},
            index=local_prices.index,
        )
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=incomplete_fx,
        )
        converter.fit(local_prices)
        result = converter.transform(local_prices)

        # SPY should remain unchanged (missing USD rate)
        pd.testing.assert_series_equal(result["SPY"], local_prices["SPY"])
        # LLOY.L should be converted
        assert not result["LLOY.L"].equals(local_prices["LLOY.L"])

    def test_transform_not_fitted_raises(self) -> None:
        from sklearn.exceptions import NotFittedError

        converter = FxPriceConverter()
        with pytest.raises(NotFittedError):
            converter.transform(pd.DataFrame({"A": [1, 2]}))

    def test_transform_preserves_index(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_rates,
        )
        converter.fit(local_prices)
        result = converter.transform(local_prices)

        pd.testing.assert_index_equal(result.index, local_prices.index)
        pd.testing.assert_index_equal(result.columns, local_prices.columns)


class TestFxPriceConverterCaseInsensitive:
    """FX rate columns quoted in lower-case must still match currencies."""

    def test_lowercase_fx_columns_matched(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        fx_lower = fx_rates.rename(columns=str.lower)
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_lower,
        )
        converter.fit(local_prices)
        # No currencies should be reported missing despite lower-case cols.
        assert converter.missing_currencies_ == set()

        result = converter.transform(local_prices)
        expected_lloy = local_prices["LLOY.L"] * fx_rates["GBP"]
        pd.testing.assert_series_equal(
            result["LLOY.L"], expected_lloy, check_names=False
        )


class TestFxPriceConverterClone:
    """The transformer must survive sklearn clone (Pipeline composability)."""

    def test_clone_preserves_params(
        self,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        from sklearn.base import clone

        converter = FxPriceConverter(
            base_currency="GBP",
            currency_map=currency_map,
            fx_rates=fx_rates,
            fill_limit=7,
        )
        cloned = clone(converter)
        assert cloned.base_currency == "GBP"
        assert cloned.fill_limit == 7
        assert cloned.currency_map == currency_map


class TestFxPriceConverterSklearnAPI:
    """Tests for sklearn API compliance."""

    def test_get_params(self) -> None:
        converter = FxPriceConverter(base_currency="GBP", fill_limit=10)
        params = converter.get_params()
        assert params["base_currency"] == "GBP"
        assert params["fill_limit"] == 10

    def test_get_feature_names_out(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_rates,
        )
        converter.fit(local_prices)
        names = converter.get_feature_names_out()
        assert list(names) == list(local_prices.columns)

    def test_fit_transform(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_rates,
        )
        result = converter.fit_transform(local_prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == local_prices.shape


class TestFxPriceConverterMinorUnits:
    """Sub-unit (pence/cents/agorot) rescaling — the core 100x-bug guard."""

    def test_gbp_pence_scaled_then_fx_converted(self) -> None:
        """A ``GBp`` (pence) ticker must be divided by 100 before the GBP FX
        rate is applied — otherwise the EUR price is 100x too large."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        prices = pd.DataFrame({"LLOY.L": [50.0, 51.0, 52.0, 53.0, 54.0]}, index=dates)
        # price_unit stored verbatim as pence.
        cmap = {"LLOY.L": "GBp"}
        rate = pd.Series(np.linspace(1.15, 1.17, 5), index=dates)
        fx = pd.DataFrame({"GBP": rate})

        converter = FxPriceConverter(
            base_currency="EUR", currency_map=cmap, fx_rates=fx
        )
        result = converter.fit_transform(prices)

        # pence -> pounds (/100) -> EUR (x rate)
        expected = (prices["LLOY.L"] / 100.0) * rate
        pd.testing.assert_series_equal(result["LLOY.L"], expected, check_names=False)
        # It is treated as a foreign GBP ticker with a 100x minor-unit scale.
        assert converter.foreign_tickers_["LLOY.L"] == "GBP"
        assert converter.ticker_scale_["LLOY.L"] == 100
        assert converter.minor_unit_tickers_ == {"LLOY.L": 100}

    def test_gbp_base_pence_ticker_rescaled_without_fx(self) -> None:
        """With GBP base, a pence ticker still needs /100 even though no FX
        conversion applies (major currency == base)."""
        dates = pd.bdate_range("2024-01-02", periods=4)
        prices = pd.DataFrame({"LLOY.L": [500.0, 510.0, 520.0, 530.0]}, index=dates)
        cmap = {"LLOY.L": "GBp"}
        # No FX rates needed; base is GBP.
        converter = FxPriceConverter(
            base_currency="GBP", currency_map=cmap, fx_rates=pd.DataFrame()
        )
        result = converter.fit_transform(prices)

        pd.testing.assert_series_equal(
            result["LLOY.L"], prices["LLOY.L"] / 100.0, check_names=False
        )
        # Not foreign (major == base), but still rescaled.
        assert "LLOY.L" not in converter.foreign_tickers_
        assert converter.ticker_scale_["LLOY.L"] == 100
        assert converter.missing_currencies_ == set()

    def test_zac_cents_uses_major_zar_rate(self) -> None:
        """A ``ZAc`` ticker resolves to ZAR for the FX lookup (upper-casing to
        'ZAC' alone would miss the 'ZAR' rate column)."""
        dates = pd.bdate_range("2024-01-02", periods=4)
        prices = pd.DataFrame({"NPN.JO": [1000.0, 1010.0, 1020.0, 1030.0]}, index=dates)
        cmap = {"NPN.JO": "ZAc"}
        rate = pd.Series([0.05, 0.05, 0.051, 0.049], index=dates)
        fx = pd.DataFrame({"ZAR": rate})

        converter = FxPriceConverter(
            base_currency="EUR", currency_map=cmap, fx_rates=fx
        )
        result = converter.fit_transform(prices)

        expected = (prices["NPN.JO"] / 100.0) * rate
        pd.testing.assert_series_equal(result["NPN.JO"], expected, check_names=False)
        assert converter.foreign_tickers_["NPN.JO"] == "ZAR"
        assert converter.missing_currencies_ == set()

    def test_pounds_not_mistaken_for_pence(self) -> None:
        """Regression: an explicit 'GBP' (pounds) code must NOT be rescaled."""
        dates = pd.bdate_range("2024-01-02", periods=4)
        prices = pd.DataFrame({"X": [50.0, 51.0, 52.0, 53.0]}, index=dates)
        cmap = {"X": "GBP"}
        rate = pd.Series([1.16, 1.16, 1.16, 1.16], index=dates)
        fx = pd.DataFrame({"GBP": rate})

        converter = FxPriceConverter(
            base_currency="EUR", currency_map=cmap, fx_rates=fx
        )
        result = converter.fit_transform(prices)

        # No /100 — pounds converted directly.
        pd.testing.assert_series_equal(
            result["X"], prices["X"] * rate, check_names=False
        )
        assert converter.ticker_scale_["X"] == 1
        assert converter.minor_unit_tickers_ == {}


class TestFxPriceConverterDecimalInput:
    """DB Numeric columns arrive as Decimal (object dtype) — must be handled."""

    def test_decimal_prices_converted(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=4)
        raw = [Decimal("50.0"), Decimal("51.0"), Decimal("52.0"), Decimal("53.0")]
        prices = pd.DataFrame({"LLOY.L": raw}, index=dates)
        assert prices["LLOY.L"].dtype == object  # sanity: Decimal object dtype
        cmap = {"LLOY.L": "GBp"}
        rate = pd.Series([1.16, 1.16, 1.16, 1.16], index=dates)
        fx = pd.DataFrame({"GBP": rate})

        converter = FxPriceConverter(
            base_currency="EUR", currency_map=cmap, fx_rates=fx
        )
        result = converter.fit_transform(prices)

        assert result["LLOY.L"].dtype == np.float64
        # pence -> pounds (/100) -> EUR (x 1.16)
        expected = pd.Series(
            [v / 100.0 * 1.16 for v in (50.0, 51.0, 52.0, 53.0)], index=dates
        )
        pd.testing.assert_series_equal(result["LLOY.L"], expected, check_names=False)

    def test_decimal_fx_rates_converted(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=4)
        prices = pd.DataFrame({"LLOY.L": [50.0, 51.0, 52.0, 53.0]}, index=dates)
        fx = pd.DataFrame({"GBP": [Decimal("1.16")] * 4}, index=dates)
        assert fx["GBP"].dtype == object  # sanity: Decimal object dtype

        converter = FxPriceConverter(
            base_currency="EUR", currency_map={"LLOY.L": "GBP"}, fx_rates=fx
        )
        result = converter.fit_transform(prices)

        expected = pd.Series([v * 1.16 for v in (50.0, 51.0, 52.0, 53.0)], index=dates)
        pd.testing.assert_series_equal(result["LLOY.L"], expected, check_names=False)

    def test_decimal_fx_rates_with_gap_still_forward_fill(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=4)
        prices = pd.DataFrame({"LLOY.L": [50.0] * 4}, index=dates)
        fx = pd.DataFrame(
            {"GBP": [Decimal("1.10"), None, None, Decimal("1.20")]}, index=dates
        )

        converter = FxPriceConverter(
            base_currency="EUR", currency_map={"LLOY.L": "GBP"}, fx_rates=fx
        )
        result = converter.fit_transform(prices)

        assert result["LLOY.L"].tolist() == pytest.approx([55.0, 55.0, 55.0, 60.0])

    def test_non_numeric_fx_rates_raise_data_error(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        prices = pd.DataFrame({"LLOY.L": [50.0, 51.0]}, index=dates)
        fx = pd.DataFrame({"GBP": ["n/a", "n/a"]}, index=dates)

        converter = FxPriceConverter(
            base_currency="EUR", currency_map={"LLOY.L": "GBP"}, fx_rates=fx
        )
        with pytest.raises(DataError, match="numeric"):
            converter.fit(prices)


class TestFxPriceConverterFillLimitWarning:
    """Tests for fill_limit exhaustion NaN warning."""

    def test_transform_warns_on_fill_limit_exhaustion(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When FX rates have gaps beyond fill_limit, warn about affected tickers."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.DataFrame(
            {
                "LLOY.L": np.linspace(50, 55, 10),
                "ORA.PA": np.linspace(90, 95, 10),
            },
            index=dates,
        )
        cmap = {"LLOY.L": "GBP", "ORA.PA": "EUR"}

        # Only provide FX rate for the first day — fill_limit=2 means
        # days 4–10 will have NaN rates, causing NaN prices for LLOY.L.
        fx = pd.DataFrame(
            {"GBP": [1.16]},
            index=pd.to_datetime(["2024-01-02"]),
        )

        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=cmap,
            fx_rates=fx,
            fill_limit=2,
        )
        converter.fit(prices)

        import logging

        with caplog.at_level(logging.WARNING, logger="optimizer.fx._converter"):
            result = converter.transform(prices)

        # LLOY.L should have NaN prices where fill_limit was exhausted
        assert result["LLOY.L"].isna().any()
        # ORA.PA (base currency) should be unchanged — no NaNs introduced
        assert not (result["ORA.PA"].isna() & ~prices["ORA.PA"].isna()).any()
        # Warning should mention the affected ticker
        assert any("LLOY.L" in record.message for record in caplog.records)
        assert any("fill_limit=2" in record.message for record in caplog.records)
        # Warning should report total NaN cells introduced (7 days × 1 ticker)
        assert any("7" in record.message for record in caplog.records)
        # Warning should include the actionable hint
        assert any(
            "Consider increasing fill_limit" in record.message
            for record in caplog.records
        )

    def test_transform_warns_total_nan_count_multi_currency(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Multi-currency: GBP exhausts fill_limit, USD does not.

        GBP rate present only on days 1-2; fill_limit=2 fills days 3-4,
        days 5-10 are NaN → 6 NaN cells for LLOY.L.
        USD rate present for all 10 days → 0 NaN cells for SPY.
        Total introduced NaN = 6.
        """
        import logging

        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.DataFrame(
            {
                "LLOY.L": np.linspace(50, 55, 10),
                "ORA.PA": np.linspace(90, 95, 10),
                "SPY": np.linspace(480, 490, 10),
            },
            index=dates,
        )
        cmap = {"LLOY.L": "GBP", "ORA.PA": "EUR", "SPY": "USD"}

        # GBP: days 1-2 real; fill_limit=2 fills days 3-4; days 5-10 NaN
        # USD: full 10-day coverage, no NaN introduced
        fx = pd.DataFrame(
            {
                "GBP": [1.16, 1.16] + [np.nan] * 8,
                "USD": np.linspace(0.91, 0.93, 10),
            },
            index=dates,
        )

        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=cmap,
            fx_rates=fx,
            fill_limit=2,
        )
        converter.fit(prices)

        with caplog.at_level(logging.WARNING, logger="optimizer.fx._converter"):
            result = converter.transform(prices)

        # Only LLOY.L should have NaN introduced
        assert result["LLOY.L"].isna().sum() == 6
        assert not (result["SPY"].isna() & ~prices["SPY"].isna()).any()
        assert not (result["ORA.PA"].isna() & ~prices["ORA.PA"].isna()).any()

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("LLOY.L" in m for m in warning_messages)
        assert not any("SPY" in m for m in warning_messages)
        # Warning must report correct total NaN count (6) and include hint
        assert any("6" in m for m in warning_messages)
        assert any("Consider increasing fill_limit" in m for m in warning_messages)


class TestFxPriceConverterUnmappedTickers:
    """A column with no known currency must never be silently valued as base."""

    @staticmethod
    def _prices() -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=4)
        return pd.DataFrame(
            {"VOD.L": [70.0, 71.0, 72.0, 73.0], "AAPL": [190.0] * 4}, index=dates
        )

    @staticmethod
    def _fx(prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"USD": [0.9] * 4}, index=prices.index)

    def test_ticker_absent_from_currency_map_is_flagged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        prices = self._prices()
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map={"AAPL": "USD"},
            fx_rates=self._fx(prices),
        )
        with caplog.at_level(logging.WARNING, logger="optimizer.fx._converter"):
            converter.fit(prices)

        assert any(
            "VOD.L" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )
        assert converter.unmapped_tickers_ == ["VOD.L"]
        assert "AAPL" not in converter.unmapped_tickers_

    def test_ticker_absent_from_currency_map_raises_when_full_coverage_required(
        self,
    ) -> None:
        prices = self._prices()
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map={"AAPL": "USD"},
            fx_rates=self._fx(prices),
            require_full_coverage=True,
        )
        with pytest.raises(DataError, match=r"VOD\.L"):
            converter.fit(prices)

    @pytest.mark.parametrize("missing_code", [None, float("nan"), "", "  "])
    def test_null_currency_code_is_unknown_not_a_bogus_currency(
        self, missing_code: object
    ) -> None:
        prices = self._prices()
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map={"AAPL": "USD", "VOD.L": missing_code},  # type: ignore[dict-item]
            fx_rates=self._fx(prices),
        )
        converter.fit(prices)

        assert converter.unmapped_tickers_ == ["VOD.L"]
        assert converter.missing_currencies_ == set()
        assert "VOD.L" not in converter.foreign_tickers_

    def test_null_currency_code_raises_unmapped_error_when_full_coverage_required(
        self,
    ) -> None:
        prices = self._prices()
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map={"AAPL": "USD", "VOD.L": None},  # type: ignore[dict-item]
            fx_rates=self._fx(prices),
            require_full_coverage=True,
        )
        with pytest.raises(DataError, match=r"VOD\.L") as excinfo:
            converter.fit(prices)
        assert "NONE" not in str(excinfo.value)

    def test_fully_mapped_frame_has_no_unmapped_tickers(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map=currency_map,
            fx_rates=fx_rates,
            require_full_coverage=True,
        )
        converter.fit(local_prices)
        assert converter.unmapped_tickers_ == []


class TestFxPriceConverterTransformOtherWindow:
    """transform() must convert any window the rate frame covers, not just the
    window seen at fit time."""

    def test_transform_converts_dates_outside_fit_window(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = FxPriceConverter(
            base_currency="EUR", currency_map=currency_map, fx_rates=fx_rates
        )
        converter.fit(local_prices.iloc[:5])
        result = converter.transform(local_prices.iloc[5:])

        assert not result.isna().any().any()
        expected = local_prices.iloc[5:]["LLOY.L"] * fx_rates["GBP"].iloc[5:]
        pd.testing.assert_series_equal(result["LLOY.L"], expected, check_names=False)
        expected_spy = local_prices.iloc[5:]["SPY"] * fx_rates["USD"].iloc[5:]
        pd.testing.assert_series_equal(result["SPY"], expected_spy, check_names=False)

    def test_fit_window_does_not_change_transform_output(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        def make() -> FxPriceConverter:
            return FxPriceConverter(
                base_currency="EUR", currency_map=currency_map, fx_rates=fx_rates
            )

        fit_on_head = make().fit(local_prices.iloc[:5])
        fit_on_all = make().fit(local_prices)

        pd.testing.assert_frame_equal(
            fit_on_head.transform(local_prices.iloc[5:]),
            fit_on_all.transform(local_prices).iloc[5:],
        )

    def test_minor_unit_ticker_converted_outside_fit_window(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.DataFrame({"AAA.L": np.linspace(5000.0, 5900.0, 10)}, index=dates)
        rate = pd.Series(np.linspace(1.15, 1.17, 10), index=dates)
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map={"AAA.L": "GBp"},
            fx_rates=pd.DataFrame({"GBP": rate}),
        ).fit(prices.iloc[:5])

        result = converter.transform(prices.iloc[5:])

        expected = prices["AAA.L"].iloc[5:] / 100.0 * rate.iloc[5:]
        pd.testing.assert_series_equal(result["AAA.L"], expected, check_names=False)


class TestFxPriceConverterTransformColumns:
    """Columns the converter never resolved at fit must not pass through raw."""

    @staticmethod
    def _make(prices: pd.DataFrame) -> FxPriceConverter:
        rate = pd.DataFrame({"GBP": [1.15] * len(prices)}, index=prices.index)
        return FxPriceConverter(
            base_currency="EUR",
            currency_map={"A": "GBP", "B": "GBp"},
            fx_rates=rate,
        )

    @staticmethod
    def _prices() -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=4)
        return pd.DataFrame({"A": [10.0] * 4, "B": [1000.0] * 4}, index=dates)

    def test_column_unseen_at_fit_raises(self) -> None:
        prices = self._prices()
        converter = self._make(prices).fit(prices[["A"]])

        with pytest.raises(DataError, match="not seen during fit"):
            converter.transform(prices)

    def test_subset_and_reordered_columns_still_convert(self) -> None:
        prices = self._prices()
        converter = self._make(prices).fit(prices)

        reordered = converter.transform(prices[["B", "A"]])
        only_b = converter.transform(prices[["B"]])

        assert list(reordered.columns) == ["B", "A"]
        assert reordered["B"].iloc[0] == pytest.approx(1000.0 / 100.0 * 1.15)
        assert reordered["A"].iloc[0] == pytest.approx(10.0 * 1.15)
        assert only_b["B"].iloc[0] == pytest.approx(1000.0 / 100.0 * 1.15)


class TestFxPriceConverterInvalidRates:
    """A zero or non-finite FX print is missing data, not a 0 or inf rate."""

    @pytest.mark.parametrize("bad_rate", [0.0, np.inf, -1.2])
    def test_bad_rate_yields_nan_price_and_warning(
        self, bad_rate: float, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        dates = pd.bdate_range("2024-01-02", periods=3)
        prices = pd.DataFrame({"LLOY.L": [50.0, 51.0, 52.0]}, index=dates)
        fx = pd.DataFrame({"GBP": [bad_rate, 1.2, 1.2]}, index=dates)
        converter = FxPriceConverter(
            base_currency="EUR", currency_map={"LLOY.L": "GBP"}, fx_rates=fx
        )
        converter.fit(prices)

        with caplog.at_level(logging.WARNING, logger="optimizer.fx._converter"):
            result = converter.transform(prices)

        assert np.isnan(result["LLOY.L"].iloc[0])
        assert result["LLOY.L"].iloc[1:].tolist() == pytest.approx([61.2, 62.4])
        assert any("NaN prices" in r.message for r in caplog.records)


class TestFxPriceConverterTimezones:
    """Prices and rates may differ in tz-awareness; dates still line up."""

    @staticmethod
    def _frames(
        price_tz: str | None, fx_tz: str | None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.bdate_range("2024-01-02", periods=5)
        prices = pd.DataFrame({"LLOY.L": [50.0] * 5}, index=dates.tz_localize(price_tz))
        fx = pd.DataFrame({"GBP": [1.2] * 5}, index=dates.tz_localize(fx_tz))
        return prices, fx

    @pytest.mark.parametrize(
        ("price_tz", "fx_tz"), [("UTC", None), (None, "UTC"), ("Europe/London", None)]
    )
    def test_mixed_tz_awareness_converts_instead_of_all_nan(
        self, price_tz: str | None, fx_tz: str | None
    ) -> None:
        prices, fx = self._frames(price_tz, fx_tz)
        converter = FxPriceConverter(
            base_currency="EUR",
            currency_map={"LLOY.L": "GBP"},
            fx_rates=fx,
            require_full_coverage=True,
        )

        result = converter.fit_transform(prices)

        assert result["LLOY.L"].tolist() == pytest.approx([60.0] * 5)
        assert result.index.equals(prices.index)
