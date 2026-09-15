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
