"""Tests for FxPriceConverter."""

from __future__ import annotations

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
        pd.testing.assert_series_equal(
            result["ORA.PA"], local_prices["ORA.PA"]
        )

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
        pd.testing.assert_series_equal(
            result["SPY"], expected_spy, check_names=False
        )

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
        pd.testing.assert_series_equal(
            result["SPY"], local_prices["SPY"]
        )
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
