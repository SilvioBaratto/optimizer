"""Tests for build_fx_converter()."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.fx import (
    BaseCurrency,
    FxConfig,
    FxConversionMode,
    FxPriceConverter,
    build_fx_converter,
)


@pytest.fixture()
def prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "AAPL": [190.0, 191.0, 192.0, 190.5, 193.0],
            "SAP.DE": [140.0, 141.0, 139.0, 142.0, 143.0],
            "VOD.L": [70.0, 71.0, 72.0, 70.5, 71.5],
        },
        index=dates,
    )


@pytest.fixture()
def currency_map() -> dict[str, str]:
    return {"AAPL": "USD", "SAP.DE": "EUR", "VOD.L": "GBp"}


@pytest.fixture()
def fx_rates(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"USD": 0.92, "GBP": 1.17}, index=prices.index, dtype="float64")


class TestBuildFxConverterMode:
    def test_default_none_mode_leaves_prices_unconverted(
        self,
        prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = build_fx_converter(
            FxConfig(), fx_rates=fx_rates, currency_map=currency_map
        )

        assert isinstance(converter, FxPriceConverter)
        pd.testing.assert_frame_equal(converter.fit_transform(prices), prices)

    @pytest.mark.parametrize(
        "mode", [FxConversionMode.TO_BASE, FxConversionMode.DECOMPOSE]
    )
    def test_conversion_modes_convert_foreign_prices(
        self,
        prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
        mode: FxConversionMode,
    ) -> None:
        config = FxConfig(base_currency=BaseCurrency.EUR, mode=mode)

        out = build_fx_converter(
            config, fx_rates=fx_rates, currency_map=currency_map
        ).fit_transform(prices)

        np.testing.assert_allclose(out["AAPL"], prices["AAPL"] * 0.92)
        np.testing.assert_allclose(out["VOD.L"], prices["VOD.L"] / 100 * 1.17)
        np.testing.assert_allclose(out["SAP.DE"], prices["SAP.DE"])

    def test_none_mode_ignores_strict(self, prices: pd.DataFrame) -> None:
        config = FxConfig(mode=FxConversionMode.NONE, strict=True)

        converter = build_fx_converter(config, fx_rates=None, currency_map=None)

        pd.testing.assert_frame_equal(converter.fit_transform(prices), prices)


class TestBuildFxConverterStrict:
    def test_empty_fx_rates_raise(self, currency_map: dict[str, str]) -> None:
        with pytest.raises(ConfigurationError, match="fx_rates"):
            build_fx_converter(
                FxConfig.for_strict_conversion(),
                fx_rates=pd.DataFrame(),
                currency_map=currency_map,
            )

    def test_none_fx_rates_raise(self, currency_map: dict[str, str]) -> None:
        with pytest.raises(ConfigurationError, match="fx_rates"):
            build_fx_converter(
                FxConfig.for_strict_conversion(),
                fx_rates=None,
                currency_map=currency_map,
            )

    @pytest.mark.parametrize("empty_map", [None, {}])
    def test_missing_currency_map_raises(
        self, fx_rates: pd.DataFrame, empty_map: dict[str, str] | None
    ) -> None:
        with pytest.raises(ConfigurationError, match="currency_map"):
            build_fx_converter(
                FxConfig.for_strict_conversion(),
                fx_rates=fx_rates,
                currency_map=empty_map,
            )

    def test_missing_rate_column_raises_instead_of_mixing_currencies(
        self,
        prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        converter = build_fx_converter(
            FxConfig.for_strict_conversion(),
            fx_rates=fx_rates[["USD"]],
            currency_map=currency_map,
        )

        with pytest.raises(DataError, match="GBP"):
            converter.fit(prices)

    def test_complete_inputs_convert(
        self,
        prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        out = build_fx_converter(
            FxConfig.for_strict_conversion(),
            fx_rates=fx_rates,
            currency_map=currency_map,
        ).fit_transform(prices)

        np.testing.assert_allclose(out["AAPL"], prices["AAPL"] * 0.92)


class TestBuildFxConverterNonStrict:
    def test_missing_currency_map_warns_and_skips_conversion(
        self,
        prices: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = FxConfig(mode=FxConversionMode.TO_BASE)

        with caplog.at_level(logging.WARNING, logger="optimizer.fx"):
            converter = build_fx_converter(config, fx_rates=None, currency_map=None)

        assert "currency_map" in caplog.text
        pd.testing.assert_frame_equal(converter.fit_transform(prices), prices)

    @pytest.mark.parametrize("empty_rates", [None, pd.DataFrame()])
    def test_empty_fx_rates_for_base_only_book_does_not_warn(
        self,
        prices: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
        empty_rates: pd.DataFrame | None,
    ) -> None:
        # An all-base-currency book has no base/quote rows to load, so empty
        # rates are legitimate and must not warn on every NAV mark.
        config = FxConfig(base_currency=BaseCurrency.EUR, mode=FxConversionMode.TO_BASE)
        base_only = prices[["SAP.DE"]]

        with caplog.at_level(logging.WARNING, logger="optimizer.fx"):
            converter = build_fx_converter(
                config, fx_rates=empty_rates, currency_map={"SAP.DE": "EUR"}
            )
            out = converter.fit_transform(base_only)

        assert caplog.records == []
        pd.testing.assert_frame_equal(out, base_only)

    def test_empty_fx_rates_with_foreign_ticker_warns_and_skips_it(
        self,
        prices: pd.DataFrame,
        currency_map: dict[str, str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = FxConfig(base_currency=BaseCurrency.EUR, mode=FxConversionMode.TO_BASE)

        with caplog.at_level(logging.WARNING, logger="optimizer.fx"):
            out = build_fx_converter(
                config, fx_rates=pd.DataFrame(), currency_map=currency_map
            ).fit_transform(prices)

        assert "USD" in caplog.text
        np.testing.assert_allclose(out["AAPL"], prices["AAPL"])
