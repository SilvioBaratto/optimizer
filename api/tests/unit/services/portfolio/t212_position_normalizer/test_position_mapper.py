"""Tests for T212PositionMapper (issue #776)."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.schemas.portfolio import PositionFlag


def _make_fx(rate: float | None = 1.0) -> MagicMock:
    provider = MagicMock()
    provider.get_rate_to_base.return_value = rate
    return provider


def _raw(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ticker": "AAPL_US_EQ",
        "quantity": 10.0,
        "currentPrice": 180.0,
        "averagePrice": 170.0,
        "currencyCode": "USD",
        "ppl": 100.0,
        "fxPpl": -5.0,
        "initialFillDate": "2024-01-10T10:00:00Z",
    }
    base.update(overrides)
    return base


class TestEurNativePosition:
    def test_when_eur_position_then_no_fx_call_and_eur_value_direct(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = _raw(
            ticker="SAP_d_EQ",
            currencyCode="EUR",
            currentPrice=100.0,
            quantity=5.0,
        )

        pos = mapper.map_position(raw, yf_ticker="SAP.DE", as_of_date=date(2026, 5, 19))

        assert pos.currency == "EUR"
        assert pos.price_native == 100.0
        assert pos.qty == 5.0
        assert pos.eur_value == pytest.approx(500.0)
        assert pos.flags == []
        fx.get_rate_to_base.assert_not_called()


class TestUsdPosition:
    def test_when_usd_position_then_fx_applied(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx(rate=0.92)
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        pos = mapper.map_position(
            _raw(), yf_ticker="AAPL", as_of_date=date(2026, 5, 19)
        )

        assert pos.currency == "USD"
        assert pos.price_native == 180.0
        assert pos.eur_value == pytest.approx(10.0 * 180.0 * 0.92)
        assert pos.flags == []
        fx.get_rate_to_base.assert_called_once_with("USD", date(2026, 5, 19))


class TestGbxLondonPosition:
    def test_when_gbx_then_price_divided_by_100_and_currency_gbp(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx(rate=1.17)
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = _raw(
            ticker="VOD_l_EQ",
            currencyCode="GBX",
            currentPrice=7500.0,  # 75.00 GBP in pence
            quantity=200.0,
        )

        pos = mapper.map_position(raw, yf_ticker="VOD.L", as_of_date=date(2026, 5, 19))

        assert pos.currency == "GBP"
        assert pos.price_native == pytest.approx(75.0)
        assert pos.eur_value == pytest.approx(200.0 * 75.0 * 1.17)
        fx.get_rate_to_base.assert_called_once_with("GBP", date(2026, 5, 19))

    def test_when_gbp_lowercase_p_then_normalized_same_as_gbx(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx(rate=1.17)
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = _raw(
            ticker="VOD_l_EQ",
            currencyCode="GBp",
            currentPrice=7500.0,
            quantity=10.0,
        )

        pos = mapper.map_position(raw, yf_ticker="VOD.L", as_of_date=date(2026, 5, 19))

        assert pos.currency == "GBP"
        assert pos.price_native == pytest.approx(75.0)


class TestUnmappedFallback:
    def test_when_yf_ticker_none_then_uses_suffix_resolver(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx(rate=1.17)
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = _raw(
            ticker="VOD_l_EQ",
            currencyCode="GBP",
            currentPrice=75.0,
        )

        pos = mapper.map_position(raw, yf_ticker=None, as_of_date=date(2026, 5, 19))

        assert pos.ticker == "VOD.L"
        assert PositionFlag.UNMAPPED in pos.flags
        assert pos.t212_ticker == "VOD_l_EQ"

    def test_when_yf_ticker_none_and_unrecognized_then_passthrough_base(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = _raw(
            ticker="WEIRD",
            currencyCode="EUR",
            currentPrice=10.0,
        )

        pos = mapper.map_position(raw, yf_ticker=None, as_of_date=date(2026, 5, 19))

        assert pos.ticker == "WEIRD"
        assert PositionFlag.UNMAPPED in pos.flags


class TestMissingFx:
    def test_when_fx_returns_none_then_flag_fx_missing_and_eur_value_zero(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx(rate=None)
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        pos = mapper.map_position(
            _raw(), yf_ticker="AAPL", as_of_date=date(2026, 5, 19)
        )

        assert PositionFlag.FX_MISSING in pos.flags
        assert pos.eur_value == 0.0


class TestStalePrice:
    def test_when_current_price_zero_then_flag_stale_price(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = _raw(currencyCode="EUR", currentPrice=0.0)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=date(2026, 5, 19))

        assert PositionFlag.STALE_PRICE in pos.flags

    def test_when_current_price_none_then_flag_stale_price(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = _raw(currencyCode="EUR", currentPrice=None)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=date(2026, 5, 19))

        assert PositionFlag.STALE_PRICE in pos.flags
        assert pos.price_native == 0.0
        assert pos.eur_value == 0.0

    def test_when_current_price_negative_then_flag_stale_price(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = _raw(currencyCode="EUR", currentPrice=-1.0)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=date(2026, 5, 19))

        assert PositionFlag.STALE_PRICE in pos.flags


class TestPplEur:
    def test_when_base_eur_then_ppl_eur_is_ppl_plus_fx_ppl(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx(rate=0.92)
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        pos = mapper.map_position(
            _raw(ppl=100.0, fxPpl=-5.0),
            yf_ticker="AAPL",
            as_of_date=date(2026, 5, 19),
        )

        assert pos.ppl_eur == pytest.approx(95.0)

    def test_when_base_eur_and_fx_ppl_none_then_ppl_eur_is_ppl_only(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx(rate=0.92)
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        pos = mapper.map_position(
            _raw(ppl=42.0, fxPpl=None),
            yf_ticker="AAPL",
            as_of_date=date(2026, 5, 19),
        )

        assert pos.ppl_eur == pytest.approx(42.0)

    def test_when_base_non_eur_then_ppl_eur_is_none(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx(rate=1.0)
        mapper = T212PositionMapper(fx_provider=fx, base_currency="USD")

        pos = mapper.map_position(
            _raw(currencyCode="USD"),
            yf_ticker="AAPL",
            as_of_date=date(2026, 5, 19),
        )

        assert pos.ppl_eur is None


class TestEurWeightAlwaysZero:
    def test_when_mapped_then_eur_weight_is_zero(self) -> None:
        """Weight is the orchestrator's job; mapper sets 0.0."""
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        pos = mapper.map_position(
            _raw(currencyCode="EUR"),
            yf_ticker="AAPL",
            as_of_date=date(2026, 5, 19),
        )

        assert pos.eur_weight == 0.0


class TestT212TickerCarry:
    def test_when_mapped_then_raw_ticker_preserved_in_t212_ticker_field(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        pos = mapper.map_position(
            _raw(ticker="VOD_l_EQ", currencyCode="EUR"),
            yf_ticker="VOD.L",
            as_of_date=date(2026, 5, 19),
        )

        assert pos.t212_ticker == "VOD_l_EQ"
        assert pos.ticker == "VOD.L"


class TestStructurallyInvalidInput:
    def test_when_no_ticker_then_value_error(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        with pytest.raises(ValueError, match="ticker"):
            mapper.map_position(
                {"quantity": 1.0, "currentPrice": 1.0, "currencyCode": "EUR"},
                yf_ticker="AAPL",
                as_of_date=date(2026, 5, 19),
            )

    def test_when_no_quantity_then_value_error(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        with pytest.raises(ValueError, match="quantity"):
            mapper.map_position(
                {"ticker": "AAPL_US_EQ", "currentPrice": 1.0, "currencyCode": "EUR"},
                yf_ticker="AAPL",
                as_of_date=date(2026, 5, 19),
            )

    def test_when_empty_ticker_string_then_value_error(self) -> None:
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")

        with pytest.raises(ValueError, match="ticker"):
            mapper.map_position(
                {"ticker": "", "quantity": 1.0, "currencyCode": "EUR"},
                yf_ticker="AAPL",
                as_of_date=date(2026, 5, 19),
            )


class TestMissingCurrencyDefaultsEur:
    def test_when_currency_code_missing_then_defaults_to_eur_and_no_fx_call(
        self,
    ) -> None:
        """T212 occasionally omits currencyCode — default to base, skip FX."""
        from app.services.portfolio.t212_position_normalizer._position_mapper import (
            T212PositionMapper,
        )

        fx = _make_fx()
        mapper = T212PositionMapper(fx_provider=fx, base_currency="EUR")
        raw = {
            "ticker": "AAPL_US_EQ",
            "quantity": 1.0,
            "currentPrice": 100.0,
        }

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=date(2026, 5, 19))

        assert pos.currency == "EUR"
        fx.get_rate_to_base.assert_not_called()
