"""Time-based STALE_PRICE + FlagInstance reference tests (issue #795)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

from app.schemas.portfolio import FlagInstance, PositionFlag
from app.services.portfolio.t212_position_normalizer._position_mapper import (
    T212PositionMapper,
)
from app.services.portfolio.t212_position_normalizer.freshness_config import (
    FreshnessConfig,
)


def _make_fx(rate: float | None = 1.0) -> MagicMock:
    provider = MagicMock()
    provider.get_rate_to_base.return_value = rate
    return provider


def _raw(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ticker": "AAPL_US_EQ",
        "quantity": 10.0,
        "currentPrice": 180.0,
        "currencyCode": "USD",
        "ppl": 0.0,
        "fxPpl": 0.0,
    }
    base.update(overrides)
    return base


def _flag_for(pos, code: PositionFlag) -> FlagInstance:
    matches = [f for f in pos.flags if f.code is code]
    assert matches, f"flag {code} not present"
    return matches[0]


_AS_OF = date(2026, 5, 20)


class TestUnmappedReference:
    def test_when_unmapped_then_reference_is_t212_ticker(self) -> None:
        mapper = T212PositionMapper(fx_provider=_make_fx(rate=0.92), base_currency="EUR")
        raw = _raw(ticker="WEIRD_XX_EQ", currencyCode="USD", currentPrice=100.0)

        pos = mapper.map_position(raw, yf_ticker=None, as_of_date=_AS_OF)

        flag = _flag_for(pos, PositionFlag.UNMAPPED)
        assert flag.reference == "WEIRD_XX_EQ"
        assert "WEIRD_XX_EQ" in flag.reason


class TestFxMissingReference:
    def test_when_fx_missing_then_reference_is_currency_code(self) -> None:
        mapper = T212PositionMapper(fx_provider=_make_fx(rate=None), base_currency="EUR")

        pos = mapper.map_position(_raw(), yf_ticker="AAPL", as_of_date=_AS_OF)

        flag = _flag_for(pos, PositionFlag.FX_MISSING)
        assert flag.reference == "USD"


class TestStalePriceNullGuardReference:
    def test_when_price_none_then_reference_is_ticker(self) -> None:
        mapper = T212PositionMapper(fx_provider=_make_fx(), base_currency="EUR")
        raw = _raw(currencyCode="EUR", currentPrice=None)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=_AS_OF)

        flag = _flag_for(pos, PositionFlag.STALE_PRICE)
        assert flag.reference == "AAPL"

    def test_when_price_zero_then_reference_is_ticker(self) -> None:
        mapper = T212PositionMapper(fx_provider=_make_fx(), base_currency="EUR")
        raw = _raw(currencyCode="EUR", currentPrice=0.0)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=_AS_OF)

        flag = _flag_for(pos, PositionFlag.STALE_PRICE)
        assert flag.reference == "AAPL"


class TestTimeBasedStalePrice:
    def _mapper(
        self,
        *,
        last_tick: datetime | None,
        threshold_hours: int = 48,
    ) -> T212PositionMapper:
        return T212PositionMapper(
            fx_provider=_make_fx(),
            base_currency="EUR",
            freshness_config=FreshnessConfig(
                stale_price_threshold_hours=threshold_hours
            ),
            last_tick_lookup=lambda _: last_tick,
        )

    def test_when_last_tick_older_than_threshold_then_stale_price_fires(
        self,
    ) -> None:
        ancient = datetime.combine(_AS_OF, datetime.min.time()) - timedelta(days=10)
        mapper = self._mapper(last_tick=ancient)
        raw = _raw(currencyCode="EUR", currentPrice=100.0)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=_AS_OF)

        flag = _flag_for(pos, PositionFlag.STALE_PRICE)
        assert flag.reference == "AAPL"

    def test_when_last_tick_within_threshold_then_no_stale_price_flag(
        self,
    ) -> None:
        recent = datetime.combine(_AS_OF, datetime.min.time()) - timedelta(hours=12)
        mapper = self._mapper(last_tick=recent, threshold_hours=48)
        raw = _raw(currencyCode="EUR", currentPrice=100.0)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=_AS_OF)

        assert PositionFlag.STALE_PRICE not in pos.flags

    def test_when_last_tick_lookup_returns_none_then_stale_price_with_marker(
        self,
    ) -> None:
        mapper = self._mapper(last_tick=None)
        raw = _raw(currencyCode="EUR", currentPrice=100.0)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=_AS_OF)

        flag = _flag_for(pos, PositionFlag.STALE_PRICE)
        assert flag.reference == "no_tick_timestamp"

    def test_when_freshness_config_unset_then_no_time_based_check(self) -> None:
        """Default mapper (no freshness wiring) preserves Cycle-1 behavior."""
        mapper = T212PositionMapper(fx_provider=_make_fx(), base_currency="EUR")
        raw = _raw(currencyCode="EUR", currentPrice=100.0)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=_AS_OF)

        assert PositionFlag.STALE_PRICE not in pos.flags

    def test_when_lookup_wired_but_config_absent_then_no_time_check(self) -> None:
        mapper = T212PositionMapper(
            fx_provider=_make_fx(),
            base_currency="EUR",
            last_tick_lookup=lambda _: None,
        )
        raw = _raw(currencyCode="EUR", currentPrice=100.0)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=_AS_OF)

        assert PositionFlag.STALE_PRICE not in pos.flags

    def test_when_price_invalid_and_lookup_wired_then_only_one_stale_emitted(
        self,
    ) -> None:
        """Null/zero guard short-circuits time-based check."""
        mapper = self._mapper(last_tick=None)
        raw = _raw(currencyCode="EUR", currentPrice=None)

        pos = mapper.map_position(raw, yf_ticker="AAPL", as_of_date=_AS_OF)

        stale_flags = [f for f in pos.flags if f.code is PositionFlag.STALE_PRICE]
        assert len(stale_flags) == 1
        assert stale_flags[0].reference == "AAPL"


