"""Unit tests for YFinanceTickerMapper (suffix rules, cache, verification).

Split out of ``test_t212_builder.py`` to keep each file under the 500-line cap.
All yfinance I/O is mocked via the injected ``_yf_client`` field — no real
``YFinanceClient`` singleton and no network are touched, and no rate-limit /
retry sleep path is exercised.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from app.services.universe.trading212.cache.ticker_cache import TickerMappingCache
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.ticker_mapper import YFinanceTickerMapper

CFG = UniverseBuilderConfig()


# ---------------------------------------------------------------------------
# _build_ticker_attempts (pure, no network)
# ---------------------------------------------------------------------------


class TestYFinanceTickerMapperBuildTickerAttempts:
    def _mapper(self) -> YFinanceTickerMapper:
        return YFinanceTickerMapper(
            config=CFG,
            cache=TickerMappingCache(),
            _yf_client=MagicMock(),
        )

    def test_london_stock_exchange_appends_dot_l_suffix(self) -> None:
        mapper = self._mapper()
        attempts = mapper._build_ticker_attempts("VOD", "London Stock Exchange")
        assert attempts[0] == "VOD.L"

    def test_london_stock_exchange_also_includes_bare_symbol(self) -> None:
        mapper = self._mapper()
        attempts = mapper._build_ticker_attempts("VOD", "London Stock Exchange")
        assert "VOD" in attempts

    def test_nyse_empty_suffix_produces_single_attempt(self) -> None:
        mapper = self._mapper()
        attempts = mapper._build_ticker_attempts("AAPL", "NYSE")
        # empty suffix → preferred_ticker == clean_symbol → dedup keeps one
        assert attempts == ["AAPL"]

    def test_nasdaq_empty_suffix_produces_single_attempt(self) -> None:
        mapper = self._mapper()
        attempts = mapper._build_ticker_attempts("MSFT", "NASDAQ")
        assert attempts == ["MSFT"]

    def test_euronext_paris_appends_dot_pa(self) -> None:
        mapper = self._mapper()
        attempts = mapper._build_ticker_attempts("AIR", "Euronext Paris")
        assert attempts[0] == "AIR.PA"

    def test_deutsche_boerse_xetra_appends_dot_de(self) -> None:
        mapper = self._mapper()
        attempts = mapper._build_ticker_attempts("SAP", "Deutsche Börse Xetra")
        assert attempts[0] == "SAP.DE"

    def test_unknown_exchange_suffix_none_falls_back_to_bare_symbol(self) -> None:
        mapper = self._mapper()
        attempts = mapper._build_ticker_attempts("XYZ", "SomeUnknownMarket")
        assert attempts == ["XYZ"]

    def test_none_exchange_name_falls_back_to_bare_symbol(self) -> None:
        mapper = self._mapper()
        attempts = mapper._build_ticker_attempts("XYZ", None)
        assert attempts == ["XYZ"]

    def test_discover_replaces_slash_with_dash_before_building_attempts(self) -> None:
        """discover() cleans ``symbol.replace("/", "-")`` before
        _build_ticker_attempts, so the slash never reaches that method.
        Verified end-to-end via discover()."""
        mapper = self._mapper()
        mapper._verify_ticker = MagicMock(return_value=True)  # type: ignore[method-assign]

        result = mapper.discover("BRK/B", "NYSE")

        # NYSE has empty suffix → single attempt "BRK-B" → verified → returned
        assert result == "BRK-B"


# ---------------------------------------------------------------------------
# discover (end-to-end, _verify_ticker mocked)
# ---------------------------------------------------------------------------


class TestYFinanceTickerMapperDiscover:
    def _mapper(self, cache: TickerMappingCache) -> YFinanceTickerMapper:
        return YFinanceTickerMapper(config=CFG, cache=cache, _yf_client=MagicMock())

    def test_when_first_attempt_verified_then_discover_returns_it(self) -> None:
        mapper = self._mapper(TickerMappingCache())
        mapper._verify_ticker = MagicMock(return_value=True)  # type: ignore[method-assign]

        result = mapper.discover("VOD", "London Stock Exchange")

        assert result == "VOD.L"

    def test_when_verified_then_ticker_saved_to_cache(self) -> None:
        cache = TickerMappingCache()
        mapper = self._mapper(cache)
        mapper._verify_ticker = MagicMock(return_value=True)  # type: ignore[method-assign]
        mapper.discover("VOD", "London Stock Exchange")

        assert cache.get_mapping("VOD", "London Stock Exchange") == "VOD.L"

    def test_cache_hit_avoids_discovery_loop(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        mapper = self._mapper(cache)
        mapper._verify_ticker = MagicMock(return_value=True)  # type: ignore[method-assign]
        build_spy = MagicMock(wraps=mapper._build_ticker_attempts)
        mapper._build_ticker_attempts = build_spy  # type: ignore[method-assign]

        result = mapper.discover("AAPL", "NYSE")

        assert result == "AAPL"
        build_spy.assert_not_called()

    def test_when_all_attempts_fail_verification_then_returns_none(self) -> None:
        mapper = self._mapper(TickerMappingCache())
        mapper._verify_ticker = MagicMock(return_value=False)  # type: ignore[method-assign]

        result = mapper.discover("ZZZZ", "NYSE")

        assert result is None

    def test_when_verify_ticker_raises_then_discover_swallows_and_returns_none(
        self,
    ) -> None:
        mapper = self._mapper(TickerMappingCache())
        mapper._verify_ticker = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("unexpected")
        )

        result = mapper.discover("AAPL", "NYSE")

        assert result is None


# ---------------------------------------------------------------------------
# _verify_ticker
# ---------------------------------------------------------------------------


def _mapper_with_info(info: Any) -> YFinanceTickerMapper:
    mock_yf = MagicMock()
    mock_yf.fetch_info.return_value = info
    return YFinanceTickerMapper(
        config=CFG, cache=TickerMappingCache(), _yf_client=mock_yf
    )


class TestYFinanceTickerMapperVerifyTicker:
    _RICH = {
        "currentPrice": 150.0,
        "marketCap": 2e12,
        "sector": "Tech",
        "industry": "Software",
        "exchange": "NMS",
        "currency": "USD",
    }

    def test_when_info_has_more_than_5_keys_and_current_price_then_true(self) -> None:
        assert _mapper_with_info(dict(self._RICH))._verify_ticker("AAPL") is True

    def test_when_info_has_regular_market_price_then_true(self) -> None:
        info = {**self._RICH}
        del info["currentPrice"]
        info["regularMarketPrice"] = 150.0
        assert _mapper_with_info(info)._verify_ticker("AAPL") is True

    def test_when_info_is_none_then_false(self) -> None:
        assert _mapper_with_info(None)._verify_ticker("AAPL") is False

    def test_when_info_has_five_or_fewer_keys_then_false(self) -> None:
        info = {"currentPrice": 10.0, "a": 1, "b": 2, "c": 3, "d": 4}  # exactly 5
        assert _mapper_with_info(info)._verify_ticker("AAPL") is False

    def test_when_info_has_no_price_field_then_false(self) -> None:
        info = {**self._RICH, "country": "US"}
        del info["currentPrice"]
        assert _mapper_with_info(info)._verify_ticker("AAPL") is False

    def test_when_info_is_empty_dict_then_false(self) -> None:
        assert _mapper_with_info({})._verify_ticker("AAPL") is False

    def test_verify_ticker_uses_injected_client_not_singleton(self) -> None:
        mapper = _mapper_with_info(dict(self._RICH))
        with patch(
            "app.services.universe.trading212.ticker_mapper.YFinanceClient"
        ) as mock_cls:
            mapper._verify_ticker("AAPL")
        mock_cls.get_instance.assert_not_called()
