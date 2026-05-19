"""Tests for FxRateProvider (issue #775)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _mk_df(close: float) -> pd.DataFrame:
    return pd.DataFrame({"Close": [close]}, index=pd.DatetimeIndex(["2026-05-19"]))


def _mk_empty_df() -> pd.DataFrame:
    return pd.DataFrame({"Close": []}, index=pd.DatetimeIndex([]))


def _mk_nan_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [float("nan")]}, index=pd.DatetimeIndex(["2026-05-19"])
    )


class TestSameCurrencyShortCircuit:
    def test_when_from_equals_base_then_returns_one_exact(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        assert provider.get_rate_to_base("EUR", date(2026, 5, 19)) == 1.0

    def test_when_case_differs_then_still_short_circuits(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        assert provider.get_rate_to_base("eur", date(2026, 5, 19)) == 1.0

    def test_when_from_equals_base_then_no_fetch_no_cache_write(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            provider.get_rate_to_base("EUR", date(2026, 5, 19))
            yf_mock.download.assert_not_called()
        assert provider._cache.size() == 0  # type: ignore[attr-defined]


class TestPenceRejection:
    @pytest.mark.parametrize("pence_code", ["GBX", "GBp", "GBP_PENCE", "gbx"])
    def test_when_pence_currency_then_value_error(self, pence_code: str) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with pytest.raises(ValueError, match="pence|GBX|GBP"):
            provider.get_rate_to_base(pence_code, date(2026, 5, 19))


class TestDirectFetch:
    def test_when_usd_to_eur_then_inverts_eurusd_close(self) -> None:
        """USD→EUR: ticker EURUSD=X; rate = 1 / close."""
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_df(1.08)
            rate = provider.get_rate_to_base("USD", date(2026, 5, 19))

        assert rate is not None
        assert abs(rate - (1.0 / 1.08)) < 1e-9

    def test_when_eur_to_usd_then_uses_eurusd_close_direct(self) -> None:
        """EUR→USD: ticker EURUSD=X; rate = close directly."""
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="USD")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_df(1.08)
            rate = provider.get_rate_to_base("EUR", date(2026, 5, 19))

        assert rate == pytest.approx(1.08)


class TestCrossCurrencyFetch:
    def test_when_gbp_to_eur_then_uses_two_usd_tickers_for_cross_rate(self) -> None:
        """GBP→EUR: fetch GBPUSD and EURUSD; rate = GBPUSD / EURUSD."""
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")

        def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
            if "GBPUSD" in ticker:
                return _mk_df(1.25)
            if "EURUSD" in ticker:
                return _mk_df(1.08)
            return _mk_empty_df()

        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.side_effect = fake_download
            rate = provider.get_rate_to_base("GBP", date(2026, 5, 19))

        assert rate == pytest.approx(1.25 / 1.08)
        assert yf_mock.download.call_count == 2


class TestCachingBehavior:
    def test_when_called_twice_then_second_call_is_cache_hit(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_df(1.08)

            first = provider.get_rate_to_base("USD", date(2026, 5, 19))
            second = provider.get_rate_to_base("USD", date(2026, 5, 19))

            assert first == second
            assert yf_mock.download.call_count == 1

    def test_when_different_dates_then_separate_cache_entries(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_df(1.08)
            provider.get_rate_to_base("USD", date(2026, 5, 19))
            provider.get_rate_to_base("USD", date(2026, 5, 20))

            assert yf_mock.download.call_count == 2

    def test_when_cache_key_format_then_matches_spec(self) -> None:
        """Key format: f'{from}_{base}_{date.isoformat()}' uppercase."""
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_df(1.08)
            provider.get_rate_to_base("usd", date(2026, 5, 19))

        assert provider._cache.contains("USD_EUR_2026-05-19")  # type: ignore[attr-defined]

    def test_when_clear_cache_then_empty(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_df(1.08)
            provider.get_rate_to_base("USD", date(2026, 5, 19))

        assert provider._cache.size() == 1  # type: ignore[attr-defined]
        provider.clear_cache()
        assert provider._cache.size() == 0  # type: ignore[attr-defined]


class TestFetchFailures:
    def test_when_empty_dataframe_then_returns_none_and_no_cache(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_empty_df()
            result = provider.get_rate_to_base("USD", date(2026, 5, 19))

        assert result is None
        assert provider._cache.size() == 0  # type: ignore[attr-defined]

    def test_when_all_nan_dataframe_then_returns_none(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_nan_df()
            result = provider.get_rate_to_base("USD", date(2026, 5, 19))

        assert result is None
        assert provider._cache.size() == 0  # type: ignore[attr-defined]

    def test_when_network_exception_then_returns_none(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")
        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.side_effect = RuntimeError("boom")
            result = provider.get_rate_to_base("USD", date(2026, 5, 19))

        assert result is None
        assert provider._cache.size() == 0  # type: ignore[attr-defined]

    def test_when_cross_pair_one_leg_empty_then_returns_none(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        provider = FxRateProvider(base_currency="EUR")

        def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
            if "GBPUSD" in ticker:
                return _mk_df(1.25)
            return _mk_empty_df()

        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.side_effect = fake_download
            result = provider.get_rate_to_base("GBP", date(2026, 5, 19))

        assert result is None


class TestCircuitBreakerTripped:
    def test_when_breaker_active_then_returns_none_without_fetch(self) -> None:
        from app.services.infrastructure import CircuitBreaker
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        breaker = CircuitBreaker(service_name="FX rates", max_attempts=5)
        breaker.trigger()  # arm the breaker
        provider = FxRateProvider(base_currency="EUR", circuit_breaker=breaker)

        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            result = provider.get_rate_to_base("USD", date(2026, 5, 19))
            yf_mock.download.assert_not_called()

        assert result is None


class TestRateLimiterCall:
    def test_when_fetch_then_rate_limiter_acquire_called_with_from_ccy(self) -> None:
        from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
            FxRateProvider,
        )

        limiter = MagicMock()
        provider = FxRateProvider(base_currency="EUR", rate_limiter=limiter)

        with patch(
            "app.services.portfolio.t212_position_normalizer._fx_rate_provider.yf"
        ) as yf_mock:
            yf_mock.download.return_value = _mk_df(1.08)
            provider.get_rate_to_base("USD", date(2026, 5, 19))

        limiter.acquire.assert_called_with(key="USD")
