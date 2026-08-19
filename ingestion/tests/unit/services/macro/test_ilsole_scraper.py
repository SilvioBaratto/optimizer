"""Unit tests for IlSoleScraper (issue #822).

Covers: row parsing, empty/unknown country, _safe_float, Accept-Encoding
regression guard, and circuit-breaker trip on repeated network failure.
NO network, NO real sleeps — all external collaborators are patched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.infrastructure.circuit_breaker import CircuitBreaker
from app.services.macro.scrapers.ilsole_scraper import IlSoleScraper

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

_REAL_INDICATORS_HTML = b"""
<html><body>
<table class="mainTable">
  <tbody>
    <tr>
      <td name="Paese">Stati Uniti</td>
      <td name="Pil_TT">2,5%</td>
      <td name="ProdIndustriale_AA">1,2%</td>
      <td name="Disoccupazione_AA">3,9%</td>
      <td name="PrezziConsumo_AA">3,1%</td>
      <td name="Deficit_AA">-5,5%</td>
      <td name="Debito_AA">120,0%</td>
      <td name="TassoSconto">5,5%</td>
      <td name="TassoInteresse">4,2%</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

_FORECAST_HTML = b"""
<html><body>
<table class="mainTable">
  <tbody>
    <tr>
      <td id="Paese">Usa</td>
      <td id="UltimaInflazione">3,1%</td>
      <td id="ConsensoInflazione">2,8%</td>
      <td id="MediaInfl10Anni">2,5%</td>
      <td id="ConsensoPil">2,2%</td>
      <td id="ConsensoUtili">10,5%</td>
      <td id="ConsensoEps">5,3%</td>
      <td id="ConsensoPeg">1,9%</td>
      <td id="ConsensoTassiLungo">4,1%</td>
      <td id="DataRiferimento">2026-01</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

_NO_TABLE_HTML = b"<html><body><p>Nothing here</p></body></html>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def _make_scraper_with_patches(
    content: bytes,
) -> tuple[IlSoleScraper, MagicMock, CircuitBreaker]:
    """Return (scraper, mock_rate_limiter, fresh_circuit_breaker)."""
    scraper = IlSoleScraper()
    mock_rl = MagicMock()
    fresh_cb = CircuitBreaker(service_name="t", max_attempts=5)
    scraper.session.get = MagicMock(return_value=_mock_response(content))
    return scraper, mock_rl, fresh_cb


# ---------------------------------------------------------------------------
# get_real_indicators — success path
# ---------------------------------------------------------------------------


class TestGetRealIndicatorsSuccess:
    def test_when_usa_row_present_then_gdp_growth_parsed(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_REAL_INDICATORS_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_real_indicators("USA")

        assert result is not None
        assert result["gdp_growth_qq"] == pytest.approx(2.5)

    def test_when_usa_row_present_then_st_rate_parsed(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_REAL_INDICATORS_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_real_indicators("USA")

        assert result is not None
        assert result["st_rate"] == pytest.approx(5.5)

    def test_when_usa_row_present_then_timestamp_present(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_REAL_INDICATORS_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_real_indicators("USA")

        assert result is not None
        assert "timestamp" in result


# ---------------------------------------------------------------------------
# get_forecasts — success path
# ---------------------------------------------------------------------------


class TestGetForecastsSuccess:
    def test_when_usa_forecast_row_present_then_last_inflation_parsed(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_FORECAST_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_forecasts("USA")

        assert result is not None
        assert result["last_inflation"] == pytest.approx(3.1)

    def test_when_usa_forecast_row_present_then_gdp_growth_6m_parsed(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_FORECAST_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_forecasts("USA")

        assert result is not None
        assert result["gdp_growth_6m"] == pytest.approx(2.2)

    def test_when_usa_forecast_row_present_then_reference_date_string(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_FORECAST_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_forecasts("USA")

        assert result is not None
        assert result["reference_date"] == "2026-01"


# ---------------------------------------------------------------------------
# get_real_indicators / get_forecasts — empty / unknown paths
# ---------------------------------------------------------------------------


class TestGetIndicatorsEmpty:
    def test_when_no_table_in_html_then_real_indicators_returns_none(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_NO_TABLE_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_real_indicators("USA")

        assert result is None

    def test_when_no_table_in_html_then_forecasts_returns_none(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_NO_TABLE_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_forecasts("USA")

        assert result is None

    def test_when_unknown_country_then_real_indicators_returns_none(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_REAL_INDICATORS_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_real_indicators("Atlantis")

        assert result is None

    def test_when_unknown_country_then_forecasts_returns_none(self) -> None:
        scraper, mock_rl, fresh_cb = _make_scraper_with_patches(_FORECAST_HTML)

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                fresh_cb,
            ),
        ):
            result = scraper.get_forecasts("Atlantis")

        assert result is None


# ---------------------------------------------------------------------------
# _safe_float — parametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2,5%", 2.5),
        ("5,5%", 5.5),
        ("3,1%", 3.1),
        ("-1,5%", -1.5),
        ("-1,5", -1.5),
        ("0-4,25%", 4.25),  # range: upper bound taken
        ("-", None),
        ("—", None),
        ("N/A", None),
        ("n/a", None),
        ("", None),
        (None, None),
        (1.5, 1.5),
        (0, 0.0),
    ],
)
def test_safe_float_parametrized(value: object, expected: float | None) -> None:
    result = IlSoleScraper._safe_float(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Accept-Encoding regression guards
# ---------------------------------------------------------------------------


class TestAcceptEncodingGuards:
    def test_when_ilsole_scraper_created_then_accept_encoding_excludes_br(
        self,
    ) -> None:
        """IlSoleScraper must not advertise brotli encoding."""
        scraper = IlSoleScraper()
        enc = str(scraper.session.headers.get("Accept-Encoding", ""))
        assert "br" not in enc

    def test_when_te_scraper_created_then_accept_encoding_is_gzip_deflate_only(
        self,
    ) -> None:
        """TradingEconomicsIndicatorsScraper must use gzip/deflate only.

        Regression guard for the 16-day TE table freeze caused by advertising
        "br" Accept-Encoding with no brotli decoder installed (2026-05-01).
        TE started serving Content-Encoding: br; requests could not decompress,
        every parse matched 0 rows and the circuit breaker latched open.
        """
        from app.services.macro.scrapers.tradingeconomics_scraper import (
            TradingEconomicsIndicatorsScraper,
        )

        scraper = TradingEconomicsIndicatorsScraper()
        enc = str(scraper.session.headers["Accept-Encoding"])
        assert "br" not in enc
        assert enc == "gzip, deflate"


# ---------------------------------------------------------------------------
# Circuit breaker — trip on repeated network failure
# ---------------------------------------------------------------------------


class TestCircuitBreakerTripsOnNetworkFailure:
    def test_when_network_errors_on_every_attempt_then_trigger_called(self) -> None:
        """circuit_breaker.trigger() must be called when a transient error fires."""
        import requests as req_lib

        scraper = IlSoleScraper()
        scraper.session.get = MagicMock(
            side_effect=req_lib.exceptions.ConnectionError("Connection reset by peer")
        )
        mock_rl = MagicMock()
        mock_cb = MagicMock()
        mock_cb.check = MagicMock()  # don't raise — let retry proceed

        with (
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_rate_limiter",
                mock_rl,
            ),
            patch(
                "app.services.macro.scrapers.ilsole_scraper._ilsole_circuit_breaker",
                mock_cb,
            ),
            patch(
                "app.services.infrastructure.retry.time.sleep",
                MagicMock(),
            ),
        ):
            result = scraper.get_real_indicators("USA")

        assert result is None
        assert mock_cb.trigger.call_count >= 1

    def test_when_circuit_breaker_max_attempts_exceeded_then_check_raises(
        self,
    ) -> None:
        """After max_attempts triggers, check() must raise RuntimeError immediately.

        trigger() only increments _attempt when the breaker is not already active,
        so we force _attempt to max_attempts directly via the internal field.
        """
        cb = CircuitBreaker(service_name="t", max_attempts=2)
        # Bypass the "already active, skip" guard by directly setting the count
        cb._attempt = 2  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError):
            cb.check()
