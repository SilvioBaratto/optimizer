"""HTTP error handling of ``FredScraper.get_series_observations``.

No network: ``requests.Session.get`` and the retry sleep are patched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.macro.scrapers.fred_scraper import FredScraper

_MOD = "app.services.macro.scrapers.fred_scraper"
_BAD_KEY = "Bad Request.  The value for variable api_key is not registered."


def _response(status: int, payload: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.json.return_value = payload if payload is not None else {}

    def _raise_for_status() -> None:
        if status >= 400:
            reason = "Too Many Requests" if status == 429 else "Error"
            raise requests.HTTPError(f"{status} Client Error: {reason}", response=resp)

    resp.raise_for_status.side_effect = _raise_for_status
    return resp


@pytest.fixture
def scraper() -> FredScraper:
    return FredScraper(api_key="test-key")


@pytest.fixture(autouse=True)
def _no_sleep_no_breaker():
    with (
        patch("app.services.infrastructure.retry.time.sleep"),
        patch(f"{_MOD}._fred_circuit_breaker"),
    ):
        yield


class TestPermanentClientErrors:
    def test_bad_api_key_is_not_retried_and_keeps_fred_message(
        self, scraper: FredScraper
    ) -> None:
        bad_key = _response(400, {"error_code": 400, "error_message": _BAD_KEY})

        with patch.object(scraper.session, "get", return_value=bad_key) as get:
            result = scraper.get_series_observations("DGS10")

        assert get.call_count == 1
        assert result["status"] == "error"
        assert "api_key is not registered" in result["error"]

    def test_client_error_without_json_body_reports_the_status(
        self, scraper: FredScraper
    ) -> None:
        resp = _response(404)
        resp.json.side_effect = ValueError("no body")

        with patch.object(scraper.session, "get", return_value=resp) as get:
            result = scraper.get_series_observations("NOPE")

        assert get.call_count == 1
        assert result["status"] == "error"
        assert "404" in result["error"]


class TestTransientErrorsStillRetry:
    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_rate_limit_and_server_errors_are_retried(
        self, scraper: FredScraper, status: int
    ) -> None:
        with patch.object(
            scraper.session, "get", return_value=_response(status)
        ) as get:
            result = scraper.get_series_observations("DGS10")

        assert get.call_count == 5
        assert result["error"] == "Fetch failed after retries"
