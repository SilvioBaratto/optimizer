"""``YFINANCE_NEWS_SCRAPE_DELAY_SECONDS`` paces the article scraper of the sweep.

Validates ``app/config.py`` (``Settings.yfinance_news_scrape_delay_seconds``) and
``app/services/market_data/yfinance_data_service.py`` (``YFinanceDataService``
hands the delay to the scraper of its news step). No requirement id is registered
for this setting yet.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.config import Settings, settings
from app.services.market_data.yfinance_data_service import YFinanceDataService
from tests.unit.test_yfinance_data_service_branches import (
    _make_repo,
    _make_yf_client_silent,
)

_ENV = "YFINANCE_NEWS_SCRAPE_DELAY_SECONDS"
# The sweep builds its scraper from the class in its home module.
_SCRAPER_CLASS = "app.services.market_data.yfinance.news.scraper.ArticleScraper"


def _settings(monkeypatch: pytest.MonkeyPatch, value: str | None) -> Settings:
    if value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, value)
    # A developer's .env must not decide what the default looks like.
    return Settings(_env_file=None)


def _scraper_class_after_a_sweep(service: YFinanceDataService) -> MagicMock:
    """Sweep one ticker whose feed has one linked article; return the scraper class."""
    feed = MagicMock()
    feed.news = [{"content": {"canonicalUrl": {"url": "https://example.com/a"}}}]
    service.yf_client.get_ticker.return_value = feed
    with patch(_SCRAPER_CLASS) as scraper_class:
        scraper_class.return_value.fetch.return_value = {
            "success": False,
            "content": None,
        }
        service.fetch_and_store(
            instrument_id=uuid4(), yfinance_ticker="AAPL", mode="full"
        )
    return scraper_class


class TestNewsScrapeDelaySetting:
    """The operator's knob for the pause before each scraped article page."""

    def test_defaults_to_half_a_second(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset, the sweep keeps the pace it always had."""
        assert _settings(monkeypatch, None).yfinance_news_scrape_delay_seconds == 0.5

    def test_env_overrides_the_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The value is read from the environment."""
        assert _settings(monkeypatch, "2.5").yfinance_news_scrape_delay_seconds == 2.5

    @pytest.mark.parametrize(("value", "expected"), [("0", 0.0), ("10", 10.0)])
    def test_both_bounds_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch, value: str, expected: float
    ) -> None:
        """Zero scrapes without a pause; ten seconds is the slowest pace allowed."""
        delay = _settings(monkeypatch, value).yfinance_news_scrape_delay_seconds

        assert delay == expected

    @pytest.mark.parametrize("value", ["-0.1", "10.5"])
    def test_a_delay_outside_the_range_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """A negative pause is meaningless, and a long one would stall the sweep."""
        with pytest.raises(ValidationError, match=_ENV):
            _settings(monkeypatch, value)


class TestSweepPacesItsScraper:
    """The news step scrapes at the pace the service was given."""

    def test_an_injected_delay_reaches_the_scraper(self) -> None:
        """A caller's delay wins over the settings."""
        service = YFinanceDataService(
            _make_repo(), _make_yf_client_silent(), news_scrape_delay=2.0
        )

        scraper_class = _scraper_class_after_a_sweep(service)

        scraper_class.assert_called_once_with(delay=2.0)
        scraper_class.return_value.fetch.assert_called_once_with(
            "https://example.com/a"
        )

    def test_without_one_the_configured_delay_is_used(self) -> None:
        """A service built by the bulk sweeps takes the daemon's setting."""
        with patch.object(settings, "yfinance_news_scrape_delay_seconds", 1.5):
            service = YFinanceDataService(_make_repo(), _make_yf_client_silent())

        scraper_class = _scraper_class_after_a_sweep(service)

        scraper_class.assert_called_once_with(delay=1.5)
