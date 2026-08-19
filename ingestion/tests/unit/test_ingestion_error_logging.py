"""Data-ingestion error-surfacing tests (issue #850).

Each previously-silent per-item drop now logs before continuing. These tests
lock that behaviour: a malformed item is skipped/degraded *and* logged, never
swallowed without a trace.
"""

from __future__ import annotations

import logging
from datetime import date
from unittest.mock import MagicMock, patch

from app.services.macro.scrapers.fred_scraper import FredScraper
from app.services.macro.scrapers.ilsole_scraper import IlSoleScraper
from app.services.market_data.yfinance.news.aggregator import _parse_article_date


def test_when_article_pub_time_is_unparseable_then_none_is_returned_and_logged(
    caplog,
) -> None:
    """when a pub_time cannot be parsed, None is returned and a debug log fires."""
    with caplog.at_level(logging.DEBUG):
        result = _parse_article_date("definitely-not-a-date")
    assert result is None
    assert any("Unparseable article pub_time" in r.getMessage() for r in caplog.records)


def test_when_fred_observation_date_is_malformed_then_row_is_skipped_and_logged(
    caplog,
) -> None:
    """when one FRED observation has a bad date, it is skipped and logged."""
    raw = [
        {"date": "NOT-A-DATE", "value": "1.0"},
        {"date": "2026-01-01", "value": "2.0"},
    ]
    with caplog.at_level(logging.DEBUG):
        parsed = FredScraper._parse_observations(raw)
    assert len(parsed) == 1
    assert parsed[0]["date"] == date(2026, 1, 1)
    assert any("bad date" in r.getMessage() for r in caplog.records)


def test_when_ilsole_real_indicators_parse_raises_then_none_and_logged(
    caplog,
) -> None:
    """when the IlSole indicators page parse raises, None is returned and logged."""
    scraper = IlSoleScraper()
    bad_soup = MagicMock()
    bad_soup.find.side_effect = RuntimeError("site structure changed")
    with patch.object(scraper, "_fetch_page", return_value=bad_soup):
        with caplog.at_level(logging.WARNING):
            result = scraper.get_real_indicators("USA")
    assert result is None
    assert any("real-indicators parse failed" in r.getMessage() for r in caplog.records)


def test_when_ilsole_forecasts_parse_raises_then_none_and_logged(caplog) -> None:
    """when the IlSole forecasts page parse raises, None is returned and logged."""
    scraper = IlSoleScraper()
    bad_soup = MagicMock()
    bad_soup.find.side_effect = RuntimeError("site structure changed")
    with patch.object(scraper, "_fetch_page", return_value=bad_soup):
        with caplog.at_level(logging.WARNING):
            result = scraper.get_forecasts("USA")
    assert result is None
    assert any("forecasts parse failed" in r.getMessage() for r in caplog.records)
