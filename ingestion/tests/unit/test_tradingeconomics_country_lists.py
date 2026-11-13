"""The two Trading Economics ``country-list`` pages, answer by answer.

Validates ``app/services/macro/scrapers/tradingeconomics_scraper.py``
(``get_industrial_production_all``, ``get_capacity_utilization_all`` and their
table parsers): the request each page sends, the exact result for a parsed
page, a failed fetch, a page whose table drifted and a parser that raises, and
which rows the parsers keep. Both pages must answer the same way except for
their URL and their name in messages. No requirement id is registered for the
Trading Economics bulk pages yet.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from bs4 import BeautifulSoup

from app.services.macro.scrapers.exceptions import ParseStructureError
from app.services.macro.scrapers.tradingeconomics_scraper import (
    _MIN_CAPACITY_UTILIZATION_ROWS,
    _MIN_INDUSTRIAL_PRODUCTION_ROWS,
    TradingEconomicsIndicatorsScraper,
)

_MOD = "app.services.macro.scrapers.tradingeconomics_scraper"
_BASE = "https://tradingeconomics.com"

_PAGES = [
    pytest.param(
        "get_industrial_production_all",
        "_parse_industrial_production_table",
        f"{_BASE}/country-list/industrial-production",
        "industrial production",
        _MIN_INDUSTRIAL_PRODUCTION_ROWS,
        id="industrial-production",
    ),
    pytest.param(
        "get_capacity_utilization_all",
        "_parse_capacity_utilization_table",
        f"{_BASE}/country-list/capacity-utilization?continent=g20",
        "capacity utilization",
        _MIN_CAPACITY_UTILIZATION_ROWS,
        id="capacity-utilization",
    ),
]

_PARSERS = [
    pytest.param("_parse_industrial_production_table", "industrial production"),
    pytest.param("_parse_capacity_utilization_table", "capacity utilization"),
]


def _row(*cells: str, linked: bool = True) -> str:
    first, *rest = cells
    head = f"<td><a href='#'>{first}</a></td>" if linked else f"<td>{first}</td>"
    return "<tr>" + head + "".join(f"<td>{c}</td>" for c in rest) + "</tr>"


def _page(*rows: str) -> bytes:
    return ("<table><tbody>" + "".join(rows) + "</tbody></table>").encode()


_GOOD_ROWS = (
    _row("Germany", "-1.2", "0.4", "Aug/26", "%"),
    _row("India", "8", "6.1", "Aug/26", "%"),
    _row("Japan", "1.5", "bad", "Jul/26", "%"),
)


def _scraper() -> TradingEconomicsIndicatorsScraper:
    scraper = TradingEconomicsIndicatorsScraper(timeout=7, rate_limit_delay=0.0)
    scraper.session = MagicMock(name="session")
    return scraper


def _answer(scraper: TradingEconomicsIndicatorsScraper, body: bytes) -> None:
    response = MagicMock()
    response.content = body
    response.raise_for_status.return_value = None
    scraper.session.get.return_value = response  # type: ignore[attr-defined]


class TestGetter:
    """Each getter fetches its page once and wraps the parser's answer."""

    @pytest.mark.parametrize(("getter", "parser", "url", "label", "minimum"), _PAGES)
    def test_a_parsed_page_gives_the_rows_and_its_url(
        self, getter: str, parser: str, url: str, label: str, minimum: int
    ) -> None:
        """Request the page once and return every kept row under ``data``."""
        scraper = _scraper()
        _answer(scraper, _page(*_GOOD_ROWS))

        with patch(f"{_MOD}._te_circuit_breaker") as breaker:
            result = getattr(scraper, getter)()

        assert set(result) == {"status", "source_url", "data", "timestamp"}
        assert result["status"] == "success"
        assert result["source_url"] == url
        assert result["data"] == {
            "Germany": {
                "value": -1.2,
                "previous": 0.4,
                "reference": "Aug/26",
                "unit": "%",
            },
            "India": {
                "value": 8.0,
                "previous": 6.1,
                "reference": "Aug/26",
                "unit": "%",
            },
            "Japan": {
                "value": 1.5,
                "previous": None,
                "reference": "Jul/26",
                "unit": "%",
            },
        }
        scraper.session.get.assert_called_once_with(url, timeout=7)  # type: ignore[attr-defined]
        breaker.check.assert_called_once_with()
        breaker.reset.assert_called_once_with()
        breaker.trigger.assert_not_called()

    @pytest.mark.parametrize(("getter", "parser", "url", "label", "minimum"), _PAGES)
    def test_a_failed_fetch_is_an_error_without_a_parse_flag(
        self, getter: str, parser: str, url: str, label: str, minimum: int
    ) -> None:
        """No response after the retries gives a plain error and no parse."""
        scraper = _scraper()

        with patch(f"{_MOD}.retry_with_backoff", return_value=None) as retry:
            result = getattr(scraper, getter)()

        assert result == {"status": "error", "error": "Fetch failed after retries"}
        assert retry.call_args.kwargs["max_retries"] == 3

    @pytest.mark.parametrize(("getter", "parser", "url", "label", "minimum"), _PAGES)
    def test_a_drifted_table_trips_the_breaker(
        self, getter: str, parser: str, url: str, label: str, minimum: int
    ) -> None:
        """Too few rows is a structure change: flagged, and the breaker trips."""
        scraper = _scraper()
        _answer(scraper, _page(_row("Germany", "-1.2", "0.4", "Aug/26", "%")))

        with patch(f"{_MOD}._te_circuit_breaker") as breaker:
            result = getattr(scraper, getter)()

        assert result == {
            "status": "error",
            "error": (
                f"Parse structure failure: Expected at least {minimum} {label} "
                "rows, got 1. The Trading Economics country-list table structure "
                "may have changed."
            ),
            "parse_failure": True,
        }
        breaker.trigger.assert_called_once_with()

    @pytest.mark.parametrize(("getter", "parser", "url", "label", "minimum"), _PAGES)
    def test_a_parser_crash_is_an_error_that_leaves_the_breaker(
        self, getter: str, parser: str, url: str, label: str, minimum: int
    ) -> None:
        """Any other parser failure is reported and does not trip the breaker."""
        scraper = _scraper()
        _answer(scraper, _page(*_GOOD_ROWS))

        with (
            patch(f"{_MOD}._te_circuit_breaker") as breaker,
            patch.object(
                TradingEconomicsIndicatorsScraper,
                parser,
                side_effect=ValueError("bad cell"),
            ) as parse,
        ):
            result = getattr(scraper, getter)()

        assert result == {"status": "error", "error": "Parsing failed: bad cell"}
        assert parse.call_count == 1
        breaker.trigger.assert_not_called()

    @pytest.mark.parametrize(("getter", "parser", "url", "label", "minimum"), _PAGES)
    def test_the_getter_goes_through_the_rate_limiter_key_of_the_host(
        self, getter: str, parser: str, url: str, label: str, minimum: int
    ) -> None:
        """Every Trading Economics page shares one pacing key."""
        scraper = _scraper()
        scraper._rate_limiter = MagicMock(name="rate_limiter")
        _answer(scraper, _page(*_GOOD_ROWS))

        with patch(f"{_MOD}._te_circuit_breaker"):
            getattr(scraper, getter)()

        assert scraper._rate_limiter.acquire.call_args_list == [
            call("tradingeconomics.com")
        ]


class TestParser:
    """Which rows a country-list parser keeps, and when it gives up."""

    @pytest.mark.parametrize(("parser", "label"), _PARSERS)
    def test_rows_that_are_not_a_linked_country_with_a_value_are_skipped(
        self, parser: str, label: str
    ) -> None:
        """Keep linked rows of three or more cells whose value parses."""
        soup = BeautifulSoup(
            _page(
                *_GOOD_ROWS,
                _row("Unlinked", "1.0", "2.0", linked=False),
                _row("Short", "1.0"),
                _row("No value", "n/a", "2.0"),
                _row("Brazil", "2.5", "2.0"),
                _row("Germany", "-0.9", "-1.2", "Sep/26", "%"),
            ),
            "html.parser",
        )

        rows: dict[str, Any] = getattr(
            TradingEconomicsIndicatorsScraper(rate_limit_delay=0.0), parser
        )(soup)

        assert list(rows) == ["Germany", "India", "Japan", "Brazil"]
        assert rows["Brazil"] == {
            "value": 2.5,
            "previous": 2.0,
            "reference": "",
            "unit": "",
        }
        assert rows["Germany"]["reference"] == "Sep/26"

    @pytest.mark.parametrize(("getter", "parser", "url", "label", "minimum"), _PAGES)
    def test_too_few_rows_raise_with_the_page_named(
        self, getter: str, parser: str, url: str, label: str, minimum: int
    ) -> None:
        """The error carries the threshold, the count and the page's name."""
        soup = BeautifulSoup(_page(*_GOOD_ROWS[: minimum - 1]), "html.parser")

        with pytest.raises(ParseStructureError) as raised:
            getattr(TradingEconomicsIndicatorsScraper(rate_limit_delay=0.0), parser)(
                soup
            )

        assert str(raised.value) == (
            f"Expected at least {minimum} {label} rows, got {minimum - 1}. The "
            "Trading Economics country-list table structure may have changed."
        )
        assert raised.value.url == f"({label} page)"
        assert raised.value.rows_found == minimum - 1
