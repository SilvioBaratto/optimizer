"""Unit tests for TradingEconomicsIndicatorsScraper — GitHub issue #317.

Covers ParseStructureError propagation, circuit breaker triggering,
fetch failure handling, _extract_number edge cases, and unsupported
country early-exit behaviour.

No database, no HTTP calls — all external I/O is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from app.services.scrapers.exceptions import ParseStructureError
from app.services.scrapers.tradingeconomics_scraper import (
    _MIN_BOND_YIELDS,
    _MIN_CAPACITY_UTILIZATION_ROWS,
    _MIN_INDICATORS,
    _MIN_INDUSTRIAL_PRODUCTION_ROWS,
    TradingEconomicsIndicatorsScraper,
)

# ---------------------------------------------------------------------------
# Module path used for patching module-level names
# ---------------------------------------------------------------------------

_MOD = "app.services.scrapers.tradingeconomics_scraper"

# ---------------------------------------------------------------------------
# HTML fixture helpers
# ---------------------------------------------------------------------------


def _indicators_row(name: str, last: str, prev: str = "1.0") -> str:
    """Return a <tr> with 7 cells matching the indicators table column layout."""
    return (
        "<tr>"
        f"<td><a href='#'>{name}</a></td>"
        f"<td>{last}</td>"
        f"<td>{prev}</td>"
        "<td>3.0</td>"
        "<td>0.5</td>"
        "<td>percent</td>"
        "<td>Feb 2026</td>"
        "</tr>"
    )


def _indicators_soup(*rows: str) -> BeautifulSoup:
    """Wrap rows in a table-hover table."""
    html = (
        '<table class="table table-hover"><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )
    return BeautifulSoup(html, "html.parser")


def _bond_row(name: str, yield_val: str) -> str:
    """Return a <tr> with 7 cells matching the bond yields table layout."""
    return (
        "<tr>"
        f"<td><a href='#'>{name}</a></td>"
        f"<td>{yield_val}</td>"
        "<td>4.50</td>"
        "<td>0.02</td>"
        "<td>0.10</td>"
        "<td>-0.50</td>"
        "<td>Mar/2026</td>"
        "</tr>"
    )


def _bond_soup(*rows: str) -> BeautifulSoup:
    """Wrap rows in a table-heatmap table."""
    html = (
        '<table class="table-heatmap"><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )
    return BeautifulSoup(html, "html.parser")


def _country_row(country: str, last: str) -> str:
    """Return a <tr> for industrial-production / capacity-utilization tables."""
    return (
        "<tr>"
        f"<td><a href='#'>{country}</a></td>"
        f"<td>{last}</td>"
        "<td>2.0</td>"
        "<td>Jan 2026</td>"
        "<td>percent</td>"
        "</tr>"
    )


def _country_list_soup(*rows: str) -> BeautifulSoup:
    html = (
        '<table class="table-heatmap"><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )
    return BeautifulSoup(html, "html.parser")


def _make_scraper() -> TradingEconomicsIndicatorsScraper:
    return TradingEconomicsIndicatorsScraper(timeout=5, rate_limit_delay=0.0)


def _fake_response(content: bytes = b"<html></html>") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# TestParseStructureError — direct parser method tests
# ---------------------------------------------------------------------------


class TestParseStructureError:
    # --- _parse_indicators_table ---

    def test_no_table_hover_raises(self) -> None:
        """Soup with no table-hover table → zero matched indicators → raises."""
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError) as exc_info:
            scraper._parse_indicators_table(soup)
        assert exc_info.value.rows_found == 0

    def test_empty_soup_raises_for_indicators(self) -> None:
        """Completely empty soup → raises with rows_found == 0."""
        soup = BeautifulSoup("", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError) as exc_info:
            scraper._parse_indicators_table(soup)
        assert exc_info.value.rows_found == 0

    def test_unrecognised_indicator_names_raise(self) -> None:
        """Rows whose names match no INDICATOR_PATTERNS key produce zero matches → raises."""
        soup = _indicators_soup(
            _indicators_row("Some Unknown Indicator", "5.0"),
            _indicators_row("Another Unrecognised Stat", "3.2"),
        )
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError) as exc_info:
            scraper._parse_indicators_table(soup)
        assert exc_info.value.rows_found == 0

    def test_below_min_matched_rows_raises(self) -> None:
        """Only 1 matched row when minimum is 3 → raises with correct rows_found."""
        soup = _indicators_soup(
            _indicators_row("GDP Growth Rate", "2.1"),
        )
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError) as exc_info:
            scraper._parse_indicators_table(soup)
        assert exc_info.value.rows_found == 1
        assert exc_info.value.rows_found < _MIN_INDICATORS

    def test_indicators_parse_error_carries_url_attribute(self) -> None:
        """ParseStructureError from indicators parser exposes url=(indicators page)."""
        soup = BeautifulSoup("", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError) as exc_info:
            scraper._parse_indicators_table(soup)
        assert exc_info.value.url == "(indicators page)"

    # --- _parse_bond_yields_table ---

    def test_no_table_heatmap_raises_for_bonds(self) -> None:
        """Soup with no table-heatmap or sortable-theme-minimal → raises."""
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError) as exc_info:
            scraper._parse_bond_yields_table(soup, "USA")
        assert exc_info.value.rows_found == 0
        assert exc_info.value.rows_found < _MIN_BOND_YIELDS

    def test_empty_soup_raises_for_bonds(self) -> None:
        soup = BeautifulSoup("", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError):
            scraper._parse_bond_yields_table(soup, "UK")

    def test_non_target_maturities_skipped_and_raise(self) -> None:
        """Rows with 3M/6M maturities are filtered; threshold not met → raises."""
        soup = _bond_soup(
            _bond_row("USA 3M", "5.25"),
            _bond_row("USA 6M", "5.10"),
        )
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError):
            scraper._parse_bond_yields_table(soup, "USA")

    def test_tips_rows_skipped_causing_raise(self) -> None:
        """TIPS bond rows are excluded; if only TIPS rows exist, raises."""
        soup = _bond_soup(_bond_row("USA 10Y TIPS", "1.80"))
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError):
            scraper._parse_bond_yields_table(soup, "USA")

    # --- _parse_industrial_production_table ---

    def test_empty_soup_raises_for_industrial_production(self) -> None:
        soup = BeautifulSoup("", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError) as exc_info:
            scraper._parse_industrial_production_table(soup)
        assert exc_info.value.rows_found < _MIN_INDUSTRIAL_PRODUCTION_ROWS

    def test_no_table_heatmap_raises_for_industrial_production(self) -> None:
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError):
            scraper._parse_industrial_production_table(soup)

    # --- _parse_capacity_utilization_table ---

    def test_empty_soup_raises_for_capacity_utilization(self) -> None:
        soup = BeautifulSoup("", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError) as exc_info:
            scraper._parse_capacity_utilization_table(soup)
        assert exc_info.value.rows_found < _MIN_CAPACITY_UTILIZATION_ROWS

    def test_no_table_heatmap_raises_for_capacity_utilization(self) -> None:
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        scraper = _make_scraper()
        with pytest.raises(ParseStructureError):
            scraper._parse_capacity_utilization_table(soup)


# ---------------------------------------------------------------------------
# TestGetCountryIndicators — public method with mocked HTTP
# ---------------------------------------------------------------------------


class TestGetCountryIndicators:
    def test_parse_failure_returns_parse_failure_flag(self) -> None:
        """ParseStructureError from parser → result has status='error' and
        parse_failure=True."""
        scraper = _make_scraper()
        parse_err = ParseStructureError(
            "structure changed", url="http://x", rows_found=0
        )

        with (
            patch(f"{_MOD}.retry_with_backoff", return_value=_fake_response()),
            patch.object(scraper, "_parse_indicators_table", side_effect=parse_err),
            patch(f"{_MOD}._te_circuit_breaker"),
        ):
            result = scraper.get_country_indicators("USA", include_bonds=False)

        assert result["status"] == "error"
        assert result.get("parse_failure") is True

    def test_parse_failure_triggers_circuit_breaker(self) -> None:
        """ParseStructureError must call _te_circuit_breaker.trigger() exactly once."""
        scraper = _make_scraper()
        parse_err = ParseStructureError(
            "structure changed", url="http://x", rows_found=0
        )

        with (
            patch(f"{_MOD}.retry_with_backoff", return_value=_fake_response()),
            patch.object(scraper, "_parse_indicators_table", side_effect=parse_err),
            patch(f"{_MOD}._te_circuit_breaker") as mock_cb,
        ):
            scraper.get_country_indicators("USA", include_bonds=False)

        mock_cb.trigger.assert_called_once()

    def test_fetch_failure_returns_error_no_parse_failure(self) -> None:
        """retry_with_backoff returning None → status='error', no parse_failure key."""
        scraper = _make_scraper()

        with patch(f"{_MOD}.retry_with_backoff", return_value=None):
            result = scraper.get_country_indicators("USA")

        assert result["status"] == "error"
        assert "parse_failure" not in result

    def test_fetch_failure_error_message_mentions_retries(self) -> None:
        """Fetch failure error message signals retry exhaustion."""
        scraper = _make_scraper()

        with patch(f"{_MOD}.retry_with_backoff", return_value=None):
            result = scraper.get_country_indicators("Germany")

        error_lower = result["error"].lower()
        assert "fetch" in error_lower or "retr" in error_lower

    def test_circuit_breaker_not_triggered_on_plain_fetch_failure(self) -> None:
        """A plain fetch failure must NOT call circuit breaker trigger."""
        scraper = _make_scraper()

        with (
            patch(f"{_MOD}.retry_with_backoff", return_value=None),
            patch(f"{_MOD}._te_circuit_breaker") as mock_cb,
        ):
            scraper.get_country_indicators("USA")

        mock_cb.trigger.assert_not_called()

    def test_result_country_key_matches_input(self) -> None:
        """Error result always echoes back the requested country name."""
        scraper = _make_scraper()

        with patch(f"{_MOD}.retry_with_backoff", return_value=None):
            result = scraper.get_country_indicators("Japan")

        assert result["country"] == "Japan"


# ---------------------------------------------------------------------------
# TestExtractNumber
# ---------------------------------------------------------------------------


class TestExtractNumber:
    def setup_method(self) -> None:
        self.scraper = _make_scraper()

    def test_none_input_returns_none(self) -> None:
        assert self.scraper._extract_number(None) is None  # type: ignore[arg-type]

    def test_empty_string_returns_none(self) -> None:
        assert self.scraper._extract_number("") is None

    def test_na_string_returns_none(self) -> None:
        assert self.scraper._extract_number("N/A") is None

    def test_dash_returns_none(self) -> None:
        assert self.scraper._extract_number("-") is None

    def test_percentage_string(self) -> None:
        assert self.scraper._extract_number("5.3%") == pytest.approx(5.3)

    def test_negative_decimal(self) -> None:
        assert self.scraper._extract_number("-2.1") == pytest.approx(-2.1)

    def test_dollar_with_commas(self) -> None:
        assert self.scraper._extract_number("$1,234.5") == pytest.approx(1234.5)

    def test_plain_integer(self) -> None:
        assert self.scraper._extract_number("42") == pytest.approx(42.0)

    def test_euro_symbol(self) -> None:
        assert self.scraper._extract_number("€3.14") == pytest.approx(3.14)

    def test_whitespace_only_returns_none(self) -> None:
        assert self.scraper._extract_number("   ") is None

    def test_na_lowercase_returns_none(self) -> None:
        assert self.scraper._extract_number("na") is None


# ---------------------------------------------------------------------------
# TestUnsupportedCountry
# ---------------------------------------------------------------------------


class TestUnsupportedCountry:
    def test_invalid_country_returns_error_status(self) -> None:
        scraper = _make_scraper()
        result = scraper.get_country_indicators("INVALID")
        assert result["status"] == "error"
        assert result["country"] == "INVALID"

    def test_invalid_country_makes_no_http_call(self) -> None:
        scraper = _make_scraper()

        with patch(f"{_MOD}.retry_with_backoff") as mock_rwb:
            scraper.get_country_indicators("INVALID")

        mock_rwb.assert_not_called()

    def test_invalid_country_error_mentions_supported_list(self) -> None:
        scraper = _make_scraper()
        result = scraper.get_country_indicators("INVALID")
        assert "Available" in result["error"] or "supported" in result["error"].lower()

    def test_invalid_country_bond_yields_returns_error_without_http(self) -> None:
        scraper = _make_scraper()

        with patch(f"{_MOD}.retry_with_backoff") as mock_rwb:
            result = scraper.get_bond_yields("XYZ")

        mock_rwb.assert_not_called()
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TestHappyPathParsing — parser methods with valid HTML fixtures
# ---------------------------------------------------------------------------


class TestHappyPathParsing:
    def test_parse_indicators_returns_min_keys(self) -> None:
        """3 known indicator names → dict with at least _MIN_INDICATORS entries."""
        soup = _indicators_soup(
            _indicators_row("GDP Growth Rate", "2.1", "1.8"),
            _indicators_row("Inflation Rate", "3.5", "3.2"),
            _indicators_row("Interest Rate", "5.25", "5.00"),
        )
        result = _make_scraper()._parse_indicators_table(soup)

        assert len(result) >= _MIN_INDICATORS
        assert "gdp_growth_rate" in result
        assert "inflation_rate" in result
        assert "interest_rate" in result

    def test_parse_indicators_entry_structure(self) -> None:
        """Each parsed indicator carries value, previous, unit, reference, raw_name."""
        soup = _indicators_soup(
            _indicators_row("GDP Growth Rate", "2.1", "1.8"),
            _indicators_row("Inflation Rate", "3.5", "3.2"),
            _indicators_row("Interest Rate", "5.25", "5.00"),
        )
        result = _make_scraper()._parse_indicators_table(soup)
        entry = result["gdp_growth_rate"]

        assert entry["value"] == pytest.approx(2.1)
        assert entry["previous"] == pytest.approx(1.8)
        assert entry["unit"] == "percent"
        assert entry["reference"] == "Feb 2026"
        assert entry["raw_name"] == "GDP Growth Rate"

    def test_parse_bond_yields_accepted_maturities(self) -> None:
        """2Y and 10Y rows → both keys present with correct yield values."""
        soup = _bond_soup(
            _bond_row("USA 10Y", "4.25"),
            _bond_row("USA 2Y", "4.80"),
        )
        result = _make_scraper()._parse_bond_yields_table(soup, "USA")

        assert "10Y" in result
        assert "2Y" in result
        assert result["10Y"]["yield"] == pytest.approx(4.25)
        assert result["2Y"]["yield"] == pytest.approx(4.80)

    def test_parse_industrial_production_country_keys(self) -> None:
        soup = _country_list_soup(
            _country_row("United States", "0.5"),
            _country_row("Germany", "-0.3"),
            _country_row("Japan", "1.2"),
        )
        result = _make_scraper()._parse_industrial_production_table(soup)

        assert len(result) >= _MIN_INDUSTRIAL_PRODUCTION_ROWS
        assert "United States" in result
        assert result["Germany"]["value"] == pytest.approx(-0.3)

    def test_parse_capacity_utilization_country_keys(self) -> None:
        soup = _country_list_soup(
            _country_row("United States", "78.5"),
            _country_row("Germany", "75.2"),
            _country_row("Japan", "80.1"),
        )
        result = _make_scraper()._parse_capacity_utilization_table(soup)

        assert len(result) >= _MIN_CAPACITY_UTILIZATION_ROWS
        assert "United States" in result
        assert result["United States"]["value"] == pytest.approx(78.5)
