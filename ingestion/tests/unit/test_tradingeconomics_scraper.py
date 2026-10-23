"""Unit tests for TradingEconomicsIndicatorsScraper.

Covers ParseStructureError propagation, circuit breaker triggering,
fetch failure handling, _extract_number edge cases, and unsupported
country early-exit behaviour.

No database, no HTTP calls — all external I/O is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from app.services.macro.scrapers.exceptions import ParseStructureError
from app.services.macro.scrapers.tradingeconomics_scraper import (
    _MIN_BOND_YIELDS,
    _MIN_CAPACITY_UTILIZATION_ROWS,
    _MIN_INDICATORS,
    _MIN_INDUSTRIAL_PRODUCTION_ROWS,
    TradingEconomicsIndicatorsScraper,
)

# ---------------------------------------------------------------------------
# Module path used for patching module-level names
# ---------------------------------------------------------------------------

_MOD = "app.services.macro.scrapers.tradingeconomics_scraper"

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
        '<table class="table table-hover"><tbody>' + "".join(rows) + "</tbody></table>"
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
    html = '<table class="table-heatmap"><tbody>' + "".join(rows) + "</tbody></table>"
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
    html = '<table class="table-heatmap"><tbody>' + "".join(rows) + "</tbody></table>"
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


# ---------------------------------------------------------------------------
# TestIndicatorKeyComparability — one canonical key must hold one measure
# ---------------------------------------------------------------------------


def _unit_row(name: str, last: str, unit: str) -> str:
    return (
        "<tr>"
        f"<td><a href='#'>{name}</a></td>"
        f"<td>{last}</td>"
        "<td>1.0</td>"
        "<td>3.0</td>"
        "<td>0.5</td>"
        f"<td>{unit}</td>"
        "<td>Feb 2026</td>"
        "</tr>"
    )


_FILLER_ROWS = (
    _indicators_row("GDP Growth Rate", "2.1"),
    _indicators_row("Inflation Rate", "3.5"),
    _indicators_row("Interest Rate", "5.25"),
)


class TestIndicatorKeyComparability:
    def test_capacity_utilization_index_points_are_not_stored_as_percent(
        self,
    ) -> None:
        """Japan reports an index (103 points) under the same row label as the
        US utilisation rate (76.9 percent); only the rate may share the key."""
        index_soup = _indicators_soup(
            *_FILLER_ROWS, _unit_row("Capacity Utilization", "103", "points")
        )
        rate_soup = _indicators_soup(
            *_FILLER_ROWS, _unit_row("Capacity Utilization", "76.9", "percent")
        )
        scraper = _make_scraper()

        assert "capacity_utilization" not in scraper._parse_indicators_table(index_soup)
        kept = scraper._parse_indicators_table(rate_soup)
        assert kept["capacity_utilization"]["value"] == pytest.approx(76.9)

    def test_leading_index_percent_changes_are_not_stored_as_index_level(
        self,
    ) -> None:
        change_soup = _indicators_soup(
            *_FILLER_ROWS, _unit_row("Leading Economic Index", "0.7", "percent")
        )
        level_soup = _indicators_soup(
            *_FILLER_ROWS, _unit_row("Leading Economic Index", "99.6", "Points")
        )
        scraper = _make_scraper()

        assert "leading_economic_index" not in scraper._parse_indicators_table(
            change_soup
        )
        kept = scraper._parse_indicators_table(level_soup)
        assert kept["leading_economic_index"]["value"] == pytest.approx(99.6)

    def test_personal_savings_percent_of_gdp_is_not_a_savings_rate(self) -> None:
        soup = _indicators_soup(
            *_FILLER_ROWS, _unit_row("Personal Savings", "18.3", "percent of GDP")
        )

        assert "personal_savings" not in _make_scraper()._parse_indicators_table(soup)

    def test_us_only_labour_series_are_not_taken_from_other_countries(self) -> None:
        """France's 'Non Farm Payrolls' row is an employment level in thousands,
        not the US monthly payroll change."""
        html = (
            '<table class="table table-hover"><tbody>'
            + "".join(_FILLER_ROWS)
            + _unit_row("Non Farm Payrolls", "27021", "Thousand")
            + _unit_row("Initial Jobless Claims", "-61.3", "Thousand")
            + "</tbody></table>"
        ).encode()
        scraper = _make_scraper()

        with (
            patch(f"{_MOD}.retry_with_backoff", return_value=_fake_response(html)),
            patch(f"{_MOD}._te_circuit_breaker"),
        ):
            france = scraper.get_country_indicators("France", include_bonds=False)
            usa = scraper.get_country_indicators("USA", include_bonds=False)

        assert "non_farm_payrolls" not in france["indicators"]
        assert "initial_jobless_claims" not in france["indicators"]
        assert usa["indicators"]["non_farm_payrolls"]["value"] == pytest.approx(27021)
        assert "initial_jobless_claims" in usa["indicators"]

    @pytest.mark.parametrize(
        ("name", "key", "unit"),
        [
            ("Factory Orders", "factory_orders", "Net Balance"),
            ("Employment Change", "employment_change", "percent"),
        ],
    )
    def test_survey_balance_and_growth_rows_do_not_share_a_level_key(
        self, name: str, key: str, unit: str
    ) -> None:
        soup = _indicators_soup(*_FILLER_ROWS, _unit_row(name, "-32", unit))

        assert key not in _make_scraper()._parse_indicators_table(soup)


class TestGdpQuarterlyConvention:
    """The US page reports QoQ growth annualised; the other pages do not."""

    @staticmethod
    def _gdp_page(value: str) -> bytes:
        return (
            '<table class="table table-hover"><tbody>'
            + _unit_row("GDP Growth Rate", value, "percent")
            + "".join(_FILLER_ROWS[1:])
            + _unit_row("Unemployment Rate", "4.1", "percent")
            + "</tbody></table>"
        ).encode()

    def _indicators(self, country: str, value: str) -> dict:
        with (
            patch(
                f"{_MOD}.retry_with_backoff",
                return_value=_fake_response(self._gdp_page(value)),
            ),
            patch(f"{_MOD}._te_circuit_breaker"),
        ):
            result = _make_scraper().get_country_indicators(
                country, include_bonds=False
            )
        return result["indicators"]

    def test_us_annualised_rate_does_not_share_the_quarterly_key(self) -> None:
        indicators = self._indicators("USA", "2.2")

        assert "gdp_growth_rate" not in indicators
        assert indicators["gdp_growth_qoq_annualized"]["value"] == pytest.approx(2.2)
        assert indicators["gdp_growth_qoq_annualized"]["unit"] == "percent"

    def test_non_annualised_countries_keep_the_quarterly_key(self) -> None:
        for country in ("Germany", "Japan", "UK"):
            indicators = self._indicators(country, "0.3")

            assert indicators["gdp_growth_rate"]["value"] == pytest.approx(0.3)
            assert "gdp_growth_qoq_annualized" not in indicators

    def test_summary_labels_the_annualised_us_rate(self) -> None:
        scraper = _make_scraper()
        summary = scraper.get_indicator_summary(
            {
                "country": "USA",
                "status": "success",
                "indicators": {
                    "gdp_growth_qoq_annualized": {
                        "value": 2.2,
                        "previous": 2.5,
                        "unit": "percent",
                        "reference": "Jun/26",
                    }
                },
            }
        )

        assert "annualised" in summary
        assert "2.20" in summary
