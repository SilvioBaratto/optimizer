"""Branch-coverage tests for app/utils/date_parsing.py.

Covers every format and edge-case branch in parse_reference_date:
  - None input                       → None
  - date object passthrough          → same object
  - non-string / non-date input      → None
  - empty / blank string             → None
  - "Mon YYYY" valid                 → date(year, month, 1)
  - "Mon YYYY" with bad year         → falls through to ISO / datetime formats
  - "MM/YY" two-digit year           → date(20YY, month, 1)
  - "MM/ YY" with interior space     → same (normalised)
  - "Mon/DD" slash variant           → date(current_year, month, day)
  - Mon/DD with invalid day          → falls through
  - ISO "YYYY-MM-DD"                 → date
  - US format "M/D/YYYY"            → date
  - EU format "D/M/YYYY"            → date (fallback)
  - completely unparseable string    → None
"""

from __future__ import annotations

from datetime import date

from portopt_db.coerce import parse_reference_date

# ---------------------------------------------------------------------------
# None / non-string edge cases
# ---------------------------------------------------------------------------


class TestNoneAndNonStringInputs:
    def test_when_none_then_returns_none(self) -> None:
        assert parse_reference_date(None) is None

    def test_when_date_object_then_returns_same_object(self) -> None:
        d = date(2024, 6, 15)
        result = parse_reference_date(d)
        assert result is d

    def test_when_integer_then_returns_none(self) -> None:
        assert parse_reference_date(42) is None  # type: ignore[arg-type]

    def test_when_list_then_returns_none(self) -> None:
        assert parse_reference_date([2024, 1, 1]) is None  # type: ignore[arg-type]

    def test_when_empty_string_then_returns_none(self) -> None:
        assert parse_reference_date("") is None

    def test_when_whitespace_only_string_then_returns_none(self) -> None:
        assert parse_reference_date("   ") is None


# ---------------------------------------------------------------------------
# "Mon YYYY" format (e.g. "Dec 2024")
# ---------------------------------------------------------------------------


class TestMonYearFormat:
    def test_when_dec_2024_then_returns_first_of_december(self) -> None:
        result = parse_reference_date("Dec 2024")
        assert result == date(2024, 12, 1)

    def test_when_jan_2025_then_returns_first_of_january(self) -> None:
        result = parse_reference_date("Jan 2025")
        assert result == date(2025, 1, 1)

    def test_when_month_lowercase_then_still_parses(self) -> None:
        result = parse_reference_date("mar 2023")
        assert result == date(2023, 3, 1)

    def test_when_month_mixed_case_then_still_parses(self) -> None:
        result = parse_reference_date("JUL 2026")
        assert result == date(2026, 7, 1)

    def test_when_all_months_parse_correctly(self) -> None:
        expected_months = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }
        for abbr, month_num in expected_months.items():
            result = parse_reference_date(f"{abbr} 2024")
            assert result == date(2024, month_num, 1), f"Failed for {abbr}"

    def test_when_unknown_month_abbr_then_falls_through(self) -> None:
        # "Xyz 2024" — month not in _MONTH_ABBR, falls through to other formats
        result = parse_reference_date("Xyz 2024")
        # Not parseable by any format; should return None
        assert result is None

    def test_when_year_is_not_integer_then_falls_through(self) -> None:
        # "Dec abc" — year_str is not a valid int; ValueError caught internally
        result = parse_reference_date("Dec abc")
        assert result is None


# ---------------------------------------------------------------------------
# "MM/YY" IlSole-style (two-digit year, normalises internal spaces)
# ---------------------------------------------------------------------------


class TestMmYyFormat:
    def test_when_mm_slash_yy_then_returns_first_of_month(self) -> None:
        result = parse_reference_date("12/25")
        assert result == date(2025, 12, 1)

    def test_when_mm_slash_space_yy_then_normalised_and_parsed(self) -> None:
        # "12/ 25" → normalise spaces → "12/25"
        result = parse_reference_date("12/ 25")
        assert result == date(2025, 12, 1)

    def test_when_01_slash_26_then_january_2026(self) -> None:
        result = parse_reference_date("01/26")
        assert result == date(2026, 1, 1)

    def test_when_right_side_is_four_digits_then_not_mm_yy_branch(self) -> None:
        # "01/2025" — right side len != 2, should NOT enter MM/YY branch
        # It will eventually be parsed as a date via datetime.strptime
        result = parse_reference_date("01/2025")
        # Parsed by strptime "%m/%d/%Y" where day=2025 is invalid, or "%d/%m/%Y"
        # Both invalid day=2025; should be None (no format succeeds)
        assert result is None

    def test_when_month_value_out_of_range_then_value_error_caught(self) -> None:
        # "99/25" → month=99 → date(2025, 99, 1) raises ValueError, caught
        result = parse_reference_date("99/25")
        assert result is None


# ---------------------------------------------------------------------------
# "Mon/DD" slash format (e.g. "Jan/15")
# ---------------------------------------------------------------------------


class TestMonDdSlashFormat:
    def test_when_jan_slash_15_then_jan_15_current_year(self) -> None:
        # Mon/DD uses date.today().year — just verify it parses to Jan 15 of some year
        result = parse_reference_date("Jan/15")
        assert result is not None
        assert result.month == 1
        assert result.day == 15

    def test_when_dec_slash_31_then_dec_31_current_year(self) -> None:
        result = parse_reference_date("Dec/31")
        assert result == date(date.today().year, 12, 31)

    def test_when_mon_slash_invalid_day_then_returns_none(self) -> None:
        # "Feb/99" → day=99 invalid
        result = parse_reference_date("Feb/99")
        assert result is None

    def test_when_mon_slash_nonnumeric_day_then_returns_none(self) -> None:
        # "Jan/xx" → int("xx") raises ValueError
        result = parse_reference_date("Jan/xx")
        assert result is None


# ---------------------------------------------------------------------------
# ISO format "YYYY-MM-DD"
# ---------------------------------------------------------------------------


class TestIsoFormat:
    def test_when_iso_date_string_then_returns_date(self) -> None:
        result = parse_reference_date("2024-12-01")
        assert result == date(2024, 12, 1)

    def test_when_iso_format_with_invalid_month_then_falls_through(self) -> None:
        # "2024-13-01" — invalid ISO; fromisoformat raises, falls to strptime
        result = parse_reference_date("2024-13-01")
        assert result is None

    def test_when_iso_format_boundary_jan_1(self) -> None:
        result = parse_reference_date("2000-01-01")
        assert result == date(2000, 1, 1)


# ---------------------------------------------------------------------------
# US / EU datetime strptime formats
# ---------------------------------------------------------------------------


class TestStrptimeFormats:
    def test_when_us_format_m_d_yyyy_then_parses(self) -> None:
        result = parse_reference_date("1/15/2025")
        assert result == date(2025, 1, 15)

    def test_when_us_format_leading_zeros_then_parses(self) -> None:
        result = parse_reference_date("01/15/2025")
        assert result == date(2025, 1, 15)

    def test_when_eu_format_d_m_yyyy_then_parses_as_fallback(self) -> None:
        # "30/06/2025" — fails "%m/%d/%Y" (day=30 for month 30 is invalid),
        # but succeeds with "%d/%m/%Y" (day=30, month=6).
        result = parse_reference_date("30/06/2025")
        assert result == date(2025, 6, 30)

    def test_when_completely_unparseable_then_returns_none(self) -> None:
        result = parse_reference_date("not-a-date-at-all")
        assert result is None

    def test_when_partial_date_string_then_returns_none(self) -> None:
        result = parse_reference_date("2024-12")
        assert result is None

    def test_when_numeric_string_only_then_returns_none(self) -> None:
        # "99999999" has no valid date format match
        result = parse_reference_date("99999999")
        assert result is None
