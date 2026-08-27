"""parse_reference_date — multi-format reference-date parsing."""

from __future__ import annotations

import datetime as dt

import pytest

from portopt_db.coerce import parse_reference_date


class TestParseReferenceDate:
    def test_date_passthrough(self) -> None:
        d = dt.date(2024, 6, 1)
        assert parse_reference_date(d) is d

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Dec 2024", dt.date(2024, 12, 1)),
            ("Jan 2025", dt.date(2025, 1, 1)),
            ("12/25", dt.date(2025, 12, 1)),
            ("01/ 26", dt.date(2026, 1, 1)),
            ("2024-12-01", dt.date(2024, 12, 1)),
            ("1/15/2025", dt.date(2025, 1, 15)),
        ],
    )
    def test_supported_formats(self, value, expected) -> None:
        assert parse_reference_date(value) == expected

    def test_mon_slash_dd_uses_current_year(self) -> None:
        result = parse_reference_date("Mar/15")
        assert result is not None
        assert (result.month, result.day) == (3, 15)

    @pytest.mark.parametrize("value", [None, "", "   ", "not a date", 12345, "13/99"])
    def test_unparseable_returns_none(self, value) -> None:
        assert parse_reference_date(value) is None
