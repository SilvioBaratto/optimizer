"""Unit tests for ``app.utils.currency`` minor-unit normalisation."""

from __future__ import annotations

import pytest

from app.utils.currency import to_major_currency


class TestToMajorCurrency:
    @pytest.mark.parametrize(
        ("code", "major"),
        [
            ("GBp", "GBP"),
            ("GBX", "GBP"),
            ("GBx", "GBP"),
            ("ZAc", "ZAR"),
            ("ZAC", "ZAR"),
            ("ZAX", "ZAR"),
            ("ZAx", "ZAR"),
            ("ILA", "ILS"),
            ("ILa", "ILS"),
        ],
    )
    def test_minor_unit_spellings_map_to_their_major(
        self, code: str, major: str
    ) -> None:
        assert to_major_currency(code) == major

    @pytest.mark.parametrize("code", ["GBP", "ZAR", "ILS", "USD", "EUR", "XYZ"])
    def test_major_and_unknown_codes_pass_through(self, code: str) -> None:
        assert to_major_currency(code) == code

    def test_none_passes_through(self) -> None:
        assert to_major_currency(None) is None

    def test_pound_is_never_mistaken_for_pence(self) -> None:
        # ``GBp`` upper-cases to the pound code, so a case-insensitive match
        # must not apply to it; only the exact pence spelling is a minor unit.
        assert to_major_currency("gbp") == "gbp"
        assert to_major_currency("GBP") == "GBP"
