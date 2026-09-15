"""Tests for minor-unit (sub-currency) normalization.

These guard the single most load-bearing DB compatibility concern for the fx
module: ``price_history.price_unit`` stores sub-unit codes (GBp pence, ZAc
cents, ILA agorot) verbatim, and yfinance 1.6.0 keeps sub-unit prices as-is,
so a code-only FX conversion is a 100x error.
"""

from __future__ import annotations

import pytest

from optimizer.fx import MINOR_UNIT_SCALES, normalize_currency_code


class TestNormalizeMajorUnits:
    """Major-unit codes pass through with scale 1."""

    @pytest.mark.parametrize("code", ["USD", "EUR", "GBP", "ZAR", "ILS", "JPY"])
    def test_major_units_scale_one(self, code: str) -> None:
        assert normalize_currency_code(code) == (code, 1)

    def test_major_unit_lowercase_upper_cased(self) -> None:
        assert normalize_currency_code("usd") == ("USD", 1)
        assert normalize_currency_code("eur") == ("EUR", 1)

    def test_unknown_code_passthrough(self) -> None:
        # An unrecognised code is upper-cased and treated as a major unit.
        assert normalize_currency_code("chf") == ("CHF", 1)

    def test_whitespace_stripped(self) -> None:
        assert normalize_currency_code("  USD  ") == ("USD", 1)


class TestNormalizeMinorUnits:
    """Sub-unit codes resolve to (major, 100)."""

    def test_gbp_pence(self) -> None:
        assert normalize_currency_code("GBp") == ("GBP", 100)

    def test_gbp_pence_synonym_gbx(self) -> None:
        assert normalize_currency_code("GBX") == ("GBP", 100)
        assert normalize_currency_code("GBx") == ("GBP", 100)

    def test_zar_cents(self) -> None:
        assert normalize_currency_code("ZAc") == ("ZAR", 100)

    def test_ils_agorot(self) -> None:
        assert normalize_currency_code("ILA") == ("ILS", 100)

    def test_gbx_case_insensitive(self) -> None:
        # GBX has no major-currency collision, so casing is irrelevant.
        assert normalize_currency_code("gbx") == ("GBP", 100)

    def test_zac_case_insensitive(self) -> None:
        assert normalize_currency_code("zac") == ("ZAR", 100)


class TestGbpAmbiguity:
    """The pence/pound distinction is case-sensitive (the crux 100x trap)."""

    def test_uppercase_gbp_is_pounds_not_pence(self) -> None:
        # "GBP" (pounds) must NEVER be read as pence, else every LSE price
        # is off by 100x in the wrong direction.
        assert normalize_currency_code("GBP") == ("GBP", 1)

    def test_lowercase_p_is_pence(self) -> None:
        assert normalize_currency_code("GBp") == ("GBP", 100)

    def test_all_lowercase_gbp_is_pounds(self) -> None:
        # "gbp" is a casing of the pound code, not the pence code "GBp".
        assert normalize_currency_code("gbp") == ("GBP", 1)


class TestMinorUnitRegistry:
    """The public registry documents every sub-unit the DB can hold."""

    def test_registry_maps_known_minor_codes(self) -> None:
        assert MINOR_UNIT_SCALES["GBp"] == ("GBP", 100)
        assert MINOR_UNIT_SCALES["ZAc"] == ("ZAR", 100)
        assert MINOR_UNIT_SCALES["ILA"] == ("ILS", 100)

    def test_registry_all_scale_100(self) -> None:
        assert all(scale == 100 for _major, scale in MINOR_UNIT_SCALES.values())

    def test_major_codes_absent_from_registry(self) -> None:
        # A major code must never appear as a minor-unit key.
        for major in {major for major, _ in MINOR_UNIT_SCALES.values()}:
            assert major not in MINOR_UNIT_SCALES
