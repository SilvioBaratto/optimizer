"""Tests for resolve_t212_suffix + ResolvedSymbol (issue #774)."""

from __future__ import annotations

import pytest


class TestResolvedSymbolShape:
    def test_when_imported_then_resolved_symbol_is_frozen_dataclass(self) -> None:
        from dataclasses import FrozenInstanceError, fields

        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            ResolvedSymbol,
        )

        names = {f.name for f in fields(ResolvedSymbol)}
        assert names == {
            "base_symbol",
            "t212_suffix",
            "probable_yf_suffix",
            "is_recognized",
        }

        instance = ResolvedSymbol(
            base_symbol="AAPL",
            t212_suffix="_US_EQ",
            probable_yf_suffix="",
            is_recognized=True,
        )
        with pytest.raises(FrozenInstanceError):
            instance.base_symbol = "MSFT"  # type: ignore[misc]


class TestResolveUSSuffix:
    def test_when_aapl_us_eq_then_us_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("AAPL_US_EQ")
        assert r.base_symbol == "AAPL"
        assert r.t212_suffix == "_US_EQ"
        assert r.probable_yf_suffix == ""
        assert r.is_recognized is True

    def test_when_msft_us_eq_then_us_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("MSFT_US_EQ")
        assert r.base_symbol == "MSFT"
        assert r.probable_yf_suffix == ""
        assert r.is_recognized is True


class TestResolveLondonSuffix:
    def test_when_vod_l_eq_then_london_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("VOD_l_EQ")
        assert r.base_symbol == "VOD"
        assert r.t212_suffix == "_l_EQ"
        assert r.probable_yf_suffix == ".L"
        assert r.is_recognized is True

    def test_when_lloy_l_eq_then_london_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("LLOY_l_EQ")
        assert r.base_symbol == "LLOY"
        assert r.probable_yf_suffix == ".L"


class TestResolveParisSuffix:
    def test_when_air_p_eq_then_paris_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("AIR_p_EQ")
        assert r.base_symbol == "AIR"
        assert r.t212_suffix == "_p_EQ"
        assert r.probable_yf_suffix == ".PA"
        assert r.is_recognized is True

    def test_when_lvmh_p_eq_then_paris_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("LVMH_p_EQ")
        assert r.base_symbol == "LVMH"
        assert r.probable_yf_suffix == ".PA"

    def test_when_paris_preferred_pp_eq_then_paris_recognized_with_pp_match(
        self,
    ) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("AIRp_pp_EQ")
        assert r.base_symbol == "AIRp"
        assert r.t212_suffix == "_pp_EQ"
        assert r.probable_yf_suffix == ".PA"
        assert r.is_recognized is True


class TestResolveXetraSuffix:
    def test_when_sap_d_eq_then_xetra_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("SAP_d_EQ")
        assert r.base_symbol == "SAP"
        assert r.t212_suffix == "_d_EQ"
        assert r.probable_yf_suffix == ".DE"
        assert r.is_recognized is True

    def test_when_siemens_d_eq_then_xetra_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("SIE_d_EQ")
        assert r.probable_yf_suffix == ".DE"


class TestResolveMilanSuffix:
    def test_when_eni_m_eq_then_milan_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("ENI_m_EQ")
        assert r.base_symbol == "ENI"
        assert r.t212_suffix == "_m_EQ"
        assert r.probable_yf_suffix == ".MI"
        assert r.is_recognized is True

    def test_when_isp_m_eq_then_milan_recognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("ISP_m_EQ")
        assert r.probable_yf_suffix == ".MI"


class TestFallbackBareEQ:
    def test_when_bare_eq_then_unrecognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("XYZ_EQ")
        assert r.base_symbol == "XYZ"
        assert r.t212_suffix == "_EQ"
        assert r.probable_yf_suffix == ""
        assert r.is_recognized is False

    def test_when_long_symbol_bare_eq_then_unrecognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("ACME_EQ")
        assert r.base_symbol == "ACME"
        assert r.t212_suffix == "_EQ"
        assert r.is_recognized is False


class TestUnrecognizedNoSuffix:
    def test_when_no_eq_then_unrecognized_and_input_passthrough(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("WEIRDTKR")
        assert r.base_symbol == "WEIRDTKR"
        assert r.t212_suffix == ""
        assert r.probable_yf_suffix == ""
        assert r.is_recognized is False

    def test_when_empty_string_then_unrecognized_and_empty_base(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("")
        assert r.base_symbol == ""
        assert r.t212_suffix == ""
        assert r.probable_yf_suffix == ""
        assert r.is_recognized is False

    def test_when_unknown_venue_letter_then_unrecognized(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("XYZ_z_EQ")
        assert r.base_symbol == "XYZ_z"
        assert r.t212_suffix == "_EQ"
        assert r.probable_yf_suffix == ""
        assert r.is_recognized is False

    def test_when_random_pattern_then_passthrough(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("FOO_BAR_BAZ")
        assert r.base_symbol == "FOO_BAR_BAZ"
        assert r.is_recognized is False

    def test_when_starts_with_eq_then_passthrough(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("EQ_FOO")
        assert r.base_symbol == "EQ_FOO"
        assert r.is_recognized is False


class TestLongestMatchOrder:
    def test_when_pp_eq_collides_with_p_eq_then_pp_wins(self) -> None:
        """`AIRp_pp_EQ` must match `_pp_EQ`, not be greedily eaten by `_p_EQ`."""
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("AIRp_pp_EQ")
        assert r.t212_suffix == "_pp_EQ"
        assert r.base_symbol == "AIRp"

    def test_when_us_eq_overlaps_with_bare_eq_then_us_wins(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix("AAPL_US_EQ")
        assert r.t212_suffix == "_US_EQ"


class TestRealTickerPatterns:
    """≥20 distinct ticker patterns drawn from real T212 payloads."""

    @pytest.mark.parametrize(
        "raw,expected_base,expected_yf,recognized",
        [
            ("AAPL_US_EQ", "AAPL", "", True),
            ("MSFT_US_EQ", "MSFT", "", True),
            ("TSLA_US_EQ", "TSLA", "", True),
            ("NVDA_US_EQ", "NVDA", "", True),
            ("GOOGL_US_EQ", "GOOGL", "", True),
            ("VOD_l_EQ", "VOD", ".L", True),
            ("LLOY_l_EQ", "LLOY", ".L", True),
            ("BARC_l_EQ", "BARC", ".L", True),
            ("AIR_p_EQ", "AIR", ".PA", True),
            ("LVMH_p_EQ", "LVMH", ".PA", True),
            ("BNP_p_EQ", "BNP", ".PA", True),
            ("AIRp_pp_EQ", "AIRp", ".PA", True),
            ("SAP_d_EQ", "SAP", ".DE", True),
            ("SIE_d_EQ", "SIE", ".DE", True),
            ("BMW_d_EQ", "BMW", ".DE", True),
            ("ENI_m_EQ", "ENI", ".MI", True),
            ("ISP_m_EQ", "ISP", ".MI", True),
            ("UCG_m_EQ", "UCG", ".MI", True),
            ("XYZ_EQ", "XYZ", "", False),
            ("RANDOM_EQ", "RANDOM", "", False),
            ("WEIRD", "WEIRD", "", False),
            ("BTC_USD", "BTC_USD", "", False),
            ("", "", "", False),
        ],
    )
    def test_when_real_pattern_then_resolves_correctly(
        self,
        raw: str,
        expected_base: str,
        expected_yf: str,
        recognized: bool,
    ) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        r = resolve_t212_suffix(raw)
        assert r.base_symbol == expected_base
        assert r.probable_yf_suffix == expected_yf
        assert r.is_recognized is recognized


class TestPureFunction:
    def test_when_called_repeatedly_then_same_output(self) -> None:
        from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
            resolve_t212_suffix,
        )

        a = resolve_t212_suffix("VOD_l_EQ")
        b = resolve_t212_suffix("VOD_l_EQ")
        assert a == b
