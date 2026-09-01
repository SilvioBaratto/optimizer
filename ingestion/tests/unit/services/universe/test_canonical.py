"""Dedup-first universe core: name normalization, canonical selection, USD floor.

Pure unit tests — FX rates are injected, no network. Fixtures mirror the live
Phase-0 clusters (NVDA cross-listings, dual-class, warrants, name collisions).
"""

from __future__ import annotations

import pytest

from app.services.universe.canonical import (
    FloorBand,
    IngestionFloorConfig,
    Listing,
    Metrics,
    dedup_canonical,
    derive_metrics,
    normalize_name,
    passes_floor,
    split_currency,
)

# USD per 1 major unit.
_FX = {"USD": 1.0, "EUR": 1.1, "GBP": 1.25, "CHF": 1.1, "CAD": 0.73}


def _mk(
    symbol,
    exchange,
    name,
    ccy,
    price,
    vol,
    *,
    fin_ccy="USD",
    shares: float | None = 1e9,
    mcap: float | None = None,
):
    return Listing(
        symbol=symbol,
        exchange=exchange,
        long_name=name,
        short_name=name,
        currency=ccy,
        financial_currency=fin_ccy,
        price=price,
        avg_volume=vol,
        market_cap=mcap,
        shares_outstanding=shares,
    )


# --------------------------------------------------------------------------- #
# normalize_name
# --------------------------------------------------------------------------- #
class TestNormalizeName:
    def test_deaccent_and_uppercase(self) -> None:
        assert normalize_name("Moët Hennessy") == "MOET HENNESSY"
        assert normalize_name("Nestlé S.A.") == "NESTLE SA"

    def test_dots_and_apostrophes_collapse_without_space(self) -> None:
        # S.A. -> SA so it does not split from a plain "SA"
        assert normalize_name("Foo S.A.") == "FOO SA"
        assert normalize_name("O'Reilly Automotive") == "OREILLY AUTOMOTIVE"

    def test_legal_suffix_is_kept(self) -> None:
        # Deliberately NOT stripped (false-merge risk); identical across venues.
        assert normalize_name("NVIDIA Corporation") == "NVIDIA CORPORATION"

    def test_cross_listing_names_match(self) -> None:
        assert normalize_name("Toyota Motor Corporation") == normalize_name(
            "Toyota Motor Corporation"
        )

    def test_empty(self) -> None:
        assert normalize_name(None) == ""
        assert normalize_name("") == ""


# --------------------------------------------------------------------------- #
# split_currency
# --------------------------------------------------------------------------- #
class TestSplitCurrency:
    @pytest.mark.parametrize(
        ("code", "major", "div"),
        [
            ("GBX", "GBP", 100.0),
            ("GBp", "GBP", 100.0),
            ("ILA", "ILS", 100.0),
            ("ZAC", "ZAR", 100.0),
            ("USD", "USD", 1.0),
            ("EUR", "EUR", 1.0),
            (None, None, 1.0),
            ("XYZ", "XYZ", 1.0),  # unknown -> fail-open div=1
        ],
    )
    def test_split(self, code, major, div) -> None:
        assert split_currency(code) == (major, div)


# --------------------------------------------------------------------------- #
# derive_metrics
# --------------------------------------------------------------------------- #
class TestDeriveMetrics:
    def test_usd_line(self) -> None:
        m = derive_metrics(
            _mk("AAPL", "NMS", "Apple Inc.", "USD", 200.0, 1e6, shares=15e9), _FX
        )
        assert m.major_ccy == "USD"
        assert m.price_major == 200.0
        assert m.addv_usd == pytest.approx(200.0 * 1e6)
        assert m.mcap_usd == pytest.approx(200.0 * 15e9)

    def test_gbp_pence_price_divided_by_100(self) -> None:
        # .L quotes in pence; price/100 -> GBP major, then *fx.
        m = derive_metrics(
            _mk("SHEL.L", "LSE", "Shell plc", "GBp", 2700.0, 1e6, shares=6e9), _FX
        )
        assert m.major_ccy == "GBP"
        assert m.price_major == pytest.approx(27.0)
        assert m.addv_usd == pytest.approx(1e6 * 27.0 * 1.25)
        assert m.mcap_usd == pytest.approx(6e9 * 27.0 * 1.25)

    def test_volume_count_is_never_scaled(self) -> None:
        # divisor hits price, not the share/volume count
        m = derive_metrics(
            _mk("X.L", "LSE", "X", "GBX", 100.0, 2000.0, shares=1e9), _FX
        )
        # addv = 2000 shares * (100/100 GBP) * 1.25
        assert m.addv_usd == pytest.approx(2000.0 * 1.0 * 1.25)

    def test_missing_fx_fails_open(self) -> None:
        m = derive_metrics(_mk("ABC.XX", "XXX", "Abc", "XYZ", 10.0, 1e6), _FX)
        assert m.mcap_usd is None and m.addv_usd is None
        assert m.price_major == 10.0  # still derivable

    def test_missing_shares_falls_back_to_reported_mcap(self) -> None:
        lst = _mk("EUx", "GER", "Eu Co", "EUR", 50.0, 1e5, shares=None, mcap=2e9)
        m = derive_metrics(lst, _FX)
        assert m.mcap_usd == pytest.approx(2e9 * 1.1)  # reported * fx

    def test_gbp_mcap_gotcha_prefers_reconstruction(self) -> None:
        # Yahoo: marketCap already in GBP major (7.5e9), price in pence (2500).
        # Reconstruction (shares*price/100*fx) is authoritative; reported*fx would
        # be self-consistent here — both ~ same — so reconstruction is used.
        lst = _mk(
            "BP.L", "LSE", "BP p.l.c.", "GBp", 2500.0, 1e6, shares=3e8, mcap=7.5e9
        )
        m = derive_metrics(lst, _FX)
        assert m.mcap_usd == pytest.approx(3e8 * 25.0 * 1.25)


# --------------------------------------------------------------------------- #
# dedup_canonical
# --------------------------------------------------------------------------- #
class TestDedupCanonical:
    def test_cross_listings_collapse_to_max_addv(self) -> None:
        listings = [
            _mk("NVD.DE", "GER", "NVIDIA Corporation", "EUR", 187.0, 109_045),
            _mk("NVDA", "NMS", "NVIDIA Corporation", "USD", 220.0, 138_830_287),
            _mk("1NVDA.MI", "MIL", "NVIDIA Corporation", "EUR", 187.0, 49_765),
            _mk("NVDA.SW", "EBS", "NVIDIA Corporation", "CHF", 200.0, 93),
        ]
        out = dedup_canonical(listings, _FX)
        assert [x.symbol for x in out] == ["NVDA"]

    def test_financial_currency_guard_keeps_distinct_companies(self) -> None:
        # Same normalized name, DIFFERENT financial currency -> not merged.
        listings = [
            _mk("ACME", "NMS", "ACME Corp", "USD", 100.0, 1e6, fin_ccy="USD"),
            _mk("ACM.PA", "PAR", "ACME Corp", "EUR", 90.0, 5e5, fin_ccy="EUR"),
        ]
        out = dedup_canonical(listings, _FX)
        assert {x.symbol for x in out} == {"ACME", "ACM.PA"}

    def test_dual_class_both_kept_when_liquid_same_exchange(self) -> None:
        listings = [
            _mk("GOOGL", "NMS", "Alphabet Inc.", "USD", 165.0, 30e6),
            _mk("GOOG", "NMS", "Alphabet Inc.", "USD", 167.0, 25e6),
        ]
        out = dedup_canonical(listings, _FX)
        assert {x.symbol for x in out} == {"GOOGL", "GOOG"}

    def test_illiquid_sibling_dropped(self) -> None:
        # BRK-A (5 sh/day) below the dual-class ADDV bar -> only BRK-B survives.
        listings = [
            _mk("BRK-A", "NYQ", "Berkshire Hathaway Inc.", "USD", 700_000.0, 5),
            _mk("BRK-B", "NYQ", "Berkshire Hathaway Inc.", "USD", 470.0, 3e6),
        ]
        out = dedup_canonical(listings, _FX)
        assert [x.symbol for x in out] == ["BRK-B"]

    def test_warrant_loses_to_common(self) -> None:
        listings = [
            _mk("HUMAW", "NMS", "Humacyte Inc", "USD", 0.5, 10_000),
            _mk("HUMA", "NMS", "Humacyte Inc", "USD", 5.0, 1e6),
        ]
        out = dedup_canonical(listings, _FX)
        assert [x.symbol for x in out] == ["HUMA"]

    def test_unnamed_listing_not_merged(self) -> None:
        listings = [
            _mk("AAA", "NMS", None, "USD", 10.0, 1e6),
            _mk("BBB", "NMS", None, "USD", 20.0, 2e6),
        ]
        out = dedup_canonical(listings, _FX)
        assert {x.symbol for x in out} == {"AAA", "BBB"}

    def test_deterministic_order(self) -> None:
        listings = [
            _mk("NVDA", "NMS", "NVIDIA Corporation", "USD", 220.0, 138e6),
            _mk("AAPL", "NMS", "Apple Inc.", "USD", 200.0, 50e6),
        ]
        a = [x.symbol for x in dedup_canonical(listings, _FX)]
        b = [x.symbol for x in dedup_canonical(list(reversed(listings)), _FX)]
        assert a == b


# --------------------------------------------------------------------------- #
# passes_floor
# --------------------------------------------------------------------------- #
class TestPassesFloor:
    _CFG = IngestionFloorConfig()

    def _m(self, mcap, addv, price: float | None = 10.0):
        return Metrics(major_ccy="USD", price_major=price, mcap_usd=mcap, addv_usd=addv)

    def test_clears_entry(self) -> None:
        assert passes_floor(self._m(30e6, 60_000), self._CFG) is True

    def test_below_mcap_entry_rejected(self) -> None:
        assert passes_floor(self._m(20e6, 60_000), self._CFG) is False

    def test_member_uses_exit_band(self) -> None:
        # 15M < entry 25M but > exit 10M -> kept as member, rejected as entrant.
        m = self._m(15e6, 60_000)
        assert passes_floor(m, self._CFG, is_member=True) is True
        assert passes_floor(m, self._CFG, is_member=False) is False

    def test_below_addv_rejected(self) -> None:
        assert passes_floor(self._m(30e6, 10_000), self._CFG) is False

    def test_unknown_metrics_fail_open(self) -> None:
        assert passes_floor(self._m(None, None), self._CFG) is True

    def test_unpriced_rejected(self) -> None:
        assert passes_floor(self._m(30e6, 60_000, price=None), self._CFG) is False


# --------------------------------------------------------------------------- #
# FloorBand
# --------------------------------------------------------------------------- #
def test_floorband_rejects_exit_above_entry() -> None:
    with pytest.raises(ValueError, match="must be <="):
        FloorBand(entry=10.0, exit_=20.0)
