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
    carried_by_sibling,
    dedup_canonical,
    derive_metrics,
    is_real_security_line,
    normalize_name,
    passes_floor,
    split_currency,
)
from app.utils.currency import MINOR_TO_MAJOR

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
    fin_ccy: str | None = "USD",
    shares: float | None = 1e9,
    mcap: float | None = None,
    net_assets: float | None = None,
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
        net_assets=net_assets,
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
            ("GBx", "GBP", 100.0),
            ("ILA", "ILS", 100.0),
            ("ILa", "ILS", 100.0),
            ("ZAC", "ZAR", 100.0),
            ("ZAc", "ZAR", 100.0),  # the spelling Yahoo actually emits for JSE
            ("GBP", "GBP", 1.0),  # pounds must not be read as pence
            ("USD", "USD", 1.0),
            ("EUR", "EUR", 1.0),
            (None, None, 1.0),
            ("XYZ", "XYZ", 1.0),  # unknown -> fail-open div=1
        ],
    )
    def test_split(self, code, major, div) -> None:
        assert split_currency(code) == (major, div)

    @pytest.mark.parametrize("code", sorted(MINOR_TO_MAJOR))
    def test_every_listed_sub_unit_casing_is_split(self, code) -> None:
        # The sub-unit table is shared with the statement labeller; a code added
        # there must also be priced in major units here.
        assert split_currency(code) == (MINOR_TO_MAJOR[code], 100.0)

    @pytest.mark.parametrize("code", ["GBP", "ILS", "ZAR"])
    def test_major_codes_are_never_treated_as_sub_units(self, code) -> None:
        assert split_currency(code) == (code, 1.0)


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

    def test_zar_cents_price_divided_by_100(self) -> None:
        # JSE quotes in cents ('ZAc'); the price is normalised to rand and the
        # ZAR rate applies, like pence on the LSE.
        m = derive_metrics(
            _mk("NPN.JO", "JNB", "Naspers", "ZAc", 300000.0, 1e5, shares=1e8),
            {**_FX, "ZAR": 0.055},
        )
        assert m.major_ccy == "ZAR"
        assert m.price_major == pytest.approx(3000.0)
        assert m.addv_usd == pytest.approx(1e5 * 3000.0 * 0.055)
        assert m.mcap_usd == pytest.approx(1e8 * 3000.0 * 0.055)

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

    def test_etf_size_from_net_assets(self) -> None:
        # ETFs carry netAssets, not marketCap/shares -> mcap_usd from net_assets.
        lst = _mk(
            "SPY",
            "PCX",
            "SPDR S&P 500 ETF Trust",
            "USD",
            760.0,
            50e6,
            fin_ccy=None,
            shares=None,
            net_assets=795e9,
        )
        m = derive_metrics(lst, _FX)
        assert m.mcap_usd == pytest.approx(795e9)
        assert m.has_fundamentals is True

    def test_existence_gate_does_not_depend_on_fx(self) -> None:
        # GBP rate unresolved: shares x price still prove the line is a security.
        lst = _mk(
            "SEQI.L",
            "LSE",
            "Sequoia Economic Infrastructure Income Fund",
            "GBp",
            70.0,
            1e6,
            fin_ccy=None,
            shares=1.5e9,
        )
        m = derive_metrics(lst, {"USD": 1.0})
        assert m.mcap_usd is None
        assert m.has_fundamentals is True
        assert passes_floor(m, IngestionFloorConfig()) is True

    def test_reported_market_cap_proves_existence_without_fx(self) -> None:
        lst = _mk(
            "EUx", "GER", "Eu Co", "EUR", 50.0, 1e5, fin_ccy=None, shares=None, mcap=2e9
        )
        assert derive_metrics(lst, {"USD": 1.0}).has_fundamentals is True

    def test_certificate_stays_without_fundamentals_when_fx_is_missing(self) -> None:
        lst = _mk(
            "AT0000A3.VI",
            "VIE",
            "RBI Expr.Z./Nvidia",
            "EUR",
            100.0,
            500,
            fin_ccy=None,
            shares=None,
        )
        assert derive_metrics(lst, {"USD": 1.0}).has_fundamentals is False

    def test_certificate_has_no_fundamentals(self) -> None:
        # No financialCurrency, no marketCap, no shares, no netAssets -> not a security.
        lst = _mk(
            "AT0000A3.VI",
            "VIE",
            "RBI Expr.Z./Nvidia 25-30",
            "EUR",
            100.0,
            500,
            fin_ccy=None,
            shares=None,
        )
        m = derive_metrics(lst, _FX)
        assert m.mcap_usd is None
        assert m.has_fundamentals is False


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

    def test_missing_financial_currency_does_not_split_a_cross_listing(self) -> None:
        # Yahoo omits financialCurrency on many thin lines; None must act as a
        # wildcard, else the IOB line is persisted next to its Paris primary.
        listings = [
            _mk("VK.PA", "PAR", "Vallourec S.A.", "EUR", 20.0, 1.5e6, fin_ccy="EUR"),
            _mk("0NR2.IL", "IOB", "Vallourec S.A.", "EUR", 20.0, 5e4, fin_ccy=None),
        ]
        out = dedup_canonical(listings, _FX)
        assert [x.symbol for x in out] == ["VK.PA"]

    def test_missing_financial_currency_on_primary_still_wins(self) -> None:
        listings = [
            _mk("FISV", "NMS", "Fiserv, Inc.", "USD", 150.0, 5e6, fin_ccy=None),
            _mk("0IP9.L", "IOB", "Fiserv, Inc.", "USD", 150.0, 1e3, fin_ccy="USD"),
        ]
        out = dedup_canonical(listings, _FX)
        assert [x.symbol for x in out] == ["FISV"]

    def test_missing_financial_currency_stays_apart_when_currencies_conflict(
        self,
    ) -> None:
        # Two different reporting currencies already prove distinct companies; a
        # None line cannot be attributed to either, so it is not merged into one.
        listings = [
            _mk("ACME", "NMS", "ACME Corp", "USD", 100.0, 1e6, fin_ccy="USD"),
            _mk("ACM.PA", "PAR", "ACME Corp", "EUR", 90.0, 5e5, fin_ccy="EUR"),
            _mk("ACM.IL", "IOB", "ACME Corp", "EUR", 90.0, 5e4, fin_ccy=None),
        ]
        out = dedup_canonical(listings, _FX)
        assert {x.symbol for x in out} == {"ACME", "ACM.PA", "ACM.IL"}

    def test_incumbent_holds_against_marginally_more_liquid_sibling(self) -> None:
        # AZN.L is ~1.3x more liquid than the US line: inside the hysteresis band,
        # so the incumbent stays instead of flipping on a noisy 3-month average.
        listings = [
            _mk("AZN", "NMS", "AstraZeneca PLC", "USD", 100.0, 1e6),
            _mk("AZN.L", "LSE", "AstraZeneca PLC", "GBp", 8000.0, 1.3e6),
        ]
        assert [x.symbol for x in dedup_canonical(listings, _FX)] == ["AZN.L"]
        out = dedup_canonical(listings, _FX, incumbents={"AZN"})
        assert [x.symbol for x in out] == ["AZN"]

    def test_incumbent_is_displaced_beyond_the_hysteresis_margin(self) -> None:
        listings = [
            _mk("AZN", "NMS", "AstraZeneca PLC", "USD", 100.0, 1e6),
            _mk("AZN.L", "LSE", "AstraZeneca PLC", "GBp", 8000.0, 3e6),
        ]
        out = dedup_canonical(listings, _FX, incumbents={"AZN"})
        assert [x.symbol for x in out] == ["AZN.L"]

    def test_incumbent_without_traded_volume_does_not_hold(self) -> None:
        # No ADDV means nothing to compare: a dead incumbent must not block a live line.
        listings = [
            _mk("ACME.MU", "MUN", "Acme Corp", "EUR", 9.0, 0),
            _mk("ACME", "NMS", "Acme Corp", "USD", 10.0, 1e6),
        ]
        out = dedup_canonical(listings, _FX, incumbents={"ACME.MU"})
        assert [x.symbol for x in out] == ["ACME"]

    def test_incumbent_inside_the_member_band_holds_though_below_entry(self) -> None:
        # 40k ADDV is under the 50k entry bar but over the 25k member exit bar, and the
        # challenger (53.9k after FX) is inside the 1.5x margin: the incumbent stays.
        listings = [
            _mk("FOO", "NMS", "Foo Corp", "USD", 10.0, 4_000),
            _mk("FOO.DE", "GER", "Foo Corp", "EUR", 10.0, 4_900),
        ]
        assert [x.symbol for x in dedup_canonical(listings, _FX)] == ["FOO.DE"]
        out = dedup_canonical(listings, _FX, incumbents={"FOO"})
        assert [x.symbol for x in out] == ["FOO"]
        assert passes_floor(
            derive_metrics(out[0], _FX), IngestionFloorConfig(), is_member=True
        )

    def test_incumbent_failing_the_floor_does_not_hold_against_a_challenger(
        self,
    ) -> None:
        # FOO's market cap (5M) is under even the member exit bar, so the build would
        # drop it and take the entity with it; FOO.DE clears the floor and must carry
        # the entity instead of losing it to the hold.
        listings = [
            _mk("FOO", "NMS", "Foo Corp", "USD", 10.0, 4_000, shares=5e5),
            _mk("FOO.DE", "GER", "Foo Corp", "EUR", 10.0, 4_900),
        ]
        out = dedup_canonical(listings, _FX, incumbents={"FOO"})
        assert [x.symbol for x in out] == ["FOO.DE"]
        assert passes_floor(derive_metrics(out[0], _FX), IngestionFloorConfig())

    def test_incumbent_without_fundamentals_does_not_hold(self) -> None:
        # A line that fails the existence gate cannot pass the floor, so holding it
        # would delete the entity.
        listings = [
            _mk(
                "FOO", "NMS", "Foo Corp", "USD", 10.0, 4_000, fin_ccy=None, shares=None
            ),
            _mk("FOO.DE", "GER", "Foo Corp", "EUR", 10.0, 4_900, fin_ccy=None),
        ]
        out = dedup_canonical(listings, _FX, incumbents={"FOO"})
        assert [x.symbol for x in out] == ["FOO.DE"]

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

    def test_unresolved_fx_on_cross_listing_aborts_instead_of_demoting_primary(
        self,
    ) -> None:
        # CHF rate missing: NESN.SW would get ADDV -1 and lose to the 2k-share EUR
        # line, silently swapping the primary listing for a thin one.
        fx = {k: v for k, v in _FX.items() if k != "CHF"}
        listings = [
            _mk("NESN.SW", "EBS", "Nestle S.A.", "CHF", 100.0, 3e6, fin_ccy="CHF"),
            _mk("NSRGF.DE", "GER", "Nestle S.A.", "EUR", 90.0, 2000, fin_ccy="CHF"),
        ]
        with pytest.raises(ValueError, match="CHF"):
            dedup_canonical(listings, fx)

    def test_unresolved_fx_on_lone_listing_is_fail_open(self) -> None:
        # Nothing to rank against, so a missing rate cannot change a winner.
        lone = _mk("ABC.XX", "XXX", "Abc Ltd", "XYZ", 10.0, 1e6)
        assert [x.symbol for x in dedup_canonical([lone], _FX)] == ["ABC.XX"]

    def test_line_without_volume_still_loses_without_fx_error(self) -> None:
        # No traded volume is a dead line, not an FX gap: ranked last, no abort.
        listings = [
            _mk("ACME", "NMS", "Acme Corp", "USD", 10.0, 1e6),
            _mk("ACME.MU", "MUN", "Acme Corp", "EUR", 9.0, 0),
        ]
        fx = {"USD": 1.0}
        assert [x.symbol for x in dedup_canonical(listings, fx)] == ["ACME"]


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

    def test_no_fundamentals_rejected(self) -> None:
        # Existence gate: certificate/structured product with no size signal.
        m = Metrics(
            major_ccy="EUR",
            price_major=100.0,
            mcap_usd=None,
            addv_usd=None,
            has_fundamentals=False,
        )
        assert passes_floor(m, self._CFG) is False


# --------------------------------------------------------------------------- #
# is_real_security_line
# --------------------------------------------------------------------------- #
class TestIsRealSecurityLine:
    @pytest.mark.parametrize(
        ("price", "has_fundamentals", "expected"),
        [(10.0, True, True), (None, True, False), (10.0, False, False)],
    )
    def test_requires_price_and_fundamentals(
        self, price, has_fundamentals, expected
    ) -> None:
        m = Metrics(
            major_ccy="USD",
            price_major=price,
            mcap_usd=None,
            addv_usd=None,
            has_fundamentals=has_fundamentals,
        )
        assert is_real_security_line(m) is expected


# --------------------------------------------------------------------------- #
# carried_by_sibling
# --------------------------------------------------------------------------- #
class TestCarriedBySibling:
    def test_rejected_line_with_a_carrying_sibling_is_carried(self) -> None:
        carriers = [_mk("AGGG.L", "LSE", "Global Agg", "GBp", 500.0, 1e5, fin_ccy=None)]
        rejected = [_mk("EUNA.DE", "GER", "Global Agg", "EUR", 5.0, 3e6, fin_ccy=None)]
        assert carried_by_sibling(carriers, rejected) == {"EUNA.DE"}

    def test_lone_rejected_line_has_no_carrier(self) -> None:
        carriers = [_mk("OTHER", "NMS", "Other Corp", "USD", 10.0, 1e6)]
        rejected = [_mk("EUNA.DE", "GER", "Global Agg", "EUR", 5.0, 3e6, fin_ccy=None)]
        assert carried_by_sibling(carriers, rejected) == frozenset()

    def test_name_match_is_normalized_like_dedup(self) -> None:
        carriers = [
            _mk("NESN.SW", "EBS", "Nestlé S.A.", "CHF", 100.0, 1e6, fin_ccy="CHF")
        ]
        rejected = [
            _mk(
                "NSRGF.DE",
                "GER",
                "NESTLE SA",
                "EUR",
                90.0,
                1e3,
                fin_ccy="CHF",
                shares=None,
            )
        ]
        assert carried_by_sibling(carriers, rejected) == {"NSRGF.DE"}

    def test_same_name_in_another_reporting_currency_is_another_company(self) -> None:
        carriers = [_mk("ACME", "NMS", "ACME Corp", "USD", 100.0, 1e6, fin_ccy="USD")]
        rejected = [_mk("ACM.PA", "PAR", "ACME Corp", "EUR", 90.0, 5e5, fin_ccy="EUR")]
        assert carried_by_sibling(carriers, rejected) == frozenset()

    def test_line_without_reporting_currency_joins_a_single_currency_group(
        self,
    ) -> None:
        carriers = [_mk("ACME", "NMS", "ACME Corp", "USD", 100.0, 1e6, fin_ccy="USD")]
        rejected = [_mk("ACM.IL", "IOB", "ACME Corp", "EUR", 90.0, 5e4, fin_ccy=None)]
        assert carried_by_sibling(carriers, rejected) == {"ACM.IL"}

    def test_line_without_reporting_currency_is_unattributable_when_currencies_conflict(
        self,
    ) -> None:
        carriers = [
            _mk("ACME", "NMS", "ACME Corp", "USD", 100.0, 1e6, fin_ccy="USD"),
            _mk("ACM.PA", "PAR", "ACME Corp", "EUR", 90.0, 5e5, fin_ccy="EUR"),
        ]
        rejected = [_mk("ACM.IL", "IOB", "ACME Corp", "EUR", 90.0, 5e4, fin_ccy=None)]
        assert carried_by_sibling(carriers, rejected) == frozenset()

    def test_unnamed_line_is_never_matched(self) -> None:
        carriers = [_mk("ACME", "NMS", None, "USD", 100.0, 1e6)]
        rejected = [_mk("ACM.DE", "GER", None, "EUR", 90.0, 5e4)]
        assert carried_by_sibling(carriers, rejected) == frozenset()


# --------------------------------------------------------------------------- #
# FloorBand
# --------------------------------------------------------------------------- #
def test_floorband_rejects_exit_above_entry() -> None:
    with pytest.raises(ValueError, match="must be <="):
        FloorBand(entry=10.0, exit_=20.0)
