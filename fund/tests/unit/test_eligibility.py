"""``resolve_eligibility``: the tickers a portfolio may hold on a run date, and why not.

One pure resolver answers it once per run so every weighting tool and the PM seed
agree. Pins, over a seeded SQLite catalogue:

* an empty allowlist is unrestricted (``None``), today's behaviour;
* the allowlist is normalised itself (the mandate's validators are skipped by
  ``model_copy``) and the answer follows mandate order;
* each bar carries one stable reason: ``no_instrument``, ``delisted`` or
  ``not_priced``, judged point-in-time on ``asof``;
* the data reasons are the ones ``list_universe`` applies, and the cost is a constant
  number of queries however long the list is.
"""

from __future__ import annotations

import datetime as dt

import pytest
from _fund_fakes import make_constraint_set, make_energy_exclusion_constraint_set
from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy import event

from fund.schemas.constraint_set import EsgPolicy
from fund.schemas.enums import GicsSector
from fund.tools.eligibility import Eligibility, resolve_eligibility
from fund.tools.mandate import constraint_set_breaches
from fund.tools.universe import list_universe

_ASOF = dt.date(2024, 1, 30)


def _seed(
    session,
    ticker: str,
    *,
    priced: bool = True,
    delisted_at: dt.date | None = None,
    instrument_type: str = "ETF",
    asset_class: str = "equity",
) -> None:
    exchange = Exchange(name=f"EX-{ticker}")
    session.add(exchange)
    session.flush()
    instrument = Instrument(
        ticker=ticker,
        short_name=ticker,
        name=f"{ticker} fund",
        exchange_id=exchange.id,
        instrument_type=instrument_type,
        asset_class=asset_class,
        yfinance_ticker=ticker,
        delisted_at=delisted_at,
    )
    session.add(instrument)
    session.flush()
    if priced:
        session.add(
            PriceHistory(
                instrument_id=instrument.id,
                date=_ASOF - dt.timedelta(days=1),
                close=10.0,
                volume=1000,
            )
        )
    session.flush()


def _seed_catalogue(session) -> None:
    _seed(session, "ELIG.A")
    _seed(session, "ELIG.B")
    _seed(session, "GONE.X", delisted_at=_ASOF)
    _seed(session, "LATE.X", delisted_at=_ASOF + dt.timedelta(days=5))
    _seed(session, "NOPX.X", priced=False)
    _seed(session, "GONENOPX.X", priced=False, delisted_at=_ASOF)


def test_an_empty_allowlist_is_unrestricted(db_session) -> None:
    _seed_catalogue(db_session)

    assert resolve_eligibility(db_session, (), _ASOF) is None


def test_every_investable_name_is_allowed_in_mandate_order(db_session) -> None:
    _seed_catalogue(db_session)

    result = resolve_eligibility(db_session, ("ELIG.B", "ELIG.A"), _ASOF)

    assert result == Eligibility(allowed=("ELIG.B", "ELIG.A"), barred=())


def test_each_bar_carries_its_reason_and_keeps_mandate_order(db_session) -> None:
    _seed_catalogue(db_session)

    result = resolve_eligibility(
        db_session,
        ("NOPX.X", "ELIG.A", "MISSING.X", "GONE.X", "ELIG.B"),
        _ASOF,
    )

    assert result is not None
    assert result.allowed == ("ELIG.A", "ELIG.B")
    assert result.barred == (
        ("NOPX.X", "not_priced"),
        ("MISSING.X", "no_instrument"),
        ("GONE.X", "delisted"),
    )


def test_a_delisted_unpriced_name_is_reported_as_delisted(db_session) -> None:
    _seed_catalogue(db_session)

    result = resolve_eligibility(db_session, ("GONENOPX.X",), _ASOF)

    assert result is not None
    assert result.barred == (("GONENOPX.X", "delisted"),)


def test_delisting_is_judged_on_the_bar_not_on_today(db_session) -> None:
    _seed_catalogue(db_session)

    after = resolve_eligibility(db_session, ("LATE.X",), _ASOF)
    before = resolve_eligibility(db_session, ("LATE.X",), _ASOF + dt.timedelta(days=5))

    assert after is not None
    assert after.allowed == ("LATE.X",)
    assert before is not None
    assert before.barred == (("LATE.X", "delisted"),)


def test_prices_after_the_bar_do_not_make_a_name_priced(db_session) -> None:
    _seed_catalogue(db_session)

    result = resolve_eligibility(db_session, ("ELIG.A",), _ASOF - dt.timedelta(days=30))

    assert result is not None
    assert result.barred == (("ELIG.A", "not_priced"),)


def test_the_allowlist_is_normalised_whatever_the_mandate_validators_did(
    db_session,
) -> None:
    _seed_catalogue(db_session)

    result = resolve_eligibility(
        db_session, (" elig.a ", "ELIG.A", "", "  ", "elig.b", "Elig.B"), _ASOF
    )

    assert result == Eligibility(allowed=("ELIG.A", "ELIG.B"), barred=())


def test_an_allowlist_of_only_blanks_is_unrestricted(db_session) -> None:
    assert resolve_eligibility(db_session, ("", "   "), _ASOF) is None


def test_the_data_reasons_agree_with_list_universe(db_session) -> None:
    _seed_catalogue(db_session)
    names = ("ELIG.A", "ELIG.B", "GONE.X", "LATE.X", "NOPX.X", "MISSING.X")

    result = resolve_eligibility(db_session, names, _ASOF)
    listed = list_universe(db_session, _ASOF, limit=500)

    assert result is not None
    in_universe = {row["ticker"] for row in listed["data"]["instruments"]}
    assert set(result.allowed) == set(names) & in_universe


def test_the_cost_is_a_constant_number_of_queries(db_session) -> None:
    _seed_catalogue(db_session)
    engine = db_session.get_bind().engine
    counted: list[str] = []

    def _count(conn, cursor, statement, parameters, context, executemany) -> None:
        counted.append(statement)

    event.listen(engine, "before_cursor_execute", _count)
    try:
        resolve_eligibility(db_session, ("ELIG.A",), _ASOF)
        small = len(counted)
        counted.clear()
        resolve_eligibility(
            db_session, ("ELIG.A", "ELIG.B", "GONE.X", "NOPX.X", "MISSING.X"), _ASOF
        )
        large = len(counted)
    finally:
        event.remove(engine, "before_cursor_execute", _count)

    assert small == large == 1


class TestEligibility:
    def _eligibility(self) -> Eligibility:
        return Eligibility(
            allowed=("A", "B"), barred=(("C", "delisted"), ("D", "esg:energy"))
        )

    def test_allowed_set_holds_the_allowed_names(self) -> None:
        assert self._eligibility().allowed_set == frozenset({"A", "B"})

    @pytest.mark.parametrize(
        ("ticker", "reason"),
        [("C", "delisted"), ("D", "esg:energy"), ("A", None), ("Z", None)],
    )
    def test_reason_for_reports_the_bar_of_a_barred_name(
        self, ticker: str, reason: str | None
    ) -> None:
        assert self._eligibility().reason_for(ticker) == reason

    def test_offenders_names_every_ticker_outside_the_allowed_set(self) -> None:
        offenders = self._eligibility().offenders(["A", "C", "Z", "D", "B"])

        assert offenders == {
            "C": "delisted",
            "Z": "not_in_allowlist",
            "D": "esg:energy",
        }

    def test_offenders_is_empty_when_every_ticker_is_allowed(self) -> None:
        assert self._eligibility().offenders(["B", "A"]) == {}

    def test_it_is_frozen_and_hashable(self) -> None:
        eligibility = self._eligibility()

        assert hash(eligibility) == hash(self._eligibility())
        with pytest.raises(AttributeError):
            eligibility.allowed = ()  # type: ignore[misc]


# --- ESG rule for single stocks ---------------------------------------------
#
# A stock is judged by its stored sector, failing closed: an excluded sector and a
# sector the resolver cannot place both bar it, an unrelated sector passes. ETFs are
# not judged by this rule.


def _seed_stock(session, ticker: str, *, sector: str | None, profile: bool = True):
    _seed(session, ticker, instrument_type="STOCK")
    if profile:
        instrument = (
            session.query(Instrument).filter(Instrument.yfinance_ticker == ticker).one()
        )
        session.add(TickerProfile(instrument_id=instrument.id, sector=sector))
        session.flush()


def _esg_constraint_set(*sectors: GicsSector):
    return make_constraint_set().model_copy(
        update={"esg": EsgPolicy(exclusions=sectors)}
    )


def test_a_stock_in_an_excluded_sector_is_barred_with_the_sector(db_session) -> None:
    _seed_stock(db_session, "OIL.S", sector="Energy")

    result = resolve_eligibility(
        db_session,
        ("OIL.S",),
        _ASOF,
        constraint_set=make_energy_exclusion_constraint_set(),
    )

    assert result is not None
    assert result.allowed == ()
    assert result.barred == (("OIL.S", "esg:energy"),)


def test_a_stock_with_no_sector_is_barred_as_unclassified(db_session) -> None:
    _seed_stock(db_session, "NOPROFILE.S", sector=None, profile=False)
    _seed_stock(db_session, "BLANK.S", sector=None)

    result = resolve_eligibility(
        db_session,
        ("NOPROFILE.S", "BLANK.S"),
        _ASOF,
        constraint_set=make_energy_exclusion_constraint_set(),
    )

    assert result is not None
    assert result.barred == (
        ("NOPROFILE.S", "esg_unclassified"),
        ("BLANK.S", "esg_unclassified"),
    )


def test_a_stock_with_a_sector_outside_the_eleven_labels_fails_closed(
    db_session,
) -> None:
    _seed_stock(db_session, "ODD.S", sector="Crypto Assets")

    result = resolve_eligibility(
        db_session,
        ("ODD.S",),
        _ASOF,
        constraint_set=make_energy_exclusion_constraint_set(),
    )

    assert result is not None
    assert result.barred == (("ODD.S", "esg_unclassified"),)


def test_a_stock_outside_every_exclusion_passes(db_session) -> None:
    _seed_stock(db_session, "TECH.S", sector="Technology")
    _seed_stock(db_session, "OIL.S", sector="Energy")

    result = resolve_eligibility(
        db_session,
        ("TECH.S", "OIL.S"),
        _ASOF,
        constraint_set=make_energy_exclusion_constraint_set(),
    )

    assert result is not None
    assert result.allowed == ("TECH.S",)


def test_a_real_estate_label_without_the_underscore_is_still_real_estate(
    db_session,
) -> None:
    _seed_stock(db_session, "REIT.S", sector="realestate")

    result = resolve_eligibility(
        db_session,
        ("REIT.S",),
        _ASOF,
        constraint_set=_esg_constraint_set(GicsSector.REAL_ESTATE),
    )

    assert result is not None
    assert result.barred == (("REIT.S", "esg:real_estate"),)


def test_without_exclusions_nothing_gets_an_esg_reason(db_session) -> None:
    _seed_stock(db_session, "OIL.S", sector="Energy")
    _seed_stock(db_session, "NOSECTOR.S", sector=None, profile=False)

    result = resolve_eligibility(
        db_session,
        ("OIL.S", "NOSECTOR.S"),
        _ASOF,
        constraint_set=make_constraint_set(),
    )

    assert result is not None
    assert result.allowed == ("OIL.S", "NOSECTOR.S")


def test_without_a_constraint_set_nothing_gets_an_esg_reason(db_session) -> None:
    _seed_stock(db_session, "OIL.S", sector="Energy")

    result = resolve_eligibility(db_session, ("OIL.S",), _ASOF)

    assert result is not None
    assert result.allowed == ("OIL.S",)


def test_a_data_bar_takes_precedence_over_an_esg_bar(db_session) -> None:
    _seed_stock(db_session, "OIL.S", sector="Energy")
    _seed(db_session, "GONEOIL.S", instrument_type="STOCK", delisted_at=_ASOF)

    result = resolve_eligibility(
        db_session,
        ("GONEOIL.S", "OIL.S"),
        _ASOF,
        constraint_set=make_energy_exclusion_constraint_set(),
    )

    assert result is not None
    assert result.barred == (("GONEOIL.S", "delisted"), ("OIL.S", "esg:energy"))


def test_an_etf_is_not_judged_by_the_stock_sector_rule(db_session) -> None:
    _seed(db_session, "FUND.E", instrument_type="ETF")

    result = resolve_eligibility(
        db_session,
        ("FUND.E",),
        _ASOF,
        constraint_set=make_energy_exclusion_constraint_set(),
    )

    assert result is not None
    assert result.allowed == ("FUND.E",)


def test_resolver_list_universe_and_mandate_screen_differ_on_an_unplaceable_sector(
    db_session,
) -> None:
    # Pinned on purpose: list_universe and the resolver fail closed on a sector label
    # outside the eleven, while mandate_breaches waves it through. A future change to
    # any of the three must be a deliberate decision, not a side effect.
    _seed_stock(db_session, "ODD.S", sector="Crypto Assets")
    constraint_set = make_energy_exclusion_constraint_set()

    listed = list_universe(db_session, _ASOF, exclusions=[GicsSector.ENERGY])
    resolved = resolve_eligibility(
        db_session, ("ODD.S",), _ASOF, constraint_set=constraint_set
    )
    breaches = constraint_set_breaches(db_session, constraint_set, ["ODD.S"])

    assert [row["ticker"] for row in listed["data"]["instruments"]] == []
    assert resolved is not None
    assert resolved.barred == (("ODD.S", "esg_unclassified"),)
    assert breaches == []
