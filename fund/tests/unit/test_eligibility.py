"""``resolve_eligibility``: the tickers a portfolio may hold on a run date, and why not.

One pure resolver answers it once per run so every weighting tool and the PM seed
agree. Pins, over a seeded SQLite catalogue:

* an empty allowlist is unrestricted (``None``), today's behaviour;
* the allowlist is normalised itself (the mandate's validators are skipped by
  ``model_copy``) and the answer follows mandate order;
* each bar carries one stable reason: ``no_instrument``, ``delisted`` or
  ``not_priced``, judged point-in-time on ``asof``;
* the data reasons are the ones ``list_universe`` applies, and the cost is a constant
  number of queries however long the list is;
* under an ESG exclusion a single stock is judged by its stored sector and an ETF by
  the look-through of its latest sector snapshot against the client's tolerance,
  failing closed on stale, partial or unmappable data.
"""

from __future__ import annotations

import datetime as dt

import pytest
from _fund_fakes import make_constraint_set, make_energy_exclusion_constraint_set
from portopt_db.models.market_data.etf_metadata import ETFSectorWeight
from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy import event

from fund.config import load_config
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
    fi_subclass: str | None = None,
    tradeable: bool | None = None,
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
        fi_subclass=fi_subclass,
        yfinance_ticker=ticker,
        delisted_at=delisted_at,
        is_t212_tradeable=tradeable,
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


def _seed_sectors(
    session,
    ticker: str,
    weights: dict[str, float | None],
    *,
    age_days: int = 0,
) -> None:
    """Store a sector snapshot for ``ticker``, ``age_days`` before the decision bar."""
    instrument = (
        session.query(Instrument).filter(Instrument.yfinance_ticker == ticker).one()
    )
    as_of = _ASOF - dt.timedelta(days=age_days)
    for sector, weight in weights.items():
        session.add(
            ETFSectorWeight(
                instrument_id=instrument.id, as_of=as_of, sector=sector, weight=weight
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
    instrument = db_session.query(Instrument).filter_by(yfinance_ticker="FUND.E").one()
    db_session.add(TickerProfile(instrument_id=instrument.id, sector="Energy"))
    _seed_sectors(db_session, "FUND.E", {"technology": 1.0})

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


# --- ESG look-through for ETFs ------------------------------------------------
#
# An ETF has no single sector, so it is judged by the share of its latest sector
# snapshot that sits in an excluded sector, against the client's tolerance. Commodity
# and government-type bond funds hold no company exposure and pass; every other fund
# needs a recent, complete, mappable snapshot or stays barred.


def _policy(*sectors: GicsSector, tolerance: float = 0.0):
    return make_constraint_set().model_copy(
        update={"esg": EsgPolicy(exclusions=sectors, look_through_tolerance=tolerance)}
    )


def _resolve_one(session, ticker: str, constraint_set, **kwargs):
    result = resolve_eligibility(
        session, (ticker,), _ASOF, constraint_set=constraint_set, **kwargs
    )
    assert result is not None
    return result


def _outcome(result: Eligibility) -> str:
    """``allowed`` or the single bar reason of a one-ticker resolution."""
    return "allowed" if result.allowed else result.barred[0][1]


@pytest.mark.parametrize(
    ("weights", "tolerance", "outcome"),
    [
        pytest.param({"energy": 0.03, "technology": 0.97}, 0.05, "allowed", id="below"),
        pytest.param({"energy": 0.05, "technology": 0.95}, 0.05, "allowed", id="at"),
        pytest.param(
            {"energy": 0.08, "technology": 0.92}, 0.05, "esg:energy", id="above"
        ),
        pytest.param(
            {"energy": 0.001, "technology": 0.999}, 0.0, "esg:energy", id="strict-bars"
        ),
        pytest.param(
            {"energy": 0.0, "technology": 1.0}, 0.0, "allowed", id="strict-zero-passes"
        ),
        pytest.param({"technology": 1.0}, 0.0, "allowed", id="no-excluded-weight"),
    ],
)
def test_an_etf_passes_while_its_excluded_exposure_stays_within_the_tolerance(
    db_session, weights, tolerance, outcome
) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(db_session, "FUND.E", weights)

    result = _resolve_one(
        db_session, "FUND.E", _policy(GicsSector.ENERGY, tolerance=tolerance)
    )

    assert _outcome(result) == outcome


def test_exposure_sums_every_excluded_sector_and_names_the_largest(db_session) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(
        db_session, "FUND.E", {"energy": 0.1, "utilities": 0.2, "technology": 0.7}
    )
    both = _policy(GicsSector.ENERGY, GicsSector.UTILITIES, tolerance=0.05)

    result = _resolve_one(db_session, "FUND.E", both)

    assert result.barred == (("FUND.E", "esg:utilities"),)


def test_float_noise_does_not_push_an_exposure_over_its_tolerance(db_session) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(
        db_session, "FUND.E", {"energy": 0.1, "utilities": 0.2, "technology": 0.7}
    )
    # 0.1 + 0.2 is 0.30000000000000004.
    both = _policy(GicsSector.ENERGY, GicsSector.UTILITIES, tolerance=0.3)

    assert _outcome(_resolve_one(db_session, "FUND.E", both)) == "allowed"


def test_the_realestate_spelling_counts_as_real_estate(db_session) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(db_session, "FUND.E", {"realestate": 0.2, "technology": 0.8})

    result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.REAL_ESTATE))

    assert result.barred == (("FUND.E", "esg:real_estate"),)


@pytest.mark.parametrize(
    ("total", "outcome"),
    [
        (0.94, "esg_unclassified"),
        (0.96, "allowed"),
        (1.04, "allowed"),
        (1.06, "esg_unclassified"),
    ],
)
def test_a_snapshot_must_sum_to_one_within_five_points(
    db_session, total, outcome
) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(db_session, "FUND.E", {"technology": total})

    result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

    assert _outcome(result) == outcome


@pytest.mark.parametrize(
    ("age_days", "outcome"),
    [(179, "allowed"), (180, "allowed"), (181, "esg_unclassified")],
)
def test_a_snapshot_older_than_the_max_age_is_stale(
    db_session, age_days, outcome
) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(db_session, "FUND.E", {"technology": 1.0}, age_days=age_days)

    result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

    assert _outcome(result) == outcome


def test_the_max_age_is_a_parameter_of_the_resolver(db_session) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(db_session, "FUND.E", {"technology": 1.0}, age_days=31)

    strict = _resolve_one(
        db_session, "FUND.E", _policy(GicsSector.ENERGY), max_sector_age_days=30
    )
    lenient = _resolve_one(
        db_session, "FUND.E", _policy(GicsSector.ENERGY), max_sector_age_days=31
    )

    assert _outcome(strict) == "esg_unclassified"
    assert _outcome(lenient) == "allowed"


def test_the_default_max_age_comes_from_the_fund_config(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    from fund.tools import eligibility

    monkeypatch.setattr(
        eligibility,
        "settings",
        dataclasses.replace(eligibility.settings, esg_sector_weights_max_age_days=10),
    )
    _seed(db_session, "FUND.E")
    _seed_sectors(db_session, "FUND.E", {"technology": 1.0}, age_days=11)

    result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

    assert _outcome(result) == "esg_unclassified"


def test_null_weights_are_ignored_but_must_leave_a_complete_snapshot(
    db_session,
) -> None:
    _seed(db_session, "FUND.PART")
    _seed_sectors(db_session, "FUND.PART", {"energy": None, "technology": 1.0})
    _seed(db_session, "FUND.EMPTY")
    _seed_sectors(db_session, "FUND.EMPTY", {"energy": None, "technology": None})

    result = resolve_eligibility(
        db_session,
        ("FUND.PART", "FUND.EMPTY"),
        _ASOF,
        constraint_set=_policy(GicsSector.ENERGY),
    )

    assert result is not None
    assert result.allowed == ("FUND.PART",)
    assert result.barred == (("FUND.EMPTY", "esg_unclassified"),)


def test_a_label_outside_the_eleven_sectors_fails_closed(db_session) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(db_session, "FUND.E", {"crypto_assets": 0.1, "technology": 0.9})

    result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

    assert result.barred == (("FUND.E", "esg_unclassified"),)


def test_an_etf_with_no_snapshot_at_all_is_unclassified(db_session) -> None:
    _seed(db_session, "FUND.E")

    result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

    assert result.barred == (("FUND.E", "esg_unclassified"),)


def test_a_commodity_etf_passes_whatever_its_stored_collateral_looks_like(
    db_session,
) -> None:
    _seed(db_session, "GOLD.E", asset_class="commodity")
    _seed(db_session, "BARS.E", asset_class="commodity")
    _seed_sectors(db_session, "BARS.E", {"energy": 0.9, "basic_materials": 0.1})

    result = resolve_eligibility(
        db_session,
        ("GOLD.E", "BARS.E"),
        _ASOF,
        constraint_set=_policy(GicsSector.ENERGY),
    )

    assert result is not None
    assert result.allowed == ("GOLD.E", "BARS.E")


@pytest.mark.parametrize(
    ("asset_class", "fi_subclass", "outcome"),
    [
        ("fixed_income", "government", "allowed"),
        ("fixed_income", "inflation_linked", "allowed"),
        ("fixed_income", "Government", "allowed"),
        ("fixed_income", "corporate", "esg_unclassified"),
        ("fixed_income", "aggregate", "esg_unclassified"),
        ("fixed_income", "high_yield", "esg_unclassified"),
        ("fixed_income", "em", "esg_unclassified"),
        ("fixed_income", None, "esg_unclassified"),
        ("multi_asset", None, "esg_unclassified"),
        ("equity", None, "esg_unclassified"),
        # government is a fixed-income subclass; on any other class it means nothing
        ("equity", "government", "esg_unclassified"),
    ],
)
def test_only_government_type_bond_funds_pass_without_a_snapshot(
    db_session, asset_class, fi_subclass, outcome
) -> None:
    _seed(db_session, "FUND.E", asset_class=asset_class, fi_subclass=fi_subclass)

    result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

    assert _outcome(result) == outcome


def test_a_corporate_bond_fund_with_a_clean_snapshot_passes(db_session) -> None:
    _seed(db_session, "CORP.E", asset_class="fixed_income", fi_subclass="corporate")
    _seed_sectors(db_session, "CORP.E", {"financial_services": 0.6, "industrials": 0.4})

    result = _resolve_one(db_session, "CORP.E", _policy(GicsSector.ENERGY))

    assert _outcome(result) == "allowed"


def test_a_government_bond_fund_is_not_judged_by_its_snapshot(db_session) -> None:
    _seed(db_session, "GOV.E", asset_class="fixed_income", fi_subclass="government")
    _seed_sectors(db_session, "GOV.E", {"energy": 0.5, "utilities": 0.5})

    result = _resolve_one(db_session, "GOV.E", _policy(GicsSector.ENERGY))

    assert _outcome(result) == "allowed"


def test_the_latest_snapshot_decides_not_an_older_one(db_session) -> None:
    _seed(db_session, "FUND.E")
    _seed_sectors(db_session, "FUND.E", {"energy": 0.9, "technology": 0.1}, age_days=90)
    _seed_sectors(db_session, "FUND.E", {"technology": 1.0}, age_days=10)

    result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

    assert _outcome(result) == "allowed"


def test_without_exclusions_or_a_constraint_set_no_etf_is_screened(db_session) -> None:
    _seed(db_session, "FUND.E")

    plain = resolve_eligibility(
        db_session, ("FUND.E",), _ASOF, constraint_set=make_constraint_set()
    )
    unset = resolve_eligibility(db_session, ("FUND.E",), _ASOF)

    assert plain is not None
    assert plain.allowed == ("FUND.E",)
    assert unset is not None
    assert unset.allowed == ("FUND.E",)


def test_a_data_bar_takes_precedence_over_the_etf_look_through(db_session) -> None:
    _seed(db_session, "GONE.E", delisted_at=_ASOF)
    _seed_sectors(db_session, "GONE.E", {"energy": 1.0})

    result = _resolve_one(db_session, "GONE.E", _policy(GicsSector.ENERGY))

    assert result.barred == (("GONE.E", "delisted"),)


def test_stocks_and_funds_are_screened_together_in_mandate_order(db_session) -> None:
    _seed_stock(db_session, "OIL.S", sector="Energy")
    _seed_stock(db_session, "TECH.S", sector="Technology")
    _seed(db_session, "OILFUND.E")
    _seed_sectors(db_session, "OILFUND.E", {"energy": 1.0})
    _seed(db_session, "TECHFUND.E")
    _seed_sectors(db_session, "TECHFUND.E", {"technology": 1.0})

    result = resolve_eligibility(
        db_session,
        ("OILFUND.E", "TECH.S", "OIL.S", "TECHFUND.E"),
        _ASOF,
        constraint_set=_policy(GicsSector.ENERGY),
    )

    assert result is not None
    assert result.allowed == ("TECH.S", "TECHFUND.E")
    assert result.barred == (("OILFUND.E", "esg:energy"), ("OIL.S", "esg:energy"))


def test_the_fund_snapshots_are_read_in_one_batch(db_session) -> None:
    for index in range(6):
        _seed(db_session, f"FUND.{index}")
        _seed_sectors(db_session, f"FUND.{index}", {"technology": 1.0})
    engine = db_session.get_bind().engine
    counted: list[str] = []

    def _count(conn, cursor, statement, parameters, context, executemany) -> None:
        counted.append(statement)

    event.listen(engine, "before_cursor_execute", _count)
    try:
        resolve_eligibility(
            db_session, ("FUND.0",), _ASOF, constraint_set=_policy(GicsSector.ENERGY)
        )
        small = len(counted)
        counted.clear()
        resolve_eligibility(
            db_session,
            tuple(f"FUND.{index}" for index in range(6)),
            _ASOF,
            constraint_set=_policy(GicsSector.ENERGY),
        )
        large = len(counted)
    finally:
        event.remove(engine, "before_cursor_execute", _count)

    assert small == large


def test_the_look_through_max_age_defaults_to_half_a_year_and_reads_the_env() -> None:
    default = load_config(env={}, load_dotenv_file=False)
    overridden = load_config(
        env={"FUND_ESG_SECTOR_WEIGHTS_MAX_AGE_DAYS": "30"}, load_dotenv_file=False
    )

    assert default.esg_sector_weights_max_age_days == 180
    assert overridden.esg_sector_weights_max_age_days == 30


# --- the violation records the risk gate and the cockpit read-out share ---------


def test_not_eligible_violations_name_every_held_outside_ticker_and_reason() -> None:
    from fund.tools.eligibility import not_eligible_violations

    eligibility = Eligibility(allowed=("A", "B"), barred=(("C", "delisted"),))

    records = not_eligible_violations(
        eligibility, {"A": 0.5, "C": 0.3, "Z": 0.2, "B": 0.0}
    )

    assert records == [
        {"type": "not_eligible", "asset": "C", "reason": "delisted"},
        {"type": "not_eligible", "asset": "Z", "reason": "not_in_allowlist"},
    ]


def test_not_eligible_violations_ignore_solver_dust_and_zero_weights() -> None:
    from fund.tools.eligibility import not_eligible_violations

    eligibility = Eligibility(allowed=("A",), barred=())

    assert not_eligible_violations(eligibility, {"A": 1.0, "Z": 1e-9, "Y": 0.0}) == []


def test_not_eligible_violations_are_empty_without_an_allowlist() -> None:
    from fund.tools.eligibility import not_eligible_violations

    assert not_eligible_violations(None, {"Z": 1.0}) == []


# --- tradeable_only: scope to the Trading 212 catalogue -----------------------
#
# The flag is a tri-state per instrument: True (carried), False (matched and absent),
# NULL (never evaluated). Only an explicit False bars a name. When no instrument has
# been evaluated at all the restriction cannot be applied and the full universe stays,
# with a note, rather than silently emptying it.


def _seed_tradeable_catalogue(session) -> None:
    _seed(session, "ON.T", tradeable=True)
    _seed(session, "OFF.T", tradeable=False)
    _seed(session, "NEW.T")
    _seed(session, "OFF2.T", tradeable=False)


def test_tradeable_only_is_off_by_default_whatever_the_flags_say(db_session) -> None:
    _seed_tradeable_catalogue(db_session)

    result = resolve_eligibility(db_session, ("ON.T", "OFF.T", "NEW.T"), _ASOF)

    assert result == Eligibility(allowed=("ON.T", "OFF.T", "NEW.T"), barred=())


def test_tradeable_only_bars_an_explicit_false_and_keeps_true_and_unevaluated(
    db_session,
) -> None:
    _seed_tradeable_catalogue(db_session)

    result = resolve_eligibility(
        db_session, ("OFF.T", "ON.T", "NEW.T", "OFF2.T"), _ASOF, tradeable_only=True
    )

    assert result == Eligibility(
        allowed=("ON.T", "NEW.T"),
        barred=(("OFF.T", "not_t212_tradeable"), ("OFF2.T", "not_t212_tradeable")),
    )


def test_tradeable_only_composes_with_the_allowlist(db_session) -> None:
    """A tradeable name outside the allowlist stays outside it."""
    _seed_tradeable_catalogue(db_session)

    result = resolve_eligibility(db_session, ("ON.T",), _ASOF, tradeable_only=True)

    assert result is not None
    assert result.allowed == ("ON.T",)
    assert result.offenders(["ON.T", "NEW.T", "OFF.T"]) == {
        "NEW.T": "not_in_allowlist",
        "OFF.T": "not_in_allowlist",
    }


def test_a_data_bar_takes_precedence_over_the_tradeable_bar(db_session) -> None:
    _seed(db_session, "GONE.T", delisted_at=_ASOF, tradeable=False)
    _seed(db_session, "NOPX.T", priced=False, tradeable=False)
    _seed(db_session, "OFF.T", tradeable=False)

    result = resolve_eligibility(
        db_session,
        ("GONE.T", "NOPX.T", "MISSING.T", "OFF.T"),
        _ASOF,
        tradeable_only=True,
    )

    assert result is not None
    assert result.barred == (
        ("GONE.T", "delisted"),
        ("NOPX.T", "not_priced"),
        ("MISSING.T", "no_instrument"),
        ("OFF.T", "not_t212_tradeable"),
    )


def test_both_conditions_must_hold_under_an_esg_exclusion(db_session) -> None:
    """An untradeable name keeps its tradeable reason and is not ESG-screened."""
    _seed_stock(db_session, "OIL.OFF", sector="Energy")
    _seed_stock(db_session, "OIL.ON", sector="Energy")
    _seed_stock(db_session, "TECH.OFF", sector="Technology")
    _seed_stock(db_session, "TECH.ON", sector="Technology")
    for ticker, flag in (
        ("OIL.OFF", False),
        ("OIL.ON", True),
        ("TECH.OFF", False),
        ("TECH.ON", True),
    ):
        instrument = (
            db_session.query(Instrument)
            .filter(Instrument.yfinance_ticker == ticker)
            .one()
        )
        instrument.is_t212_tradeable = flag
    db_session.flush()

    result = resolve_eligibility(
        db_session,
        ("OIL.OFF", "OIL.ON", "TECH.OFF", "TECH.ON"),
        _ASOF,
        constraint_set=_esg_constraint_set(GicsSector.ENERGY),
        tradeable_only=True,
    )

    assert result is not None
    assert result.allowed == ("TECH.ON",)
    assert result.barred == (
        ("OIL.OFF", "not_t212_tradeable"),
        ("OIL.ON", "esg:energy"),
        ("TECH.OFF", "not_t212_tradeable"),
    )


def test_tradeable_only_keeps_the_full_allowlist_with_a_note_when_nothing_is_evaluated(
    db_session,
) -> None:
    _seed(db_session, "ELIG.A")
    _seed(db_session, "ELIG.B")

    result = resolve_eligibility(
        db_session, ("ELIG.A", "ELIG.B"), _ASOF, tradeable_only=True
    )

    assert result is not None
    assert result.allowed == ("ELIG.A", "ELIG.B")
    assert result.barred == ()
    assert len(result.notes) == 1
    assert "Trading 212" in result.notes[0]


def test_tradeable_only_bars_everything_when_the_overlay_matched_nothing(
    db_session,
) -> None:
    """The overlay ran and the broker carries none of them: an honest empty universe."""
    _seed(db_session, "OFF.A", tradeable=False)
    _seed(db_session, "OFF.B", tradeable=False)

    result = resolve_eligibility(
        db_session, ("OFF.A", "OFF.B"), _ASOF, tradeable_only=True
    )

    assert result is not None
    assert result.allowed == ()
    assert result.barred == (
        ("OFF.A", "not_t212_tradeable"),
        ("OFF.B", "not_t212_tradeable"),
    )
    assert result.notes == ()


def test_tradeable_only_engages_without_an_allowlist(db_session) -> None:
    """An empty allowlist no longer short-circuits to unrestricted."""
    _seed_tradeable_catalogue(db_session)

    result = resolve_eligibility(db_session, (), _ASOF, tradeable_only=True)

    assert result is not None
    assert result.has_allowlist is False
    assert result.allowed == ()
    assert result.barred == ()
    assert result.untradeable == frozenset({"OFF.T", "OFF2.T"})
    assert result.offenders(["ON.T", "NEW.T", "OFF.T", "ANYTHING.ELSE"]) == {
        "OFF.T": "not_t212_tradeable"
    }
    assert result.reason_for("OFF.T") == "not_t212_tradeable"
    assert result.reason_for("ON.T") is None


def test_tradeable_only_without_an_allowlist_and_nothing_evaluated_restricts_nothing(
    db_session,
) -> None:
    _seed(db_session, "ELIG.A")

    result = resolve_eligibility(db_session, (), _ASOF, tradeable_only=True)

    assert result is not None
    assert result.has_allowlist is False
    assert result.untradeable == frozenset()
    assert result.offenders(["ELIG.A", "ANYTHING"]) == {}
    assert len(result.notes) == 1
    assert "Trading 212" in result.notes[0]


def test_an_empty_allowlist_without_tradeable_only_is_still_unrestricted(
    db_session,
) -> None:
    _seed_tradeable_catalogue(db_session)

    assert resolve_eligibility(db_session, (), _ASOF, tradeable_only=False) is None


def test_the_tradeable_scope_costs_a_constant_number_of_extra_queries(
    db_session,
) -> None:
    _seed_tradeable_catalogue(db_session)
    engine = db_session.get_bind().engine
    counted: list[str] = []

    def _count(conn, cursor, statement, parameters, context, executemany) -> None:
        counted.append(statement)

    event.listen(engine, "before_cursor_execute", _count)
    try:
        resolve_eligibility(db_session, ("ON.T",), _ASOF, tradeable_only=True)
        small = len(counted)
        counted.clear()
        resolve_eligibility(
            db_session, ("ON.T", "OFF.T", "NEW.T", "OFF2.T"), _ASOF, tradeable_only=True
        )
        large = len(counted)
    finally:
        event.remove(engine, "before_cursor_execute", _count)

    assert small == large


def test_resolve_mandate_eligibility_reads_the_flag_off_the_mandate(db_session) -> None:
    from _fund_fakes import make_mandate

    from fund.tools.eligibility import resolve_mandate_eligibility

    _seed_tradeable_catalogue(db_session)
    mandate = make_mandate().model_copy(update={"tradeable_only": True})

    scoped = resolve_mandate_eligibility(db_session, mandate, _ASOF)
    plain = resolve_mandate_eligibility(
        db_session, mandate.model_copy(update={"tradeable_only": False}), _ASOF
    )

    assert scoped is not None
    assert scoped.untradeable == frozenset({"OFF.T", "OFF2.T"})
    assert plain is None


def test_not_eligible_violations_report_the_tradeable_reason() -> None:
    from fund.tools.eligibility import not_eligible_violations

    open_scope = Eligibility(
        allowed=(),
        barred=(),
        has_allowlist=False,
        untradeable=frozenset({"OFF.T"}),
    )

    assert not_eligible_violations(open_scope, {"OFF.T": 0.4, "ON.T": 0.6}) == [
        {"type": "not_eligible", "asset": "OFF.T", "reason": "not_t212_tradeable"}
    ]
