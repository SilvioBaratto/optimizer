"""``list_universe``: the allocator's view of the DB universe.

The allocator must choose its candidates from the instruments the database actually
holds, never from model memory. Pins, over a seeded SQLite catalogue:

* only active, priced instruments are listed, biggest market cap first, each with
  the sector the optimizer's sector bands use;
* an instrument delisted *after* the decision bar was still investable on it;
* a client's ESG exclusions drop the matching sectors and, failing closed, every
  instrument whose sector is unknown;
* paging (``limit``/``offset``/``total``) and a ``sector`` filter narrow the list;
  the filter takes the stored sector names and the GICS names a ConstraintSet
  speaks, and an unknown name is an error that lists the valid ones rather than a
  silently empty page;
"""

from __future__ import annotations

import datetime as dt

import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.schemas.enums import GicsSector
from fund.tools.universe import list_universe

_ASOF = dt.date(2024, 1, 30)


def _seed(
    db_session,
    ticker: str,
    *,
    sector: str | None,
    cap: int | None,
    priced: bool = True,
    delisted_at: dt.date | None = None,
    profile: bool = True,
) -> None:
    exchange = Exchange(name=f"EX-{ticker}")
    db_session.add(exchange)
    db_session.flush()
    instrument = Instrument(
        ticker=ticker,
        short_name=ticker,
        name=f"{ticker} Inc",
        exchange_id=exchange.id,
        instrument_type="STOCK",
        asset_class="equity",
        yfinance_ticker=ticker,
        delisted_at=delisted_at,
    )
    db_session.add(instrument)
    db_session.flush()
    if profile:
        db_session.add(
            TickerProfile(instrument_id=instrument.id, sector=sector, market_cap=cap)
        )
    if priced:
        db_session.add(
            PriceHistory(
                instrument_id=instrument.id,
                date=_ASOF - dt.timedelta(days=1),
                close=10.0,
                volume=1000,
            )
        )
    db_session.flush()


def _seed_catalogue(db_session) -> None:
    _seed(db_session, "TECH", sector="Technology", cap=300)
    _seed(db_session, "OIL", sector="Energy", cap=200)
    _seed(db_session, "PHRM", sector="Healthcare", cap=100)
    _seed(db_session, "NOCAP", sector="Utilities", cap=None)
    _seed(db_session, "GONE", sector="Technology", cap=999, delisted_at=_ASOF)
    _seed(
        db_session,
        "LATE",
        sector="Technology",
        cap=50,
        delisted_at=_ASOF + dt.timedelta(days=5),
    )
    _seed(db_session, "NOPX", sector="Technology", cap=500, priced=False)
    _seed(db_session, "NOSEC", sector=None, cap=None, profile=False)


def _tickers(result) -> list[str]:
    assert result["ok"] is True, result
    return [row["ticker"] for row in result["data"]["instruments"]]


def test_lists_active_priced_instruments_largest_market_cap_first(db_session) -> None:
    _seed_catalogue(db_session)

    result = list_universe(db_session, _ASOF)

    assert _tickers(result) == ["TECH", "OIL", "PHRM", "LATE", "NOCAP", "NOSEC"]
    assert result["data"]["total"] == 6
    tech = result["data"]["instruments"][0]
    assert tech == {
        "ticker": "TECH",
        "name": "TECH Inc",
        "sector": "Technology",
        "market_cap": 300,
        "instrument_type": "STOCK",
        "asset_class": "equity",
    }


def test_delisted_before_the_bar_or_unpriced_names_are_not_investable(db_session):
    _seed_catalogue(db_session)

    listed = _tickers(list_universe(db_session, _ASOF))

    assert "GONE" not in listed  # delisted on the bar itself
    assert "NOPX" not in listed  # no price on or before the bar
    assert "LATE" in listed  # delisted only after the bar: investable on it


def test_esg_exclusions_drop_the_sector_and_fail_closed_on_unknown_sectors(db_session):
    _seed_catalogue(db_session)

    result = list_universe(db_session, _ASOF, exclusions=(GicsSector.ENERGY,))

    assert _tickers(result) == ["TECH", "PHRM", "LATE", "NOCAP"]


def test_every_gics_sector_maps_to_the_sector_names_the_optimizer_bands_use(db_session):
    from optimizer.factors._sector_bands import ALL_11_SECTORS

    from fund.tools.universe import _SECTOR_NAME_BY_GICS

    assert set(_SECTOR_NAME_BY_GICS) == set(GicsSector)
    assert set(_SECTOR_NAME_BY_GICS.values()) == set(ALL_11_SECTORS)


def test_sector_filter_and_paging_narrow_the_listing(db_session) -> None:
    _seed_catalogue(db_session)

    tech = list_universe(db_session, _ASOF, sector="technology")
    page = list_universe(db_session, _ASOF, limit=2, offset=1)

    assert _tickers(tech) == ["TECH", "LATE"]
    assert _tickers(page) == ["OIL", "PHRM"]
    assert page["data"]["total"] == 6
    assert (page["data"]["limit"], page["data"]["offset"]) == (2, 1)


@pytest.mark.parametrize(
    "spelling",
    ["information_technology", "Information Technology", "information-technology"],
)
def test_sector_filter_accepts_the_gics_names_a_constraint_set_uses(
    db_session, spelling: str
) -> None:
    _seed_catalogue(db_session)

    result = list_universe(db_session, _ASOF, sector=spelling)

    assert _tickers(result) == ["TECH", "LATE"]
    assert result["data"]["total"] == 2


def test_sector_filter_maps_every_gics_sector_to_its_stored_name(db_session) -> None:
    from fund.tools.universe import _SECTOR_NAME_BY_GICS

    for index, stored in enumerate(_SECTOR_NAME_BY_GICS.values()):
        _seed(db_session, f"S{index}", sector=stored, cap=index + 1)

    for index, gics in enumerate(_SECTOR_NAME_BY_GICS):
        result = list_universe(db_session, _ASOF, sector=gics.value)
        assert _tickers(result) == [f"S{index}"], gics


def test_unknown_sector_is_an_error_naming_the_valid_sectors(db_session) -> None:
    _seed_catalogue(db_session)

    result = list_universe(db_session, _ASOF, sector="Tech")

    assert result["ok"] is False
    assert "'Tech'" in result["error"]
    assert "Technology" in result["error"]
    assert "information_technology" in result["error"]


def test_blank_sector_means_no_filter(db_session) -> None:
    _seed_catalogue(db_session)

    assert _tickers(list_universe(db_session, _ASOF, sector="  ")) == _tickers(
        list_universe(db_session, _ASOF)
    )


def test_limit_is_clamped_so_one_call_cannot_dump_the_whole_universe(db_session):
    _seed_catalogue(db_session)

    result = list_universe(db_session, _ASOF, limit=10_000)

    assert result["data"]["limit"] == 500


# --- restricted to an eligible universe ---------------------------------------
#
# With an allowlist the listing is exactly that set, in the mandate's order: ETFs
# have neither sector nor market cap, so the market-cap ranking would bury them.


def _seed_etf(db_session, ticker: str, *, asset_class: str = "equity") -> None:
    _seed(db_session, ticker, sector=None, cap=None, profile=False)
    instrument = (
        db_session.query(Instrument).filter(Instrument.yfinance_ticker == ticker).one()
    )
    instrument.instrument_type = "ETF"
    instrument.asset_class = asset_class
    db_session.flush()


def test_tickers_restrict_the_listing_to_exactly_that_set_in_the_given_order(
    db_session,
) -> None:
    _seed_catalogue(db_session)
    _seed_etf(db_session, "VWCE")
    _seed_etf(db_session, "IGLO", asset_class="fixed_income")

    result = list_universe(db_session, _ASOF, tickers=["IGLO", "TECH", "VWCE", "PHRM"])

    assert _tickers(result) == ["IGLO", "TECH", "VWCE", "PHRM"]
    assert result["data"]["total"] == 4


def test_a_restricted_listing_tells_etfs_from_stocks(db_session) -> None:
    _seed_catalogue(db_session)
    _seed_etf(db_session, "IGLO", asset_class="fixed_income")

    rows = list_universe(db_session, _ASOF, tickers=["IGLO", "TECH"])["data"][
        "instruments"
    ]

    assert [(r["ticker"], r["instrument_type"], r["asset_class"]) for r in rows] == [
        ("IGLO", "ETF", "fixed_income"),
        ("TECH", "STOCK", "equity"),
    ]


def test_a_restricted_listing_leaves_the_sql_esg_filter_to_the_resolver(
    db_session,
) -> None:
    _seed_catalogue(db_session)
    _seed_etf(db_session, "VWCE")

    result = list_universe(
        db_session,
        _ASOF,
        exclusions=(GicsSector.ENERGY,),
        tickers=["VWCE", "OIL"],
    )

    # The SQL rule fails closed on a missing sector and would drop every ETF; the
    # resolver already applied the exclusions to the tickers it hands over.
    assert _tickers(result) == ["VWCE", "OIL"]


def test_a_restricted_listing_pages_the_given_order(db_session) -> None:
    _seed_catalogue(db_session)
    names = ["PHRM", "TECH", "OIL", "NOCAP", "LATE"]

    page = list_universe(db_session, _ASOF, tickers=names, limit=2, offset=1)

    assert _tickers(page) == ["TECH", "OIL"]
    assert page["data"]["total"] == 5
    assert (page["data"]["limit"], page["data"]["offset"]) == (2, 1)


def test_a_sector_filter_narrows_within_the_restricted_listing(db_session) -> None:
    _seed_catalogue(db_session)
    _seed_etf(db_session, "VWCE")

    result = list_universe(
        db_session, _ASOF, tickers=["VWCE", "LATE", "OIL", "TECH"], sector="Technology"
    )

    assert _tickers(result) == ["LATE", "TECH"]
    assert result["data"]["total"] == 2


def test_a_restricted_listing_omits_names_the_database_cannot_serve(db_session) -> None:
    _seed_catalogue(db_session)

    result = list_universe(
        db_session, _ASOF, tickers=["GONE", "NOPX", "MISSING", "TECH"]
    )

    assert _tickers(result) == ["TECH"]
    assert result["data"]["total"] == 1


def test_an_empty_ticker_set_lists_nothing_rather_than_everything(db_session) -> None:
    _seed_catalogue(db_session)

    result = list_universe(db_session, _ASOF, tickers=[])

    assert _tickers(result) == []
    assert result["data"]["total"] == 0
