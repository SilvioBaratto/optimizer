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
