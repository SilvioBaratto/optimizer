"""``fund.tools.mandate``: ESG-sector and K&E instrument screening.

The screen runs against the instrument table, never against model output, and fails
closed: a restriction that is active but cannot be evaluated is a breach. These tests
pin the GICS → Yahoo sector bridge (drift-free against the optimizer's sector list),
the leverage name heuristic's false-positive guard, and the no-restriction short cut.
"""

from __future__ import annotations

import pytest
from optimizer.factors._sector_bands import ALL_11_SECTORS
from portopt_db.models.market_data.yfinance_data import TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.schemas.enums import GicsSector
from fund.tools.mandate import YAHOO_SECTOR_BY_GICS, mandate_breaches


def _instrument(
    db_session,
    ticker: str,
    *,
    instrument_type: str | None = "STOCK",
    name: str | None = None,
    sector: str | None = None,
    long_name: str | None = None,
) -> None:
    exchange = Exchange(name=f"EX-{ticker}")
    db_session.add(exchange)
    db_session.flush()
    instrument = Instrument(
        ticker=ticker,
        short_name=ticker,
        name=name,
        exchange_id=exchange.id,
        instrument_type=instrument_type,
        asset_class="equity",
        yfinance_ticker=ticker,
    )
    db_session.add(instrument)
    db_session.flush()
    if sector is not None or long_name is not None:
        db_session.add(
            TickerProfile(
                instrument_id=instrument.id, sector=sector, long_name=long_name
            )
        )
    db_session.flush()


def test_sector_bridge_covers_every_gics_sector() -> None:
    """Every GICS sector has a Yahoo label, so no exclusion is silently unmapped."""
    assert set(YAHOO_SECTOR_BY_GICS) == set(GicsSector)


def test_sector_bridge_matches_the_optimizer_sector_vocabulary() -> None:
    """The Yahoo labels are exactly the optimizer's 11 sector names."""
    assert set(YAHOO_SECTOR_BY_GICS.values()) == set(ALL_11_SECTORS)


def test_no_active_restriction_never_queries_the_database() -> None:
    """With nothing to screen, the function returns before opening a query."""

    # A session that explodes on first use proves the short cut.
    class _ExplodingSession:
        def execute(self, *args: object, **kwargs: object) -> None:
            """Fail the test on any query."""
            raise AssertionError("database touched without an active restriction")

    assert mandate_breaches(_ExplodingSession(), ["AAA", "BBB"]) == []  # type: ignore[arg-type]


def test_sector_match_is_case_and_whitespace_insensitive(db_session) -> None:
    """A profile sector of " energy " still matches the excluded "Energy"."""
    _instrument(db_session, "AAA", sector=" energy ")

    (breach,) = mandate_breaches(db_session, ["AAA"], excluded_sectors=["Energy"])

    assert breach["type"] == "esg_exclusion"


def test_unknown_ticker_is_unclassified_under_an_esg_screen(db_session) -> None:
    """A ticker with no instrument row cannot be proven compliant, so it blocks."""
    (breach,) = mandate_breaches(db_session, ["ZZZ"], excluded_sectors=["Energy"])

    assert breach == {"type": "unclassified_sector", "asset": "ZZZ"}


def test_stock_named_like_a_leveraged_product_is_not_leveraged(db_session) -> None:
    """A common stock called "Bull ... Ultra" is not a leveraged product."""
    _instrument(db_session, "AAA", instrument_type="STOCK", name="Bull Run Ultra Corp")

    assert mandate_breaches(db_session, ["AAA"], no_leverage=True) == []


@pytest.mark.parametrize(
    "name",
    [
        "Direxion Daily S&P 500 Bull 3X Shares",
        "ProShares UltraPro QQQ",
        "ProShares Short QQQ -1x Daily",
        "Leverage Shares 2x Tesla",
        "Inverse VIX Short-Term ETN",
    ],
)
def test_leveraged_and_inverse_etf_names_are_flagged(db_session, name: str) -> None:
    """Leveraged and inverse ETF names trip the leverage restriction."""
    _instrument(db_session, "AAA", instrument_type="ETF", name=name)

    (breach,) = mandate_breaches(db_session, ["AAA"], no_leverage=True)

    assert breach["type"] == "leveraged_instrument"


def test_leveraged_etf_is_also_complex(db_session) -> None:
    """A leveraged ETF is complex under the no-complex restriction as well."""
    _instrument(db_session, "AAA", instrument_type="ETF", name="ProShares Ultra QQQ")

    kinds = {b["type"] for b in mandate_breaches(db_session, ["AAA"], no_complex=True)}

    assert kinds == {"complex_instrument"}


def test_plain_etf_is_not_complex(db_session) -> None:
    """A plain index ETF passes both the no-complex and no-leverage screens."""
    _instrument(
        db_session, "AAA", instrument_type="ETF", name="Vanguard FTSE All-World"
    )

    assert (
        mandate_breaches(db_session, ["AAA"], no_complex=True, no_leverage=True) == []
    )


def test_missing_instrument_type_is_complex(db_session) -> None:
    """An unknown instrument type fails closed under no-complex."""
    _instrument(db_session, "AAA", instrument_type=None)

    (breach,) = mandate_breaches(db_session, ["AAA"], no_complex=True)

    assert breach["type"] == "complex_instrument"
