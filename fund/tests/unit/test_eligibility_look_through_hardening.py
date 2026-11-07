"""Gaps in the ESG look-through that let a hard exclusion pass.

Each case here used to return ``allowed`` for a fund that may hold an excluded sector:
a commodity-labelled fund that actually owns stocks, a sector snapshot whose weights are
not finite, a strict tolerance that quietly allowed the unreported remainder of a
snapshot, a snapshot dated after the decision bar, and a venue twin chosen by row order.
"""

from __future__ import annotations

import datetime as dt

import pytest
from portopt_db.models.market_data.etf_metadata import ETFAssetClass, ETFSectorWeight
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.etf_metadata_repository import SectorSnapshot
from test_eligibility import (
    _ASOF,
    _outcome,
    _policy,
    _resolve_one,
    _seed,
    _seed_sectors,
)

from fund.schemas.enums import GicsSector
from fund.tools.eligibility import _look_through


def _seed_stock_share(
    session, ticker: str, stock_pct: float | None, *, age_days: int = 0
) -> None:
    """Store an asset-class allocation for ``ticker``."""
    instrument = (
        session.query(Instrument).filter(Instrument.yfinance_ticker == ticker).one()
    )
    session.add(
        ETFAssetClass(
            instrument_id=instrument.id,
            as_of=_ASOF - dt.timedelta(days=age_days),
            stock_pct=stock_pct,
        )
    )
    session.flush()


class TestCommodityLabelIsNotAnExemptionForAStockFund:
    """A name-derived ``commodity`` label cannot excuse a fund that owns equities."""

    def test_a_commodity_labelled_fund_holding_stocks_is_looked_through(
        self, db_session
    ) -> None:
        """A natural-gas equity ETF is labelled commodity by name but is 97% energy."""
        _seed(db_session, "GAS.E", asset_class="commodity")
        _seed_sectors(db_session, "GAS.E", {"energy": 0.97, "utilities": 0.03})
        _seed_stock_share(db_session, "GAS.E", 0.99)

        result = _resolve_one(db_session, "GAS.E", _policy(GicsSector.ENERGY))

        assert result.barred == (("GAS.E", "esg:energy"),)

    def test_a_commodity_labelled_stock_fund_without_a_snapshot_is_unclassified(
        self, db_session
    ) -> None:
        """Owning stocks removes the exemption; no sector data then fails closed."""
        _seed(db_session, "GAS.E", asset_class="commodity")
        _seed_stock_share(db_session, "GAS.E", 0.99)

        result = _resolve_one(db_session, "GAS.E", _policy(GicsSector.ENERGY))

        assert result.barred == (("GAS.E", "esg_unclassified"),)

    @pytest.mark.parametrize("stock_pct", [0.0, None])
    def test_a_commodity_fund_reporting_no_stocks_stays_exempt(
        self, db_session, stock_pct
    ) -> None:
        """Collateral-only funds, and funds with no allocation data, keep the pass."""
        _seed(db_session, "BARS.E", asset_class="commodity")
        _seed_sectors(db_session, "BARS.E", {"energy": 0.9, "basic_materials": 0.1})
        _seed_stock_share(db_session, "BARS.E", stock_pct)

        result = _resolve_one(db_session, "BARS.E", _policy(GicsSector.ENERGY))

        assert _outcome(result) == "allowed"

    def test_a_stock_share_inside_the_tolerance_cannot_breach_it(
        self, db_session
    ) -> None:
        """Three points of stocks cannot put more than three points in energy."""
        _seed(db_session, "BARS.E", asset_class="commodity")
        _seed_stock_share(db_session, "BARS.E", 0.03)

        result = _resolve_one(
            db_session, "BARS.E", _policy(GicsSector.ENERGY, tolerance=0.05)
        )

        assert _outcome(result) == "allowed"

    def test_a_government_bond_fund_holding_stocks_is_looked_through(
        self, db_session
    ) -> None:
        """The same rule covers the other exempt class."""
        _seed(db_session, "GOV.E", asset_class="fixed_income", fi_subclass="government")
        _seed_sectors(db_session, "GOV.E", {"energy": 0.5, "utilities": 0.5})
        _seed_stock_share(db_session, "GOV.E", 0.4)

        result = _resolve_one(db_session, "GOV.E", _policy(GicsSector.ENERGY))

        assert _outcome(result).startswith("esg:")

    def test_an_allocation_dated_after_the_bar_is_not_used(self, db_session) -> None:
        """A later stock share must not revoke an exemption on an earlier run."""
        _seed(db_session, "BARS.E", asset_class="commodity")
        _seed_stock_share(db_session, "BARS.E", 0.99, age_days=-10)

        result = _resolve_one(db_session, "BARS.E", _policy(GicsSector.ENERGY))

        assert _outcome(result) == "allowed"


class TestNonFiniteWeightsFailClosed:
    """A NaN or infinite weight is not a number the completeness check can sum.

    SQLite stores NaN as NULL, so the resolver is driven with a snapshot directly: a
    PostgreSQL ``numeric`` column keeps NaN and hands it back.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_weight_leaves_the_fund_unclassified(self, bad) -> None:
        """NaN poisons the total, and a NaN comparison reads as "complete"."""
        snapshot = SectorSnapshot(
            as_of=_ASOF, weights={"technology": 0.6, "healthcare": bad}
        )

        reason = _look_through(
            snapshot,
            frozenset({GicsSector.ENERGY}),
            0.0,
            asof=_ASOF,
            max_age_days=180,
        )

        assert reason == "esg_unclassified"


class TestStrictToleranceCountsTheUnreportedRemainder:
    """What a snapshot does not account for may be sitting in an excluded sector."""

    def test_strict_tolerance_bars_a_snapshot_with_an_unreported_remainder(
        self, db_session
    ) -> None:
        """Four points are missing and energy is not listed: up to four could be it."""
        _seed(db_session, "FUND.E")
        _seed_sectors(db_session, "FUND.E", {"technology": 0.96})

        result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

        assert result.barred == (("FUND.E", "esg_unclassified"),)

    def test_a_remainder_within_the_tolerance_passes(self, db_session) -> None:
        """Four unreported points sit inside a five point tolerance."""
        _seed(db_session, "FUND.E")
        _seed_sectors(db_session, "FUND.E", {"technology": 0.96})

        result = _resolve_one(
            db_session, "FUND.E", _policy(GicsSector.ENERGY, tolerance=0.05)
        )

        assert _outcome(result) == "allowed"

    def test_an_excluded_sector_reported_as_zero_leaves_no_remainder_to_fear(
        self, db_session
    ) -> None:
        """The provider said energy is zero, so the gap is not energy's."""
        _seed(db_session, "FUND.E")
        _seed_sectors(db_session, "FUND.E", {"technology": 0.96, "energy": 0.0})

        result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

        assert _outcome(result) == "allowed"

    def test_a_remainder_that_breaches_alongside_real_exposure_names_the_sector(
        self, db_session
    ) -> None:
        """Real excluded weight above the tolerance is still reported by sector."""
        _seed(db_session, "FUND.E")
        _seed_sectors(
            db_session, "FUND.E", {"technology": 0.9, "energy": 0.06, "utilities": 0.0}
        )

        result = _resolve_one(
            db_session, "FUND.E", _policy(GicsSector.ENERGY, tolerance=0.05)
        )

        assert result.barred == (("FUND.E", "esg:energy"),)


class TestSnapshotsFromTheFutureAreIgnored:
    """A run replayed at an earlier bar is judged on what was known then."""

    def test_a_snapshot_dated_after_the_bar_is_not_the_funds_look_through(
        self, db_session
    ) -> None:
        """Only a snapshot from the future exists, so the fund is unclassified."""
        _seed(db_session, "FUND.E")
        _seed_sectors(db_session, "FUND.E", {"technology": 1.0}, age_days=-3)

        result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

        assert result.barred == (("FUND.E", "esg_unclassified"),)

    def test_the_latest_snapshot_on_or_before_the_bar_decides(self, db_session) -> None:
        """A later, energy-heavy snapshot does not override the one known at the bar."""
        _seed(db_session, "FUND.E")
        _seed_sectors(db_session, "FUND.E", {"technology": 1.0}, age_days=5)
        _seed_sectors(
            db_session, "FUND.E", {"energy": 0.9, "technology": 0.1}, age_days=-5
        )

        result = _resolve_one(db_session, "FUND.E", _policy(GicsSector.ENERGY))

        assert _outcome(result) == "allowed"


class TestVenueTwinsAreResolvedDeterministically:
    """Two instruments sharing a Yahoo symbol are judged by the active listing."""

    def test_the_active_listing_decides_the_route_not_a_delisted_twin(
        self, db_session
    ) -> None:
        """The delisted twin is commodity (exempt); the live listing is equity."""
        live_exchange = Exchange(name="EX-LIVE")
        old_exchange = Exchange(name="EX-OLD")
        db_session.add_all([live_exchange, old_exchange])
        db_session.flush()
        for ticker, exchange, asset_class, delisted_at in (
            ("TWIN.OLD", old_exchange, "commodity", _ASOF - dt.timedelta(days=400)),
            ("TWIN.NEW", live_exchange, "equity", None),
        ):
            db_session.add(
                Instrument(
                    ticker=ticker,
                    short_name=ticker,
                    exchange_id=exchange.id,
                    instrument_type="ETF",
                    asset_class=asset_class,
                    yfinance_ticker="TWIN",
                    delisted_at=delisted_at,
                )
            )
        db_session.flush()
        live = db_session.query(Instrument).filter_by(ticker="TWIN.NEW").one()
        db_session.add(
            PriceHistory(
                instrument_id=live.id,
                date=_ASOF - dt.timedelta(days=1),
                close=10.0,
                volume=1000,
            )
        )
        for sector, weight in (("energy", 0.9), ("technology", 0.1)):
            db_session.add(
                ETFSectorWeight(
                    instrument_id=live.id, as_of=_ASOF, sector=sector, weight=weight
                )
            )
        db_session.flush()

        result = _resolve_one(db_session, "TWIN", _policy(GicsSector.ENERGY))

        assert result.barred == (("TWIN", "esg:energy"),)
