"""``universe_filter``: pre-selection over a seeded price panel.

Pins the tool as a pure, deterministic function over a seeded SQLite panel:

* it wraps ``optimizer.pre_selection`` and returns the **surviving tickers** as a
  ``list[str]`` in requested order — never the returns frame;
* an empty universe (or one with nothing priced) is a valid empty result
  (``{ok: true, data: []}``), not an error;
* a bad selection criterion (unknown ``select_k_measure`` enum) degrades to
  ``{ok: false, error}`` via ``PreSelectionConfig`` validation, never raised;
* ``top_k`` is a real filter lever: ``top_k=1`` prunes the universe to one asset;
* same seeded data + same criteria ⇒ identical list;
* ``exclusions`` is enforced against each instrument's profile sector, failing
  closed for an instrument whose sector is unknown; a criterion the tool cannot
  apply is reported in ``ignored_criteria`` instead of vanishing behind ``ok``.
"""

from __future__ import annotations

import datetime as dt
import math

from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.tools.universe import universe_filter

_START = dt.date(2024, 1, 1)
_N_DAYS = 60
_ASOF = _START + dt.timedelta(days=_N_DAYS - 1)
_UNIVERSE = ["AAA", "BBB", "CCC", "DUP"]
# Deterministic per-ticker (start, drift, wave, freq, phase) — no RNG. ``DUP`` is a
# scaled clone of ``AAA`` (identical returns ⇒ correlation 1.0) so DropCorrelated
# has something to prune; the others use distinct frequencies/phases.
_SERIES = {
    "AAA": (100.0, 0.0008, 0.015, 5.0, 0.0),
    "BBB": (50.0, 0.0004, 0.030, 3.0, 1.0),
    "CCC": (25.0, 0.0011, 0.020, 7.0, 2.0),
    "DUP": (200.0, 0.0008, 0.015, 5.0, 0.0),  # clone of AAA, scaled 2x
}


def _seed_instrument(db_session, ticker: str) -> Instrument:
    ex = Exchange(name=f"EX-{ticker}")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker=ticker,
        short_name=ticker,
        exchange_id=ex.id,
        instrument_type="EQUITY",
        asset_class="equity",
        yfinance_ticker=ticker,
    )
    db_session.add(inst)
    db_session.flush()
    return inst


def _seed_panel(db_session, tickers: list[str] | None = None, n_days: int = _N_DAYS):
    """Seed an ``n_days`` close panel for ``tickers`` (default: full universe)."""
    for ticker in tickers if tickers is not None else list(_SERIES):
        start, drift, wave, freq, phase = _SERIES[ticker]
        inst = _seed_instrument(db_session, ticker)
        for i in range(n_days):
            close = start * (1.0 + drift * i + wave * math.sin(i / freq + phase))
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=round(close, 6),
                    volume=1000,
                )
            )
    db_session.flush()


_THREE = ["AAA", "BBB", "CCC"]


def _seed_profile(db_session, ticker: str, sector: str | None) -> None:
    inst = db_session.query(Instrument).filter_by(yfinance_ticker=ticker).one()
    db_session.add(TickerProfile(instrument_id=inst.id, symbol=ticker, sector=sector))
    db_session.flush()


class TestUniverseFilter:
    def test_returns_subset_of_universe(self, db_session) -> None:
        _seed_panel(db_session)

        result = universe_filter(db_session, _ASOF, _UNIVERSE)

        assert result["ok"] is True
        survivors = result["data"]
        assert isinstance(survivors, list)
        assert set(survivors) <= set(_UNIVERSE)
        # order follows the requested universe
        assert survivors == [t for t in _UNIVERSE if t in survivors]

    def test_drops_perfectly_correlated_clone(self, db_session) -> None:
        _seed_panel(db_session)

        survivors = universe_filter(db_session, _ASOF, _UNIVERSE)["data"]

        # AAA and DUP have identical returns (corr 1.0); at most one survives.
        assert not ("AAA" in survivors and "DUP" in survivors)

    def test_top_k_prunes_to_one(self, db_session) -> None:
        _seed_panel(db_session)

        result = universe_filter(db_session, _ASOF, _UNIVERSE, criteria={"top_k": 1})

        assert result["ok"] is True
        assert len(result["data"]) == 1
        assert result["data"][0] in _UNIVERSE

    def test_unknown_criteria_keys_are_ignored(self, db_session) -> None:
        _seed_panel(db_session)

        result = universe_filter(
            db_session, _ASOF, _UNIVERSE, criteria={"not_a_field": 123}
        )

        assert result["ok"] is True

    def test_sector_mapping_is_accepted(self, db_session) -> None:
        _seed_panel(db_session)

        result = universe_filter(
            db_session,
            _ASOF,
            _UNIVERSE,
            sector_mapping=dict.fromkeys(_UNIVERSE, "Tech"),
        )

        assert result["ok"] is True
        assert set(result["data"]) <= set(_UNIVERSE)

    def test_empty_universe_is_empty_result_not_error(self, db_session) -> None:
        result = universe_filter(db_session, _ASOF, [])

        assert result == {"ok": True, "data": []}

    def test_no_priced_assets_is_empty_result_not_error(self, db_session) -> None:
        result = universe_filter(db_session, _ASOF, ["ZZZ", "YYY"])

        assert result == {"ok": True, "data": []}

    def test_insufficient_history_is_empty_result(self, db_session) -> None:
        _seed_panel(db_session, tickers=["AAA"], n_days=1)

        result = universe_filter(db_session, _ASOF, ["AAA"])

        assert result == {"ok": True, "data": []}

    def test_bad_criteria_enum_is_error(self, db_session) -> None:
        _seed_panel(db_session)

        result = universe_filter(
            db_session, _ASOF, _UNIVERSE, criteria={"select_k_measure": "bogus"}
        )

        assert result["ok"] is False
        assert "error" in result

    def test_deterministic(self, db_session) -> None:
        _seed_panel(db_session)

        first = universe_filter(db_session, _ASOF, _UNIVERSE)
        second = universe_filter(db_session, _ASOF, _UNIVERSE)

        assert first == second


class TestDegenerateUniverse:
    def _seed_constant(self, db_session, ticker: str) -> None:
        inst = _seed_instrument(db_session, ticker)
        for i in range(_N_DAYS):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=10.0,
                    volume=1000,
                )
            )
        db_session.flush()

    def test_single_asset_universe_survives(self, db_session) -> None:
        _seed_panel(db_session, tickers=["AAA"])

        result = universe_filter(db_session, _ASOF, ["AAA"])

        assert result == {"ok": True, "data": ["AAA"]}

    def test_pipeline_reduction_to_one_asset_is_a_valid_result(
        self, db_session
    ) -> None:
        # The constant-price asset is dropped as zero variance, leaving one name.
        _seed_panel(db_session, tickers=["AAA"])
        self._seed_constant(db_session, "FLAT")

        result = universe_filter(db_session, _ASOF, ["AAA", "FLAT"])

        assert result == {"ok": True, "data": ["AAA"]}

    def test_every_asset_dropped_is_an_empty_result_not_error(self, db_session) -> None:
        self._seed_constant(db_session, "FLAT")

        result = universe_filter(db_session, _ASOF, ["FLAT"])

        assert result == {"ok": True, "data": []}


class TestCriteriaEnforcement:
    def test_esg_exclusion_drops_instruments_in_the_excluded_sector(
        self, db_session
    ) -> None:
        _seed_panel(db_session, tickers=_THREE)
        _seed_profile(db_session, "AAA", "Energy")
        _seed_profile(db_session, "BBB", "Technology")
        _seed_profile(db_session, "CCC", "Healthcare")

        result = universe_filter(
            db_session, _ASOF, _THREE, criteria={"exclusions": ["energy"]}
        )

        assert result["ok"] is True
        assert "AAA" not in result["data"]
        assert set(result["data"]) == {"BBB", "CCC"}
        assert result["excluded_by_esg"] == {"AAA": "energy"}

    def test_esg_exclusion_maps_yahoo_sector_labels_onto_gics(self, db_session) -> None:
        _seed_panel(db_session)
        _seed_profile(db_session, "AAA", "Consumer Cyclical")
        _seed_profile(db_session, "BBB", "Financial Services")
        _seed_profile(db_session, "CCC", "Healthcare")
        _seed_profile(db_session, "DUP", "Technology")

        result = universe_filter(
            db_session,
            _ASOF,
            _UNIVERSE,
            criteria={"exclusions": ["consumer_discretionary", "financials"]},
        )

        assert result["data"] == ["CCC", "DUP"]
        assert result["excluded_by_esg"] == {
            "AAA": "consumer_discretionary",
            "BBB": "financials",
        }

    def test_esg_exclusion_fails_closed_when_sector_is_unknown(
        self, db_session
    ) -> None:
        # No profile row (AAA) and a profile without a sector (BBB) cannot be shown
        # to sit outside the excluded sector, so neither is allowed through.
        _seed_panel(db_session)
        _seed_profile(db_session, "BBB", None)
        _seed_profile(db_session, "CCC", "Healthcare")
        _seed_profile(db_session, "DUP", "Technology")

        result = universe_filter(
            db_session, _ASOF, _UNIVERSE, criteria={"exclusions": ["energy"]}
        )

        assert result["data"] == ["CCC", "DUP"]
        assert result["excluded_by_esg"] == {
            "AAA": "unclassified",
            "BBB": "unclassified",
        }

    def test_unknown_exclusion_sector_is_error(self, db_session) -> None:
        _seed_panel(db_session, tickers=_THREE)

        result = universe_filter(
            db_session, _ASOF, _THREE, criteria={"exclusions": ["tobacco"]}
        )

        assert result["ok"] is False
        assert "tobacco" in result["error"]

    def test_unapplied_criteria_are_reported_not_silently_dropped(
        self, db_session
    ) -> None:
        _seed_panel(db_session, tickers=_THREE)

        result = universe_filter(
            db_session,
            _ASOF,
            _THREE,
            criteria={"max_position_cap": 0.05, "no_complex": True},
        )

        assert result["ok"] is True
        assert result["ignored_criteria"] == ["max_position_cap", "no_complex"]

    def test_clean_result_carries_no_extra_keys(self, db_session) -> None:
        _seed_panel(db_session, tickers=_THREE)

        result = universe_filter(db_session, _ASOF, _THREE)

        assert set(result) == {"ok", "data"}


class TestShortHistory:
    _LONG_DAYS = 120
    _YOUNG_DAYS = 6

    def _seed_quotes(
        self,
        db_session,
        ticker: str,
        days: range,
        *,
        freq: float = 3.0,
        phase: float = 0.5,
    ) -> None:
        """Quote ``ticker`` only on the given bar offsets from ``_START``."""
        inst = _seed_instrument(db_session, ticker)
        for i in days:
            close = 40.0 * (1.0 + 0.001 * i + 0.025 * math.sin(i / freq + phase))
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=round(close, 6),
                    volume=1000,
                )
            )
        db_session.flush()

    def _seed_with_young_listing(self, db_session) -> dt.date:
        """Three long histories plus NEW, quoted only for the last few bars."""
        _seed_panel(db_session, tickers=_THREE, n_days=self._LONG_DAYS)
        self._seed_quotes(
            db_session,
            "NEW",
            range(self._LONG_DAYS - self._YOUNG_DAYS, self._LONG_DAYS),
        )
        return _START + dt.timedelta(days=self._LONG_DAYS - 1)

    def test_short_history_listing_is_dropped_not_kept_by_a_truncated_sample(
        self, db_session
    ) -> None:
        # Truncating every asset to NEW's first quote hides its leading NaNs from
        # SelectComplete, so the pipeline would keep it and judge the rest on 5 rows.
        asof = self._seed_with_young_listing(db_session)

        result = universe_filter(db_session, asof, [*_THREE, "NEW"])

        assert result["ok"] is True
        assert result["data"] == _THREE

    def test_asset_that_stopped_quoting_is_dropped_not_kept_by_a_stale_price(
        self, db_session
    ) -> None:
        # Forward-filling STOP's last price gives it zero returns for the 60 bars it
        # was not quoted, so SelectComplete would see a complete history.
        _seed_panel(db_session, tickers=_THREE, n_days=self._LONG_DAYS)
        self._seed_quotes(db_session, "STOP", range(60), freq=4.0, phase=1.5)
        asof = _START + dt.timedelta(days=self._LONG_DAYS - 1)

        result = universe_filter(db_session, asof, [*_THREE, "STOP"])

        assert result["ok"] is True
        assert result["data"] == _THREE

    def test_asset_missing_only_the_last_bars_to_a_closure_is_kept(
        self, db_session
    ) -> None:
        # A venue shut for a couple of sessions (holiday) is stale, not delisted.
        _seed_panel(db_session, tickers=_THREE, n_days=self._LONG_DAYS)
        self._seed_quotes(
            db_session, "HOLI", range(self._LONG_DAYS - 2), freq=4.0, phase=1.5
        )
        asof = _START + dt.timedelta(days=self._LONG_DAYS - 1)

        result = universe_filter(db_session, asof, [*_THREE, "HOLI"])

        assert result["ok"] is True
        assert result["data"] == [*_THREE, "HOLI"]
