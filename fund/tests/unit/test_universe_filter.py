"""T3.3 — ``universe_filter``: pre-selection over a seeded price panel.

Pins the tool as a pure, deterministic function over a seeded SQLite panel:

* it wraps ``optimizer.pre_selection`` and returns the **surviving tickers** as a
  ``list[str]`` in requested order — never the returns frame;
* an empty universe (or one with nothing priced) is a valid empty result
  (``{ok: true, data: []}``), not an error;
* a bad selection criterion (unknown ``select_k_measure`` enum) degrades to
  ``{ok: false, error}`` via ``PreSelectionConfig`` validation, never raised;
* ``top_k`` is a real filter lever: ``top_k=1`` prunes the universe to one asset;
* same seeded data + same criteria ⇒ identical list.
"""

from __future__ import annotations

import datetime as dt
import math

from portopt_db.models.market_data.yfinance_data import PriceHistory
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

        result = universe_filter(
            db_session, _ASOF, _UNIVERSE, criteria={"top_k": 1}
        )

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
