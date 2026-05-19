"""Unit tests for drift_service join + broker-universe flagging (issue #780)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas.portfolio import (
    BaseToggle,
    DriftResponse,
    NormalizedPosition,
    PositionFlag,
)
from app.services.portfolio.drift_service import (
    _build_broker_universe,
    _join_holdings_and_target,
    compute_drift,
)
from app.services.portfolio.t212_position_normalizer.normalizer_service import (
    NormalizationResult,
)
from tests.unit.services.portfolio.t212_position_normalizer._fixtures import (
    seed_instruments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pos(
    ticker: str,
    *,
    eur_value: float = 1000.0,
    eur_weight: float = 0.5,
    flags: list[PositionFlag] | None = None,
) -> NormalizedPosition:
    return NormalizedPosition(
        ticker=ticker,
        t212_ticker=f"{ticker}_US_EQ",
        qty=10.0,
        price_native=100.0,
        currency="USD",
        eur_value=eur_value,
        eur_weight=eur_weight,
        flags=flags or [],
    )


# ---------------------------------------------------------------------------
# _join_holdings_and_target
# ---------------------------------------------------------------------------


class TestJoinHoldingsAndTarget:
    def test_when_ticker_in_both_then_row_has_both_weights(self) -> None:
        positions = [_pos("AAPL", eur_weight=0.4, eur_value=4000.0)]
        target = {"AAPL": 0.5}
        rows = _join_holdings_and_target(positions, target, frozenset({"AAPL"}))

        assert len(rows) == 1
        row = rows[0]
        assert row.ticker == "AAPL"
        assert row.current_weight == pytest.approx(0.4)
        assert row.target_weight == pytest.approx(0.5)
        assert row.delta_weight == pytest.approx(0.1)
        assert row.eur_value == pytest.approx(4000.0)
        assert row.flags == []

    def test_when_held_but_not_in_target_then_target_weight_zero_no_flag(
        self,
    ) -> None:
        positions = [_pos("AAPL", eur_weight=0.4, eur_value=4000.0)]
        rows = _join_holdings_and_target(positions, {}, frozenset())

        assert len(rows) == 1
        assert rows[0].current_weight == pytest.approx(0.4)
        assert rows[0].target_weight == 0.0
        assert rows[0].delta_weight == pytest.approx(-0.4)
        assert PositionFlag.TARGET_NOT_ON_BROKER not in rows[0].flags

    def test_when_target_only_in_broker_universe_then_no_flag(self) -> None:
        rows = _join_holdings_and_target(
            [], {"AAPL": 0.5}, frozenset({"AAPL"})
        )
        assert len(rows) == 1
        assert rows[0].ticker == "AAPL"
        assert rows[0].current_weight == 0.0
        assert rows[0].target_weight == pytest.approx(0.5)
        assert rows[0].delta_weight == pytest.approx(0.5)
        assert rows[0].eur_value == 0.0
        assert PositionFlag.TARGET_NOT_ON_BROKER not in rows[0].flags

    def test_when_target_only_not_in_broker_universe_then_flag_set(self) -> None:
        rows = _join_holdings_and_target(
            [], {"AAPL": 0.5}, frozenset()
        )
        assert len(rows) == 1
        assert rows[0].flags == [PositionFlag.TARGET_NOT_ON_BROKER]

    def test_when_holding_has_flags_then_propagated_verbatim(self) -> None:
        positions = [
            _pos(
                "AAPL",
                eur_weight=0.4,
                eur_value=4000.0,
                flags=[PositionFlag.FX_MISSING, PositionFlag.STALE_PRICE],
            )
        ]
        rows = _join_holdings_and_target(positions, {"AAPL": 0.5}, frozenset())
        assert rows[0].flags == [PositionFlag.FX_MISSING, PositionFlag.STALE_PRICE]

    def test_when_mixed_universe_then_ordering_is_sorted(self) -> None:
        positions = [
            _pos("MSFT", eur_weight=0.3, eur_value=3000.0),
            _pos("AAPL", eur_weight=0.4, eur_value=4000.0),
        ]
        target = {"GOOG": 0.2, "AAPL": 0.5}
        rows = _join_holdings_and_target(
            positions, target, frozenset({"AAPL", "MSFT", "GOOG"})
        )
        assert [r.ticker for r in rows] == ["AAPL", "GOOG", "MSFT"]

    def test_when_empty_inputs_then_empty_output(self) -> None:
        rows = _join_holdings_and_target([], {}, frozenset())
        assert rows == []


# ---------------------------------------------------------------------------
# _build_broker_universe
# ---------------------------------------------------------------------------


class TestBuildBrokerUniverse:
    def test_when_resolved_then_yfinance_set_returned(self, db_session) -> None:
        seed_instruments(db_session)
        client = MagicMock()
        client.get_instruments.return_value = [
            {"ticker": "AAPL_US_EQ"},
            {"ticker": "MSFT_US_EQ"},
            {"ticker": "VOD_l_EQ"},
        ]

        universe = _build_broker_universe(client, db_session)

        assert isinstance(universe, frozenset)
        assert universe == frozenset({"AAPL", "MSFT", "VOD.L"})

    def test_when_t212_ticker_unmapped_then_excluded(self, db_session) -> None:
        seed_instruments(db_session)
        client = MagicMock()
        client.get_instruments.return_value = [
            {"ticker": "AAPL_US_EQ"},
            {"ticker": "UNKNOWN_XX_EQ"},
        ]

        universe = _build_broker_universe(client, db_session)
        assert "AAPL" in universe
        assert len(universe) == 1


# ---------------------------------------------------------------------------
# compute_drift orchestrator
# ---------------------------------------------------------------------------


class TestComputeDrift:
    def test_when_called_then_returns_drift_response(
        self, db_session, monkeypatch
    ) -> None:
        seed_instruments(db_session)
        client = MagicMock()
        client.get_instruments.return_value = [{"ticker": "AAPL_US_EQ"}]

        fake_result = NormalizationResult(
            positions=[_pos("AAPL", eur_weight=0.4, eur_value=4000.0)],
            reconciliation_ok=True,
            reconciliation_delta_pct=0.001,
            base_currency="EUR",
            unmapped_count=0,
            fx_missing_count=0,
        )
        monkeypatch.setattr(
            "app.services.portfolio.drift_service.normalize_live_positions",
            lambda *a, **kw: fake_result,
        )

        resp = compute_drift(
            client,
            db_session,
            target_weights={"AAPL": 0.5},
            base=BaseToggle.INVESTED,
            request_id=7,
        )

        assert isinstance(resp, DriftResponse)
        assert resp.request_id == 7
        assert len(resp.holdings) == 1
        assert resp.holdings[0].ticker == "AAPL"
        assert len(resp.drift) == 1
        assert resp.drift[0].ticker == "AAPL"
        assert resp.drift[0].delta_weight == pytest.approx(0.1)
        assert resp.diagnostics.base_currency == "EUR"

    def test_when_target_not_on_broker_then_flagged_and_counted(
        self, db_session, monkeypatch
    ) -> None:
        seed_instruments(db_session)
        client = MagicMock()
        client.get_instruments.return_value = []  # empty broker universe

        fake_result = NormalizationResult(
            positions=[],
            reconciliation_ok=True,
            reconciliation_delta_pct=0.0,
            base_currency="EUR",
            unmapped_count=0,
            fx_missing_count=0,
        )
        monkeypatch.setattr(
            "app.services.portfolio.drift_service.normalize_live_positions",
            lambda *a, **kw: fake_result,
        )

        resp = compute_drift(
            client,
            db_session,
            target_weights={"GHOST": 0.3},
            base=BaseToggle.INVESTED,
        )

        assert len(resp.drift) == 1
        assert resp.drift[0].ticker == "GHOST"
        assert resp.drift[0].flags == [PositionFlag.TARGET_NOT_ON_BROKER]
        assert resp.diagnostics.target_not_on_broker_count == 1

    def test_when_called_then_get_instruments_called_once(
        self, db_session, monkeypatch
    ) -> None:
        seed_instruments(db_session)
        client = MagicMock()
        client.get_instruments.return_value = [{"ticker": "AAPL_US_EQ"}]

        fake_result = NormalizationResult(
            positions=[],
            reconciliation_ok=True,
            reconciliation_delta_pct=0.0,
            base_currency="EUR",
            unmapped_count=0,
            fx_missing_count=0,
        )
        monkeypatch.setattr(
            "app.services.portfolio.drift_service.normalize_live_positions",
            lambda *a, **kw: fake_result,
        )

        compute_drift(
            client,
            db_session,
            target_weights={"AAPL": 0.5},
            base=BaseToggle.INVESTED,
        )

        assert client.get_instruments.call_count == 1

    def test_when_no_target_and_no_holdings_then_empty_drift(
        self, db_session, monkeypatch
    ) -> None:
        client = MagicMock()
        client.get_instruments.return_value = []

        fake_result = NormalizationResult(
            positions=[],
            reconciliation_ok=True,
            reconciliation_delta_pct=0.0,
            base_currency="EUR",
            unmapped_count=0,
            fx_missing_count=0,
        )
        monkeypatch.setattr(
            "app.services.portfolio.drift_service.normalize_live_positions",
            lambda *a, **kw: fake_result,
        )

        resp = compute_drift(
            client,
            db_session,
            target_weights={},
            base=BaseToggle.INVESTED,
        )

        assert resp.drift == []
        assert resp.diagnostics.target_not_on_broker_count == 0
