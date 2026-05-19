"""Unit tests for drift_service trade math + totals + diagnostics (issue #781)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas.portfolio import (
    BaseToggle,
    DriftDiagnostics,
    DriftResponse,
    DriftRow,
    DriftTotals,
    NormalizedPosition,
    PositionFlag,
    TradeAction,
)
from app.services.portfolio.drift_service import (
    _aggregate_diagnostics,
    _aggregate_totals,
    _compute_trades,
    _resolve_deployable_eur,
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
    qty: float = 10.0,
    eur_value: float = 1000.0,
    eur_weight: float = 0.5,
    flags: list[PositionFlag] | None = None,
) -> NormalizedPosition:
    return NormalizedPosition(
        ticker=ticker,
        t212_ticker=f"{ticker}_US_EQ",
        qty=qty,
        price_native=100.0,
        currency="USD",
        eur_value=eur_value,
        eur_weight=eur_weight,
        flags=flags or [],
    )


def _row(
    ticker: str,
    *,
    current_weight: float = 0.0,
    target_weight: float = 0.0,
    delta_weight: float = 0.0,
    eur_value: float = 0.0,
    flags: list[PositionFlag] | None = None,
) -> DriftRow:
    return DriftRow(
        ticker=ticker,
        current_weight=current_weight,
        target_weight=target_weight,
        delta_weight=delta_weight,
        eur_value=eur_value,
        flags=flags or [],
    )


# ---------------------------------------------------------------------------
# _resolve_deployable_eur
# ---------------------------------------------------------------------------


class TestResolveDeployableEur:
    def test_when_base_invested_then_invested_returned(self) -> None:
        client = MagicMock()
        client.get_account_cash.return_value = {"invested": 5000.0, "total": 7000.0}
        assert _resolve_deployable_eur(client, BaseToggle.INVESTED) == 5000.0

    def test_when_base_total_then_total_returned(self) -> None:
        client = MagicMock()
        client.get_account_cash.return_value = {"invested": 5000.0, "total": 7000.0}
        assert _resolve_deployable_eur(client, BaseToggle.TOTAL) == 7000.0

    def test_when_invested_missing_then_zero(self) -> None:
        client = MagicMock()
        client.get_account_cash.return_value = {}
        assert _resolve_deployable_eur(client, BaseToggle.INVESTED) == 0.0

    def test_when_total_missing_then_zero(self) -> None:
        client = MagicMock()
        client.get_account_cash.return_value = {}
        assert _resolve_deployable_eur(client, BaseToggle.TOTAL) == 0.0

    def test_when_cash_none_then_zero(self) -> None:
        client = MagicMock()
        client.get_account_cash.return_value = None
        assert _resolve_deployable_eur(client, BaseToggle.INVESTED) == 0.0


# ---------------------------------------------------------------------------
# _compute_trades
# ---------------------------------------------------------------------------


class TestComputeTrades:
    def test_when_delta_positive_then_action_is_buy(self) -> None:
        rows = [_row("AAPL", delta_weight=0.1, eur_value=400.0)]
        positions = {"AAPL": _pos("AAPL", qty=4.0, eur_value=400.0)}
        trades = _compute_trades(rows, 10_000.0, positions, commission_rate=0.0)

        assert len(trades) == 1
        assert trades[0].ticker == "AAPL"
        assert trades[0].action == TradeAction.BUY
        assert trades[0].delta_eur == pytest.approx(1000.0)

    def test_when_delta_negative_then_action_is_sell(self) -> None:
        rows = [_row("AAPL", delta_weight=-0.1, eur_value=400.0)]
        positions = {"AAPL": _pos("AAPL", qty=4.0, eur_value=400.0)}
        trades = _compute_trades(rows, 10_000.0, positions, commission_rate=0.0)

        assert len(trades) == 1
        assert trades[0].action == TradeAction.SELL
        assert trades[0].delta_eur == pytest.approx(-1000.0)

    def test_when_delta_below_threshold_then_hold_excluded(self) -> None:
        # delta_eur = 1e-11 * 10_000 = 1e-7 < 1e-6 threshold -> HOLD
        rows = [_row("AAPL", delta_weight=1e-11, eur_value=400.0)]
        positions = {"AAPL": _pos("AAPL", qty=4.0, eur_value=400.0)}
        trades = _compute_trades(rows, 10_000.0, positions, commission_rate=0.0)
        assert trades == []

    def test_when_zero_delta_then_excluded(self) -> None:
        rows = [_row("AAPL", delta_weight=0.0, eur_value=400.0)]
        positions = {"AAPL": _pos("AAPL", qty=4.0, eur_value=400.0)}
        trades = _compute_trades(rows, 10_000.0, positions, commission_rate=0.0)
        assert trades == []

    def test_when_qty_zero_then_est_shares_zero(self) -> None:
        rows = [_row("GHOST", delta_weight=0.1)]
        trades = _compute_trades(rows, 10_000.0, {}, commission_rate=0.0)

        assert len(trades) == 1
        assert trades[0].est_shares == 0.0

    def test_when_position_qty_positive_then_est_shares_uses_price(self) -> None:
        rows = [_row("AAPL", delta_weight=0.1)]
        # eur_value 400 / qty 4 = 100 EUR/share. delta_eur = 1000. shares = 10.
        positions = {"AAPL": _pos("AAPL", qty=4.0, eur_value=400.0)}
        trades = _compute_trades(rows, 10_000.0, positions, commission_rate=0.0)

        assert trades[0].est_shares == pytest.approx(10.0)

    def test_when_commission_rate_then_cost_inflated(self) -> None:
        rows = [_row("AAPL", delta_weight=0.1)]
        positions = {"AAPL": _pos("AAPL", qty=4.0, eur_value=400.0)}
        trades = _compute_trades(rows, 10_000.0, positions, commission_rate=0.005)

        # abs(1000) * (1 + 0.005) = 1005
        assert trades[0].est_cost_eur == pytest.approx(1005.0)

    def test_when_sell_then_cost_uses_abs_delta(self) -> None:
        rows = [_row("AAPL", delta_weight=-0.1)]
        positions = {"AAPL": _pos("AAPL", qty=4.0, eur_value=400.0)}
        trades = _compute_trades(rows, 10_000.0, positions, commission_rate=0.0)

        assert trades[0].est_cost_eur == pytest.approx(1000.0)

    def test_when_multiple_trades_then_sorted_by_abs_delta_desc(self) -> None:
        rows = [
            _row("AAPL", delta_weight=0.05),
            _row("MSFT", delta_weight=-0.20),
            _row("GOOG", delta_weight=0.10),
        ]
        positions = {
            "AAPL": _pos("AAPL", qty=4.0, eur_value=400.0),
            "MSFT": _pos("MSFT", qty=2.0, eur_value=200.0),
            "GOOG": _pos("GOOG", qty=1.0, eur_value=100.0),
        }
        trades = _compute_trades(rows, 10_000.0, positions, commission_rate=0.0)

        assert [t.ticker for t in trades] == ["MSFT", "GOOG", "AAPL"]


# ---------------------------------------------------------------------------
# _aggregate_totals
# ---------------------------------------------------------------------------


class TestAggregateTotals:
    def test_when_called_then_totals_computed(self) -> None:
        positions = [
            _pos("AAPL", eur_value=4000.0, eur_weight=0.4),
            _pos("MSFT", eur_value=3000.0, eur_weight=0.3),
        ]
        rows = [
            _row("AAPL", delta_weight=0.1, current_weight=0.4, target_weight=0.5),
            _row("MSFT", delta_weight=-0.05, current_weight=0.3, target_weight=0.25),
            _row("GHOST", delta_weight=0.25, current_weight=0.0, target_weight=0.25),
        ]
        totals = _aggregate_totals(positions, rows, deployable_eur=10_000.0)

        assert isinstance(totals, DriftTotals)
        assert totals.deployable_eur == 10_000.0
        assert totals.total_holdings_eur == pytest.approx(7000.0)
        assert totals.total_drift_abs == pytest.approx(0.40)
        assert totals.buy_eur == pytest.approx(0.1 * 10_000 + 0.25 * 10_000)
        assert totals.sell_eur == pytest.approx(0.05 * 10_000)

    def test_when_no_buys_or_sells_then_zero(self) -> None:
        totals = _aggregate_totals([], [], deployable_eur=0.0)
        assert totals.deployable_eur == 0.0
        assert totals.total_holdings_eur == 0.0
        assert totals.total_drift_abs == 0.0
        assert totals.buy_eur == 0.0
        assert totals.sell_eur == 0.0


# ---------------------------------------------------------------------------
# _aggregate_diagnostics
# ---------------------------------------------------------------------------


class TestAggregateDiagnostics:
    def test_when_flags_present_then_count_derived(self) -> None:
        norm = NormalizationResult(
            positions=[],
            reconciliation_ok=False,
            reconciliation_delta_pct=0.02,
            base_currency="EUR",
            unmapped_count=2,
            fx_missing_count=1,
        )
        rows = [
            _row("A", flags=[PositionFlag.TARGET_NOT_ON_BROKER]),
            _row("B", flags=[PositionFlag.TARGET_NOT_ON_BROKER]),
            _row("C", flags=[PositionFlag.FX_MISSING]),
        ]
        diag = _aggregate_diagnostics(norm, rows)

        assert isinstance(diag, DriftDiagnostics)
        assert diag.target_not_on_broker_count == 2
        assert diag.unmapped_count == 2
        assert diag.fx_missing_count == 1
        assert diag.reconciliation_ok is False
        assert diag.reconciliation_delta_pct == 0.02
        assert diag.base_currency == "EUR"


# ---------------------------------------------------------------------------
# compute_drift end-to-end
# ---------------------------------------------------------------------------


class TestComputeDriftEndToEnd:
    def test_when_fully_populated_then_response_carries_all_fields(
        self, db_session, monkeypatch
    ) -> None:
        seed_instruments(db_session)
        client = MagicMock()
        client.get_instruments.return_value = [
            {"ticker": "AAPL_US_EQ"},
            {"ticker": "MSFT_US_EQ"},
        ]
        client.get_account_cash.return_value = {
            "invested": 10_000.0,
            "total": 11_000.0,
        }

        fake_result = NormalizationResult(
            positions=[
                _pos("AAPL", qty=4.0, eur_value=4000.0, eur_weight=0.4),
                _pos("MSFT", qty=10.0, eur_value=3000.0, eur_weight=0.3),
            ],
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
            target_weights={"AAPL": 0.5, "MSFT": 0.25},
            base=BaseToggle.INVESTED,
            request_id=42,
        )

        assert isinstance(resp, DriftResponse)
        assert resp.request_id == 42
        assert resp.totals.deployable_eur == 10_000.0
        assert resp.totals.total_holdings_eur == pytest.approx(7000.0)
        # AAPL +0.10, MSFT -0.05 → buys 1000, sells 500
        assert resp.totals.buy_eur == pytest.approx(1000.0)
        assert resp.totals.sell_eur == pytest.approx(500.0)
        assert resp.totals.total_drift_abs == pytest.approx(0.15)

        # 2 trades (no HOLD), sorted by abs(delta_eur) desc
        assert len(resp.trades) == 2
        assert resp.trades[0].ticker == "AAPL"
        assert resp.trades[0].action == TradeAction.BUY
        assert resp.trades[1].ticker == "MSFT"
        assert resp.trades[1].action == TradeAction.SELL

    def test_when_base_total_then_total_used(
        self, db_session, monkeypatch
    ) -> None:
        client = MagicMock()
        client.get_instruments.return_value = []
        client.get_account_cash.return_value = {
            "invested": 5000.0,
            "total": 8000.0,
        }
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
            base=BaseToggle.TOTAL,
        )

        assert resp.totals.deployable_eur == 8000.0

    def test_when_zero_delta_row_then_no_trade_emitted(
        self, db_session, monkeypatch
    ) -> None:
        client = MagicMock()
        client.get_instruments.return_value = [{"ticker": "AAPL_US_EQ"}]
        client.get_account_cash.return_value = {
            "invested": 10_000.0,
            "total": 10_000.0,
        }
        seed_instruments(db_session)

        fake_result = NormalizationResult(
            positions=[_pos("AAPL", qty=4.0, eur_value=5000.0, eur_weight=0.5)],
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
            target_weights={"AAPL": 0.5},  # zero drift
            base=BaseToggle.INVESTED,
        )

        assert resp.trades == []
        assert resp.drift[0].delta_weight == pytest.approx(0.0)
