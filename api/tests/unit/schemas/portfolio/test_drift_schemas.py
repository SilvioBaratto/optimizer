"""Unit tests for Cycle-2 Drift API Pydantic v2 schemas (issue #779)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.portfolio import (
    BaseToggle,
    DriftDiagnostics,
    DriftResponse,
    DriftRow,
    DriftTotals,
    NormalizedPosition,
    PositionFlag,
    TargetWeight,
    TradeAction,
    TradeRow,
)

# ---------------------------------------------------------------------------
# PositionFlag extension
# ---------------------------------------------------------------------------


class TestPositionFlagExtension:
    def test_target_not_on_broker_value(self):
        assert PositionFlag.TARGET_NOT_ON_BROKER.value == "target_not_on_broker"

    def test_existing_flags_preserved(self):
        assert PositionFlag.UNMAPPED.value == "unmapped"
        assert PositionFlag.FX_MISSING.value == "fx_missing"
        assert PositionFlag.STALE_PRICE.value == "stale_price"
        assert PositionFlag.RECONCILIATION_MISMATCH.value == "reconciliation_mismatch"


# ---------------------------------------------------------------------------
# BaseToggle enum
# ---------------------------------------------------------------------------


class TestBaseToggle:
    def test_members(self):
        assert BaseToggle.INVESTED.value == "invested"
        assert BaseToggle.TOTAL.value == "total"

    def test_is_str_enum(self):
        assert isinstance(BaseToggle.INVESTED, str)


# ---------------------------------------------------------------------------
# TradeAction enum
# ---------------------------------------------------------------------------


class TestTradeAction:
    def test_members(self):
        assert TradeAction.BUY.value == "buy"
        assert TradeAction.SELL.value == "sell"
        assert TradeAction.HOLD.value == "hold"

    def test_is_str_enum(self):
        assert isinstance(TradeAction.BUY, str)


# ---------------------------------------------------------------------------
# TargetWeight
# ---------------------------------------------------------------------------


class TestTargetWeight:
    def test_valid_construction(self):
        tw = TargetWeight(ticker="AAPL", weight=0.5)
        assert tw.ticker == "AAPL"
        assert tw.weight == 0.5

    def test_is_frozen(self):
        tw = TargetWeight(ticker="AAPL", weight=0.5)
        with pytest.raises(ValidationError):
            tw.weight = 0.3  # type: ignore[misc]

    def test_weight_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            TargetWeight(ticker="AAPL", weight=-0.1)

    def test_weight_above_one_rejected(self):
        with pytest.raises(ValidationError):
            TargetWeight(ticker="AAPL", weight=1.1)

    def test_weight_boundaries_accepted(self):
        TargetWeight(ticker="AAPL", weight=0.0)
        TargetWeight(ticker="AAPL", weight=1.0)


# ---------------------------------------------------------------------------
# DriftRow
# ---------------------------------------------------------------------------


class TestDriftRow:
    def _row(self, **overrides):
        defaults = {
            "ticker": "AAPL",
            "current_weight": 0.4,
            "target_weight": 0.5,
            "delta_weight": -0.1,
            "eur_value": 1000.0,
        }
        defaults.update(overrides)
        return DriftRow(**defaults)

    def test_valid_construction(self):
        row = self._row()
        assert row.ticker == "AAPL"
        assert row.current_weight == 0.4
        assert row.target_weight == 0.5
        assert row.delta_weight == pytest.approx(-0.1)
        assert row.eur_value == 1000.0
        assert row.flags == []

    def test_is_frozen(self):
        row = self._row()
        with pytest.raises(ValidationError):
            row.eur_value = 999.0  # type: ignore[misc]

    def test_default_flags_is_empty_list(self):
        row = self._row()
        assert row.flags == []

    def test_flags_accept_position_flag(self):
        row = self._row(flags=[PositionFlag.TARGET_NOT_ON_BROKER, PositionFlag.UNMAPPED])
        assert PositionFlag.TARGET_NOT_ON_BROKER in row.flags
        assert PositionFlag.UNMAPPED in row.flags

    def test_current_weight_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            self._row(current_weight=-0.01)

    def test_current_weight_above_one_rejected(self):
        with pytest.raises(ValidationError):
            self._row(current_weight=1.01)

    def test_target_weight_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            self._row(target_weight=-0.01)

    def test_target_weight_above_one_rejected(self):
        with pytest.raises(ValidationError):
            self._row(target_weight=1.01)

    def test_delta_weight_below_minus_one_rejected(self):
        with pytest.raises(ValidationError):
            self._row(delta_weight=-1.01)

    def test_delta_weight_above_one_rejected(self):
        with pytest.raises(ValidationError):
            self._row(delta_weight=1.01)

    def test_delta_weight_boundaries_accepted(self):
        self._row(delta_weight=-1.0)
        self._row(delta_weight=1.0)


# ---------------------------------------------------------------------------
# TradeRow
# ---------------------------------------------------------------------------


class TestTradeRow:
    def test_valid_construction(self):
        trade = TradeRow(
            ticker="AAPL",
            action=TradeAction.BUY,
            delta_eur=500.0,
            est_shares=2.5,
            est_cost_eur=502.5,
        )
        assert trade.ticker == "AAPL"
        assert trade.action == TradeAction.BUY
        assert trade.delta_eur == 500.0
        assert trade.est_shares == 2.5
        assert trade.est_cost_eur == 502.5

    def test_is_frozen(self):
        trade = TradeRow(
            ticker="AAPL",
            action=TradeAction.BUY,
            delta_eur=500.0,
            est_shares=2.5,
            est_cost_eur=502.5,
        )
        with pytest.raises(ValidationError):
            trade.delta_eur = 0.0  # type: ignore[misc]

    def test_sell_negative_delta_eur(self):
        trade = TradeRow(
            ticker="AAPL",
            action=TradeAction.SELL,
            delta_eur=-300.0,
            est_shares=-1.5,
            est_cost_eur=-301.5,
        )
        assert trade.action == TradeAction.SELL
        assert trade.delta_eur == -300.0


# ---------------------------------------------------------------------------
# DriftTotals
# ---------------------------------------------------------------------------


class TestDriftTotals:
    def test_valid_construction(self):
        totals = DriftTotals(
            deployable_eur=10_000.0,
            total_holdings_eur=9_500.0,
            total_drift_abs=0.15,
            buy_eur=600.0,
            sell_eur=-100.0,
        )
        assert totals.deployable_eur == 10_000.0
        assert totals.total_holdings_eur == 9_500.0
        assert totals.total_drift_abs == 0.15
        assert totals.buy_eur == 600.0
        assert totals.sell_eur == -100.0

    def test_is_frozen(self):
        totals = DriftTotals(
            deployable_eur=10_000.0,
            total_holdings_eur=9_500.0,
            total_drift_abs=0.15,
            buy_eur=600.0,
            sell_eur=-100.0,
        )
        with pytest.raises(ValidationError):
            totals.buy_eur = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DriftDiagnostics
# ---------------------------------------------------------------------------


class TestDriftDiagnostics:
    def test_valid_construction(self):
        diag = DriftDiagnostics(
            reconciliation_ok=True,
            reconciliation_delta_pct=0.001,
            unmapped_count=0,
            fx_missing_count=0,
            target_not_on_broker_count=0,
            base_currency="EUR",
        )
        assert diag.reconciliation_ok is True
        assert diag.reconciliation_delta_pct == 0.001
        assert diag.base_currency == "EUR"

    def test_reconciliation_delta_pct_optional(self):
        diag = DriftDiagnostics(
            reconciliation_ok=False,
            reconciliation_delta_pct=None,
            unmapped_count=2,
            fx_missing_count=1,
            target_not_on_broker_count=3,
            base_currency="EUR",
        )
        assert diag.reconciliation_delta_pct is None

    def test_is_frozen(self):
        diag = DriftDiagnostics(
            reconciliation_ok=True,
            reconciliation_delta_pct=0.0,
            unmapped_count=0,
            fx_missing_count=0,
            target_not_on_broker_count=0,
            base_currency="EUR",
        )
        with pytest.raises(ValidationError):
            diag.unmapped_count = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DriftResponse
# ---------------------------------------------------------------------------


def _make_position(ticker: str = "AAPL") -> NormalizedPosition:
    return NormalizedPosition(
        ticker=ticker,
        t212_ticker=f"{ticker}_US_EQ",
        qty=10.0,
        price_native=150.0,
        currency="USD",
        eur_value=1400.0,
        eur_weight=0.14,
    )


def _make_totals() -> DriftTotals:
    return DriftTotals(
        deployable_eur=10_000.0,
        total_holdings_eur=10_000.0,
        total_drift_abs=0.0,
        buy_eur=0.0,
        sell_eur=0.0,
    )


def _make_diagnostics() -> DriftDiagnostics:
    return DriftDiagnostics(
        reconciliation_ok=True,
        reconciliation_delta_pct=0.0,
        unmapped_count=0,
        fx_missing_count=0,
        target_not_on_broker_count=0,
        base_currency="EUR",
    )


class TestDriftResponse:
    def test_valid_construction(self):
        resp = DriftResponse(
            holdings=[_make_position()],
            target=[TargetWeight(ticker="AAPL", weight=0.14)],
            drift=[
                DriftRow(
                    ticker="AAPL",
                    current_weight=0.14,
                    target_weight=0.14,
                    delta_weight=0.0,
                    eur_value=1400.0,
                )
            ],
            trades=[
                TradeRow(
                    ticker="AAPL",
                    action=TradeAction.HOLD,
                    delta_eur=0.0,
                    est_shares=0.0,
                    est_cost_eur=0.0,
                )
            ],
            totals=_make_totals(),
            diagnostics=_make_diagnostics(),
            request_id=1,
        )
        assert resp.request_id == 1
        assert len(resp.holdings) == 1
        assert len(resp.target) == 1
        assert len(resp.drift) == 1
        assert len(resp.trades) == 1

    def test_is_frozen(self):
        resp = DriftResponse(
            holdings=[],
            target=[],
            drift=[],
            trades=[],
            totals=_make_totals(),
            diagnostics=_make_diagnostics(),
            request_id=1,
        )
        with pytest.raises(ValidationError):
            resp.request_id = 2  # type: ignore[misc]

    def test_empty_lists_accepted(self):
        resp = DriftResponse(
            holdings=[],
            target=[],
            drift=[],
            trades=[],
            totals=_make_totals(),
            diagnostics=_make_diagnostics(),
            request_id=42,
        )
        assert resp.holdings == []
        assert resp.target == []
        assert resp.drift == []
        assert resp.trades == []
