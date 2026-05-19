"""Scenario-level unit tests for `compute_drift` (issue #782).

Covers reconciliation, full-outer-join, base toggle, all Cycle 1 flags,
trade math edge cases, sorting, and `request_id` echo. Each AC has its own
test function; base toggle and Cycle 1 flag presence are parametrised.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas.portfolio import (
    BaseToggle,
    NormalizedPosition,
    PositionFlag,
    TradeAction,
)
from app.services.portfolio.drift_service import compute_drift
from app.services.portfolio.t212_position_normalizer.normalizer_service import (
    NormalizationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pos(
    ticker: str,
    *,
    qty: float = 10.0,
    eur_value: float = 1000.0,
    eur_weight: float = 0.1,
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


def _norm_result(
    positions: list[NormalizedPosition],
    *,
    reconciliation_ok: bool = True,
    reconciliation_delta_pct: float | None = 0.0,
    unmapped_count: int = 0,
    fx_missing_count: int = 0,
    base_currency: str = "EUR",
) -> NormalizationResult:
    return NormalizationResult(
        positions=positions,
        reconciliation_ok=reconciliation_ok,
        reconciliation_delta_pct=reconciliation_delta_pct,
        base_currency=base_currency,
        unmapped_count=unmapped_count,
        fx_missing_count=fx_missing_count,
    )


def _patch_normalize(monkeypatch, result: NormalizationResult) -> None:
    monkeypatch.setattr(
        "app.services.portfolio.drift_service.normalize_live_positions",
        lambda *a, **kw: result,
    )


def _mock_client(
    *,
    instruments: list[dict] | None = None,
    cash: dict | None = None,
) -> MagicMock:
    client = MagicMock()
    client.get_instruments.return_value = instruments or []
    client.get_account_cash.return_value = cash or {
        "invested": 10_000.0,
        "total": 12_000.0,
    }
    return client


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_when_reconciliation_within_tolerance_then_ok_true(
    db_session, monkeypatch
) -> None:
    _patch_normalize(
        monkeypatch,
        _norm_result(
            [_pos("AAPL", eur_value=10_000.0, eur_weight=1.0)],
            reconciliation_ok=True,
            reconciliation_delta_pct=0.01,
        ),
    )
    resp = compute_drift(
        _mock_client(),
        db_session,
        target_weights={},
        base=BaseToggle.INVESTED,
    )
    assert resp.diagnostics.reconciliation_ok is True
    assert resp.diagnostics.reconciliation_delta_pct is not None
    assert resp.diagnostics.reconciliation_delta_pct < 0.015


def test_when_reconciliation_mismatch_then_ok_false_and_flag_on_row(
    db_session, monkeypatch
) -> None:
    _patch_normalize(
        monkeypatch,
        _norm_result(
            [
                _pos(
                    "AAPL",
                    eur_value=10_000.0,
                    eur_weight=1.0,
                    flags=[PositionFlag.RECONCILIATION_MISMATCH],
                )
            ],
            reconciliation_ok=False,
            reconciliation_delta_pct=0.05,
        ),
    )
    resp = compute_drift(
        _mock_client(),
        db_session,
        target_weights={"AAPL": 1.0},
        base=BaseToggle.INVESTED,
    )
    assert resp.diagnostics.reconciliation_ok is False
    assert resp.diagnostics.reconciliation_delta_pct == pytest.approx(0.05)
    assert PositionFlag.RECONCILIATION_MISMATCH in resp.drift[0].flags


# ---------------------------------------------------------------------------
# Join behavior
# ---------------------------------------------------------------------------


def test_when_holding_absent_from_target_then_target_weight_zero_no_broker_flag(
    db_session, monkeypatch
) -> None:
    _patch_normalize(
        monkeypatch,
        _norm_result([_pos("AAPL", eur_value=4000.0, eur_weight=0.4)]),
    )
    resp = compute_drift(
        _mock_client(),
        db_session,
        target_weights={},
        base=BaseToggle.INVESTED,
    )
    assert len(resp.drift) == 1
    assert resp.drift[0].ticker == "AAPL"
    assert resp.drift[0].target_weight == 0.0
    assert PositionFlag.TARGET_NOT_ON_BROKER not in resp.drift[0].flags


def test_when_target_absent_from_holdings_and_broker_then_flag_set(
    db_session, monkeypatch
) -> None:
    _patch_normalize(monkeypatch, _norm_result([]))
    resp = compute_drift(
        _mock_client(instruments=[]),
        db_session,
        target_weights={"GHOST": 0.5},
        base=BaseToggle.INVESTED,
    )
    assert len(resp.drift) == 1
    assert resp.drift[0].ticker == "GHOST"
    assert resp.drift[0].current_weight == 0.0
    assert resp.drift[0].eur_value == 0.0
    assert resp.drift[0].flags == [PositionFlag.TARGET_NOT_ON_BROKER]
    assert resp.diagnostics.target_not_on_broker_count == 1


def test_when_target_absent_from_holdings_but_in_broker_then_no_flag(
    db_session, monkeypatch
) -> None:
    from tests.unit.services.portfolio.t212_position_normalizer._fixtures import (
        seed_instruments,
    )

    seed_instruments(db_session)
    _patch_normalize(monkeypatch, _norm_result([]))
    resp = compute_drift(
        _mock_client(instruments=[{"ticker": "AAPL_US_EQ"}]),
        db_session,
        target_weights={"AAPL": 0.5},
        base=BaseToggle.INVESTED,
    )
    assert resp.drift[0].ticker == "AAPL"
    assert PositionFlag.TARGET_NOT_ON_BROKER not in resp.drift[0].flags
    assert resp.diagnostics.target_not_on_broker_count == 0


# ---------------------------------------------------------------------------
# Base toggle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base, expected_deployable",
    [
        (BaseToggle.INVESTED, 10_000.0),
        (BaseToggle.TOTAL, 12_000.0),
    ],
)
def test_when_base_toggled_then_deployable_matches_cash_field(
    base, expected_deployable, db_session, monkeypatch
) -> None:
    _patch_normalize(monkeypatch, _norm_result([]))
    resp = compute_drift(
        _mock_client(cash={"invested": 10_000.0, "total": 12_000.0}),
        db_session,
        target_weights={},
        base=base,
    )
    assert resp.totals.deployable_eur == expected_deployable


def test_when_base_toggled_then_trade_delta_eur_scales(
    db_session, monkeypatch
) -> None:
    _patch_normalize(
        monkeypatch,
        _norm_result([_pos("AAPL", qty=4.0, eur_value=4000.0, eur_weight=0.4)]),
    )
    invested_resp = compute_drift(
        _mock_client(cash={"invested": 10_000.0, "total": 12_000.0}),
        db_session,
        target_weights={"AAPL": 0.5},
        base=BaseToggle.INVESTED,
    )
    total_resp = compute_drift(
        _mock_client(cash={"invested": 10_000.0, "total": 12_000.0}),
        db_session,
        target_weights={"AAPL": 0.5},
        base=BaseToggle.TOTAL,
    )
    assert invested_resp.trades[0].delta_eur == pytest.approx(0.1 * 10_000.0)
    assert total_resp.trades[0].delta_eur == pytest.approx(0.1 * 12_000.0)


# ---------------------------------------------------------------------------
# Cycle 1 flag surfacing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag",
    [
        PositionFlag.UNMAPPED,
        PositionFlag.FX_MISSING,
        PositionFlag.STALE_PRICE,
        PositionFlag.RECONCILIATION_MISMATCH,
    ],
)
def test_when_cycle1_flag_on_position_then_surfaces_on_drift_row(
    flag, db_session, monkeypatch
) -> None:
    _patch_normalize(
        monkeypatch,
        _norm_result(
            [_pos("AAPL", eur_value=4000.0, eur_weight=0.4, flags=[flag])]
        ),
    )
    resp = compute_drift(
        _mock_client(),
        db_session,
        target_weights={"AAPL": 0.5},
        base=BaseToggle.INVESTED,
    )
    assert flag in resp.drift[0].flags


# ---------------------------------------------------------------------------
# Trade math edge cases
# ---------------------------------------------------------------------------


def test_when_qty_zero_then_est_shares_zero(db_session, monkeypatch) -> None:
    _patch_normalize(
        monkeypatch,
        _norm_result([_pos("ZERO", qty=0.0, eur_value=0.0, eur_weight=0.0)]),
    )
    resp = compute_drift(
        _mock_client(),
        db_session,
        target_weights={"ZERO": 0.5},
        base=BaseToggle.INVESTED,
    )
    assert resp.trades[0].est_shares == 0.0


def test_when_trades_emitted_then_sorted_by_abs_delta_eur_desc(
    db_session, monkeypatch
) -> None:
    _patch_normalize(
        monkeypatch,
        _norm_result(
            [
                _pos("AAPL", qty=4.0, eur_value=4000.0, eur_weight=0.4),
                _pos("MSFT", qty=10.0, eur_value=3000.0, eur_weight=0.3),
                _pos("GOOG", qty=5.0, eur_value=1000.0, eur_weight=0.1),
            ]
        ),
    )
    resp = compute_drift(
        _mock_client(),
        db_session,
        target_weights={"AAPL": 0.45, "MSFT": 0.05, "GOOG": 0.15},
        base=BaseToggle.INVESTED,
    )
    deltas = [abs(t.delta_eur) for t in resp.trades]
    assert deltas == sorted(deltas, reverse=True)


def test_when_zero_delta_then_hold_excluded_from_trades(
    db_session, monkeypatch
) -> None:
    _patch_normalize(
        monkeypatch,
        _norm_result([_pos("AAPL", qty=4.0, eur_value=5000.0, eur_weight=0.5)]),
    )
    resp = compute_drift(
        _mock_client(),
        db_session,
        target_weights={"AAPL": 0.5},
        base=BaseToggle.INVESTED,
    )
    assert resp.trades == []
    assert all(t.action is not TradeAction.HOLD for t in resp.trades)


# ---------------------------------------------------------------------------
# request_id echo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("request_id", [0, 1, 999, 2**31 - 1])
def test_when_request_id_passed_then_echoed(
    request_id, db_session, monkeypatch
) -> None:
    _patch_normalize(monkeypatch, _norm_result([]))
    resp = compute_drift(
        _mock_client(),
        db_session,
        target_weights={},
        base=BaseToggle.INVESTED,
        request_id=request_id,
    )
    assert resp.request_id == request_id
