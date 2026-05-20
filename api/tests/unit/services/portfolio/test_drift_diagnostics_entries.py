"""Tests for `_aggregate_diagnostics` entries + reconciliation aggregate (#796)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas.portfolio import (
    BaseToggle,
    DriftDiagnostics,
    DriftRow,
    FlagInstance,
    NormalizedPosition,
    PositionFlag,
)
from app.services.portfolio.drift_service import (
    _aggregate_diagnostics,
    compute_drift,
)
from app.services.portfolio.t212_position_normalizer.normalizer_service import (
    RECONCILIATION_TOLERANCE,
    NormalizationResult,
)


def _pos(
    ticker: str,
    *,
    qty: float = 1.0,
    eur_value: float = 1000.0,
    eur_weight: float = 0.1,
    flags: list[FlagInstance] | None = None,
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
    flags: list[FlagInstance] | None = None,
) -> DriftRow:
    return DriftRow(
        ticker=ticker,
        current_weight=0.1,
        target_weight=0.1,
        delta_weight=0.0,
        eur_value=1000.0,
        flags=flags or [],
    )


def _norm(
    *,
    positions: list[NormalizedPosition] | None = None,
    reconciliation_ok: bool = True,
    reconciliation_delta_pct: float | None = 0.0,
    unmapped_count: int = 0,
    fx_missing_count: int = 0,
    stale_price_count: int = 0,
    base_currency: str = "EUR",
    dropped_raw_tickers: list[str] | None = None,
) -> NormalizationResult:
    return NormalizationResult(
        positions=positions or [],
        reconciliation_ok=reconciliation_ok,
        reconciliation_delta_pct=reconciliation_delta_pct,
        base_currency=base_currency,
        unmapped_count=unmapped_count,
        fx_missing_count=fx_missing_count,
        stale_price_count=stale_price_count,
        dropped_raw_tickers=dropped_raw_tickers or [],
    )


class TestReconciliationAggregateFields:
    def test_when_aggregate_called_then_sum_invested_delta_populated(self) -> None:
        positions = [
            _pos("AAPL", eur_value=4000.0),
            _pos("MSFT", eur_value=3000.0),
        ]
        diag = _aggregate_diagnostics(
            _norm(positions=positions), [], invested_eur=6500.0
        )
        assert diag.sum_eur == pytest.approx(7000.0)
        assert diag.invested_eur == pytest.approx(6500.0)
        assert diag.delta_eur == pytest.approx(500.0)
        assert diag.tolerance_pct == pytest.approx(RECONCILIATION_TOLERANCE)

    def test_when_invested_omitted_then_defaults_to_zero(self) -> None:
        positions = [_pos("AAPL", eur_value=4000.0)]
        diag = _aggregate_diagnostics(_norm(positions=positions), [])
        assert diag.invested_eur == 0.0
        assert diag.delta_eur == pytest.approx(4000.0)

    def test_when_stale_price_count_set_then_propagated(self) -> None:
        diag = _aggregate_diagnostics(
            _norm(stale_price_count=7), [], invested_eur=0.0
        )
        assert diag.stale_price_count == 7


class TestEntriesFromRowFlags:
    def test_when_row_has_flag_then_entry_per_flag(self) -> None:
        rows = [
            _row(
                "AAPL",
                flags=[
                    FlagInstance(code=PositionFlag.STALE_PRICE, reason="r1"),
                    FlagInstance(code=PositionFlag.FX_MISSING, reason="r2"),
                ],
            )
        ]
        diag = _aggregate_diagnostics(_norm(), rows)
        codes = {(e.code, e.ticker) for e in diag.entries}
        assert (PositionFlag.STALE_PRICE, "AAPL") in codes
        assert (PositionFlag.FX_MISSING, "AAPL") in codes

    def test_when_multiple_rows_have_flags_then_all_emitted(self) -> None:
        rows = [
            _row("AAPL", flags=[FlagInstance(code=PositionFlag.UNMAPPED)]),
            _row(
                "MSFT",
                flags=[FlagInstance(code=PositionFlag.TARGET_NOT_ON_BROKER)],
            ),
        ]
        diag = _aggregate_diagnostics(_norm(), rows)
        tickers = {e.ticker for e in diag.entries}
        assert tickers == {"AAPL", "MSFT"}

    def test_when_no_row_flags_and_recon_ok_then_no_entries(self) -> None:
        rows = [_row("AAPL"), _row("MSFT")]
        diag = _aggregate_diagnostics(_norm(), rows)
        assert diag.entries == []

    def test_when_entry_built_then_carries_flag_reason_and_reference(self) -> None:
        rows = [
            _row(
                "AAPL",
                flags=[
                    FlagInstance(
                        code=PositionFlag.STALE_PRICE,
                        reason="custom-reason",
                        reference="AAPL",
                    )
                ],
            )
        ]
        diag = _aggregate_diagnostics(_norm(), rows)
        entry = diag.entries[0]
        assert entry.reason == "custom-reason"
        assert entry.reference == "AAPL"
        assert entry.ticker == "AAPL"


class TestEntriesFromDroppedTickers:
    def test_when_dropped_raw_tickers_present_then_unmapped_entries_emitted(
        self,
    ) -> None:
        diag = _aggregate_diagnostics(
            _norm(dropped_raw_tickers=["GHOST_XX_EQ", "BOO_YY_EQ"]),
            [],
        )
        unmapped = [e for e in diag.entries if e.code is PositionFlag.UNMAPPED]
        assert {e.ticker for e in unmapped} == {"GHOST_XX_EQ", "BOO_YY_EQ"}
        assert all(e.reference == e.ticker for e in unmapped)

    def test_when_no_dropped_tickers_then_no_unmapped_entries_from_drops(
        self,
    ) -> None:
        diag = _aggregate_diagnostics(_norm(), [])
        assert all(e.code is not PositionFlag.UNMAPPED for e in diag.entries)


class TestReconciliationEntry:
    def test_when_reconciliation_ok_false_then_single_reconciliation_entry(
        self,
    ) -> None:
        positions = [_pos("AAPL", eur_value=10_000.0)]
        norm = _norm(
            positions=positions,
            reconciliation_ok=False,
            reconciliation_delta_pct=0.05,
        )
        diag = _aggregate_diagnostics(norm, [], invested_eur=9_500.0)
        recon_entries = [
            e for e in diag.entries if e.code is PositionFlag.RECONCILIATION_MISMATCH
        ]
        assert len(recon_entries) == 1
        entry = recon_entries[0]
        assert entry.ticker is None
        assert "sum_eur" in (entry.reference or "")
        assert "invested" in (entry.reference or "")
        assert "delta" in (entry.reference or "")

    def test_when_reconciliation_ok_true_then_no_reconciliation_entry(self) -> None:
        diag = _aggregate_diagnostics(_norm(reconciliation_ok=True), [])
        assert all(
            e.code is not PositionFlag.RECONCILIATION_MISMATCH for e in diag.entries
        )


class TestExistingFieldsPreserved:
    def test_when_existing_counts_set_then_propagated_unchanged(self) -> None:
        norm = _norm(unmapped_count=3, fx_missing_count=2)
        rows = [
            _row("GHOST", flags=[FlagInstance(code=PositionFlag.TARGET_NOT_ON_BROKER)]),
        ]
        diag = _aggregate_diagnostics(norm, rows)
        assert diag.unmapped_count == 3
        assert diag.fx_missing_count == 2
        assert diag.target_not_on_broker_count == 1


class TestNoSilentDropsInvariant:
    def test_when_input_tickers_then_holdings_plus_unmapped_entries_match(
        self,
    ) -> None:
        """len(input) == len(holdings) + len(UNMAPPED entries from drops)."""
        positions = [_pos("AAPL"), _pos("MSFT")]
        dropped = ["GHOST_XX_EQ"]  # 1 dropped
        norm = _norm(positions=positions, dropped_raw_tickers=dropped)
        diag = _aggregate_diagnostics(norm, [])
        unmapped_drop_entries = [
            e
            for e in diag.entries
            if e.code is PositionFlag.UNMAPPED and e.ticker in dropped
        ]
        # 2 holdings + 1 dropped == 3 inputs accounted for
        assert len(positions) + len(unmapped_drop_entries) == 3


class TestComputeDriftEndToEndAggregate:
    def _client(
        self,
        *,
        invested: float = 10_000.0,
        total: float = 12_000.0,
    ) -> MagicMock:
        c = MagicMock()
        c.get_instruments.return_value = []
        c.get_account_cash.return_value = {"invested": invested, "total": total}
        return c

    def test_when_compute_drift_then_invested_eur_from_get_account_cash(
        self, db_session, monkeypatch
    ) -> None:
        norm = _norm(
            positions=[_pos("AAPL", eur_value=10_000.0, eur_weight=1.0)],
            reconciliation_ok=True,
            reconciliation_delta_pct=0.0,
        )
        monkeypatch.setattr(
            "app.services.portfolio.drift_service.normalize_live_positions",
            lambda *a, **kw: norm,
        )
        resp = compute_drift(
            self._client(invested=10_000.0),
            db_session,
            target_weights={},
            base=BaseToggle.INVESTED,
        )
        assert isinstance(resp.diagnostics, DriftDiagnostics)
        assert resp.diagnostics.invested_eur == 10_000.0
        assert resp.diagnostics.sum_eur == 10_000.0
        assert resp.diagnostics.delta_eur == 0.0
        assert resp.diagnostics.tolerance_pct == pytest.approx(
            RECONCILIATION_TOLERANCE
        )

    def test_when_compute_drift_then_dropped_raw_tickers_surfaced_as_entries(
        self, db_session, monkeypatch
    ) -> None:
        norm = _norm(
            positions=[_pos("AAPL", eur_value=5000.0, eur_weight=0.5)],
            dropped_raw_tickers=["MAL_FORMED"],
        )
        monkeypatch.setattr(
            "app.services.portfolio.drift_service.normalize_live_positions",
            lambda *a, **kw: norm,
        )
        resp = compute_drift(
            self._client(),
            db_session,
            target_weights={},
            base=BaseToggle.INVESTED,
        )
        dropped_entries = [
            e
            for e in resp.diagnostics.entries
            if e.code is PositionFlag.UNMAPPED and e.ticker == "MAL_FORMED"
        ]
        assert len(dropped_entries) == 1


