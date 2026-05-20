"""Round-trip tests for FlagInstance + DiagnosticEntry + extended DriftDiagnostics (issue #792)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.portfolio import (
    DiagnosticEntry,
    DriftDiagnostics,
    DriftResponse,
    DriftRow,
    DriftTotals,
    FlagInstance,
    NormalizedPosition,
    PositionFlag,
    TargetWeight,
    TradeAction,
    TradeRow,
)


def _every_code() -> tuple[PositionFlag, ...]:
    return tuple(PositionFlag)


# ---------------------------------------------------------------------------
# FlagInstance
# ---------------------------------------------------------------------------


class TestFlagInstanceShape:
    def test_when_minimal_then_constructs_with_defaults(self) -> None:
        fi = FlagInstance(code=PositionFlag.UNMAPPED)
        assert fi.code is PositionFlag.UNMAPPED
        assert fi.reason == ""
        assert fi.reference is None

    def test_when_full_then_carries_reason_and_reference(self) -> None:
        fi = FlagInstance(
            code=PositionFlag.FX_MISSING,
            reason="No FX rate for USD on 2026-05-20",
            reference="docs/fx.md",
        )
        assert fi.reason == "No FX rate for USD on 2026-05-20"
        assert fi.reference == "docs/fx.md"

    def test_when_constructed_then_is_frozen(self) -> None:
        fi = FlagInstance(code=PositionFlag.STALE_PRICE)
        with pytest.raises(ValidationError):
            fi.reason = "mutated"  # type: ignore[misc]


class TestFlagInstanceEquality:
    def test_when_compared_to_same_code_position_flag_then_equal(self) -> None:
        fi = FlagInstance(code=PositionFlag.UNMAPPED)
        assert fi == PositionFlag.UNMAPPED
        # Reflected: PositionFlag on the left, FlagInstance on the right.
        assert PositionFlag.UNMAPPED == fi  # noqa: SIM300

    def test_when_in_list_membership_with_position_flag_then_found(self) -> None:
        flags = [
            FlagInstance(code=PositionFlag.UNMAPPED),
            FlagInstance(code=PositionFlag.FX_MISSING),
        ]
        assert PositionFlag.UNMAPPED in flags
        assert PositionFlag.STALE_PRICE not in flags

    def test_when_list_equality_against_position_flag_list_then_equal(self) -> None:
        flags = [FlagInstance(code=PositionFlag.STALE_PRICE)]
        assert flags == [PositionFlag.STALE_PRICE]


class TestFlagListCoercion:
    def test_when_position_flag_passed_then_wrapped_in_flag_instance(self) -> None:
        pos = NormalizedPosition(
            ticker="AAPL",
            t212_ticker="AAPL_US_EQ",
            qty=1.0,
            price_native=100.0,
            currency="USD",
            eur_value=92.0,
            eur_weight=0.1,
            flags=[PositionFlag.UNMAPPED, PositionFlag.FX_MISSING],  # type: ignore[arg-type]
        )
        assert all(isinstance(f, FlagInstance) for f in pos.flags)
        assert [f.code for f in pos.flags] == [
            PositionFlag.UNMAPPED,
            PositionFlag.FX_MISSING,
        ]


# ---------------------------------------------------------------------------
# DiagnosticEntry
# ---------------------------------------------------------------------------


class TestDiagnosticEntryShape:
    def test_when_minimal_then_defaults(self) -> None:
        entry = DiagnosticEntry(code=PositionFlag.UNMAPPED)
        assert entry.code is PositionFlag.UNMAPPED
        assert entry.reason == ""
        assert entry.reference is None
        assert entry.ticker is None

    def test_when_ticker_set_then_carried(self) -> None:
        entry = DiagnosticEntry(
            code=PositionFlag.STALE_PRICE,
            reason="Price older than threshold",
            ticker="VOD.L",
        )
        assert entry.ticker == "VOD.L"

    def test_when_constructed_then_is_frozen(self) -> None:
        entry = DiagnosticEntry(code=PositionFlag.UNMAPPED)
        with pytest.raises(ValidationError):
            entry.ticker = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DriftDiagnostics extension
# ---------------------------------------------------------------------------


class TestDriftDiagnosticsExtension:
    def _minimal_diagnostics(self, **overrides) -> DriftDiagnostics:
        defaults = {
            "reconciliation_ok": True,
            "reconciliation_delta_pct": 0.0,
            "unmapped_count": 0,
            "fx_missing_count": 0,
            "target_not_on_broker_count": 0,
            "base_currency": "EUR",
        }
        defaults.update(overrides)
        return DriftDiagnostics(**defaults)

    def test_when_legacy_fields_only_then_new_fields_default(self) -> None:
        diag = self._minimal_diagnostics()
        assert diag.sum_eur == 0.0
        assert diag.invested_eur == 0.0
        assert diag.delta_eur == 0.0
        assert diag.tolerance_pct == 0.0
        assert diag.stale_price_count == 0
        assert diag.entries == []

    def test_when_full_aggregate_set_then_carries_values(self) -> None:
        diag = self._minimal_diagnostics(
            sum_eur=9_950.0,
            invested_eur=10_000.0,
            delta_eur=-50.0,
            tolerance_pct=0.015,
            stale_price_count=2,
            entries=[
                DiagnosticEntry(code=PositionFlag.STALE_PRICE, ticker="AAPL"),
            ],
        )
        assert diag.sum_eur == 9_950.0
        assert diag.invested_eur == 10_000.0
        assert diag.delta_eur == -50.0
        assert diag.tolerance_pct == 0.015
        assert diag.stale_price_count == 2
        assert len(diag.entries) == 1
        assert diag.entries[0].ticker == "AAPL"

    def test_when_existing_fields_then_preserved(self) -> None:
        diag = self._minimal_diagnostics(
            reconciliation_ok=False,
            reconciliation_delta_pct=0.05,
            unmapped_count=1,
            fx_missing_count=2,
            target_not_on_broker_count=3,
            base_currency="EUR",
        )
        assert diag.reconciliation_ok is False
        assert diag.reconciliation_delta_pct == 0.05
        assert diag.unmapped_count == 1
        assert diag.fx_missing_count == 2
        assert diag.target_not_on_broker_count == 3


# ---------------------------------------------------------------------------
# Full DriftResponse round-trip
# ---------------------------------------------------------------------------


def _entry(code: PositionFlag) -> DiagnosticEntry:
    return DiagnosticEntry(
        code=code,
        reason=f"reason-for-{code.value}",
        reference=None,
        ticker="AAPL",
    )


def _build_response_with_every_flag() -> DriftResponse:
    return DriftResponse(
        holdings=[
            NormalizedPosition(
                ticker="AAPL",
                t212_ticker="AAPL_US_EQ",
                qty=10.0,
                price_native=100.0,
                currency="USD",
                eur_value=920.0,
                eur_weight=0.1,
                flags=[FlagInstance(code=code) for code in _every_code()],
            ),
        ],
        target=[TargetWeight(ticker="AAPL", weight=0.1)],
        drift=[
            DriftRow(
                ticker="AAPL",
                current_weight=0.1,
                target_weight=0.1,
                delta_weight=0.0,
                eur_value=920.0,
                flags=[FlagInstance(code=code) for code in _every_code()],
            ),
        ],
        trades=[
            TradeRow(
                ticker="AAPL",
                action=TradeAction.HOLD,
                delta_eur=0.0,
                est_shares=0.0,
                est_cost_eur=0.0,
            ),
        ],
        totals=DriftTotals(
            deployable_eur=9200.0,
            total_holdings_eur=920.0,
            total_drift_abs=0.0,
            buy_eur=0.0,
            sell_eur=0.0,
        ),
        diagnostics=DriftDiagnostics(
            reconciliation_ok=False,
            reconciliation_delta_pct=0.02,
            unmapped_count=1,
            fx_missing_count=1,
            target_not_on_broker_count=1,
            base_currency="EUR",
            sum_eur=9180.0,
            invested_eur=9000.0,
            delta_eur=180.0,
            tolerance_pct=0.015,
            stale_price_count=1,
            entries=[_entry(code) for code in _every_code()],
        ),
        request_id=42,
    )


class TestDriftResponseFullRoundTrip:
    def test_when_response_dumped_and_reloaded_then_equal(self) -> None:
        original = _build_response_with_every_flag()
        payload = original.model_dump_json()
        rebuilt = DriftResponse.model_validate_json(payload)

        assert rebuilt == original

    def test_when_response_has_every_flag_code_then_entries_carry_each(self) -> None:
        original = _build_response_with_every_flag()
        codes = {entry.code for entry in original.diagnostics.entries}
        assert codes == set(_every_code())
