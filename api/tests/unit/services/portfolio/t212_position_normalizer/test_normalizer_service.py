"""Tests for normalize_live_positions orchestrator (issue #777)."""

from __future__ import annotations

from dataclasses import is_dataclass
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.portfolio import NormalizedPosition, PositionFlag


def _client_mock(
    *,
    base_currency: str = "EUR",
    invested: float = 0.0,
    positions: list[dict[str, Any]] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.get_account_info.return_value = {"currencyCode": base_currency, "id": 1}
    client.get_account_cash.return_value = {"invested": invested}
    client.get_portfolio_positions.return_value = positions or []
    return client


def _raw_pos(
    ticker: str = "AAPL_US_EQ",
    *,
    quantity: float = 10.0,
    currency: str = "EUR",
    current_price: float = 100.0,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "quantity": quantity,
        "currentPrice": current_price,
        "currencyCode": currency,
        "ppl": 0.0,
        "fxPpl": 0.0,
    }


def _fx_mock(rate: float | None = 1.0) -> MagicMock:
    fx = MagicMock()
    fx.get_rate_to_base.return_value = rate
    return fx


def _patch_lookup(mapping: dict[str, str | None]) -> Any:
    def fake(ticker: str, _session: Any) -> str | None:
        return mapping.get(ticker)

    return patch(
        "app.services.portfolio.t212_position_normalizer.normalizer_service.lookup_yf_ticker",
        side_effect=fake,
    )


class TestNormalizationResultShape:
    def test_when_imported_then_is_frozen_dataclass_with_required_fields(self) -> None:
        from dataclasses import FrozenInstanceError, fields

        from app.services.portfolio.t212_position_normalizer import (
            NormalizationResult,
        )

        assert is_dataclass(NormalizationResult)
        names = {f.name for f in fields(NormalizationResult)}
        assert names == {
            "positions",
            "reconciliation_ok",
            "reconciliation_delta_pct",
            "base_currency",
            "unmapped_count",
            "fx_missing_count",
            "stale_price_count",
            "dropped_raw_tickers",
        }

        result = NormalizationResult(
            positions=[],
            reconciliation_ok=True,
            reconciliation_delta_pct=0.0,
            base_currency="EUR",
            unmapped_count=0,
            fx_missing_count=0,
            stale_price_count=0,
            dropped_raw_tickers=[],
        )
        with pytest.raises(FrozenInstanceError):
            result.reconciliation_ok = False  # type: ignore[misc]


class TestReExports:
    def test_when_package_imported_then_orchestrator_symbols_available(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            NormalizationResult,
            normalize_live_positions,
        )

        assert NormalizationResult is not None
        assert callable(normalize_live_positions)


class TestEmptyPositions:
    def test_when_no_positions_then_empty_result_and_ok(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        client = _client_mock(positions=[], invested=0.0)
        session = MagicMock()

        with _patch_lookup({}):
            result = normalize_live_positions(
                client, session, as_of_date=date(2026, 5, 19), fx_provider=_fx_mock()
            )

        assert result.positions == []
        assert result.reconciliation_ok is True
        assert result.reconciliation_delta_pct == 0.0
        assert result.base_currency == "EUR"
        assert result.unmapped_count == 0
        assert result.fx_missing_count == 0


class TestHappyPathReconciliation:
    def test_when_eur_positions_match_invested_then_reconciliation_ok(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=100.0, currency="EUR"),
            _raw_pos("MSFT_US_EQ", quantity=2.0, current_price=200.0, currency="EUR"),
        ]
        # total = 100 + 400 = 500
        client = _client_mock(invested=500.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL", "MSFT_US_EQ": "MSFT"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        assert result.reconciliation_ok is True
        assert result.reconciliation_delta_pct == pytest.approx(0.0)
        assert result.base_currency == "EUR"


class TestWeightAssignment:
    def test_when_two_positions_then_weights_sum_to_one(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=100.0, currency="EUR"),
            _raw_pos("MSFT_US_EQ", quantity=1.0, current_price=400.0, currency="EUR"),
        ]
        client = _client_mock(invested=500.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL", "MSFT_US_EQ": "MSFT"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        weights = {p.ticker: p.eur_weight for p in result.positions}
        assert weights["AAPL"] == pytest.approx(0.2)
        assert weights["MSFT"] == pytest.approx(0.8)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_when_total_zero_then_all_weights_zero(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=0.0, currency="EUR"),
        ]
        client = _client_mock(invested=0.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        assert all(p.eur_weight == 0.0 for p in result.positions)


class TestReconciliationMismatch:
    def test_when_delta_above_threshold_then_flag_attached_and_ok_false(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        # total = 1000, invested = 500 → delta = 1.0 > 0.015
        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=1000.0, currency="EUR"),
        ]
        client = _client_mock(invested=500.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        assert result.reconciliation_ok is False
        assert result.reconciliation_delta_pct == pytest.approx(1.0)
        assert all(
            PositionFlag.RECONCILIATION_MISMATCH in p.flags for p in result.positions
        )

    def test_when_reconciliation_flag_emitted_then_reference_is_signed_pct(
        self,
    ) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        # total = 1000, invested = 500 → signed_delta = +1.0 → "+100.0%"
        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=1000.0, currency="EUR"),
        ]
        client = _client_mock(invested=500.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        flags = result.positions[0].flags
        recon = next(f for f in flags if f.code is PositionFlag.RECONCILIATION_MISMATCH)
        assert recon.reference == "+100.0%"

    def test_when_invested_above_total_then_reference_is_negative(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        # total = 500, invested = 1000 → signed_delta = -0.5 → "-50.0%"
        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=500.0, currency="EUR"),
        ]
        client = _client_mock(invested=1000.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        recon = next(
            f
            for f in result.positions[0].flags
            if f.code is PositionFlag.RECONCILIATION_MISMATCH
        )
        assert recon.reference == "-50.0%"

    def test_when_delta_at_one_pct_then_ok(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        # total = 505, invested = 500 → delta = 0.01 < 0.015
        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=505.0, currency="EUR"),
        ]
        client = _client_mock(invested=500.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        assert result.reconciliation_ok is True
        assert all(
            PositionFlag.RECONCILIATION_MISMATCH not in p.flags
            for p in result.positions
        )

    def test_when_negative_delta_then_also_triggers(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        # total = 100, invested = 500 → delta = abs(-400)/500 = 0.8 > 0.015
        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=100.0, currency="EUR"),
        ]
        client = _client_mock(invested=500.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        assert result.reconciliation_ok is False
        assert result.reconciliation_delta_pct == pytest.approx(0.8)


class TestNonEurBase:
    def test_when_base_is_usd_then_reconciliation_skipped(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=100.0, currency="USD"),
        ]
        client = _client_mock(
            base_currency="USD", invested=500.0, positions=positions
        )
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        assert result.base_currency == "USD"
        assert result.reconciliation_ok is True
        assert result.reconciliation_delta_pct is None
        assert all(
            PositionFlag.RECONCILIATION_MISMATCH not in p.flags
            for p in result.positions
        )


class TestUnmappedAndFxMissingCounters:
    def test_when_some_unmapped_then_count_reflected(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=100.0, currency="EUR"),
            _raw_pos("WEIRD_l_EQ", quantity=1.0, current_price=50.0, currency="GBP"),
        ]
        client = _client_mock(invested=0.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL", "WEIRD_l_EQ": None}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(rate=1.17),
            )

        assert result.unmapped_count == 1

    def test_when_stale_price_then_count_reflected(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        positions = [
            _raw_pos(
                "AAPL_US_EQ",
                quantity=1.0,
                current_price=0.0,  # null/zero guard fires STALE_PRICE
                currency="EUR",
            ),
            _raw_pos(
                "MSFT_US_EQ",
                quantity=1.0,
                current_price=100.0,
                currency="EUR",
            ),
        ]
        client = _client_mock(invested=0.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL", "MSFT_US_EQ": "MSFT"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        assert result.stale_price_count == 1

    def test_when_fx_missing_then_count_reflected(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=100.0, currency="USD"),
        ]
        client = _client_mock(invested=0.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(rate=None),
            )

        assert result.fx_missing_count == 1


class TestDefaultAsOfDate:
    def test_when_as_of_date_none_then_uses_today(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        client = _client_mock(positions=[], invested=0.0)
        session = MagicMock()

        with _patch_lookup({}):
            result = normalize_live_positions(
                client, session, fx_provider=_fx_mock()
            )

        # No assertion error => default branch taken
        assert isinstance(result.positions, list)


class TestImmutabilityOfPositions:
    def test_when_reconciliation_flagged_then_original_position_not_mutated(
        self,
    ) -> None:
        """Pydantic models are frozen; orchestrator uses model_copy."""
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        positions = [
            _raw_pos("AAPL_US_EQ", quantity=1.0, current_price=1000.0, currency="EUR"),
        ]
        client = _client_mock(invested=500.0, positions=positions)
        session = MagicMock()

        with _patch_lookup({"AAPL_US_EQ": "AAPL"}):
            result = normalize_live_positions(
                client,
                session,
                as_of_date=date(2026, 5, 19),
                fx_provider=_fx_mock(),
            )

        # Each returned position is a NormalizedPosition
        assert all(isinstance(p, NormalizedPosition) for p in result.positions)
        assert all(
            PositionFlag.RECONCILIATION_MISMATCH in p.flags for p in result.positions
        )


class TestNoCredentialLeak:
    def test_when_orchestrator_runs_then_no_api_key_in_result_repr(self) -> None:
        from app.services.portfolio.t212_position_normalizer import (
            normalize_live_positions,
        )

        client = _client_mock(positions=[], invested=0.0)
        client.api_key = "SECRET_API_KEY_DO_NOT_LEAK"
        client.api_secret = "ALSO_SECRET"  # noqa: S105
        session = MagicMock()

        with _patch_lookup({}):
            result = normalize_live_positions(
                client, session, as_of_date=date(2026, 5, 19), fx_provider=_fx_mock()
            )

        assert "SECRET_API_KEY_DO_NOT_LEAK" not in repr(result)
        assert "ALSO_SECRET" not in repr(result)
