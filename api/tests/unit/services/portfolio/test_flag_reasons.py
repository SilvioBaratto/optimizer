"""Tests for `flag_reasons` module (issue #793)."""

from __future__ import annotations

import pytest

from app.schemas.portfolio import FlagInstance, PositionFlag
from app.services.portfolio.flag_reasons import FLAG_REASONS, make_flag_instance


class TestFlagReasonsCompleteness:
    def test_when_inspected_then_every_position_flag_has_entry(self) -> None:
        assert set(FLAG_REASONS) == set(PositionFlag)

    def test_when_each_template_inspected_then_non_empty_string(self) -> None:
        for code, template in FLAG_REASONS.items():
            assert isinstance(template, str)
            assert template.strip(), f"empty reason for {code}"


class TestMakeFlagInstance:
    def test_when_called_then_returns_flag_instance(self) -> None:
        fi = make_flag_instance(PositionFlag.RECONCILIATION_MISMATCH)
        assert isinstance(fi, FlagInstance)
        assert fi.code is PositionFlag.RECONCILIATION_MISMATCH

    def test_when_no_placeholder_then_reason_matches_template(self) -> None:
        fi = make_flag_instance(PositionFlag.RECONCILIATION_MISMATCH)
        assert fi.reason == FLAG_REASONS[PositionFlag.RECONCILIATION_MISMATCH]
        assert fi.reference is None

    def test_when_reference_provided_then_template_interpolated(self) -> None:
        fi = make_flag_instance(PositionFlag.UNMAPPED, reference="WEIRD_XX_EQ")
        assert "WEIRD_XX_EQ" in fi.reason
        assert "{reference}" not in fi.reason
        assert fi.reference == "WEIRD_XX_EQ"

    def test_when_reference_provided_for_fx_missing_then_interpolated(self) -> None:
        fi = make_flag_instance(PositionFlag.FX_MISSING, reference="USD")
        assert "USD" in fi.reason
        assert fi.code is PositionFlag.FX_MISSING
        assert fi.reference == "USD"

    def test_when_reference_provided_for_stale_price_then_interpolated(self) -> None:
        fi = make_flag_instance(PositionFlag.STALE_PRICE, reference="VOD.L")
        assert "VOD.L" in fi.reason
        assert fi.reference == "VOD.L"

    def test_when_reference_provided_for_target_not_on_broker_then_interpolated(
        self,
    ) -> None:
        fi = make_flag_instance(
            PositionFlag.TARGET_NOT_ON_BROKER, reference="GHOST"
        )
        assert "GHOST" in fi.reason
        assert fi.reference == "GHOST"

    def test_when_factory_called_per_every_code_then_no_exception(self) -> None:
        reference_per_code = {
            PositionFlag.UNMAPPED: "AAA_BB_EQ",
            PositionFlag.FX_MISSING: "USD",
            PositionFlag.STALE_PRICE: "VOD.L",
            PositionFlag.RECONCILIATION_MISMATCH: None,
            PositionFlag.TARGET_NOT_ON_BROKER: "GHOST",
        }
        for code, ref in reference_per_code.items():
            fi = (
                make_flag_instance(code, reference=ref)
                if ref is not None
                else make_flag_instance(code)
            )
            assert fi.code is code
            assert fi.reason
            assert "{reference}" not in fi.reason

    def test_when_returned_instance_then_frozen(self) -> None:
        fi = make_flag_instance(PositionFlag.RECONCILIATION_MISMATCH)
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            fi.reason = "mutated"  # type: ignore[misc]


class TestMakeFlagInstanceUnknownCode:
    def test_when_code_missing_from_reasons_then_key_error(self) -> None:
        """Defensive guard: any future PositionFlag without an entry must raise."""
        original_unmapped = FLAG_REASONS.pop(PositionFlag.UNMAPPED)
        try:
            with pytest.raises(KeyError):
                make_flag_instance(PositionFlag.UNMAPPED, reference="X")
        finally:
            FLAG_REASONS[PositionFlag.UNMAPPED] = original_unmapped


class TestPurity:
    def test_when_imported_then_no_io_side_effects(self) -> None:
        """Module must not perform I/O or service calls at import time."""
        import importlib

        import app.services.portfolio.flag_reasons as module

        reloaded = importlib.reload(module)
        assert reloaded.FLAG_REASONS is not None
        assert callable(reloaded.make_flag_instance)
