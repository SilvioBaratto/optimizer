"""Tests for NormalizedPosition + PositionFlag (issue #772)."""

from enum import Enum

import pytest
from pydantic import ValidationError


class TestPositionFlagEnum:
    def test_when_imported_then_position_flag_is_str_enum(self):
        from app.schemas.portfolio import PositionFlag

        assert issubclass(PositionFlag, str)
        assert issubclass(PositionFlag, Enum)

    def test_when_inspected_then_members_have_lowercase_string_values(self):
        from app.schemas.portfolio import PositionFlag

        assert PositionFlag.UNMAPPED.value == "unmapped"
        assert PositionFlag.FX_MISSING.value == "fx_missing"
        assert PositionFlag.STALE_PRICE.value == "stale_price"
        assert PositionFlag.RECONCILIATION_MISMATCH.value == "reconciliation_mismatch"

    def test_when_listed_then_only_four_members_exist(self):
        from app.schemas.portfolio import PositionFlag

        assert {m.name for m in PositionFlag} == {
            "UNMAPPED",
            "FX_MISSING",
            "STALE_PRICE",
            "RECONCILIATION_MISMATCH",
        }


def _valid_payload(**overrides):
    base = {
        "ticker": "VOD.L",
        "t212_ticker": "VOD_l_EQ",
        "qty": 10.0,
        "price_native": 75.0,
        "currency": "GBP",
        "eur_value": 875.5,
        "eur_weight": 0.05,
        "ppl_eur": 12.3,
    }
    base.update(overrides)
    return base


class TestNormalizedPositionHappyPath:
    def test_when_valid_payload_then_model_constructs(self):
        from app.schemas.portfolio import NormalizedPosition

        pos = NormalizedPosition(**_valid_payload())

        assert pos.ticker == "VOD.L"
        assert pos.t212_ticker == "VOD_l_EQ"
        assert pos.qty == 10.0
        assert pos.price_native == 75.0
        assert pos.currency == "GBP"
        assert pos.eur_value == 875.5
        assert pos.eur_weight == 0.05
        assert pos.ppl_eur == 12.3

    def test_when_flags_omitted_then_defaults_to_empty_list(self):
        from app.schemas.portfolio import NormalizedPosition

        pos = NormalizedPosition(**_valid_payload())

        assert pos.flags == []
        assert isinstance(pos.flags, list)

    def test_when_ppl_eur_omitted_then_defaults_to_none(self):
        from app.schemas.portfolio import NormalizedPosition

        payload = _valid_payload()
        del payload["ppl_eur"]

        pos = NormalizedPosition(**payload)

        assert pos.ppl_eur is None

    def test_when_flag_provided_then_stored_as_position_flag(self):
        from app.schemas.portfolio import NormalizedPosition, PositionFlag

        pos = NormalizedPosition(**_valid_payload(flags=[PositionFlag.UNMAPPED]))

        assert pos.flags == [PositionFlag.UNMAPPED]


class TestCurrencyValidator:
    def test_when_currency_is_gbx_then_validation_error(self):
        from app.schemas.portfolio import NormalizedPosition

        with pytest.raises(ValidationError):
            NormalizedPosition(**_valid_payload(currency="GBX"))

    def test_when_currency_is_gbp_lowercase_pence_then_validation_error(self):
        from app.schemas.portfolio import NormalizedPosition

        with pytest.raises(ValidationError):
            NormalizedPosition(**_valid_payload(currency="GBp"))

    def test_when_currency_is_lowercase_then_validation_error(self):
        from app.schemas.portfolio import NormalizedPosition

        with pytest.raises(ValidationError):
            NormalizedPosition(**_valid_payload(currency="eur"))

    def test_when_currency_length_not_three_then_validation_error(self):
        from app.schemas.portfolio import NormalizedPosition

        with pytest.raises(ValidationError):
            NormalizedPosition(**_valid_payload(currency="EURO"))

    def test_when_currency_is_valid_iso_then_accepted(self):
        from app.schemas.portfolio import NormalizedPosition

        for code in ("EUR", "USD", "GBP", "JPY", "CHF"):
            pos = NormalizedPosition(**_valid_payload(currency=code))
            assert pos.currency == code


class TestEurWeightClamp:
    def test_when_weight_above_one_then_validation_error(self):
        from app.schemas.portfolio import NormalizedPosition

        with pytest.raises(ValidationError):
            NormalizedPosition(**_valid_payload(eur_weight=1.1))

    def test_when_weight_below_zero_then_validation_error(self):
        from app.schemas.portfolio import NormalizedPosition

        with pytest.raises(ValidationError):
            NormalizedPosition(**_valid_payload(eur_weight=-0.01))

    def test_when_weight_is_zero_then_accepted(self):
        from app.schemas.portfolio import NormalizedPosition

        pos = NormalizedPosition(**_valid_payload(eur_weight=0.0))
        assert pos.eur_weight == 0.0

    def test_when_weight_is_one_then_accepted(self):
        from app.schemas.portfolio import NormalizedPosition

        pos = NormalizedPosition(**_valid_payload(eur_weight=1.0))
        assert pos.eur_weight == 1.0


class TestShortPositionForwardCompat:
    def test_when_qty_negative_then_accepted(self):
        from app.schemas.portfolio import NormalizedPosition

        pos = NormalizedPosition(**_valid_payload(qty=-5.0))
        assert pos.qty == -5.0

    def test_when_eur_value_negative_then_accepted(self):
        from app.schemas.portfolio import NormalizedPosition

        pos = NormalizedPosition(**_valid_payload(eur_value=-100.0))
        assert pos.eur_value == -100.0

    def test_when_qty_zero_then_accepted(self):
        from app.schemas.portfolio import NormalizedPosition

        pos = NormalizedPosition(**_valid_payload(qty=0.0))
        assert pos.qty == 0.0


class TestReexportSurface:
    def test_when_imported_from_package_then_both_symbols_exposed(self):
        from app.schemas.portfolio import NormalizedPosition, PositionFlag

        assert NormalizedPosition is not None
        assert PositionFlag is not None

    def test_when_imported_from_module_then_same_objects(self):
        from app.schemas.portfolio import (
            NormalizedPosition as FromPkg,
        )
        from app.schemas.portfolio import (
            PositionFlag as FlagFromPkg,
        )
        from app.schemas.portfolio.normalized_position import (
            NormalizedPosition as FromMod,
        )
        from app.schemas.portfolio.normalized_position import (
            PositionFlag as FlagFromMod,
        )

        assert FromPkg is FromMod
        assert FlagFromPkg is FlagFromMod
