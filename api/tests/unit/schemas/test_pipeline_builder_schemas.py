"""Unit tests for pipeline_builder Pydantic schemas (issue #711)."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas.pipeline_builder import (
    BaseCurrencyEnum,
    CostStepRequest,
    CreateSessionResponse,
    EmptyStepRequest,
    LoadStepRequest,
    RebalanceDecisionStepRequest,
    RegimeStepRequest,
    RunLevelConfig,
    ScreenPresetEnum,
    ScreenStepRequest,
    StepPollResponse,
    StepRunResponse,
)


class TestRunLevelConfig:
    def test_when_no_args_then_defaults_are_applied(self) -> None:
        config = RunLevelConfig()
        assert config.rebalance_freq == 63
        assert config.n_selected == 20
        assert config.cost_bps == 10.0
        assert config.tax_rate == 0.26
        assert config.base_currency == BaseCurrencyEnum.EUR
        assert config.robust is False
        assert config.persist is False
        assert config.start_date is None
        assert config.end_date is None
        assert config.seed == 42

    def test_when_n_selected_below_15_then_validation_error_raised(self) -> None:
        with pytest.raises(ValidationError):
            RunLevelConfig(n_selected=14)

    def test_when_n_selected_above_30_then_validation_error_raised(self) -> None:
        with pytest.raises(ValidationError):
            RunLevelConfig(n_selected=31)

    def test_when_n_selected_in_range_then_accepted(self) -> None:
        assert RunLevelConfig(n_selected=15).n_selected == 15
        assert RunLevelConfig(n_selected=30).n_selected == 30

    def test_when_base_currency_invalid_then_validation_error_raised(self) -> None:
        with pytest.raises(ValidationError):
            RunLevelConfig(base_currency="JPY")

    def test_when_base_currency_valid_then_enum_value_set(self) -> None:
        assert RunLevelConfig(base_currency="USD").base_currency == BaseCurrencyEnum.USD
        assert RunLevelConfig(base_currency="GBP").base_currency == BaseCurrencyEnum.GBP

    def test_when_dates_iso_strings_then_coerced_to_date(self) -> None:
        config = RunLevelConfig(start_date="2024-01-15", end_date="2025-06-30")
        assert config.start_date == date(2024, 1, 15)
        assert config.end_date == date(2025, 6, 30)

    def test_when_unknown_field_passed_then_validation_error_raised(self) -> None:
        with pytest.raises(ValidationError):
            RunLevelConfig(unknown_field="x")


class TestBaseCurrencyEnum:
    def test_when_enum_members_then_three_values_present(self) -> None:
        assert BaseCurrencyEnum.EUR.value == "EUR"
        assert BaseCurrencyEnum.GBP.value == "GBP"
        assert BaseCurrencyEnum.USD.value == "USD"


class TestScreenPresetEnum:
    def test_when_enum_members_then_four_presets_present(self) -> None:
        assert ScreenPresetEnum.DEVELOPED_MARKETS.value == "developed_markets"
        assert ScreenPresetEnum.BROAD_UNIVERSE.value == "broad_universe"
        assert ScreenPresetEnum.SMALL_CAP.value == "small_cap"
        assert ScreenPresetEnum.LARGE_CAP.value == "large_cap"


class TestLoadStepRequest:
    def test_when_no_args_then_defaults_applied(self) -> None:
        req = LoadStepRequest()
        assert req.include_delisted is True
        assert req.macro_country == "USA"

    def test_when_unknown_field_then_validation_error_raised(self) -> None:
        with pytest.raises(ValidationError):
            LoadStepRequest(foo="bar")


class TestScreenStepRequest:
    def test_when_no_args_then_default_preset_developed_markets(self) -> None:
        req = ScreenStepRequest()
        assert req.preset == ScreenPresetEnum.DEVELOPED_MARKETS

    def test_when_invalid_preset_then_validation_error_raised(self) -> None:
        with pytest.raises(ValidationError):
            ScreenStepRequest(preset="not_a_preset")

    def test_when_unknown_field_then_validation_error_raised(self) -> None:
        with pytest.raises(ValidationError):
            ScreenStepRequest(extra="x")


class TestRegimeStepRequest:
    def test_when_no_args_then_defaults_applied(self) -> None:
        req = RegimeStepRequest()
        assert req.enable_tilts is True
        assert req.persist_regime is False


class TestPlaceholderStepRequests:
    def test_when_rebalance_decision_empty_body_then_constructs(self) -> None:
        assert RebalanceDecisionStepRequest() is not None

    def test_when_cost_empty_body_then_constructs(self) -> None:
        assert CostStepRequest() is not None

    def test_when_empty_step_request_then_constructs(self) -> None:
        assert EmptyStepRequest() is not None

    def test_when_unknown_field_on_empty_then_validation_error_raised(self) -> None:
        with pytest.raises(ValidationError):
            EmptyStepRequest(x=1)


class TestCreateSessionResponse:
    def test_when_session_id_provided_then_response_constructed(self) -> None:
        resp = CreateSessionResponse(session_id="abc123")
        assert resp.session_id == "abc123"

    def test_when_serialized_then_camel_case_key(self) -> None:
        resp = CreateSessionResponse(session_id="abc123")
        dumped = resp.model_dump(by_alias=True)
        assert dumped == {"sessionId": "abc123"}


class TestStepRunResponse:
    def test_when_async_job_fields_provided_then_response_constructed(self) -> None:
        resp = StepRunResponse(job_id="j1", status="pending", message="queued")
        assert resp.job_id == "j1"
        assert resp.status == "pending"


class TestStepPollResponse:
    def test_when_all_fields_provided_then_constructed(self) -> None:
        resp = StepPollResponse(
            status="running",
            progress={"current": 1, "total": 10},
            result=None,
            error=None,
            gate_reason=None,
        )
        assert resp.status == "running"
        assert resp.progress == {"current": 1, "total": 10}
        assert resp.gate_reason is None

    def test_when_gate_reason_omitted_then_defaults_to_none(self) -> None:
        resp = StepPollResponse(status="failed", progress={}, result=None, error="boom")
        assert resp.gate_reason is None

    def test_when_serialized_then_camel_case_keys(self) -> None:
        resp = StepPollResponse(
            status="completed",
            progress={},
            result={"weights": {}},
            error=None,
            gate_reason="insufficient_factors",
        )
        dumped = resp.model_dump(by_alias=True)
        assert "gateReason" in dumped
        assert dumped["gateReason"] == "insufficient_factors"
