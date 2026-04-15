"""Tests for schema extraction issue #352.

Verifies that all inline schemas have been moved to dedicated schema modules
with correct base classes and preserved validators.
"""

from __future__ import annotations

import inspect
import pytest
from pydantic import BaseModel


# ===========================================================================
# views.py schema file
# ===========================================================================


class TestViewsSchemasModuleExists:
    """api/app/schemas/views.py must exist and contain the expected schemas."""

    def test_generate_views_request_importable(self):
        from app.schemas.views import GenerateViewsRequest  # noqa: F401

    def test_asset_view_response_importable(self):
        from app.schemas.views import AssetViewResponse  # noqa: F401

    def test_generate_views_response_importable(self):
        from app.schemas.views import GenerateViewsResponse  # noqa: F401

    def test_ic_history_importable(self):
        from app.schemas.views import ICHistory  # noqa: F401

    def test_expert_view_summary_importable(self):
        from app.schemas.views import ExpertViewSummary  # noqa: F401

    def test_opinion_pool_request_importable(self):
        from app.schemas.views import OpinionPoolRequest  # noqa: F401

    def test_opinion_pool_response_importable(self):
        from app.schemas.views import OpinionPoolResponse  # noqa: F401


class TestViewsResponseSchemasExtendCamelCaseModel:
    """Response schemas must extend CamelCaseModel for camelCase JSON."""

    def test_asset_view_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.views import AssetViewResponse

        assert issubclass(AssetViewResponse, CamelCaseModel)

    def test_generate_views_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.views import GenerateViewsResponse

        assert issubclass(GenerateViewsResponse, CamelCaseModel)

    def test_expert_view_summary_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.views import ExpertViewSummary

        assert issubclass(ExpertViewSummary, CamelCaseModel)

    def test_opinion_pool_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.views import OpinionPoolResponse

        assert issubclass(OpinionPoolResponse, CamelCaseModel)


class TestViewsRequestSchemasKeepBaseModel:
    """Request schemas must use BaseModel, not CamelCaseModel."""

    def test_generate_views_request_uses_base_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.views import GenerateViewsRequest

        assert issubclass(GenerateViewsRequest, BaseModel)
        assert not issubclass(GenerateViewsRequest, CamelCaseModel)

    def test_ic_history_uses_base_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.views import ICHistory

        assert issubclass(ICHistory, BaseModel)
        assert not issubclass(ICHistory, CamelCaseModel)

    def test_opinion_pool_request_uses_base_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.views import OpinionPoolRequest

        assert issubclass(OpinionPoolRequest, BaseModel)
        assert not issubclass(OpinionPoolRequest, CamelCaseModel)


class TestViewsSchemasValidatorsPreserved:
    """Field validators must remain intact after migration."""

    def test_generate_views_request_normalises_tickers(self):
        from app.schemas.views import GenerateViewsRequest

        req = GenerateViewsRequest(tickers=["  aapl  ", "msft"])
        assert req.tickers == ["AAPL", "MSFT"]

    def test_generate_views_request_rejects_empty_tickers(self):
        from app.schemas.views import GenerateViewsRequest

        with pytest.raises(Exception):
            GenerateViewsRequest(tickers=["AAPL", ""])

    def test_opinion_pool_request_normalises_tickers(self):
        from app.schemas.views import OpinionPoolRequest

        req = OpinionPoolRequest(tickers=["  spy  ", "qqq"])
        assert req.tickers == ["SPY", "QQQ"]

    def test_opinion_pool_request_validates_personas(self):
        from app.schemas.views import OpinionPoolRequest

        with pytest.raises(Exception):
            OpinionPoolRequest(tickers=["SPY", "QQQ"], personas=["INVALID_PERSONA"])

    def test_generate_views_response_serialises_to_camel_case(self):
        from app.schemas.views import AssetViewResponse, GenerateViewsResponse

        resp = GenerateViewsResponse(
            n_views=1,
            n_assets=2,
            view_strings=["SPY == 0.02"],
            P=[[1.0, 0.0]],
            Q=[0.02],
            view_confidences=[0.8],
            idzorek_alphas={"SPY": 0.8},
            views=[
                AssetViewResponse(
                    asset="SPY",
                    direction=1,
                    magnitude_bps=200.0,
                    confidence=0.8,
                    reasoning="Bullish",
                )
            ],
            rationale="Test rationale",
            tickers_with_data=["SPY"],
            tickers_missing_data=[],
        )
        data = resp.model_dump(by_alias=True)
        assert "nViews" in data
        assert "nAssets" in data
        assert "viewStrings" in data
        assert "viewConfidences" in data
        assert "idzorekAlphas" in data
        assert "tickersWithData" in data
        assert "tickersMissingData" in data


# ===========================================================================
# macro_regime.py — MacroCalibrationResponse added
# ===========================================================================


class TestMacroCalibrationResponseInMacroRegimeModule:
    """MacroCalibrationResponse must live in api/app/schemas/macro_regime."""

    def test_macro_calibration_response_importable(self):
        from app.schemas.macro_regime import MacroCalibrationResponse  # noqa: F401

    def test_macro_calibration_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.macro_regime import MacroCalibrationResponse

        assert issubclass(MacroCalibrationResponse, CamelCaseModel)

    def test_macro_calibration_response_serialises_to_camel_case(self):
        from app.schemas.macro_regime import MacroCalibrationResponse

        resp = MacroCalibrationResponse(
            phase="MID_EXPANSION",
            delta=2.5,
            tau=0.025,
            confidence=0.9,
            rationale="Solid growth.",
            macro_summary="GDP up 2%.",
            bl_config={"tau": 0.025},
        )
        data = resp.model_dump(by_alias=True)
        assert "macroSummary" in data
        assert "blConfig" in data


# ===========================================================================
# stress_scenarios.py schema file
# ===========================================================================


class TestStressScenariosSchemasModuleExists:
    """api/app/schemas/stress_scenarios.py must exist with expected schemas."""

    def test_stress_scenario_request_importable(self):
        from app.schemas.stress_scenarios import StressScenarioRequest  # noqa: F401

    def test_stress_scenario_item_importable(self):
        from app.schemas.stress_scenarios import StressScenarioItem  # noqa: F401

    def test_stress_scenario_response_importable(self):
        from app.schemas.stress_scenarios import StressScenarioResponse  # noqa: F401


class TestStressSchemasBaseClasses:
    """Correct base classes for stress scenario schemas."""

    def test_stress_scenario_request_uses_base_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.stress_scenarios import StressScenarioRequest

        assert issubclass(StressScenarioRequest, BaseModel)
        assert not issubclass(StressScenarioRequest, CamelCaseModel)

    def test_stress_scenario_item_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.stress_scenarios import StressScenarioItem

        assert issubclass(StressScenarioItem, CamelCaseModel)

    def test_stress_scenario_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.stress_scenarios import StressScenarioResponse

        assert issubclass(StressScenarioResponse, CamelCaseModel)


class TestStressSchemasValidatorsPreserved:
    """Validators preserved after migration."""

    def test_stress_scenario_request_normalises_portfolio_tickers(self):
        from app.schemas.stress_scenarios import StressScenarioRequest

        req = StressScenarioRequest(
            current_portfolio={"  spy  ": 0.5, "qqq": 0.5},
            macro_context="US CPI at 4%, Fed on hold.",
            n_scenarios=2,
        )
        assert "SPY" in req.current_portfolio
        assert "QQQ" in req.current_portfolio

    def test_stress_scenario_request_rejects_empty_portfolio(self):
        from app.schemas.stress_scenarios import StressScenarioRequest

        with pytest.raises(Exception):
            StressScenarioRequest(
                current_portfolio={"  ": 0.5},
                macro_context="US CPI at 4%, Fed on hold.",
            )


# ===========================================================================
# risk_budget.py schema file
# ===========================================================================


class TestRiskBudgetSchemasModuleExists:
    """api/app/schemas/risk_budget.py must exist with expected schemas."""

    def test_risk_budget_request_importable(self):
        from app.schemas.risk_budget import RiskBudgetRequest  # noqa: F401

    def test_risk_budget_response_importable(self):
        from app.schemas.risk_budget import RiskBudgetResponse  # noqa: F401


class TestRiskBudgetSchemasBaseClasses:
    """Correct base classes for risk budget schemas."""

    def test_risk_budget_request_uses_base_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.risk_budget import RiskBudgetRequest

        assert issubclass(RiskBudgetRequest, BaseModel)
        assert not issubclass(RiskBudgetRequest, CamelCaseModel)

    def test_risk_budget_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.risk_budget import RiskBudgetResponse

        assert issubclass(RiskBudgetResponse, CamelCaseModel)


class TestRiskBudgetSchemasValidatorsPreserved:
    """Validators preserved after migration."""

    def test_risk_budget_request_normalises_asset_sector_map(self):
        from app.schemas.risk_budget import RiskBudgetRequest

        req = RiskBudgetRequest(
            sector_outlook="Overweight Technology.",
            sector_universe=["Technology"],
            asset_sector_map={"  aapl  ": "Technology"},
        )
        assert "AAPL" in req.asset_sector_map

    def test_risk_budget_request_rejects_empty_asset_map(self):
        from app.schemas.risk_budget import RiskBudgetRequest

        with pytest.raises(Exception):
            RiskBudgetRequest(
                sector_outlook="Overweight Technology.",
                sector_universe=["Technology"],
                asset_sector_map={"  ": "Technology"},
            )

    def test_risk_budget_response_serialises_to_camel_case(self):
        from app.schemas.risk_budget import RiskBudgetResponse

        resp = RiskBudgetResponse(
            n_assets=2,
            assets=["AAPL", "MSFT"],
            budget_vector=[0.5, 0.5],
            budget_sum=1.0,
        )
        data = resp.model_dump(by_alias=True)
        assert "nAssets" in data
        assert "budgetVector" in data
        assert "budgetSum" in data


# ===========================================================================
# llm_moments.py schema file
# ===========================================================================


class TestLlmMomentsSchemasModuleExists:
    """api/app/schemas/llm_moments.py must exist with expected schemas."""

    def test_calibrate_delta_request_importable(self):
        from app.schemas.llm_moments import CalibrateDeltaRequest  # noqa: F401

    def test_calibrate_delta_response_importable(self):
        from app.schemas.llm_moments import CalibrateDeltaResponse  # noqa: F401

    def test_adapt_factor_weights_request_importable(self):
        from app.schemas.llm_moments import AdaptFactorWeightsRequest  # noqa: F401

    def test_adapt_factor_weights_response_importable(self):
        from app.schemas.llm_moments import AdaptFactorWeightsResponse  # noqa: F401

    def test_select_cov_regime_request_importable(self):
        from app.schemas.llm_moments import SelectCovRegimeRequest  # noqa: F401

    def test_select_cov_regime_response_importable(self):
        from app.schemas.llm_moments import SelectCovRegimeResponse  # noqa: F401


class TestLlmMomentsSchemasBaseClasses:
    """Correct base classes for LLM moments schemas."""

    def test_calibrate_delta_request_uses_base_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.llm_moments import CalibrateDeltaRequest

        assert issubclass(CalibrateDeltaRequest, BaseModel)
        assert not issubclass(CalibrateDeltaRequest, CamelCaseModel)

    def test_calibrate_delta_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.llm_moments import CalibrateDeltaResponse

        assert issubclass(CalibrateDeltaResponse, CamelCaseModel)

    def test_adapt_factor_weights_request_uses_base_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.llm_moments import AdaptFactorWeightsRequest

        assert issubclass(AdaptFactorWeightsRequest, BaseModel)
        assert not issubclass(AdaptFactorWeightsRequest, CamelCaseModel)

    def test_adapt_factor_weights_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.llm_moments import AdaptFactorWeightsResponse

        assert issubclass(AdaptFactorWeightsResponse, CamelCaseModel)

    def test_select_cov_regime_request_uses_base_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.llm_moments import SelectCovRegimeRequest

        assert issubclass(SelectCovRegimeRequest, BaseModel)
        assert not issubclass(SelectCovRegimeRequest, CamelCaseModel)

    def test_select_cov_regime_response_extends_camel_case_model(self):
        from app.schemas.base import CamelCaseModel
        from app.schemas.llm_moments import SelectCovRegimeResponse

        assert issubclass(SelectCovRegimeResponse, CamelCaseModel)


class TestLlmMomentsSchemasValidatorsPreserved:
    """Validators preserved after migration."""

    def test_adapt_factor_weights_request_rejects_empty_group(self):
        from app.schemas.llm_moments import AdaptFactorWeightsRequest

        with pytest.raises(Exception):
            AdaptFactorWeightsRequest(
                macro_indicators="PMI at 52, yield curve positive.",
                factor_groups=["VALUE", ""],
            )

    def test_select_cov_regime_request_rejects_empty_headline(self):
        from app.schemas.llm_moments import SelectCovRegimeRequest

        with pytest.raises(Exception):
            SelectCovRegimeRequest(
                news_headlines=["Fed holds rates", ""],
                avg_sentiment_score=0.1,
                realized_vol_30d=0.15,
            )

    def test_calibrate_delta_response_serialises_to_camel_case(self):
        from app.schemas.llm_moments import CalibrateDeltaResponse

        resp = CalibrateDeltaResponse(delta=3.0, rationale="Mid-expansion.")
        data = resp.model_dump(by_alias=True)
        # delta and rationale are single-word — no transformation needed
        assert "delta" in data
        assert "rationale" in data

    def test_adapt_factor_weights_response_serialises_to_camel_case(self):
        from app.schemas.llm_moments import AdaptFactorWeightsResponse

        resp = AdaptFactorWeightsResponse(
            phase="MID_EXPANSION",
            weights={"VALUE": 1.2, "MOMENTUM": 0.8},
            rationale="Mid-expansion favors value.",
        )
        data = resp.model_dump(by_alias=True)
        assert "phase" in data
        assert "weights" in data


# ===========================================================================
# Route files no longer define schemas inline
# ===========================================================================


class TestRouteFilesNoInlineSchemas:
    """Route files must import schemas from schema modules, not define them inline."""

    def _get_source(self, module_path: str) -> str:
        import importlib

        module = importlib.import_module(module_path)
        return inspect.getsource(module)

    def test_views_route_imports_from_schema_module(self):
        source = self._get_source("app.api.v1.views")
        assert "from app.schemas.views import" in source
        # No inline class definitions for the moved schemas
        assert "class GenerateViewsRequest" not in source
        assert "class AssetViewResponse" not in source
        assert "class GenerateViewsResponse" not in source

    def test_opinion_pooling_route_imports_from_schema_module(self):
        source = self._get_source("app.api.v1.opinion_pooling")
        assert "from app.schemas.views import" in source
        assert "class ICHistory" not in source
        assert "class ExpertViewSummary" not in source
        assert "class OpinionPoolRequest" not in source
        assert "class OpinionPoolResponse" not in source

    def test_macro_calibration_route_imports_from_schema_module(self):
        source = self._get_source("app.api.v1.macro_calibration")
        assert "MacroCalibrationResponse" in source
        assert "class MacroCalibrationResponse" not in source

    def test_stress_scenarios_route_imports_from_schema_module(self):
        source = self._get_source("app.api.v1.stress_scenarios")
        assert "from app.schemas.stress_scenarios import" in source
        assert "class StressScenarioRequest" not in source
        assert "class StressScenarioItem" not in source
        assert "class StressScenarioResponse" not in source

    def test_risk_budget_route_imports_from_schema_module(self):
        source = self._get_source("app.api.v1.risk_budget")
        assert "from app.schemas.risk_budget import" in source
        assert "class RiskBudgetRequest" not in source
        assert "class RiskBudgetResponse" not in source

    def test_llm_moments_route_imports_from_schema_module(self):
        source = self._get_source("app.api.v1.llm_moments")
        assert "from app.schemas.llm_moments import" in source
        assert "class CalibrateDeltaRequest" not in source
        assert "class CalibrateDeltaResponse" not in source
        assert "class AdaptFactorWeightsRequest" not in source
        assert "class AdaptFactorWeightsResponse" not in source
        assert "class SelectCovRegimeRequest" not in source
        assert "class SelectCovRegimeResponse" not in source


# ===========================================================================
# __init__.py re-exports
# ===========================================================================


class TestInitReexportsNewSchemas:
    """api/app/schemas/__init__.py must re-export the new schemas."""

    def test_views_schemas_importable_from_package(self):
        from app.schemas import GenerateViewsRequest  # noqa: F401
        from app.schemas import AssetViewResponse  # noqa: F401
        from app.schemas import GenerateViewsResponse  # noqa: F401
        from app.schemas import ICHistory  # noqa: F401
        from app.schemas import ExpertViewSummary  # noqa: F401
        from app.schemas import OpinionPoolRequest  # noqa: F401
        from app.schemas import OpinionPoolResponse  # noqa: F401

    def test_macro_calibration_response_importable_from_package(self):
        from app.schemas import MacroCalibrationResponse  # noqa: F401

    def test_stress_scenarios_schemas_importable_from_package(self):
        from app.schemas import StressScenarioRequest  # noqa: F401
        from app.schemas import StressScenarioItem  # noqa: F401
        from app.schemas import StressScenarioResponse  # noqa: F401

    def test_risk_budget_schemas_importable_from_package(self):
        from app.schemas import RiskBudgetRequest  # noqa: F401
        from app.schemas import RiskBudgetResponse  # noqa: F401

    def test_llm_moments_schemas_importable_from_package(self):
        from app.schemas import CalibrateDeltaRequest  # noqa: F401
        from app.schemas import CalibrateDeltaResponse  # noqa: F401
        from app.schemas import AdaptFactorWeightsRequest  # noqa: F401
        from app.schemas import AdaptFactorWeightsResponse  # noqa: F401
        from app.schemas import SelectCovRegimeRequest  # noqa: F401
        from app.schemas import SelectCovRegimeResponse  # noqa: F401
