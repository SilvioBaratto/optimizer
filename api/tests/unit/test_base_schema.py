"""Tests for api/app/schemas/base.py — CamelCaseModel extraction (issue #345)."""

import pytest
from pydantic import BaseModel


class TestCamelCaseModelDefinedInBase:
    """CamelCaseModel must live in api/app/schemas/base, not only in dashboard."""

    def test_camel_case_model_importable_from_base(self):
        """Acceptance criterion: CamelCaseModel is defined in api/app/schemas/base."""
        from app.schemas.base import CamelCaseModel  # noqa: F401

    def test_camel_case_model_is_pydantic_base_model_subclass(self):
        from app.schemas.base import CamelCaseModel

        assert issubclass(CamelCaseModel, BaseModel)

    def test_camel_case_model_serialises_snake_to_camel(self):
        """Fields named in snake_case must appear as camelCase in JSON."""
        from app.schemas.base import CamelCaseModel

        class SampleResponse(CamelCaseModel):
            total_count: int
            is_active: bool

        obj = SampleResponse(total_count=5, is_active=True)
        data = obj.model_dump(by_alias=True)

        assert "totalCount" in data
        assert "isActive" in data
        assert "total_count" not in data
        assert "is_active" not in data

    def test_camel_case_model_accepts_snake_case_input(self):
        """populate_by_name=True: model can be constructed with snake_case names."""
        from app.schemas.base import CamelCaseModel

        class SampleResponse(CamelCaseModel):
            total_count: int

        obj = SampleResponse(total_count=42)
        assert obj.total_count == 42


class TestDashboardImportsCamelCaseModelFromBase:
    """dashboard.py must not define CamelCaseModel inline; it must import it."""

    def test_dashboard_imports_camel_case_model_from_base(self):
        """Acceptance criterion: dashboard.py imports CamelCaseModel from base."""
        import inspect

        import app.schemas.dashboard as dashboard_module
        import app.schemas.base as base_module

        assert dashboard_module.CamelCaseModel is base_module.CamelCaseModel

    def test_camel_case_model_not_defined_inline_in_dashboard(self):
        """CamelCaseModel must be imported, not defined in dashboard.py itself."""
        import inspect
        import app.schemas.dashboard as dashboard_module

        source = inspect.getsourcefile(dashboard_module.CamelCaseModel)
        # The class must be defined in base.py, not dashboard.py
        assert source is not None
        assert "base.py" in source, (
            f"CamelCaseModel should be defined in base.py, but found in {source}"
        )


class TestDashboardSchemasStillWork:
    """All dashboard response schemas must continue to serialize correctly."""

    def test_kpi_item_serialises_to_camel_case(self):
        from app.schemas.dashboard import KpiItem

        kpi = KpiItem(
            label="Return",
            value=0.12,
            format="percent",
            change=0.01,
            change_label="+1%",
            sparkline=[0.1, 0.11, 0.12],
        )
        data = kpi.model_dump(by_alias=True)
        assert "changeLabel" in data

    def test_equity_curve_response_serialises_to_camel_case(self):
        from app.schemas.dashboard import EquityCurveResponse

        from datetime import date

        resp = EquityCurveResponse(
            points=[],
            portfolio_total_return=0.1,
            benchmark_total_return=0.05,
        )
        data = resp.model_dump(by_alias=True)
        assert "portfolioTotalReturn" in data
        assert "benchmarkTotalReturn" in data

    def test_drift_response_serialises_to_camel_case(self):
        from app.schemas.dashboard import DriftResponse

        resp = DriftResponse(
            entries=[],
            total_drift=0.02,
            breached_count=1,
            threshold=0.05,
        )
        data = resp.model_dump(by_alias=True)
        assert "totalDrift" in data
        assert "breachedCount" in data


class TestInitReexportsCamelCaseModel:
    """api/app/schemas/__init__.py must re-export CamelCaseModel for convenience."""

    def test_camel_case_model_importable_from_schemas_package(self):
        """Acceptance criterion: __init__.py re-exports CamelCaseModel."""
        from app.schemas import CamelCaseModel  # noqa: F401

    def test_schemas_camel_case_model_is_same_object_as_base(self):
        from app.schemas import CamelCaseModel as FromInit
        from app.schemas.base import CamelCaseModel as FromBase

        assert FromInit is FromBase

    def test_no_import_cycle(self):
        """Importing base, then dashboard, then __init__ must not raise."""
        import importlib

        importlib.import_module("app.schemas.base")
        importlib.import_module("app.schemas.dashboard")
        importlib.import_module("app.schemas")
