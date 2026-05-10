"""Tests for models/ and schemas/ domain file migration (issue #609)."""

import ast
import importlib
from pathlib import Path

import pytest

API_APP = Path(__file__).resolve().parent.parent.parent / "app"

# ── Files to be moved (source → target relative to API_APP) ────────────

_MODELS_MOVES: dict[str, str] = {
    "models/api_key.py": "models/auth/api_key.py",
    "models/yfinance_data.py": "models/market_data/yfinance_data.py",
    "models/universe.py": "models/universe/universe.py",
    "models/macro_regime.py": "models/macro/macro_regime.py",
    "models/portfolio.py": "models/portfolio/portfolio.py",
    "models/execution.py": "models/execution/execution.py",
    "models/factor.py": "models/factors/factor.py",
    "models/risk.py": "models/risk/risk.py",
    "models/rebalancing.py": "models/rebalancing/rebalancing.py",
    "models/background_job.py": "models/jobs/background_job.py",
}

_SCHEMAS_MOVES: dict[str, str] = {
    "schemas/yfinance_data.py": "schemas/market_data/yfinance_data.py",
    "schemas/reference_index.py": "schemas/market_data/reference_index.py",
    "schemas/trading212.py": "schemas/universe/trading212.py",
    "schemas/universe_screen.py": "schemas/universe/universe_screen.py",
    "schemas/macro_regime.py": "schemas/macro/macro_regime.py",
    "schemas/portfolio.py": "schemas/portfolio/portfolio.py",
    "schemas/optimization.py": "schemas/optimization/optimization.py",
    "schemas/tuning.py": "schemas/optimization/tuning.py",
    "schemas/validation.py": "schemas/optimization/validation.py",
    "schemas/backtest.py": "schemas/backtest/backtest.py",
    "schemas/factors.py": "schemas/factors/factors.py",
    "schemas/risk.py": "schemas/risk/risk.py",
    "schemas/risk_analytics.py": "schemas/risk/risk_analytics.py",
    "schemas/rebalancing.py": "schemas/rebalancing/rebalancing.py",
    "schemas/views.py": "schemas/views/views.py",
    "schemas/llm_moments.py": "schemas/views/llm_moments.py",
    "schemas/attribution.py": "schemas/attribution/attribution.py",
    "schemas/dashboard.py": "schemas/dashboard/dashboard.py",
    "schemas/stress_scenarios.py": "schemas/scenarios/stress_scenarios.py",
    "schemas/synthetic.py": "schemas/scenarios/synthetic.py",
    "schemas/reports.py": "schemas/reports/reports.py",
    "schemas/jobs.py": "schemas/jobs/jobs.py",
    "schemas/scheduler.py": "schemas/jobs/scheduler.py",
}

ALL_MOVES = {**_MODELS_MOVES, **_SCHEMAS_MOVES}

# Domain dirs that should have __init__.py with re-exports after migration
_DOMAIN_DIRS_MODELS = sorted({p.split("/")[1] for p in _MODELS_MOVES.values()})
_DOMAIN_DIRS_SCHEMAS = sorted({p.split("/")[1] for p in _SCHEMAS_MOVES.values()})


# ── Helpers ────────────────────────────────────────────────────────────


def _model_move_params():
    return list(_MODELS_MOVES.items())


def _schema_move_params():
    return list(_SCHEMAS_MOVES.items())


def _all_move_params():
    return list(ALL_MOVES.items())


# ── Tests ──────────────────────────────────────────────────────────────


class TestFilesMovedToDomainDirs:
    """Acceptance criterion: All files moved to domain sub-folders via git mv."""

    @pytest.mark.parametrize("old_path, new_path", _all_move_params())
    def test_file_exists_at_new_path(self, old_path, new_path):
        assert (API_APP / new_path).is_file(), f"Missing at new path: {new_path}"

    @pytest.mark.parametrize("old_path, new_path", _all_move_params())
    def test_file_no_longer_at_old_path(self, old_path, new_path):
        assert not (API_APP / old_path).exists(), (
            f"Still exists at old path: {old_path}"
        )


class TestDomainInitPyReExports:
    """Each domain __init__.py must have docstring and re-export contained modules."""

    @pytest.mark.parametrize("domain", _DOMAIN_DIRS_MODELS)
    def test_models_domain_init_py_exists_with_docstring(self, domain):
        init = API_APP / "models" / domain / "__init__.py"
        assert init.is_file(), f"Missing __init__.py in models/{domain}"
        tree = ast.parse(init.read_text())
        doc = ast.get_docstring(tree)
        assert doc is not None, f"models/{domain}/__init__.py has no docstring"
        assert len(doc.strip()) > 0

    @pytest.mark.parametrize("domain", _DOMAIN_DIRS_SCHEMAS)
    def test_schemas_domain_init_py_exists_with_docstring(self, domain):
        init = API_APP / "schemas" / domain / "__init__.py"
        assert init.is_file(), f"Missing __init__.py in schemas/{domain}"
        tree = ast.parse(init.read_text())
        doc = ast.get_docstring(tree)
        assert doc is not None, f"schemas/{domain}/__init__.py has no docstring"
        assert len(doc.strip()) > 0


class TestTopLevelInitPyUpdated:
    """models/__init__.py and schemas/__init__.py use new sub-package paths."""

    def test_models_init_imports_from_sub_packages(self):
        src = (API_APP / "models" / "__init__.py").read_text()
        # Must not import from old flat paths like "from app.models.universe.universe import"
        for old in _MODELS_MOVES:
            # old is like "models/universe.py" → flat import "from app.models.universe.universe import"
            module_name = Path(old).stem
            flat_import = f"from app.models.{module_name} import"
            assert flat_import not in src, (
                f"models/__init__.py still has old flat import: {flat_import}"
            )

    def test_schemas_init_imports_from_sub_packages(self):
        src = (API_APP / "schemas" / "__init__.py").read_text()
        for old in _SCHEMAS_MOVES:
            module_name = Path(old).stem
            flat_import = f"from app.schemas.{module_name} import"
            assert flat_import not in src, (
                f"schemas/__init__.py still has old flat import: {flat_import}"
            )


class TestAppBootsAfterReorganization:
    """Ensure critical imports and app factory work after migration."""

    def test_import_models_from_new_paths(self):
        """from app.models import PriceHistory, Portfolio, MacroNews"""
        from app.models import MacroNews, Portfolio, PriceHistory

        assert Portfolio is not None
        assert PriceHistory is not None
        assert MacroNews is not None

    def test_import_schemas_from_new_paths(self):
        """from app.schemas import OptimizeRequest, PortfolioResponse"""
        from app.schemas import OptimizeRequest, PortfolioResponse

        assert OptimizeRequest is not None
        assert PortfolioResponse is not None

    def test_main_imports_succeed(self):
        importlib.import_module("app.main")

    def test_models_init_py_imports_resolve(self):
        importlib.import_module("app.models")

    def test_schemas_init_py_imports_resolve(self):
        importlib.import_module("app.schemas")
