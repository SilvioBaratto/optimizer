"""Tests for repositories/ domain file migration (issue #610)."""

import ast
import importlib
from pathlib import Path

import pytest

API_APP = Path(__file__).resolve().parent.parent.parent / "app"

# ── Files to be moved (source → target relative to API_APP) ────────────

_REPO_MOVES: dict[str, str] = {
    "repositories/yfinance_repository.py": "repositories/market_data/yfinance_repository.py",
    "repositories/universe_repository.py": "repositories/universe/universe_repository.py",
    "repositories/macro_regime_repository.py": "repositories/macro/macro_regime_repository.py",
    "repositories/sentiment_repository.py": "repositories/macro/sentiment_repository.py",
    "repositories/portfolio_repository.py": "repositories/portfolio/portfolio_repository.py",
    "repositories/execution_repository.py": "repositories/execution/execution_repository.py",
    "repositories/factor_repository.py": "repositories/factors/factor_repository.py",
    "repositories/risk_repository.py": "repositories/risk/risk_repository.py",
    "repositories/rebalancing_repository.py": "repositories/rebalancing/rebalancing_repository.py",
    "repositories/view_generation_repository.py": "repositories/views/view_generation_repository.py",
    "repositories/dashboard_repository.py": "repositories/dashboard/dashboard_repository.py",
    "repositories/background_job_repository.py": "repositories/jobs/background_job_repository.py",
}

_DOMAIN_DIRS = sorted({p.split("/")[1] for p in _REPO_MOVES.values()})


def _all_move_params():
    return list(_REPO_MOVES.items())


# ── Tests ──────────────────────────────────────────────────────────────


class TestFilesMovedToDomainDirs:
    """All repo files moved to correct domain sub-folders via git mv."""

    @pytest.mark.parametrize("old_path, new_path", _all_move_params())
    def test_file_exists_at_new_path(self, old_path, new_path):
        assert (API_APP / new_path).is_file(), f"Missing at new path: {new_path}"

    @pytest.mark.parametrize("old_path, new_path", _all_move_params())
    def test_file_no_longer_at_old_path(self, old_path, new_path):
        assert not (API_APP / old_path).exists(), (
            f"Still exists at old path: {old_path}"
        )


class TestDomainInitPyReExports:
    """Each domain __init__.py re-exports contained repository classes."""

    @pytest.mark.parametrize("domain", _DOMAIN_DIRS)
    def test_domain_init_py_exists_with_docstring(self, domain):
        init = API_APP / "repositories" / domain / "__init__.py"
        assert init.is_file(), f"Missing __init__.py in repositories/{domain}"
        tree = ast.parse(init.read_text())
        doc = ast.get_docstring(tree)
        assert doc is not None, f"repositories/{domain}/__init__.py has no docstring"
        assert len(doc.strip()) > 0


class TestTopLevelInitPyUpdated:
    """repositories/__init__.py uses new sub-package paths, no old flat imports."""

    def test_no_old_flat_imports(self):
        src = (API_APP / "repositories" / "__init__.py").read_text()
        for old in _REPO_MOVES:
            module_name = Path(old).stem
            flat_import = f"from app.repositories.{module_name} import"
            assert flat_import not in src, (
                f"repositories/__init__.py still has old import: {flat_import}"
            )


class TestAppBootsAfterReorganization:
    """Critical imports and app factory work after migration."""

    def test_repos_importable_from_package(self):
        from app.repositories import (
            MacroRegimeRepository,
            PortfolioRepository,
            YFinanceRepository,
        )

        assert YFinanceRepository is not None
        assert PortfolioRepository is not None
        assert MacroRegimeRepository is not None

    def test_main_imports_succeed(self):
        importlib.import_module("app.main")

    def test_repos_init_imports_resolve(self):
        importlib.import_module("app.repositories")

    def test_background_job_repo_import_via_new_path(self):
        from app.repositories.jobs.background_job_repository import (
            BackgroundJobRepository,
        )

        assert BackgroundJobRepository is not None
