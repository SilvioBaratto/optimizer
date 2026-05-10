"""Tests for domain directory scaffolding (issue #607)."""

import ast
import importlib
from pathlib import Path

import pytest

API_APP = Path(__file__).resolve().parent.parent.parent / "app"

# ── Expected directories per layer (from issue #607 body) ──────────────

_API_V1_DIRS = [
    "_shared",
    "market_data",
    "universe",
    "macro",
    "portfolio",
    "optimization",
    "backtest",
    "factors",
    "risk",
    "rebalancing",
    "views",
    "attribution",
    "dashboard",
    "scenarios",
    "reports",
    "jobs",
]

_MODELS_DIRS = [
    "_shared",
    "auth",
    "market_data",
    "universe",
    "macro",
    "portfolio",
    "execution",
    "factors",
    "risk",
    "rebalancing",
    "jobs",
]

_REPOSITORIES_DIRS = [
    "_shared",
    "market_data",
    "universe",
    "macro",
    "portfolio",
    "execution",
    "factors",
    "risk",
    "rebalancing",
    "views",
    "dashboard",
    "jobs",
]

_SERVICES_DIRS = _API_V1_DIRS  # same 16

_SCHEMAS_DIRS = _API_V1_DIRS  # same 16

EXPECTED_LAYERS = {
    "api/v1": _API_V1_DIRS,
    "models": _MODELS_DIRS,
    "repositories": _REPOSITORIES_DIRS,
    "services": _SERVICES_DIRS,
    "schemas": _SCHEMAS_DIRS,
}

# Directories whose name collides with an existing .py module file.
# Adding __init__.py would shadow the .py file and break imports until
# Cycle 2 moves the files into their domain homes.
_DIRS_WITH_DEFERRED_INIT = {
    (layer, domain)
    for layer, domains in EXPECTED_LAYERS.items()
    for domain in domains
    if (API_APP / layer / f"{domain}.py").is_file()
}


# ── Helper ─────────────────────────────────────────────────────────────


def _all_expected_paths():
    """Yield (layer, domain, absolute_path) for every expected directory."""
    for layer, domains in EXPECTED_LAYERS.items():
        for domain in domains:
            yield layer, domain, API_APP / layer / domain


# ── Tests ──────────────────────────────────────────────────────────────


class TestAllDomainDirectoriesExist:
    """Acceptance criterion: all 85+ domain directories exist."""

    @pytest.mark.parametrize(
        "layer, domain, path",
        list(_all_expected_paths()),
    )
    def test_domain_directory_exists(self, layer, domain, path):
        _ = layer, domain  # parametrize fixture labels
        assert path.is_dir(), f"Missing directory: {path}"


class TestInitPyInEveryFolder:
    """Acceptance criterion: every new folder has __init__.py with docstring."""

    @pytest.mark.parametrize(
        "layer, domain, path",
        list(_all_expected_paths()),
    )
    def test_init_py_exists(self, layer, domain, path):
        if (layer, domain) in _DIRS_WITH_DEFERRED_INIT:
            pytest.skip("__init__.py deferred to Cycle 2 (shadows existing .py)")
        init = path / "__init__.py"
        assert init.is_file(), f"Missing __init__.py in {path}"

    @pytest.mark.parametrize(
        "layer, domain, path",
        list(_all_expected_paths()),
    )
    def test_init_py_has_docstring(self, layer, domain, path):
        _ = layer, domain  # parametrize fixture labels
        init = path / "__init__.py"
        if not init.is_file():
            pytest.skip("__init__.py missing")
        tree = ast.parse(init.read_text())
        doc = ast.get_docstring(tree)
        assert doc is not None, f"__init__.py in {path} has no docstring"
        assert len(doc.strip()) > 0, f"__init__.py in {path} has empty docstring"


class TestImportsSucceed:
    """Acceptance criterion: package imports succeed."""

    def test_import_app_succeeds(self):
        importlib.import_module("app")

    def test_import_api_v1_succeeds(self):
        importlib.import_module("app.api.v1")

    def test_import_models_succeeds(self):
        importlib.import_module("app.models")

    def test_import_repositories_succeeds(self):
        importlib.import_module("app.repositories")

    def test_import_services_succeeds(self):
        importlib.import_module("app.services")

    def test_import_schemas_succeeds(self):
        importlib.import_module("app.schemas")


class TestExistingInitPyNotModified:
    """Acceptance criterion: existing __init__.py files are untouched.

    We verify by checking the existing layer __init__.py files still
    have their original content (they export symbols).
    """

    def test_api_v1_init_py_still_exports(self):
        path = API_APP / "api" / "v1" / "__init__.py"
        assert path.is_file()
        assert len(path.read_text()) > 0

    def test_models_init_py_still_exports_all(self):
        path = API_APP / "models" / "__init__.py"
        assert path.is_file()
        tree = ast.parse(path.read_text())
        # __all__ should still be present
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        return
        pytest.fail("models/__init__.py missing __all__ export")

    def test_repositories_init_py_still_exports_all(self):
        path = API_APP / "repositories" / "__init__.py"
        assert path.is_file()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        return
        pytest.fail("repositories/__init__.py missing __all__ export")

    def test_schemas_init_py_still_exports_all(self):
        path = API_APP / "schemas" / "__init__.py"
        assert path.is_file()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        return
        pytest.fail("schemas/__init__.py missing __all__ export")

    def test_services_init_py_still_exports(self):
        path = API_APP / "services" / "__init__.py"
        assert path.is_file()
        assert len(path.read_text()) > 0


class TestNoFilesMoved:
    """Acceptance criterion: files moved by #608 are at their new _shared/ paths;
    files not yet moved are still at their original locations."""

    def test_route_file_at_new_shared_path(self):
        router = API_APP / "api" / "v1" / "_shared" / "router.py"
        assert router.is_file(), "router.py should be at _shared/router.py"

    def test_model_file_at_new_shared_path(self):
        base = API_APP / "models" / "_shared" / "base.py"
        assert base.is_file(), "models/base.py should be at _shared/base.py"

    def test_repository_file_at_new_shared_path(self):
        base = API_APP / "repositories" / "_shared" / "base.py"
        assert base.is_file(), "repositories/base.py should be at _shared/base.py"

    def test_existing_service_files_still_in_place(self):
        bg = API_APP / "services" / "jobs" / "background_job.py"
        assert bg.is_file(), (
            "services/background_job.py should be at jobs/background_job.py"
        )

    def test_schema_file_at_new_shared_path(self):
        base = API_APP / "schemas" / "_shared" / "base.py"
        assert base.is_file(), "schemas/base.py should be at _shared/base.py"
