"""Tests for _shared_ file migration (issue #608)."""

import importlib
from pathlib import Path

import pytest

API_APP = Path(__file__).resolve().parent.parent.parent / "app"

# ── Files to be moved (source → target relative to API_APP) ────────────

MOVE_MAP: dict[str, str] = {
    # api/v1/ (4 files)
    "api/v1/router.py": "api/v1/_shared/router.py",
    "api/v1/metrics.py": "api/v1/_shared/metrics.py",
    "api/v1/database.py": "api/v1/_shared/database.py",
    "api/v1/test.py": "api/v1/_shared/test.py",
    # models/ (1 file)
    "models/base.py": "models/_shared/base.py",
    # repositories/ (2 files)
    "repositories/base.py": "repositories/_shared/base.py",
    "repositories/database_admin_repository.py": "repositories/_shared/database_admin_repository.py",
    # services/ (7 files)
    "services/_price_fetcher.py": "services/_shared/_price_fetcher.py",
    "services/_sector_resolver.py": "services/_shared/_sector_resolver.py",
    "services/_json_safe.py": "services/_shared/_json_safe.py",
    "services/_progress.py": "services/_shared/_progress.py",
    "services/_benchmark_bootstrap.py": "services/_shared/_benchmark_bootstrap.py",
    "services/trading_calendar.py": "services/_shared/trading_calendar.py",
    "services/notifications.py": "services/_shared/notifications.py",
    # schemas/ (2 files)
    "schemas/base.py": "schemas/_shared/base.py",
    "schemas/base_job.py": "schemas/_shared/base_job.py",
}


# ── Helpers ────────────────────────────────────────────────────────────


def _all_targets():
    return list(MOVE_MAP.items())


# ── Tests ──────────────────────────────────────────────────────────────


class TestFilesMovedToShared:
    """Acceptance criterion: All 16 files moved to _shared_ via git mv."""

    @pytest.mark.parametrize("old_path, new_path", _all_targets())
    def test_file_exists_at_new_path(self, old_path, new_path):
        assert (API_APP / new_path).is_file(), f"Missing at new path: {new_path}"

    @pytest.mark.parametrize("old_path, new_path", _all_targets())
    def test_file_no_longer_at_old_path(self, old_path, new_path):
        assert not (API_APP / old_path).exists(), (
            f"Still exists at old path: {old_path}"
        )


class TestSharedInitPyReexports:
    """Each _shared/__init__.py must re-export contained modules."""

    def test_api_v1_shared_re_exports_router(self):
        from app.api.v1._shared.router import api_router

        assert api_router is not None

    def test_services_shared_re_exports_benchmark(self):
        from app.services._shared import bootstrap_benchmarks

        assert bootstrap_benchmarks is not None

    def test_services_shared_re_exports_trading_calendar(self):
        from app.services._shared import trading_calendar

        assert trading_calendar is not None

    def test_models_shared_exports_base_classes(self):
        from app.models._shared import Base, BaseModel, TimestampMixin

        assert Base is not None
        assert BaseModel is not None
        assert TimestampMixin is not None

    def test_repositories_shared_exports_base(self):
        from app.repositories._shared import BaseRepository, RepositoryBase

        assert BaseRepository is not None
        assert RepositoryBase is not None

    def test_repositories_shared_exports_admin_repo(self):
        from app.repositories._shared import DatabaseAdminRepository

        assert DatabaseAdminRepository is not None

    def test_schemas_shared_exports_base(self):
        from app.schemas._shared import CamelCaseModel, StrFromUUID

        assert CamelCaseModel is not None
        assert StrFromUUID is not None

    def test_schemas_shared_exports_base_job(self):
        from app.schemas._shared import AsyncJobCreateResponse, AsyncJobProgress

        assert AsyncJobCreateResponse is not None
        assert AsyncJobProgress is not None


class TestAppBootsAfterReorganization:
    """Ensure the app factory can be imported without error."""

    def test_main_imports_succeed(self):
        importlib.import_module("app.main")

    def test_models_init_py_imports_resolve(self):
        importlib.import_module("app.models")

    def test_repositories_init_py_imports_resolve(self):
        importlib.import_module("app.repositories")

    def test_schemas_init_py_imports_resolve(self):
        importlib.import_module("app.schemas")

    def test_services_init_py_imports_resolve(self):
        importlib.import_module("app.services")

    def test_api_router_imports_resolve(self):
        importlib.import_module("app.api.v1._shared.router")
