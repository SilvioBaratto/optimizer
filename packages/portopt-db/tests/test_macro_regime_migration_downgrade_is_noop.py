"""
The migration module is loaded dynamically by file path and both upgrade() and
downgrade() are invoked with no Alembic runtime context (no ``op``/connection
available). This only succeeds if both bodies are pure no-ops — exactly the
invariant the test verifies.
"""

import importlib.util
from pathlib import Path

import pytest

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "ecf0a9a2bfdc_add_macro_regime_tables.py"
)


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "ecf0a9a2bfdc_add_macro_regime_tables", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.criterion("scope-1")
def test_when_macro_regime_migration_is_upgraded_then_downgraded_then_no_error_is_raised():
    module = _load_migration_module()

    module.upgrade()
    module.downgrade()
