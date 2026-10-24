"""Guard: the Postgres integration test can never touch the developer's live DB.

``fund.config`` runs ``load_dotenv(override=True)`` at import, so on a machine with
the repo-root ``.env`` the ambient ``DATABASE_URL`` is the compose (live) database
no matter what the shell exported. The integration test drops a schema before and
after each case, so it must be keyed on its own ``FUND_TEST_DATABASE_URL`` and a
throwaway schema, and must refuse a target that is the live database.

The runs below are subprocesses that simulate the ``.env`` override and point
``DATABASE_URL`` at a port nothing listens on, so even a regression fails with a
connection error instead of reaching a real database.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from fund.config import FundConfig

_FUND_ROOT = Path(__file__).resolve().parents[2]
_INTEGRATION_FILE = (
    _FUND_ROOT / "tests" / "integration" / "test_langgraph_persistence.py"
)

# Port 1 is never bound: a connect attempt is refused instantly.
_LIVE_URL = "postgresql://nobody:nopass@127.0.0.1:1/live_db"

_RUNNER = """
import os
import sys

import dotenv


def _simulated_repo_dotenv(*_args, **_kwargs):
    os.environ["DATABASE_URL"] = os.environ["SIMULATED_DOTENV_DATABASE_URL"]
    return True


dotenv.load_dotenv = _simulated_repo_dotenv

import pytest

sys.exit(pytest.main([sys.argv[1], "-q", "-rs", "-p", "no:cacheprovider"]))
"""


def _run_integration_module(
    module: Path = _INTEGRATION_FILE, **extra_env: str
) -> tuple[int, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"DATABASE_URL", "FUND_TEST_DATABASE_URL"}
    }
    env["SIMULATED_DOTENV_DATABASE_URL"] = _LIVE_URL
    env.update(extra_env)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _RUNNER, str(module)],
        cwd=_FUND_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _load_integration_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_integration_langgraph_persistence_under_audit", _INTEGRATION_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_integration_module_skips_when_only_the_ambient_database_url_is_set() -> None:
    """Without ``FUND_TEST_DATABASE_URL`` every integration case is skipped."""
    code, output = _run_integration_module()

    assert code == 0, output
    assert "skipped" in output, output
    assert "error" not in output.lower(), output


def test_integration_module_refuses_a_target_that_is_the_live_database() -> None:
    """A target resolving to the ambient ``DATABASE_URL`` fails before any connect."""
    # Same server and database spelled differently: driver suffix, localhost alias.
    aliased_live = "postgresql+psycopg2://other:secret@localhost:1/live_db"

    code, output = _run_integration_module(FUND_TEST_DATABASE_URL=aliased_live)

    assert code != 0, output
    assert "refusing" in output.lower(), output


def test_integration_schema_is_not_the_schema_the_running_fund_uses() -> None:
    """The schema the fixture drops is a throwaway, never the fund's ``langgraph``."""
    module = _load_integration_module()

    assert FundConfig().langgraph_schema != module.SCHEMA


def test_integration_accepts_a_scratch_database_on_the_live_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different database on the same server is a legitimate scratch target."""
    module = _load_integration_module()
    scratch = "postgresql://nobody:nopass@127.0.0.1:1/scratch_db"
    monkeypatch.setenv("DATABASE_URL", _LIVE_URL)
    monkeypatch.setenv("FUND_TEST_DATABASE_URL", scratch)
    monkeypatch.setattr(
        module, "settings", replace(module.settings, database_url=_LIVE_URL)
    )

    assert module._target_url() == scratch


@pytest.mark.parametrize(
    "module",
    sorted((_FUND_ROOT / "tests" / "integration").glob("test_*.py")),
    ids=lambda path: path.name,
)
def test_every_integration_module_skips_without_its_own_scratch_url(
    module: Path,
) -> None:
    """No integration module may run on the ambient ``DATABASE_URL`` alone.

    That variable is the developer's live database whenever the repo-root ``.env``
    exists or the shell exports it for the daemon, so only an explicit
    ``FUND_TEST_DATABASE_URL`` may enable a module.
    """
    code, output = _run_integration_module(module, DATABASE_URL=_LIVE_URL)

    assert code == 0, output
    assert "skipped" in output, output
    assert "error" not in output.lower(), output
