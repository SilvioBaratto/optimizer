"""The unit suite must never resolve (and wipe) the developer's real checkout.

``lifecycle.cleanup_rendered`` deletes ``<checkout>/secrets/*`` and ``.env.fund``,
and the checkout comes from ``path_install.locate_repo`` (``OPTIMIZER_REPO``, then
``repo_path`` in ``~/.portopt/config.toml``, then a cwd walk-up). A test that fails
the ``portopt setup`` path without patching that resolution would otherwise delete
the plaintext secrets the running scheduler and fund bind-mount.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.setup import path_install
from app.setup.wizard import SetupError

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]

runner = CliRunner()


def test_checkout_resolution_lands_in_a_throwaway_tree(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Unpatched resolution must never reach the real checkout the tests run in."""
    repo = path_install.locate_repo()
    assert repo is not None
    assert repo.resolve() != _REPO_ROOT
    assert repo.resolve().is_relative_to(tmp_path_factory.getbasetemp().resolve())


def test_failed_setup_never_wipes_the_checkout_found_from_the_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cwd walk-up is the last resolution step, so a test that runs from inside
    a checkout (as ``pytest`` does from the repo) must not reach it by default."""
    real = tmp_path / "real-checkout"
    (real / "secrets").mkdir(parents=True)
    (real / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    secret = real / "secrets" / "fred_api_key"
    secret.write_text("live", encoding="utf-8")
    env_fund = real / ".env.fund"
    env_fund.write_text("LLM_PROVIDER=x\n", encoding="utf-8")
    monkeypatch.chdir(real)

    with (
        patch("app.cli.init_db"),
        patch(
            "app.setup.wizard.run_setup_noninteractive",
            side_effect=SetupError("bad key"),
        ),
    ):
        result = runner.invoke(app, ["setup", "--non-interactive", "--fred-key", "x"])

    assert result.exit_code == 1
    assert secret.read_text(encoding="utf-8") == "live"
    assert env_fund.exists()
