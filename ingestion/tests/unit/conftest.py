"""Unit-suite fixtures shared across modules."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_checkout(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Pin ``OPTIMIZER_REPO`` to a throwaway checkout for every unit test.

    ``path_install.locate_repo`` falls back to ``repo_path`` in the developer's
    ``~/.portopt/config.toml`` and then to a cwd walk-up, so a test that reaches
    ``lifecycle.cleanup_rendered`` or ``docker_bootstrap.compose_dir`` without
    patching them would resolve, and delete the rendered plaintext secrets of, the
    real checkout. The env var outranks both fallbacks; tests that exercise the
    resolution order still override it with their own ``monkeypatch`` calls.
    """
    repo = tmp_path_factory.mktemp("isolated-checkout")
    (repo / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("OPTIMIZER_REPO", str(repo))
    return repo
