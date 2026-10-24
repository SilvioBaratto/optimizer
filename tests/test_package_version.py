"""Guard: ``optimizer.__version__`` must come from the ``portopt-core`` dist.

The ``optimizer`` import package ships in the ``portopt-core`` distribution. The
ingestion daemon is a different dist (``portopt``) that shares the workspace venv,
so reading its metadata reports the wrong number and, in a standalone
``portopt-core`` install, falls back to the dev placeholder.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata

import pytest

import optimizer


@pytest.fixture
def reload_optimizer(monkeypatch: pytest.MonkeyPatch):
    """Re-execute ``optimizer/__init__`` against a fake dist-metadata table."""

    def _reload(installed: dict[str, str]) -> str:
        def fake_version(name: str) -> str:
            try:
                return installed[name]
            except KeyError:
                raise importlib_metadata.PackageNotFoundError(name) from None

        monkeypatch.setattr(importlib_metadata, "version", fake_version)
        return importlib.reload(optimizer).__version__

    yield _reload
    monkeypatch.undo()
    importlib.reload(optimizer)


def test_version_comes_from_portopt_core_dist(reload_optimizer):
    version = reload_optimizer({"portopt-core": "9.9.9", "portopt": "1.2.3"})

    assert version == "9.9.9"


def test_version_resolves_when_only_portopt_core_is_installed(reload_optimizer):
    version = reload_optimizer({"portopt-core": "4.5.6"})

    assert version == "4.5.6"


def test_version_falls_back_when_portopt_core_is_not_installed(reload_optimizer):
    version = reload_optimizer({"portopt": "1.2.3"})

    assert version == "0.0.0.dev0"
