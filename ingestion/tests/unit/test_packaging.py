"""Packaging contract for the ``portopt`` distribution (SPEC D2/D7, task T1).

The ingestion daemon becomes the ``portopt`` console app; the optimizer library
is renamed ``portopt-core``. These tests pin that contract so a future edit to
either ``pyproject.toml`` that breaks the console script or drops a wizard
runtime dependency fails loudly.
"""

import importlib.metadata as im
import importlib.util

import pytest


def test_library_distribution_renamed_to_portopt_core() -> None:
    # Root optimizer library ships as ``portopt-core``, freeing ``portopt`` for the CLI.
    assert im.version("portopt-core")


def test_portopt_console_script_targets_cli_app() -> None:
    scripts = {ep.name: ep.value for ep in im.entry_points(group="console_scripts")}
    assert scripts.get("portopt") == "app.cli:app"


@pytest.mark.parametrize("pkg", ["questionary", "rich", "cryptography"])
def test_cli_runtime_dependencies_importable(pkg: str) -> None:
    assert importlib.util.find_spec(pkg) is not None
