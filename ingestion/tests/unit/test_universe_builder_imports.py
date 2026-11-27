"""The universe builder lives under a neutral package name, not a broker's."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REMOVED_PACKAGE = "app.services.universe.trading212"


def _source_files_of(name: str) -> list[Path]:
    """Return the Python source files that would back module ``name``.

    A checkout that once held the package keeps its untracked ``__pycache__``
    directory after git deletes the sources, and that directory alone still
    resolves as a namespace package; only a real ``.py`` file counts as present.
    """
    spec = importlib.util.find_spec(name)
    if spec is None:
        return []
    files = {Path(spec.origin)} if spec.origin else set()
    for location in spec.submodule_search_locations or ():
        files.update(Path(location).rglob("*.py"))
    return sorted(files)


@pytest.mark.criterion("REQ-ING-012")
def test_builder_package_has_no_trading212_module() -> None:
    """Validates ``app.services.universe.builder.__all__`` exports the builder API and
    no ``app.services.universe.trading212`` source module remains."""
    assert _source_files_of(_REMOVED_PACKAGE) == []
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{_REMOVED_PACKAGE}.client")

    builder = importlib.import_module("app.services.universe.builder")
    assert {"UniverseBuilder", "UniverseBuilderConfig", "classify_instrument"} <= set(
        builder.__all__
    )
