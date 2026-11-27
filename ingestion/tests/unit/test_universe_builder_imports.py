"""The universe builder lives under a neutral package name, not a broker's."""

from __future__ import annotations

import importlib
import importlib.util

import pytest


@pytest.mark.criterion("REQ-ING-012")
def test_builder_package_has_no_trading212_module() -> None:
    """Validates that the builder imports from app.services.universe.builder and no
    broker-named package remains, in app/services/universe/builder/__init__.py."""
    assert importlib.util.find_spec("app.services.universe.trading212") is None

    builder = importlib.import_module("app.services.universe.builder")
    assert {"UniverseBuilder", "UniverseBuilderConfig", "classify_instrument"} <= set(
        builder.__all__
    )
