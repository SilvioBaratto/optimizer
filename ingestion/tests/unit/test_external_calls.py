"""The external-call breakers cover only the sources the daemon still calls."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.services.infrastructure.external_calls import ExternalService

_REMOVED_SERVICE_MEMBERS = ("TRADING_ECONOMICS", "IL_SOLE")
_REMOVED_SETTING_MARKERS = ("tradingeconomics", "ilsole")


@pytest.mark.criterion("REQ-ING-024")
def test_no_breaker_for_removed_scrapers() -> None:
    """Validates ``ExternalService`` and ``Settings``: no breaker, attempt count or
    timeout setting remains for a removed scraper."""
    members = {service.name for service in ExternalService}
    leftover_settings = [
        field
        for field in Settings.model_fields
        if any(marker in field for marker in _REMOVED_SETTING_MARKERS)
    ]

    assert members.isdisjoint(_REMOVED_SERVICE_MEMBERS)
    assert leftover_settings == []
