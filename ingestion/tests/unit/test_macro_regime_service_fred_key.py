"""Tests for FRED_API_KEY environment-aware messaging (issue #564)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from app.services.macro import macro_regime_service as svc_mod
from app.services.macro.macro_regime_service import MacroRegimeService


def _build_service() -> MacroRegimeService:
    return MacroRegimeService(
        repo=MagicMock(),
        ilsole_scraper=MagicMock(),
        te_scraper=MagicMock(),
        fred_scraper=None,
    )


class TestFredKeyEnvironmentAwareMessage:
    """fred_scraper property + fetch_fred_series surface a docker-aware diagnostic."""

    def test_when_key_empty_property_logs_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "fred_api_key", "", raising=False)
        svc = _build_service()
        with caplog.at_level(logging.WARNING, logger=svc_mod.__name__):
            assert svc.fred_scraper is None
        assert any(
            "docker exec" in rec.message and "settings.fred_api_key" in rec.message
            for rec in caplog.records
        )

    def test_when_key_empty_fetch_returns_environment_aware_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "fred_api_key", "", raising=False)
        svc = _build_service()
        result = svc.fetch_fred_series()
        assert result["counts"] == {}
        assert len(result["errors"]) == 1
        msg = result["errors"][0]
        assert "settings.fred_api_key=" in msg
        assert "docker exec" in msg

    def test_when_key_present_property_does_not_warn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "fred_api_key", "real-key", raising=False)
        fake_scraper_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(svc_mod, "FredScraper", fake_scraper_cls)
        svc = _build_service()
        with caplog.at_level(logging.WARNING, logger=svc_mod.__name__):
            scraper = svc.fred_scraper
        assert scraper is not None
        fake_scraper_cls.assert_called_once_with(api_key="real-key")
        assert not any("docker exec" in rec.message for rec in caplog.records)
