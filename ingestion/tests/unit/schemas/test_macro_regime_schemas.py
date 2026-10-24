"""Contract tests for the macro request schemas in app/schemas/macro/macro_regime.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from app.schemas.macro.macro_regime import MacroFetchRequest


class _ReadRecorder:
    """Proxy that records which attributes of the wrapped request are read."""

    def __init__(self, request: Any) -> None:
        self._request = request
        self.read: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        self.read.add(name)
        return getattr(self._request, name)


class TestMacroFetchRequestContract:
    """A request field must change what the bulk fetch does, or not exist."""

    def test_every_request_field_is_honoured_by_bulk_macro_fetch(self) -> None:
        """Fails when the schema advertises a toggle the fetch never reads."""
        from app.services.macro.macro_regime_service import run_bulk_macro_fetch

        recorder = _ReadRecorder(MacroFetchRequest(countries=["USA"]))
        mock_svc = MagicMock()
        mock_svc.fetch_country.return_value = {"counts": {}, "errors": []}

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = MagicMock()
            mock_db.get_session.return_value.__exit__.return_value = False

            run_bulk_macro_fetch(recorder)

        ignored = set(MacroFetchRequest.model_fields) - recorder.read
        assert not ignored, f"request fields never read by the fetch: {sorted(ignored)}"
