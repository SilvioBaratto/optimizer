"""Tests for ApiClient.start_fetch — verifies workers field name (issue #577)."""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("typer")

from cli.client import ApiClient


def _client_with_capture() -> tuple[ApiClient, list[httpx.Request]]:
    """Build an ApiClient backed by a MockTransport that records requests."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"job_id": "abc-123"})

    client = ApiClient(base_url="http://test")
    client._client = httpx.Client(
        base_url=client._base_url + ApiClient.API_PREFIX,
        transport=httpx.MockTransport(handler),
    )
    return client, captured


class TestStartFetchPayload:
    def test_when_called_then_json_body_uses_workers_key(self) -> None:
        client, captured = _client_with_capture()

        client.start_fetch(workers=8, period="2y", mode="full")

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        assert body["workers"] == 8

    def test_when_called_then_legacy_max_workers_key_absent(self) -> None:
        client, captured = _client_with_capture()

        client.start_fetch(workers=4)

        body = json.loads(captured[0].content)
        assert "max_workers" not in body

    def test_when_called_then_period_and_mode_forwarded(self) -> None:
        client, captured = _client_with_capture()

        client.start_fetch(workers=2, period="1y", mode="incremental")

        body = json.loads(captured[0].content)
        assert body == {"workers": 2, "period": "1y", "mode": "incremental"}

    def test_when_called_with_defaults_then_workers_is_one(self) -> None:
        client, captured = _client_with_capture()

        client.start_fetch()

        body = json.loads(captured[0].content)
        assert body["workers"] == 1

    def test_when_called_then_endpoint_is_yfinance_fetch(self) -> None:
        client, captured = _client_with_capture()

        client.start_fetch(workers=1)

        assert captured[0].url.path == "/api/v1/yfinance-data/fetch"
        assert captured[0].method == "POST"


class TestCliFetchSurface:
    """Structural guards that the yfinance CLI surface no longer references max_workers.

    The universe build path (cli/client.py:build_universe) intentionally still uses
    max_workers because the trading212 universe-build schema retains that field name.
    """

    def test_when_yfinance_module_inspected_then_no_max_workers_reference(self) -> None:
        from pathlib import Path

        yfinance_text = (
            Path(__file__).resolve().parents[2] / "cli" / "yfinance.py"
        ).read_text()
        assert "max_workers" not in yfinance_text

    def test_when_start_fetch_block_inspected_then_no_max_workers(self) -> None:
        from pathlib import Path

        client_text = (
            Path(__file__).resolve().parents[2] / "cli" / "client.py"
        ).read_text()
        start_idx = client_text.index("def start_fetch(")
        end_idx = client_text.index("\n    def ", start_idx + 1)
        assert "max_workers" not in client_text[start_idx:end_idx]
