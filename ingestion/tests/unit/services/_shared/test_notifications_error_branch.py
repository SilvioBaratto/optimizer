"""Gap-fill: webhook error branch is LOGGED, not silently dropped.

``test_notifications.py`` already asserts the network exception is swallowed
(no raise). The missing branch is the 'no silent drop' guarantee: the failure
must surface in the logs. This file asserts the WARNING is emitted.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import httpx
import pytest

from app.services._shared.notifications import notify_failure


class TestWebhookErrorIsLogged:
    def test_when_post_raises_then_warning_logged_and_not_reraised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            caplog.at_level(logging.WARNING),
        ):
            notify_failure(
                "https://example.com/hook", "yfinance_fetch", "job-1", "boom"
            )

        assert any(
            "Failed to send webhook notification" in r.message for r in caplog.records
        )

    def test_when_timeout_then_swallowed_no_raise(self) -> None:
        with patch("httpx.post", side_effect=httpx.TimeoutException("slow")):
            notify_failure("https://example.com/hook", "d", "job-1", "boom")  # no raise


class TestWebhookRejectionIsLogged:
    @pytest.mark.parametrize("status", [401, 404, 429, 500])
    def test_when_webhook_answers_non_2xx_then_warning_logged_without_secret(
        self, status: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        url = "https://discord.com/api/webhooks/123/SECRET-TOKEN"
        response = httpx.Response(status, request=httpx.Request("POST", url))

        with (
            patch("httpx.post", return_value=response),
            caplog.at_level(logging.WARNING),
        ):
            notify_failure(url, "yfinance_fetch", "job-1", "boom")

        assert any(str(status) in r.getMessage() for r in caplog.records)
        assert all("SECRET-TOKEN" not in r.getMessage() for r in caplog.records)

    def test_when_webhook_answers_2xx_then_nothing_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        url = "https://example.com/hook"
        response = httpx.Response(204, request=httpx.Request("POST", url))

        with (
            patch("httpx.post", return_value=response),
            caplog.at_level(logging.WARNING),
        ):
            notify_failure(url, "d", "job-1", "boom")

        assert caplog.records == []
