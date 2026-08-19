"""Gap-fill: webhook error branch is LOGGED, not silently dropped (issue #826).

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
