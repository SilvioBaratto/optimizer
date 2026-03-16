"""Unit tests for the webhook notification service."""

from __future__ import annotations

from unittest.mock import patch

import httpx

from app.services.notifications import notify_failure


class TestNotifyFailure:
    def test_posts_to_webhook_url(self) -> None:
        with patch("app.services.notifications.httpx.post") as mock_post:
            notify_failure(
                "https://discord.com/api/webhooks/test",
                "yfinance_fetch",
                "abcdef12-0000-0000-0000-000000000000",
                "Connection timeout",
            )
        mock_post.assert_called_once()
        (url,) = mock_post.call_args.args
        assert url == "https://discord.com/api/webhooks/test"

    def test_payload_contains_domain_and_short_job_id(self) -> None:
        with patch("app.services.notifications.httpx.post") as mock_post:
            notify_failure(
                "https://example.com/hook",
                "macro_fetch",
                "abcdef12-1234-5678-9abc-def012345678",
                "Some error",
            )
        payload = mock_post.call_args.kwargs["json"]
        assert "macro_fetch" in payload["content"]
        assert "abcdef12" in payload["content"]

    def test_error_truncated_at_500_chars(self) -> None:
        long_error = "x" * 600
        with patch("app.services.notifications.httpx.post") as mock_post:
            notify_failure("https://example.com/hook", "d", "job-id-000", long_error)
        payload = mock_post.call_args.kwargs["json"]
        assert "x" * 501 not in payload["content"]
        assert "x" * 500 in payload["content"]

    def test_empty_error_shows_fallback_message(self) -> None:
        with patch("app.services.notifications.httpx.post") as mock_post:
            notify_failure("https://example.com/hook", "d", "job-id", "")
        payload = mock_post.call_args.kwargs["json"]
        assert "(no error message)" in payload["content"]

    def test_network_exception_is_swallowed(self) -> None:
        with patch(
            "app.services.notifications.httpx.post",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            notify_failure("https://example.com/hook", "d", "job-id", "err")

    def test_timeout_kwarg_is_10_seconds(self) -> None:
        with patch("app.services.notifications.httpx.post") as mock_post:
            notify_failure("https://example.com/hook", "d", "job-id", "err")
        assert mock_post.call_args.kwargs["timeout"] == 10
