"""Webhook notification service for pipeline failure alerts.

Supports any service that accepts a Discord-compatible JSON payload
(Discord webhooks, Slack incoming webhooks via compatibility mode, etc.).

This module has no imports from the rest of the application so it can
be imported anywhere without circular-import risk.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

_MAX_ERROR_LEN = 500


def notify_failure(
    webhook_url: str,
    domain: str,
    job_id: str,
    error: str,
) -> None:
    """POST a failure notification to a Discord/Slack-compatible webhook.

    Args:
        webhook_url: Full HTTPS URL of the incoming webhook endpoint.
        domain: Job type / pipeline domain label (e.g. ``"yfinance_fetch"``).
        job_id: UUID string of the failed job.
        error: Human-readable error message; truncated to 500 characters.
    """
    truncated = error[:_MAX_ERROR_LEN] if error else "(no error message)"
    payload = {
        "content": (
            f"**Pipeline failure**: `{domain}` (job `{job_id[:8]}`)\n"
            f"```{truncated}```"
        ),
    }
    try:
        httpx.post(webhook_url, json=payload, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to send webhook notification: %s", exc)
