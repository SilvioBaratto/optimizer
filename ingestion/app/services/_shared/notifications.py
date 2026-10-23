"""Webhook notification service for pipeline failure alerts.

The JSON body carries the message under both ``content`` (Discord) and
``text`` (Slack incoming webhooks); each service ignores the key it does not
use.

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

    A non-2xx answer (deleted webhook, rate limit, bad URL) is logged as a
    warning with its status code and never raised.

    Args:
        webhook_url: Full HTTPS URL of the incoming webhook endpoint.
        domain: Job type / pipeline domain label (e.g. ``"yfinance_fetch"``).
        job_id: UUID string of the failed job.
        error: Human-readable error message; truncated to 500 characters.
    """
    truncated = error[:_MAX_ERROR_LEN] if error else "(no error message)"
    message = (
        f"**Pipeline failure**: `{domain}` (job `{job_id[:8]}`)\n```{truncated}```"
    )
    payload = {"content": message, "text": message}
    try:
        response = httpx.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # httpx puts the full request URL in the exception text, and a webhook
        # URL embeds its secret token, so only the status code is logged.
        logger.warning(
            "Webhook notification rejected with HTTP %s", exc.response.status_code
        )
    except Exception as exc:
        # Notification is best-effort: a failed webhook is logged but must
        # never propagate and crash the job reporter that called us.
        logger.warning("Failed to send webhook notification: %s", exc)
