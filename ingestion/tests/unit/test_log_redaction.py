"""Credentials in URL query strings must not survive into log or error text."""

from __future__ import annotations

import pytest

from app.utils.log_redaction import redact_secrets


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "GET /fred/series?series_id=GNPCA&api_key=S3CRET&file_type=json",
            "GET /fred/series?series_id=GNPCA&api_key=***&file_type=json",
        ),
        (
            "GET https://x.example/v1beta/models?key=AIzaS3CRET HTTP/1.1",
            "GET https://x.example/v1beta/models?key=*** HTTP/1.1",
        ),
        ("/path?a=1&token=S3CRET&b=2", "/path?a=1&token=***&b=2"),
        ("/path?access_token=S3CRET", "/path?access_token=***"),
        ("url: /x?API_KEY=S3CRET)", "url: /x?API_KEY=***)"),
        ("(Caused by 'api-key=S3CRET')", "(Caused by 'api-key=***')"),
    ],
)
def test_redact_secrets_masks_credential_query_params(text: str, expected: str) -> None:
    assert redact_secrets(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no credentials here",
        "cache key=AAPL hit",
        "series_id=GNPCA&file_type=json",
        "monkey=banana",
    ],
)
def test_redact_secrets_leaves_other_text_untouched(text: str) -> None:
    assert redact_secrets(text) == text
