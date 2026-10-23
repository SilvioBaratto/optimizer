"""Credentials in URL query strings must not survive into log or error text."""

from __future__ import annotations

import http.server
import io
import logging
import threading
from unittest.mock import patch

import pytest

from app.services.macro.scrapers.fred_scraper import FredScraper
from app.utils.log_redaction import (
    SecretRedactingFilter,
    install_secret_redaction,
    redact_secrets,
)


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


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b'{"observations": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None


def test_fred_scraper_request_line_is_masked_in_urllib3_debug_log() -> None:
    """The real FRED client path: urllib3 DEBUG echoes the request URL."""
    leaked = "FREDSECRETKEY0123456789abcdef"
    server = http.server.HTTPServer(("127.0.0.1", 0), _OkHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    urllib3_logger = logging.getLogger("urllib3")
    previous_level = urllib3_logger.level
    root.addHandler(handler)
    urllib3_logger.setLevel(logging.DEBUG)
    try:
        install_secret_redaction()
        with patch(
            "app.services.macro.scrapers.fred_scraper._FRED_BASE_URL",
            f"http://127.0.0.1:{server.server_port}",
        ):
            scraper = FredScraper(api_key=leaked)
            scraper.session.trust_env = False
            result = scraper.get_series_observations("T10Y2Y")
    finally:
        urllib3_logger.setLevel(previous_level)
        root.removeHandler(handler)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["status"] == "success"
    logged = stream.getvalue()
    assert "GET /series/observations?" in logged
    assert "api_key=***" in logged
    assert leaked not in logged


def test_redacting_filter_never_breaks_a_malformed_log_call() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s %s", ("only",), None)
    assert SecretRedactingFilter().filter(record) is True
