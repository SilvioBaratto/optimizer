"""Unit tests for ``news.scraper.ArticleScraper`` against a hostile local server.

Coverage target
---------------
* app.services.market_data.yfinance.news.scraper

Design
------
* A real ``ThreadingHTTPServer`` on 127.0.0.1 serves the hostile bodies, so the
  assertions hold for the actual ``requests`` streaming behaviour (gzip
  inflation, per-recv timeouts) instead of a mocked response object.
* ``delay=0`` on every scraper: the politeness sleep is irrelevant here.
"""

from __future__ import annotations

import gzip
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("bs4")

from app.services.market_data.yfinance.news.scraper import ArticleScraper

_EIGHT_MIB = 8 * 1024 * 1024
_DRIP_STEPS = 12
_DRIP_INTERVAL_S = 0.5
_STALL_S = 3.0


def _page(paragraphs: str) -> bytes:
    return f"<html><body><article>{paragraphs}</article></body></html>".encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        """Serve the hostile bodies keyed by request path."""
        if self.path == "/inflate":
            body = gzip.compress(_page("<p>" + "a" * _EIGHT_MIB + "</p>"))
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/drip":
            head = b"<html><body><article><p>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(head) + _DRIP_STEPS))
            self.end_headers()
            try:
                self.wfile.write(head)
                for _ in range(_DRIP_STEPS):
                    self.wfile.flush()
                    time.sleep(_DRIP_INTERVAL_S)
                    self.wfile.write(b"x")
            except OSError:
                return
        elif self.path == "/long-text":
            body = _page("".join(f"<p>{'word ' * 200}</p>" for _ in range(100)))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/control-chars":
            body = _page("<p>hello\x00world</p><p>\x00\x01</p><p>tab\there</p>")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/ok":
            body = gzip.compress(_page("<p>First.</p><p>Second.</p>"))
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/truncated":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"<html><body><article><p>cut off")
        elif self.path == "/bad-gzip":
            body = b"this is not a gzip stream"
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stall":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            try:
                self.wfile.write(b"<html><body><article><p>")
                self.wfile.flush()
                time.sleep(_STALL_S)
            except OSError:
                return
        else:
            self.send_error(404)


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    """Run the hostile server for the module and yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _bypass_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")


class TestWithinCaps:
    """Caps must not disturb a normal compressed article page."""

    def test_when_page_is_within_caps_then_paragraphs_are_extracted(
        self, server_url: str
    ) -> None:
        """A small gzip page is decoded and its paragraphs joined."""
        result = ArticleScraper(delay=0).fetch(f"{server_url}/ok")

        assert result["success"] is True
        assert result["content"] == "First.\n\nSecond."

    def test_when_server_returns_error_status_then_fetch_reports_failure(
        self, server_url: str
    ) -> None:
        """A 404 surfaces as a request failure, not as parsed content."""
        result = ArticleScraper(delay=0).fetch(f"{server_url}/missing")

        assert result["success"] is False
        assert "404" in (result["error"] or "")


class TestResponseSizeCap:
    """A small compressed body must not be inflated without bound."""

    def test_when_gzip_body_inflates_past_cap_then_fetch_fails_without_content(
        self, server_url: str
    ) -> None:
        """An 8 MiB inflated body against a 1 MB cap yields a too-large failure."""
        scraper = ArticleScraper(delay=0, max_bytes=1_000_000)

        result = scraper.fetch(f"{server_url}/inflate")

        assert result["success"] is False
        assert result["content"] is None
        assert "too large" in (result["error"] or "").lower()


class TestBodyReadFailures:
    """A body that breaks mid-stream is a request failure, not a parsing error."""

    def test_when_server_closes_before_content_length_then_failure_is_a_request_error(
        self, server_url: str
    ) -> None:
        """A truncated body is reported like any other transport failure."""
        result = ArticleScraper(delay=0).fetch(f"{server_url}/truncated")

        assert result["success"] is False
        assert (result["error"] or "").startswith("Request failed")

    def test_when_body_is_not_valid_gzip_then_failure_is_a_request_error(
        self, server_url: str
    ) -> None:
        """A body that cannot be decoded is a transport failure, not a parse error."""
        result = ArticleScraper(delay=0).fetch(f"{server_url}/bad-gzip")

        assert result["success"] is False
        assert (result["error"] or "").startswith("Request failed")

    def test_when_server_stalls_mid_body_then_failure_is_a_timeout(
        self, server_url: str
    ) -> None:
        """A per-recv stall is labelled like the wall-clock deadline, not parsing."""
        scraper = ArticleScraper(delay=0, timeout=1)

        started = time.monotonic()
        result = scraper.fetch(f"{server_url}/stall")
        elapsed = time.monotonic() - started

        assert result["success"] is False
        assert result["error"] == "Request timeout"
        assert elapsed < _STALL_S


class TestWallClockDeadline:
    """Per-recv timeouts alone never bound a server that keeps trickling."""

    def test_when_server_drips_slowly_then_fetch_gives_up_at_deadline(
        self, server_url: str
    ) -> None:
        """A byte every 0.5 s never trips the 5 s recv timeout but must hit 1 s."""
        scraper = ArticleScraper(delay=0, timeout=5, max_seconds=1.0)

        started = time.monotonic()
        result = scraper.fetch(f"{server_url}/drip")
        elapsed = time.monotonic() - started

        assert result["success"] is False
        assert elapsed < _DRIP_STEPS * _DRIP_INTERVAL_S / 2
        assert "timeout" in (result["error"] or "").lower()


class TestControlCharacters:
    """Text bound for Postgres must not carry characters it rejects."""

    def test_when_page_contains_nul_bytes_then_content_has_none(
        self, server_url: str
    ) -> None:
        """NUL and other C0 controls are dropped; tabs and newlines survive."""
        result = ArticleScraper(delay=0).fetch(f"{server_url}/control-chars")

        assert result["success"] is True
        assert result["content"] == "helloworld\n\ntab\there"


class TestTextCap:
    """The stored article text is bounded even when the page itself is small."""

    def test_when_extracted_text_exceeds_cap_then_content_is_truncated(
        self, server_url: str
    ) -> None:
        """Text beyond ``max_text_chars`` is cut so content_length matches."""
        scraper = ArticleScraper(delay=0, max_text_chars=1_000)

        result = scraper.fetch(f"{server_url}/long-text")

        assert result["success"] is True
        assert result["content"] is not None
        assert len(result["content"]) == 1_000
        assert result["content_length"] == 1_000
