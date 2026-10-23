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
* The server lives on loopback, which the scraper refuses by default, so the
  tests that talk to it opt in with ``allow_private_hosts``. The destination
  guard itself is tested with the default scraper against a faked DNS and HTTP
  layer, where the resolved addresses are under the test's control.
"""

from __future__ import annotations

import gzip
import ipaddress
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import requests

pytest.importorskip("bs4")

from app.services.market_data.yfinance.news.scraper import ArticleScraper

_EIGHT_MIB = 8 * 1024 * 1024
_DRIP_STEPS = 12
_DRIP_INTERVAL_S = 0.5
_STALL_S = 3.0


def _page(paragraphs: str) -> bytes:
    return f"<html><body><article>{paragraphs}</article></body></html>".encode()


def _local_scraper(**kwargs: Any) -> ArticleScraper:
    """Scraper allowed to reach the loopback test server."""
    return ArticleScraper(delay=0, allow_private_hosts=True, **kwargs)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    hits: list[str] = []

    def do_GET(self) -> None:
        """Serve the hostile bodies keyed by request path."""
        type(self).hits.append(self.path)
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
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/cookie-hop":
            self.send_response(302)
            self.send_header("Set-Cookie", "consent=1; Path=/")
            self.send_header("Location", "/needs-cookie")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/needs-cookie":
            if "consent=1" not in (self.headers.get("Cookie") or ""):
                self.send_error(403)
                return
            body = _page("<p>Consented.</p>")
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
        result = _local_scraper().fetch(f"{server_url}/ok")

        assert result["success"] is True
        assert result["content"] == "First.\n\nSecond."

    def test_when_server_returns_error_status_then_fetch_reports_failure(
        self, server_url: str
    ) -> None:
        """A 404 surfaces as a request failure, not as parsed content."""
        result = _local_scraper().fetch(f"{server_url}/missing")

        assert result["success"] is False
        assert "404" in (result["error"] or "")


class TestResponseSizeCap:
    """A small compressed body must not be inflated without bound."""

    def test_when_gzip_body_inflates_past_cap_then_fetch_fails_without_content(
        self, server_url: str
    ) -> None:
        """An 8 MiB inflated body against a 1 MB cap yields a too-large failure."""
        scraper = _local_scraper(max_bytes=1_000_000)

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
        result = _local_scraper().fetch(f"{server_url}/truncated")

        assert result["success"] is False
        assert (result["error"] or "").startswith("Request failed")

    def test_when_body_is_not_valid_gzip_then_failure_is_a_request_error(
        self, server_url: str
    ) -> None:
        """A body that cannot be decoded is a transport failure, not a parse error."""
        result = _local_scraper().fetch(f"{server_url}/bad-gzip")

        assert result["success"] is False
        assert (result["error"] or "").startswith("Request failed")

    def test_when_server_stalls_mid_body_then_failure_is_a_timeout(
        self, server_url: str
    ) -> None:
        """A per-recv stall is labelled like the wall-clock deadline, not parsing."""
        scraper = _local_scraper(timeout=1)

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
        scraper = _local_scraper(timeout=5, max_seconds=1.0)

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
        result = _local_scraper().fetch(f"{server_url}/control-chars")

        assert result["success"] is True
        assert result["content"] == "helloworld\n\ntab\there"


class TestTextCap:
    """The stored article text is bounded even when the page itself is small."""

    def test_when_extracted_text_exceeds_cap_then_content_is_truncated(
        self, server_url: str
    ) -> None:
        """Text beyond ``max_text_chars`` is cut so content_length matches."""
        scraper = _local_scraper(max_text_chars=1_000)

        result = scraper.fetch(f"{server_url}/long-text")

        assert result["success"] is True
        assert result["content"] is not None
        assert len(result["content"]) == 1_000
        assert result["content_length"] == 1_000


class _Raw:
    """Stands in for the urllib3 response ``ArticleScraper`` reads from."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self._pos = 0

    def read1(self, size: int, decode_content: bool = False) -> bytes:
        chunk = self._body[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


class _FakeResponse:
    def __init__(
        self, status: int = 200, location: str | None = None, body: bytes = b""
    ) -> None:
        self.status_code = status
        self.headers = {"Location": location} if location else {}
        self.raw = _Raw(body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def close(self) -> None:
        return

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _FakeNetwork:
    """Offline DNS and HTTP so the destination guard sees chosen addresses."""

    def __init__(self) -> None:
        self.hosts: dict[str, list[str]] = {}
        self.routes: dict[str, _FakeResponse] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def getaddrinfo(self, host: str, port: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            ipaddress.ip_address(host)
            addresses = [host]
        except ValueError:
            if host not in self.hosts:
                raise socket.gaierror(f"unknown host {host}") from None
            addresses = self.hosts[host]
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (address, port or 0),
            )
            for address in addresses
        ]

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((url, kwargs))
        return self.routes[url]


@pytest.fixture
def fake_network(monkeypatch: pytest.MonkeyPatch) -> _FakeNetwork:
    network = _FakeNetwork()
    monkeypatch.setattr(socket, "getaddrinfo", network.getaddrinfo)
    monkeypatch.setattr(
        requests.Session, "get", lambda _session, url, **kw: network.get(url, **kw)
    )
    return network


_ARTICLE_BODY = _page("<p>Article text.</p>")


class TestDestinationGuard:
    """Feed-supplied links must not steer the daemon at internal services."""

    def test_when_url_targets_loopback_then_fetch_is_refused_without_a_request(
        self, server_url: str
    ) -> None:
        """A loopback literal is refused before any connection is made."""
        _Handler.hits.clear()

        result = ArticleScraper(delay=0).fetch(f"{server_url}/ok")

        assert result["success"] is False
        assert "blocked" in (result["error"] or "").lower()
        assert _Handler.hits == []

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://news.example/story",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/admin",
            "http://172.16.4.1/",
            "http://192.168.1.1/",
            "http://0.0.0.0/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[fe80::1]/",
            "http://127.0.0.1:8000\\@93.184.216.34/",
            "http://[::1]\\@93.184.216.34/",
        ],
    )
    def test_when_url_is_not_public_http_then_fetch_is_refused(
        self, fake_network: _FakeNetwork, url: str
    ) -> None:
        """Non-http schemes and private, link-local or mapped addresses are blocked."""
        result = ArticleScraper(delay=0).fetch(url)

        assert result["success"] is False
        assert "blocked" in (result["error"] or "").lower()
        assert fake_network.calls == []

    def test_when_host_resolves_to_private_address_then_fetch_is_refused(
        self, fake_network: _FakeNetwork
    ) -> None:
        """A public-looking name that resolves to an internal address is blocked."""
        fake_network.hosts["adminer.internal.example"] = ["172.18.0.3"]

        result = ArticleScraper(delay=0).fetch("http://adminer.internal.example/")

        assert result["success"] is False
        assert fake_network.calls == []

    def test_when_any_resolved_address_is_private_then_fetch_is_refused(
        self, fake_network: _FakeNetwork
    ) -> None:
        """One internal record among public ones is enough to refuse the host."""
        fake_network.hosts["mixed.example"] = ["93.184.216.34", "10.1.2.3"]

        result = ArticleScraper(delay=0).fetch("http://mixed.example/")

        assert result["success"] is False
        assert fake_network.calls == []

    def test_when_host_does_not_resolve_then_fetch_fails_without_a_request(
        self, fake_network: _FakeNetwork
    ) -> None:
        """An unresolvable name is a failed fetch, not an unchecked connection."""
        result = ArticleScraper(delay=0).fetch("http://nowhere.example/")

        assert result["success"] is False
        assert fake_network.calls == []

    def test_when_redirect_targets_private_host_then_it_is_not_followed(
        self, fake_network: _FakeNetwork
    ) -> None:
        """Every hop is validated, so a public page cannot bounce into the VPC."""
        fake_network.hosts["news.example"] = ["93.184.216.34"]
        fake_network.routes["http://news.example/story"] = _FakeResponse(
            302, location="http://169.254.169.254/latest/meta-data/"
        )

        result = ArticleScraper(delay=0).fetch("http://news.example/story")

        assert result["success"] is False
        assert [url for url, _ in fake_network.calls] == ["http://news.example/story"]

    def test_when_backslash_hides_private_redirect_target_then_it_is_not_followed(
        self, fake_network: _FakeNetwork
    ) -> None:
        """A hop whose ``\\`` makes parsers disagree on the host is still checked."""
        fake_network.hosts["news.example"] = ["93.184.216.34"]
        fake_network.routes["http://news.example/story"] = _FakeResponse(
            302, location="http://127.0.0.1:8000\\@93.184.216.34/latest/meta-data/"
        )

        result = ArticleScraper(delay=0).fetch("http://news.example/story")

        assert result["success"] is False
        assert "blocked" in (result["error"] or "").lower()
        assert [url for url, _ in fake_network.calls] == ["http://news.example/story"]

    def test_when_backslash_hides_loopback_then_no_request_reaches_the_server(
        self, server_url: str
    ) -> None:
        """A ``\\`` makes urlsplit read the public host while urllib3 dials loopback."""
        _Handler.hits.clear()

        result = ArticleScraper(delay=0).fetch(f"{server_url}\\@93.184.216.34/")

        assert result["success"] is False
        assert "blocked" in (result["error"] or "").lower()
        assert _Handler.hits == []

    def test_when_page_redirects_then_requests_never_follows_it_on_its_own(
        self, fake_network: _FakeNetwork
    ) -> None:
        """Automatic redirects stay off so no hop escapes validation."""
        fake_network.hosts["news.example"] = ["93.184.216.34"]
        fake_network.routes["http://news.example/story"] = _FakeResponse(
            200, body=_ARTICLE_BODY
        )

        ArticleScraper(delay=0).fetch("http://news.example/story")

        assert all(kw.get("allow_redirects") is False for _, kw in fake_network.calls)

    def test_when_public_redirect_is_followed_then_final_page_is_extracted(
        self, fake_network: _FakeNetwork
    ) -> None:
        """A relative redirect between public pages still yields the article."""
        fake_network.hosts["news.example"] = ["93.184.216.34"]
        fake_network.routes["http://news.example/story"] = _FakeResponse(
            301, location="/final"
        )
        fake_network.routes["http://news.example/final"] = _FakeResponse(
            200, body=_ARTICLE_BODY
        )

        result = ArticleScraper(delay=0).fetch("http://news.example/story")

        assert result["success"] is True
        assert result["content"] == "Article text."

    def test_when_redirect_chain_exceeds_cap_then_fetch_fails(
        self, fake_network: _FakeNetwork
    ) -> None:
        """A redirect loop stops after ``max_redirects`` hops."""
        fake_network.hosts["news.example"] = ["93.184.216.34"]
        fake_network.routes["http://news.example/loop"] = _FakeResponse(
            302, location="/loop"
        )

        result = ArticleScraper(delay=0, max_redirects=3).fetch(
            "http://news.example/loop"
        )

        assert result["success"] is False
        assert "redirect" in (result["error"] or "").lower()
        assert len(fake_network.calls) == 4


class TestRedirectsOverTheWire:
    """Manual redirect handling against real ``requests`` behaviour."""

    def test_when_server_redirects_with_relative_location_then_target_is_fetched(
        self, server_url: str
    ) -> None:
        """A relative ``Location`` resolves against the current URL."""
        result = _local_scraper().fetch(f"{server_url}/redirect")

        assert result["success"] is True
        assert result["content"] == "First.\n\nSecond."

    def test_when_server_redirects_forever_then_hops_are_capped(
        self, server_url: str
    ) -> None:
        """A self-redirecting URL costs ``max_redirects`` + 1 requests, no more."""
        _Handler.hits.clear()

        result = _local_scraper(max_redirects=2).fetch(f"{server_url}/loop")

        assert result["success"] is False
        assert _Handler.hits == ["/loop"] * 3

    def test_when_redirect_sets_cookie_then_next_hop_receives_it(
        self, server_url: str
    ) -> None:
        """Consent and cookie-check hops need the cookie set by the redirecting page."""
        result = _local_scraper().fetch(f"{server_url}/cookie-hop")

        assert result["success"] is True
        assert result["content"] == "Consented."

    def test_when_cookie_was_set_by_earlier_fetch_then_later_fetch_does_not_send_it(
        self, server_url: str
    ) -> None:
        """Cookies live for one fetch, so one article cannot influence the next."""
        scraper = _local_scraper()
        scraper.fetch(f"{server_url}/cookie-hop")

        result = scraper.fetch(f"{server_url}/needs-cookie")

        assert result["success"] is False
        assert "403" in (result["error"] or "")
