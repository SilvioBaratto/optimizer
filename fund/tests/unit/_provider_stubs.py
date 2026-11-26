"""Shared, fully-local plumbing for the provider tests that talk to a stub.

The model tests build the real provider SDK classes and aim them at an HTTP server
on the loopback interface, so a changed SDK default shows up where it crosses the
wire. Each provider file keeps its own request recording and reply body, because the
OpenAI, Anthropic, Groq, Gemini and Ollama shapes differ. What does not differ lives
here: the guard that keeps a stray request off the network, and the lifecycle of the
stub server.

Not a ``test_*`` module, so pytest does not collect it; the provider test files
import the helpers from here.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

# The bind address, the NO_PROXY exemption and the base URL a test hands the SDK must
# all name the same host, or a request goes through a proxy or past the stub.
STUB_HOST = "127.0.0.1"
LOOPBACK_HOSTS = frozenset({STUB_HOST, "::1", "localhost"})

_EPHEMERAL_PORT = 0
_SERVER_JOIN_SECONDS = 5


def loopback_only(
    monkeypatch: pytest.MonkeyPatch, *, refuse_lookups: bool = False
) -> list[str]:
    """Fail the test on any connection, and optionally lookup, beyond loopback.

    The refusal is ``pytest.fail``, not an ``OSError``: an SDK's retry code catches
    ``Exception`` and would turn a refused real request into an ordinary error that a
    test could then pass over. Each refusal is also appended to the returned list
    before it fails, for a test that wants to show that none happened, or to see one
    that a background thread swallowed.

    Args:
        monkeypatch: The test's monkeypatch, which restores the socket afterwards.
        refuse_lookups: Also refuse a name lookup for a host outside loopback. Off by
            default because a lookup alone reaches no remote host and most callers
            only care about the connection.

    Returns:
        The refusals so far, as ``"connect <host>"`` or ``"lookup <host>"``.
    """
    refused: list[str] = []
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def connect(sock: socket.socket, address: Any) -> None:
        """Connect over loopback only; record the refusal and fail on any other host."""
        # A non-tuple address is a unix-domain path, which cannot leave the machine.
        host = address[0] if isinstance(address, tuple) else "localhost"
        if host not in LOOPBACK_HOSTS:
            refused.append(f"connect {host}")
            pytest.fail(f"a unit test opened a connection to {host}")
        real_connect(sock, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        """Resolve a loopback name only; record the refusal and fail on any other."""
        if host is not None and host not in LOOPBACK_HOSTS:
            refused.append(f"lookup {host}")
            pytest.fail(f"a unit test looked up the host {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    if refuse_lookups:
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return refused


def loopback_url(port: int) -> str:
    """Return the base URL of a stub that ``serve_on_loopback`` serves on ``port``."""
    return f"http://{STUB_HOST}:{port}"


@contextmanager
def serve_on_loopback(
    monkeypatch: pytest.MonkeyPatch, handler: type[BaseHTTPRequestHandler]
) -> Iterator[int]:
    """Serve ``handler`` on a free loopback port for the length of the ``with`` block.

    Also exempts the loopback host from any ambient HTTP proxy, which would otherwise
    take the request and answer it from outside the test.

    Args:
        monkeypatch: The test's monkeypatch, which restores ``NO_PROXY`` afterwards.
        handler: The request handler class holding the provider's own recording and
            reply.

    Yields:
        The port the stub listens on; ``loopback_url`` turns it into a base URL.
    """
    monkeypatch.setenv("NO_PROXY", STUB_HOST)
    server = ThreadingHTTPServer((STUB_HOST, _EPHEMERAL_PORT), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=_SERVER_JOIN_SECONDS)
