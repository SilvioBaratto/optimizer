"""What the Anthropic chat model puts on the wire, through the real SDK.

Validates ``fund.agents.model._build_anthropic``.

``test_model_factory`` drives the builder with a recording fake and
``test_model_anthropic_secret_not_in_repr`` builds the real class but never calls it,
so neither notices a change in what the ``anthropic`` SDK sends. Its 2.x line moved
the API key from ``X-Api-Key`` to ``Authorization: Bearer``, defaults the
``anthropic-version`` header to a version that answers a ``temperature`` with HTTP 400
(``langchain-anthropic`` overrides it), and adds OpenTelemetry spans and trace-context
headers. These tests build the real classes and talk to a localhost stub, so each of
those is checked where it crosses the wire. Nothing leaves the machine: a connection
to any other host is refused and recorded.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from langchain_core.load import dumpd, dumps

from fund.agents import model
from fund.config import FundConfig

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("langchain_anthropic")

_KEY = "sk-ant-S3CRET-wire"
# Not a model the SDK lists as deprecated: it warns on a request that names one.
_MODEL = "claude-sonnet-4-6"
_SERVER_JOIN_SECONDS = 5
_STUB_HOST = "127.0.0.1"
_EPHEMERAL_PORT = 0

# The first API version that answers a request carrying ``temperature``, ``top_p`` or
# ``top_k`` with HTTP 400. ISO dates order the same as strings.
_FIRST_VERSION_WITHOUT_SAMPLING = "2026-10-21"

_TRACE_CONTEXT_HEADERS = ("traceparent", "tracestate", "baggage")
_LOOPBACK_HOSTS = frozenset({_STUB_HOST, "::1", "localhost"})

# Every variable the SDK or langchain-anthropic would read in place of a value the
# builder did not pass, so a developer's shell cannot change what is asserted. The
# tracing variables are here for a second reason: with one set, each call to the stub
# would also send a trace to LangSmith.
_AMBIENT_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_URL",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_PROXY",
    "ANTHROPIC_OPEN_TELEMETRY",
    "ANTHROPIC_OPEN_TELEMETRY_PROPAGATION",
    "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE",
    "OTEL_PYTHON_TRACER_PROVIDER",
    "OTEL_PROPAGATORS",
    "LANGSMITH_GATEWAY",
    "LANGSMITH_GATEWAY_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

_MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": _MODEL,
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_AUTH_ERROR: dict[str, Any] = {
    "type": "error",
    "error": {"type": "authentication_error", "message": "invalid credentials"},
}


@dataclass(frozen=True)
class _RecordedRequest:
    """One request the stub received, header names lower-cased."""

    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class _Stub:
    """A localhost Messages endpoint and what it was asked."""

    base_url: str
    requests: list[_RecordedRequest] = field(default_factory=list)
    status: HTTPStatus = HTTPStatus.OK


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AMBIENT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", _STUB_HOST)


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Refuse and record every connection or lookup that is not the loopback stub."""
    refused: list[str] = []
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def connect(self: socket.socket, address: Any) -> None:
        """Connect to the loopback stub; record and refuse any other host."""
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK_HOSTS:
            refused.append(f"connect {host}")
            raise OSError("only the loopback stub may be reached")
        real_connect(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        """Resolve a loopback name; record and refuse any other lookup."""
        if host is not None and host not in _LOOPBACK_HOSTS:
            refused.append(f"lookup {host}")
            raise socket.gaierror("only the loopback stub may be resolved")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return refused


def _recording_handler(state: _Stub) -> type[BaseHTTPRequestHandler]:
    """Build a request handler that records each POST on ``state`` and answers it."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            """Record the request and answer it with the stub's canned reply."""
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            headers = {name.lower(): value for name, value in self.headers.items()}
            state.requests.append(_RecordedRequest(self.path, headers, json.loads(raw)))
            reply = _MESSAGE if state.status == HTTPStatus.OK else _AUTH_ERROR
            payload = json.dumps(reply).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            """Keep the test output free of per-request access lines."""

    return _Handler


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Stub]:
    """A localhost Anthropic-style ``/v1/messages`` endpoint, set as the base URL."""
    state = _Stub(base_url="")
    server = ThreadingHTTPServer(
        (_STUB_HOST, _EPHEMERAL_PORT), _recording_handler(state)
    )
    state.base_url = f"http://{_STUB_HOST}:{server.server_port}"
    monkeypatch.setenv("ANTHROPIC_BASE_URL", state.base_url)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=_SERVER_JOIN_SECONDS)


def _chat_model() -> Any:
    cfg = FundConfig(llm_provider="anthropic", anthropic_api_key=_KEY)
    return model.build_chat_model(cfg, model_name=_MODEL)


@pytest.mark.criterion("REQ-FUND-029")
def test_the_key_reaches_the_configured_host_in_the_authorization_header_only(
    stub: _Stub,
) -> None:
    """Validates ``fund.agents.model._build_anthropic``."""
    reply = _chat_model().invoke("hi")

    assert reply.content == "ok"
    (seen,) = stub.requests
    assert seen.path == "/v1/messages"
    # httpx2 masks only Authorization when it renders headers, so a key sent in any
    # other header (the 1.x SDK used X-Api-Key) would show in an error's request.
    assert seen.headers["authorization"] == f"Bearer {_KEY}"
    elsewhere = {k: v for k, v in seen.headers.items() if k != "authorization"}
    assert _KEY not in json.dumps(elsewhere)
    assert _KEY not in json.dumps(seen.body)


@pytest.mark.criterion("REQ-FUND-029")
def test_the_api_version_still_accepts_the_temperature_the_request_carries(
    stub: _Stub,
) -> None:
    """The ``temperature=0`` contract needs a version that does not reject it.

    Validates ``fund.agents.model._build_anthropic``.
    """
    _chat_model().invoke("hi")

    (seen,) = stub.requests
    assert "temperature" in seen.body
    assert seen.headers["anthropic-version"] < _FIRST_VERSION_WITHOUT_SAMPLING


@pytest.mark.criterion("REQ-FUND-029")
def test_no_trace_context_is_sent_while_no_tracer_provider_is_registered(
    stub: _Stub,
) -> None:
    """The SDK's OpenTelemetry spans are inert without a provider and inject nothing.

    Validates ``fund.agents.model._build_anthropic``.
    """
    _chat_model().invoke("hi")

    (seen,) = stub.requests
    assert not [name for name in _TRACE_CONTEXT_HEADERS if name in seen.headers]


@pytest.mark.criterion("REQ-FUND-029")
def test_one_call_makes_one_request_and_contacts_no_other_host(
    stub: _Stub, _loopback_only: list[str]
) -> None:
    """Building and calling the model starts no update check, telemetry or lookup.

    Validates ``fund.agents.model._build_anthropic``.
    """
    _chat_model().invoke("hi")

    assert len(stub.requests) == 1
    assert _loopback_only == []


@pytest.mark.criterion("REQ-FUND-029")
def test_a_failed_request_shows_nothing_of_the_key_in_its_error_or_logs(
    stub: _Stub, caplog: pytest.LogCaptureFixture
) -> None:
    """Neither the error, nor its request, nor a DEBUG log record carries the key.

    Validates ``fund.agents.model._build_anthropic``.
    """
    stub.status = HTTPStatus.UNAUTHORIZED
    llm = _chat_model()
    # After the build: building a chat model puts the SDK loggers back to the level
    # they have without ANTHROPIC_LOG, and this test needs their DEBUG records.
    for name in ("anthropic", "httpx2", "httpcore2"):
        caplog.set_level(logging.DEBUG, logger=name)

    with pytest.raises(anthropic.AuthenticationError) as raised:
        llm.invoke("hi")

    error = raised.value
    rendered = [
        str(error),
        repr(error),
        repr(error.request),
        repr(error.request.headers),
    ]
    assert not [text for text in rendered if _KEY in text]
    assert caplog.records, "the DEBUG records must exist, or this test proves nothing"
    assert _KEY not in caplog.text


@pytest.mark.criterion("REQ-FUND-020")
def test_no_rendering_of_the_chat_model_shows_the_key_once_the_sdk_clients_exist(
    stub: _Stub,
) -> None:
    """The model caches its SDK client on first use; the key must stay out of it.

    Validates ``fund.agents.model._build_anthropic``.
    """
    llm = _chat_model()
    llm.invoke("hi")

    renderings = [
        repr(llm),
        str(llm),
        dumps(llm),
        json.dumps(dumpd(llm)),
        str(llm.model_dump()),
    ]
    assert not [text for text in renderings if _KEY in text]
