"""What the ``groq`` SDK puts on the wire for the ``groq`` provider.

``groq`` reaches the lockfile only through ``langchain-groq``, so a release of it can
change every request this repo makes without a line here changing: 1.3 added an
environment variable that injects headers into each request. These tests send one
request through the real ``ChatGroq`` to a localhost stub and read what arrived, so a
header, an endpoint or a new place for the key that nobody reviewed fails here.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

import pytest
from _provider_stubs import loopback_only, loopback_url, serve_on_loopback

from fund.agents import model
from fund.config import FundConfig

_MODEL = "openai/gpt-oss-20b"

_SENTINEL = "S3NTINEL-groq-api-key"

_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_HTTP_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "authorization",
        "connection",
        "content-length",
        "content-type",
        "host",
        "user-agent",
    }
)

# What the SDK says about itself on every request: package version, OS, CPU
# architecture and Python runtime, plus the call's own mode (async or not) and retry
# count. Exactly these eight; a ninth is a change to review, not to wave through.
_SDK_DESCRIPTOR_HEADERS = frozenset(
    {
        "x-stainless-arch",
        "x-stainless-async",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-retry-count",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
    }
)

# Sent only when the client has a timeout, which ChatGroq leaves unset unless the
# builder passes one; the builder always does.
_TIMEOUT_HEADER = frozenset({"x-stainless-read-timeout"})

# The deadline the other provider builders already give a request.
_DEADLINE_SECONDS = "180.0"

# Everything in the environment that can redirect or decorate a Groq request.
_AMBIENT_GROQ_VARIABLES = (
    "GROQ_API_KEY",
    "GROQ_API_BASE",
    "GROQ_BASE_URL",
    "GROQ_CUSTOM_HEADERS",
    "GROQ_LOG",
    "GROQ_PROXY",
    "LANGSMITH_GATEWAY",
)

# (path, headers with lower-cased names, body) of one request the stub received.
Request = tuple[str, dict[str, str], bytes]


def _recording_handler(requests: list[Request]) -> type[BaseHTTPRequestHandler]:
    """Return a handler that records each POST and answers with a chat completion."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            """Record the request and answer it with the canned completion."""
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            headers = {name.lower(): value for name, value in self.headers.items()}
            requests.append((self.path, headers, body))
            reply = json.dumps(_COMPLETION).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            """Keep the stub's access log out of the test output."""

    return _Handler


@pytest.fixture
def fake_groq(monkeypatch) -> Generator[list[Request], None, None]:
    """A localhost stub of Groq's chat-completions route; yields what it received.

    ``GROQ_API_BASE`` points the model at the stub, because ``build_chat_model``
    offers no other way to aim it at a loopback host. The loopback guard keeps a
    request that bypasses the stub from reaching the real host.
    """
    for name in _AMBIENT_GROQ_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    loopback_only(monkeypatch)
    requests: list[Request] = []
    with serve_on_loopback(monkeypatch, _recording_handler(requests)) as port:
        monkeypatch.setenv("GROQ_API_BASE", loopback_url(port))
        yield requests


@pytest.mark.criterion("REQ-FUND-030")
@pytest.mark.parametrize(
    ("custom_headers", "operator_header_names"),
    [
        (None, frozenset()),
        (
            "X-Operator-Trace: trace-1\nX-Second: 2",
            frozenset({"x-operator-trace", "x-second"}),
        ),
    ],
    ids=["no-custom-headers", "operator-custom-headers"],
)
def test_groq_request_sends_the_key_only_as_bearer_and_adds_no_unreviewed_header(
    monkeypatch, fake_groq, custom_headers, operator_header_names
):
    """Validates ``fund.agents.model._build_groq`` against the real ``ChatGroq``.

    ``GROQ_CUSTOM_HEADERS`` is the SDK's own way to add headers to every request, so
    the second case shows that what an operator sets there is the only addition.
    """
    if custom_headers is not None:
        monkeypatch.setenv("GROQ_CUSTOM_HEADERS", custom_headers)
    llm = model.build_chat_model(
        FundConfig(llm_provider="groq", groq_api_key=_SENTINEL), model_name=_MODEL
    )

    llm.invoke("hello")

    assert len(fake_groq) == 1
    path, headers, body = fake_groq[0]
    expected = _HTTP_HEADERS | _SDK_DESCRIPTOR_HEADERS | operator_header_names
    assert set(headers) - _TIMEOUT_HEADER == expected
    assert path == "/openai/v1/chat/completions"
    assert headers["authorization"] == f"Bearer {_SENTINEL}"
    elsewhere = {
        name: value for name, value in headers.items() if name != "authorization"
    }
    assert not [name for name, value in elsewhere.items() if _SENTINEL in value]
    assert _SENTINEL not in path
    assert _SENTINEL not in body.decode()
    assert headers["content-type"] == "application/json"
    payload = json.loads(body)
    assert payload["model"] == _MODEL
    assert payload["messages"][-1]["content"] == "hello"


@pytest.mark.criterion("REQ-FUND-040")
def test_groq_request_carries_a_deadline(fake_groq):
    """Validates ``fund.agents.model._build_groq``: a hung call cannot block a run.

    ``ChatGroq`` has no timeout unless the builder gives one, and the SDK tells the
    server the read timeout it works to in ``X-Stainless-Read-Timeout``.
    """
    llm = model.build_chat_model(
        FundConfig(llm_provider="groq", groq_api_key=_SENTINEL), model_name=_MODEL
    )

    llm.invoke("hello")

    assert fake_groq[0][1]["x-stainless-read-timeout"] == _DEADLINE_SECONDS


@pytest.mark.criterion("REQ-FUND-030")
def test_groq_clients_default_to_the_groq_host(monkeypatch):
    """Validates ``fund.agents.model._build_groq``: it chooses no endpoint for the SDK.

    Construction only, so nothing is sent. With every endpoint variable cleared, the
    synchronous and asynchronous clients must both fall back to Groq's own HTTPS host.
    """
    for name in _AMBIENT_GROQ_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    llm = model.build_chat_model(
        FundConfig(llm_provider="groq", groq_api_key=_SENTINEL),
        model_name=_MODEL,
    )

    for resource in (llm.client, llm.async_client):
        url = resource._client.base_url
        assert (url.scheme, url.host) == ("https", "api.groq.com")
