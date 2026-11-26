"""The ``openai``, ``openrouter`` and ``microsoft`` providers through the real SDK.

Validates ``fund.agents.model._build_openai``, ``_build_openrouter`` and
``_build_microsoft``, the structured-output path of
``fund.schemas.structured.structured_call`` over them, and the finish-reason read of
``fund.agents.graph_hitl._ungated_end_reason`` on their replies.

All three builders build a ``langchain-openai`` class, and ``test_model_factory``
drives them with a recording fake, which cannot notice a renamed constructor argument
or a changed default in the SDK. These tests build the real classes and talk to a
localhost stub, so the key, base URL, deployment and API version are checked where
they cross the wire. Nothing leaves the machine.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest
from _provider_stubs import loopback_url, serve_on_loopback
from langchain_core.load import dumpd
from pydantic import BaseModel

from fund.agents import model
from fund.agents.graph_hitl import _ungated_end_reason
from fund.config import FundConfig
from fund.schemas.structured import StructuredOutputError, structured_call

pytest.importorskip("langchain_openai")

_OPENAI_KEY = "sk-openai-S3CRET"
_OPENROUTER_KEY = "sk-or-S3CRET"
_AZURE_KEY = "az-S3CRET"

_AZURE_DEPLOYMENT = "my-deploy"
_AZURE_API_VERSION = "2024-06-01"

# Every variable langchain-openai or the OpenAI SDK would read in place of a value
# the builder did not pass, so a developer's shell cannot change what is asserted.
# The tracing variables are here for a second reason: with one set, each call to the
# stub would also send a trace to LangSmith.
_AMBIENT_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_ORG_ID",
    "OPENAI_PROXY",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_AD_TOKEN",
    "LANGSMITH_GATEWAY",
    "LANGSMITH_GATEWAY_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
)


class _Answer(BaseModel):
    """The structured payload the stub hands back as a forced tool call."""

    verdict: str


@dataclass(frozen=True)
class _Seen:
    """One request the stub received, header names lower-cased."""

    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class _Stub:
    """A localhost chat-completions server and what it was asked."""

    base_url: str
    requests: list[_Seen] = field(default_factory=list)
    finish_reason: str = "stop"
    tool_arguments: str | None = None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AMBIENT_ENV:
        monkeypatch.delenv(name, raising=False)


def _completion(state: _Stub, body: dict[str, Any]) -> bytes:
    """The stub's reply: a forced tool call when one was asked for, else ``ok``."""
    message: dict[str, Any] = {"role": "assistant", "content": "ok"}
    forced = (body.get("tool_choice") or {}).get("function", {}).get("name")
    if forced and state.tool_arguments is not None:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": forced, "arguments": state.tool_arguments},
                }
            ],
        }
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model", "m"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": state.finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Stub]:
    """A localhost OpenAI-style ``chat/completions`` endpoint."""
    state = _Stub(base_url="")

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            """Record the request and answer it with the stub's completion."""
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(raw)
            headers = {name.lower(): value for name, value in self.headers.items()}
            state.requests.append(_Seen(self.path, headers, body))
            payload = _completion(state, body)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            """Keep the test output free of per-request access lines."""

    with serve_on_loopback(monkeypatch, _Handler) as port:
        state.base_url = loopback_url(port)
        yield state


def _openai_cfg(stub: _Stub) -> FundConfig:
    return FundConfig(
        llm_provider="openai",
        openai_api_key=_OPENAI_KEY,
        openai_base_url=f"{stub.base_url}/v1",
    )


def _azure_cfg(endpoint: str) -> FundConfig:
    return FundConfig(
        llm_provider="microsoft",
        azure_openai_api_key=_AZURE_KEY,
        azure_openai_endpoint=endpoint,
        azure_openai_api_version=_AZURE_API_VERSION,
        azure_openai_deployment_name=_AZURE_DEPLOYMENT,
    )


# --- the wire ---------------------------------------------------------------


def test_openai_sends_the_key_as_bearer_to_the_configured_base_url(stub: _Stub) -> None:
    """The key and model reach the host named by ``OPENAI_BASE_URL``."""
    cfg = _openai_cfg(stub)
    llm = model.build_chat_model(cfg, model_name="gpt-4o")

    reply = llm.invoke("hi")

    assert reply.content == "ok"
    (seen,) = stub.requests
    assert seen.path == "/v1/chat/completions"
    assert seen.headers["authorization"] == f"Bearer {_OPENAI_KEY}"
    assert seen.body["model"] == "gpt-4o"
    assert seen.body["temperature"] == cfg.model_temperature


def test_openai_reasoning_model_request_carries_no_temperature(stub: _Stub) -> None:
    """An o-series model is sent no sampling temperature, which it would reject."""
    llm = model.build_chat_model(_openai_cfg(stub), model_name="o3-mini")

    llm.invoke("hi")

    assert "temperature" not in stub.requests[0].body


def test_azure_routes_by_deployment_and_authenticates_with_the_api_key_header(
    stub: _Stub,
) -> None:
    """Azure is addressed by deployment and API version, not by a bearer token."""
    llm = model.build_chat_model(_azure_cfg(stub.base_url), model_name="gpt-4o")

    reply = llm.invoke("hi")

    assert reply.content == "ok"
    (seen,) = stub.requests
    assert seen.path == (
        f"/openai/deployments/{_AZURE_DEPLOYMENT}/chat/completions"
        f"?api-version={_AZURE_API_VERSION}"
    )
    assert seen.headers["api-key"] == _AZURE_KEY
    assert "authorization" not in seen.headers


def test_openrouter_targets_the_openrouter_host_with_its_own_key() -> None:
    """OpenRouter reuses ``ChatOpenAI`` at a fixed host with its own key."""
    cfg = FundConfig(llm_provider="openrouter", openrouter_api_key=_OPENROUTER_KEY)

    llm = model.build_chat_model(cfg, model_name="anthropic/claude-3.5")

    assert str(llm.root_client.base_url) == "https://openrouter.ai/api/v1/"
    assert llm.root_client.api_key == _OPENROUTER_KEY
    assert llm.temperature == cfg.model_temperature


def test_a_reply_cut_off_by_the_output_cap_is_still_read_as_truncated(
    stub: _Stub,
) -> None:
    """The finish reason ChatOpenAI reports is the one the PM's end-of-run read uses."""
    stub.finish_reason = "length"
    llm = model.build_chat_model(_openai_cfg(stub), model_name="gpt-4o")

    reply = llm.invoke("hi")

    assert reply.response_metadata["finish_reason"] == "length"
    assert _ungated_end_reason(reply, pm_round_cap_reached=False) == "output_truncated"


# --- the LangSmith gateway ----------------------------------------------------
#
# langchain-core reads LANGSMITH_GATEWAY and the provider classes that apply it send a
# model whose endpoint the builder did not pass to the gateway host, key included. The
# guard sits in build_chat_model, so one test over every registered provider covers
# the whole registry, including a provider whose package starts reading it later.

_GATEWAY_URL = "https://user:pass@gateway.example.test"


@pytest.mark.criterion("REQ-FUND-027")
@pytest.mark.parametrize("gateway", ["true", "1", "yes", _GATEWAY_URL])
@pytest.mark.parametrize("provider", sorted(model._BUILDERS))
def test_an_enabled_langsmith_gateway_stops_every_provider_before_it_is_built(
    monkeypatch: pytest.MonkeyPatch, provider: str, gateway: str
) -> None:
    """No provider builds while the gateway is on, whatever else is configured.

    No key is set, so a builder that ran first would fail on its own missing-key
    error instead: the match on the variable name proves the guard came first.
    """
    monkeypatch.setenv("LANGSMITH_GATEWAY", gateway)

    with pytest.raises(RuntimeError, match="LANGSMITH_GATEWAY") as excinfo:
        model.build_chat_model(FundConfig(llm_provider=provider), model_name="m")

    assert "gateway.example.test" not in str(excinfo.value)


@pytest.mark.criterion("REQ-FUND-027")
@pytest.mark.parametrize("gateway", ["", "false", "FALSE", "0", "no"])
def test_a_disabled_langsmith_gateway_builds_as_usual(
    monkeypatch: pytest.MonkeyPatch, gateway: str
) -> None:
    """A value langchain-core reads as off leaves the OpenAI host in place."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", gateway)
    cfg = FundConfig(llm_provider="openai", openai_api_key=_OPENAI_KEY)

    llm = model.build_chat_model(cfg, model_name="gpt-4o")

    assert str(llm.root_client.base_url) == "https://api.openai.com/v1/"


# --- secrets ----------------------------------------------------------------


@pytest.mark.criterion("REQ-FUND-018")
@pytest.mark.parametrize(
    ("cfg", "model_name", "secret"),
    [
        (
            FundConfig(llm_provider="openai", openai_api_key=_OPENAI_KEY),
            "gpt-4o",
            _OPENAI_KEY,
        ),
        (
            FundConfig(llm_provider="openrouter", openrouter_api_key=_OPENROUTER_KEY),
            "anthropic/claude-3.5",
            _OPENROUTER_KEY,
        ),
        (_azure_cfg("https://example.openai.azure.com"), "gpt-4o", _AZURE_KEY),
    ],
    ids=["openai", "openrouter", "microsoft"],
)
def test_api_key_is_absent_from_repr_and_serialized_form(
    cfg: FundConfig, model_name: str, secret: str
) -> None:
    """No provider key is readable from the model's string or serialized form."""
    # LangChain hands repr() and dumpd() to every callback and tracer, so a key held
    # in a plain field would reach LangSmith and the debug logs.
    llm = model.build_chat_model(cfg, model_name=model_name)

    assert secret not in repr(llm)
    assert secret not in str(llm)
    assert secret not in json.dumps(dumpd(llm), default=str)


# --- structured output -------------------------------------------------------


def test_structured_call_returns_the_forced_tool_call_as_the_schema(
    stub: _Stub,
) -> None:
    """A forced tool call comes back as a validated schema instance."""
    stub.tool_arguments = json.dumps({"verdict": "approve"})
    llm = model.build_chat_model(_openai_cfg(stub), model_name="gpt-4o")

    answer = structured_call(llm, _Answer, "decide")

    assert answer == _Answer(verdict="approve")
    assert stub.requests[0].body["tool_choice"]["function"]["name"] == "_Answer"


def test_structured_call_reports_filtered_output_as_a_structured_output_error(
    stub: _Stub,
) -> None:
    """A filtered answer surfaces as ``StructuredOutputError``."""
    # A content filter answers with no tool call. The helper's contract is that every
    # failure leaves as StructuredOutputError, whatever error class the SDK raises.
    stub.finish_reason = "content_filter"
    llm = model.build_chat_model(_openai_cfg(stub), model_name="gpt-4o")

    with pytest.raises(StructuredOutputError):
        structured_call(llm, _Answer, "decide", retries=0)
