"""``fund.agents.model`` — the provider-registry chat-model factory (unit slice).

Construction only: the SDK class is built but never invoked, so no network is
touched. The tests assert the frozen ``FundConfig`` is threaded onto the
model — base URL, model id, ``temperature``, non-thinking ``reasoning`` route, and
the ``OLLAMA_API_KEY`` bearer header — that ``build_chat_model`` dispatches on
``config.llm_provider`` (unknown provider / uninstalled extra raise clear
``RuntimeError``\\s), that the Ollama cloud-vs-local key rule holds, and that a
missing key raises at *build* time (never at import, so a bare
``import fund.agents.model`` stays green in CI).
"""

from __future__ import annotations

import json
import re
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import tomllib
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd

from fund.agents import model
from fund.config import FundConfig, load_config

# A config carrying every default plus a key, so build never hits use-time None.
_CFG = FundConfig(ollama_api_key="sk-ollama-test")

# unit -> tests -> fund.
_FUND_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


class _SerializedSpy(BaseCallbackHandler):
    """Records the ``serialized`` payload LangChain hands every callback/tracer."""

    def __init__(self) -> None:
        self.serialized: list[str] = []

    def on_chat_model_start(self, serialized, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.serialized.append(json.dumps(serialized, default=str))


@pytest.fixture
def fake_ollama(monkeypatch):
    """A localhost Ollama ``/api/chat`` stub; yields ``(base_url, auth_headers)``.

    ``auth_headers`` collects the ``Authorization`` header of every request (``None``
    when absent), so a test can assert what actually crossed the wire.
    """
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    # The ollama SDK itself turns an ambient OLLAMA_API_KEY into a bearer header.
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    auth_headers: list[str | None] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            auth_headers.append(self.headers.get("Authorization"))
            body = (
                json.dumps(
                    {
                        "model": "m",
                        "created_at": "2026-01-01T00:00:00Z",
                        "message": {"role": "assistant", "content": "ok"},
                        "done": True,
                        "done_reason": "stop",
                    }
                ).encode()
                + b"\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", auth_headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_build_chat_model_threads_config_onto_chatollama():
    llm = model.build_chat_model(_CFG, model_name="deepseek-v4.1-flash:cloud")

    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == "deepseek-v4.1-flash:cloud"
    assert llm.base_url == _CFG.ollama_base_url
    assert llm.temperature == _CFG.model_temperature
    assert llm.reasoning is _CFG.model_reasoning


def test_build_chat_model_sends_api_key_as_bearer_header(fake_ollama):
    base_url, auth_headers = fake_ollama
    cfg = FundConfig(ollama_api_key="sk-ollama-test", ollama_base_url=base_url)
    llm = model.build_chat_model(cfg, model_name="deepseek-v4-pro:cloud")

    llm.invoke("hi")

    assert auth_headers == ["Bearer sk-ollama-test"]


def test_build_chat_model_without_api_key_raises_runtimeerror():
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        model.build_chat_model(
            FundConfig(ollama_api_key=None), model_name="deepseek-v4.1-flash:cloud"
        )


def test_build_primary_uses_the_primary_model_id():
    llm = model.build_primary(_CFG)

    assert llm.model == _CFG.primary_model


def test_build_fallback_uses_the_fallback_model_id():
    llm = model.build_fallback(_CFG)

    assert llm.model == _CFG.fallback_model


def test_import_needs_no_env_and_defers_key_validation_to_build_time():
    # Importing the module (done at file top) must not require OLLAMA_API_KEY; the
    # keyless default `settings` only fails when a builder is actually called.
    assert callable(model.build_chat_model)
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        model.build_primary(FundConfig(ollama_api_key=None))


# --- provider registry / dispatch -------------------------------------------


def test_ollama_is_the_registered_default_provider():
    # The env-free default keeps today's DeepSeek-on-Ollama-Cloud path.
    assert FundConfig().llm_provider == "ollama"
    assert "ollama" in model._BUILDERS


def test_build_chat_model_dispatches_on_llm_provider():
    # An unknown provider must not fall through to Ollama; it fails fast.
    cfg = FundConfig(llm_provider="does-not-exist", ollama_api_key="sk")
    with pytest.raises(RuntimeError, match="Unknown LLM provider"):
        model.build_chat_model(cfg, model_name="whatever")


def test_unknown_provider_error_lists_the_valid_providers():
    cfg = FundConfig(llm_provider="bogus")
    with pytest.raises(RuntimeError, match="ollama"):
        model.build_chat_model(cfg, model_name="m")


def test_uninstalled_provider_extra_reraises_as_runtimeerror_with_hint(monkeypatch):
    def _boom(_config, _model_name):
        raise ImportError("No module named 'langchain_openai'")

    monkeypatch.setitem(model._BUILDERS, "faux", _boom)
    cfg = FundConfig(llm_provider="faux")
    with pytest.raises(RuntimeError, match="--extra faux"):
        model.build_chat_model(cfg, model_name="m")


def _install_hint_extra(provider: str, monkeypatch: pytest.MonkeyPatch) -> str | None:
    """Fail `build_chat_model` on a missing SDK; return the extra the hint names."""

    def _missing_sdk(_config, _model_name):
        raise ImportError("No module named 'provider_sdk'")

    monkeypatch.setitem(model._BUILDERS, provider, _missing_sdk)
    with pytest.raises(RuntimeError) as err:
        model.build_chat_model(FundConfig(llm_provider=provider), model_name="m")
    match = re.search(r"--extra (\S+?)`", str(err.value))
    return match[1] if match else None


@pytest.mark.parametrize(
    ("provider", "extra"), [("openrouter", "openai"), ("microsoft", "openai")]
)
def test_providers_without_their_own_extra_hint_the_openai_extra(
    provider, extra, monkeypatch
):
    # openrouter/microsoft ride on ChatOpenAI; `--extra openrouter` does not exist.
    assert _install_hint_extra(provider, monkeypatch) == extra


@pytest.mark.parametrize("provider", sorted(model._BUILDERS))
def test_every_install_hint_names_an_extra_declared_in_pyproject(provider, monkeypatch):
    declared = tomllib.loads(_FUND_PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    hinted = _install_hint_extra(provider, monkeypatch)
    assert hinted is None or hinted in declared, (
        f"`--extra {hinted}` is not a defined extra"
    )


def test_ollama_is_a_core_dependency_so_its_hint_names_no_extra(monkeypatch):
    assert _install_hint_extra("ollama", monkeypatch) is None


# --- Ollama cloud-vs-local key rule -----------------------------------------


def test_ollama_cloud_default_without_key_raises_naming_the_env_var():
    # Default base_url is the cloud host, so a missing key still raises here.
    cfg = FundConfig(ollama_api_key=None)
    assert "ollama.com" in cfg.ollama_base_url
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        model.build_chat_model(cfg, model_name="deepseek-v4.1-flash:cloud")


def test_ollama_local_base_url_builds_without_a_key(fake_ollama):
    base_url, auth_headers = fake_ollama
    cfg = FundConfig(ollama_api_key=None, ollama_base_url=base_url)

    llm = model.build_chat_model(cfg, model_name="llama3")
    llm.invoke("hi")

    assert type(llm).__name__ == "ChatOllama"
    assert auth_headers == [None]  # no key ⇒ no bearer header


def test_ollama_local_with_key_still_sends_the_bearer_header(fake_ollama):
    base_url, auth_headers = fake_ollama
    cfg = FundConfig(ollama_api_key="sk-local", ollama_base_url=base_url)

    llm = model.build_chat_model(cfg, model_name="llama3")
    llm.invoke("hi")

    assert auth_headers == ["Bearer sk-local"]


def test_ollama_api_key_is_absent_from_repr_and_serialized_form():
    # LangChain falls back to repr() for a model it cannot serialize, so a key held
    # in a plain pydantic field reaches every tracer and debug log.
    llm = model.build_chat_model(_CFG, model_name="deepseek-v4.1-flash:cloud")

    assert "sk-ollama-test" not in repr(llm)
    assert "sk-ollama-test" not in json.dumps(dumpd(llm))


def test_ollama_api_key_is_not_handed_to_callbacks_during_a_call(fake_ollama):
    base_url, _ = fake_ollama
    cfg = FundConfig(ollama_api_key="sk-ollama-test", ollama_base_url=base_url)
    llm = model.build_chat_model(cfg, model_name="deepseek-v4.1-flash:cloud")
    spy = _SerializedSpy()

    llm.invoke("hi", config={"callbacks": [spy]})

    assert spy.serialized
    assert all("sk-ollama-test" not in payload for payload in spy.serialized)


def test_ollama_http_clients_have_finite_request_timeouts():
    # httpx defaults to no timeout at all; a half-open Ollama Cloud connection
    # would then block the single-worker daemon forever.
    llm = model.build_chat_model(_CFG, model_name="deepseek-v4.1-flash:cloud")

    for client in (llm._client._client, llm._async_client._client):
        assert client.timeout.connect is not None
        assert client.timeout.read is not None


def _sent_ollama_options(llm: object) -> dict[str, object]:
    """Options block ChatOllama would put in the ``/api/chat`` request."""
    from langchain_core.messages import HumanMessage

    return llm._chat_params([HumanMessage("hi")])["options"]  # type: ignore[attr-defined]


def test_ollama_local_base_url_requests_an_explicit_context_window():
    # A local Ollama server falls back to its own tiny default window and silently
    # truncates the oldest tokens (system prompt, tool schemas) of a deep-agent run.
    cfg = FundConfig(ollama_api_key=None, ollama_base_url="http://localhost:11434")

    llm = model.build_chat_model(cfg, model_name="llama3")

    assert cfg.ollama_num_ctx > 4096
    assert _sent_ollama_options(llm)["num_ctx"] == cfg.ollama_num_ctx


def test_ollama_local_context_window_is_configurable():
    cfg = FundConfig(ollama_base_url="http://localhost:11434", ollama_num_ctx=65536)

    llm = model.build_chat_model(cfg, model_name="llama3")

    assert _sent_ollama_options(llm)["num_ctx"] == 65536


def test_ollama_cloud_leaves_the_context_window_to_the_server():
    llm = model.build_chat_model(_CFG, model_name="deepseek-v4.1-flash:cloud")

    assert "num_ctx" not in _sent_ollama_options(llm)


def test_google_request_timeout_is_finite():
    # The google-genai client has no request deadline unless one is passed.
    cfg = FundConfig(llm_provider="google", google_api_key="gk-google-test")

    llm = model.build_chat_model(cfg, model_name="gemini-2.5-flash")

    assert llm.timeout is not None


# --- openai / openrouter (langchain-openai / ChatOpenAI) ---------------------


def _install_fake_chat_class(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str = "ChatOpenAI",
) -> type:
    """Inject a fake langchain provider module whose chat class records kwargs.

    The builder's lazy ``from <module_name> import <class_name>`` then resolves to
    this recorder, so the builder branch runs (and is covered) with no real SDK,
    no network, and no dependency on the optional extra actually being installed.
    Returns the recording class so a test can assert ``isinstance``.
    """

    class _RecordingChat:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake = types.ModuleType(module_name)
    setattr(fake, class_name, _RecordingChat)
    monkeypatch.setitem(sys.modules, module_name, fake)
    return _RecordingChat


def test_openai_and_openrouter_are_registered_providers():
    assert "openai" in model._BUILDERS
    assert "openrouter" in model._BUILDERS


def test_openai_builds_chatopenai_threading_model_key_and_temperature(monkeypatch):
    fake_cls = _install_fake_chat_class(monkeypatch, "langchain_openai")
    cfg = FundConfig(llm_provider="openai", openai_api_key="sk-openai-test")

    llm = model.build_chat_model(cfg, model_name="gpt-4o")

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["model"] == "gpt-4o"
    assert llm.kwargs["api_key"] == "sk-openai-test"
    assert llm.kwargs["temperature"] == cfg.model_temperature


def test_openai_honors_base_url_when_set(monkeypatch):
    _install_fake_chat_class(monkeypatch, "langchain_openai")
    cfg = FundConfig(
        llm_provider="openai",
        openai_api_key="sk-openai-test",
        openai_base_url="https://proxy.example/v1",
    )

    llm = model.build_chat_model(cfg, model_name="gpt-4o")

    assert llm.kwargs["base_url"] == "https://proxy.example/v1"


def test_openai_omits_base_url_when_unset(monkeypatch):
    _install_fake_chat_class(monkeypatch, "langchain_openai")
    cfg = FundConfig(llm_provider="openai", openai_api_key="sk-openai-test")

    llm = model.build_chat_model(cfg, model_name="gpt-4o")

    assert "base_url" not in llm.kwargs


def test_openai_blank_base_url_env_falls_back_to_the_openai_host(monkeypatch):
    # Real SDK, construction only: an exported `OPENAI_BASE_URL=` (the template
    # line left blank) must not leave ChatOpenAI with an empty base URL.
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    cfg = load_config(load_dotenv_file=False)

    llm = model.build_chat_model(
        FundConfig(llm_provider="openai", openai_api_key=cfg.openai_api_key),
        model_name="gpt-4o",
    )

    assert str(llm.root_client.base_url) == "https://api.openai.com/v1/"


def test_openai_never_passes_the_ollama_only_reasoning_kwarg(monkeypatch):
    # `reasoning=` is Ollama-only; ChatOpenAI would reject it.
    _install_fake_chat_class(monkeypatch, "langchain_openai")
    cfg = FundConfig(llm_provider="openai", openai_api_key="sk-openai-test")

    llm = model.build_chat_model(cfg, model_name="gpt-4o")

    assert "reasoning" not in llm.kwargs


def test_openai_without_key_raises_runtimeerror(monkeypatch):
    _install_fake_chat_class(monkeypatch, "langchain_openai")
    cfg = FundConfig(llm_provider="openai", openai_api_key=None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        model.build_chat_model(cfg, model_name="gpt-4o")


def test_openrouter_builds_chatopenai_at_the_openrouter_base_url(monkeypatch):
    fake_cls = _install_fake_chat_class(monkeypatch, "langchain_openai")
    cfg = FundConfig(llm_provider="openrouter", openrouter_api_key="sk-or-test")

    llm = model.build_chat_model(cfg, model_name="anthropic/claude-3.5")

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert llm.kwargs["api_key"] == "sk-or-test"
    assert llm.kwargs["model"] == "anthropic/claude-3.5"
    assert llm.kwargs["temperature"] == cfg.model_temperature


def test_openrouter_without_key_raises_runtimeerror(monkeypatch):
    _install_fake_chat_class(monkeypatch, "langchain_openai")
    cfg = FundConfig(llm_provider="openrouter", openrouter_api_key=None)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        model.build_chat_model(cfg, model_name="m")


# --- anthropic (langchain-anthropic / ChatAnthropic) ------------------------


def test_anthropic_is_a_registered_provider():
    assert "anthropic" in model._BUILDERS


def test_anthropic_builds_chatanthropic_threading_model_key_and_temperature(
    monkeypatch,
):
    fake_cls = _install_fake_chat_class(
        monkeypatch, "langchain_anthropic", class_name="ChatAnthropic"
    )
    cfg = FundConfig(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    llm = model.build_chat_model(cfg, model_name="claude-sonnet-4")

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["model"] == "claude-sonnet-4"
    assert llm.kwargs["api_key"] == "sk-ant-test"
    assert llm.kwargs["temperature"] == cfg.model_temperature


def test_anthropic_never_passes_the_ollama_only_reasoning_kwarg(monkeypatch):
    # `reasoning=` is Ollama-only; ChatAnthropic would reject it.
    _install_fake_chat_class(
        monkeypatch, "langchain_anthropic", class_name="ChatAnthropic"
    )
    cfg = FundConfig(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    llm = model.build_chat_model(cfg, model_name="claude-sonnet-4")

    assert "reasoning" not in llm.kwargs


def test_anthropic_without_key_raises_runtimeerror(monkeypatch):
    _install_fake_chat_class(
        monkeypatch, "langchain_anthropic", class_name="ChatAnthropic"
    )
    cfg = FundConfig(llm_provider="anthropic", anthropic_api_key=None)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        model.build_chat_model(cfg, model_name="claude-sonnet-4")


# --- groq (langchain-groq / ChatGroq) ---------------------------------------


def test_groq_is_a_registered_provider():
    assert "groq" in model._BUILDERS


def test_groq_builds_chatgroq_threading_model_key_and_temperature(monkeypatch):
    fake_cls = _install_fake_chat_class(
        monkeypatch, "langchain_groq", class_name="ChatGroq"
    )
    cfg = FundConfig(llm_provider="groq", groq_api_key="gsk-groq-test")

    llm = model.build_chat_model(cfg, model_name="llama-3.3-70b-versatile")

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["model"] == "llama-3.3-70b-versatile"
    assert llm.kwargs["api_key"] == "gsk-groq-test"
    assert llm.kwargs["temperature"] == cfg.model_temperature


def test_groq_never_passes_the_ollama_only_reasoning_kwarg(monkeypatch):
    # `reasoning=` is Ollama-only; ChatGroq would reject it.
    _install_fake_chat_class(monkeypatch, "langchain_groq", class_name="ChatGroq")
    cfg = FundConfig(llm_provider="groq", groq_api_key="gsk-groq-test")

    llm = model.build_chat_model(cfg, model_name="llama-3.3-70b-versatile")

    assert "reasoning" not in llm.kwargs


def test_groq_without_key_raises_runtimeerror(monkeypatch):
    _install_fake_chat_class(monkeypatch, "langchain_groq", class_name="ChatGroq")
    cfg = FundConfig(llm_provider="groq", groq_api_key=None)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        model.build_chat_model(cfg, model_name="llama-3.3-70b-versatile")


# --- google / Gemini (langchain-google-genai / ChatGoogleGenerativeAI) -------


def test_google_is_a_registered_provider():
    assert "google" in model._BUILDERS


def test_google_builds_chatgoogle_threading_model_key_and_temperature(monkeypatch):
    fake_cls = _install_fake_chat_class(
        monkeypatch, "langchain_google_genai", class_name="ChatGoogleGenerativeAI"
    )
    cfg = FundConfig(llm_provider="google", google_api_key="gk-google-test")

    llm = model.build_chat_model(cfg, model_name="gemini-2.5-flash")

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["model"] == "gemini-2.5-flash"
    # `api_key` is the accepted alias for ChatGoogleGenerativeAI.google_api_key.
    assert llm.kwargs["api_key"] == "gk-google-test"
    assert llm.kwargs["temperature"] == cfg.model_temperature


def test_google_never_passes_the_ollama_only_reasoning_kwarg(monkeypatch):
    # `reasoning=` is Ollama-only; ChatGoogleGenerativeAI would reject it.
    _install_fake_chat_class(
        monkeypatch, "langchain_google_genai", class_name="ChatGoogleGenerativeAI"
    )
    cfg = FundConfig(llm_provider="google", google_api_key="gk-google-test")

    llm = model.build_chat_model(cfg, model_name="gemini-2.5-flash")

    assert "reasoning" not in llm.kwargs


def test_google_without_key_raises_runtimeerror(monkeypatch):
    _install_fake_chat_class(
        monkeypatch, "langchain_google_genai", class_name="ChatGoogleGenerativeAI"
    )
    cfg = FundConfig(llm_provider="google", google_api_key=None)

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        model.build_chat_model(cfg, model_name="gemini-2.5-flash")


# --- aws / Bedrock Converse (langchain-aws / ChatBedrockConverse) ------------
#
# AWS is the first non-key provider: creds arrive via boto3 (AWS_ACCESS_KEY_ID /
# AWS_SECRET_ACCESS_KEY / IAM role), so the builder requires only `region_name`.


def test_aws_is_a_registered_provider():
    assert "aws" in model._BUILDERS


def test_aws_builds_chatbedrockconverse_threading_model_region_and_temperature(
    monkeypatch,
):
    fake_cls = _install_fake_chat_class(
        monkeypatch, "langchain_aws", class_name="ChatBedrockConverse"
    )
    cfg = FundConfig(llm_provider="aws", aws_region="us-east-1")

    llm = model.build_chat_model(
        cfg, model_name="anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["model"] == "anthropic.claude-3-5-sonnet-20240620-v1:0"
    # Bedrock is region-scoped, not key-scoped; creds come from boto3/IAM.
    assert llm.kwargs["region_name"] == "us-east-1"
    assert llm.kwargs["temperature"] == cfg.model_temperature
    assert "api_key" not in llm.kwargs


def test_aws_never_passes_the_ollama_only_reasoning_kwarg(monkeypatch):
    # `reasoning=` is Ollama-only; ChatBedrockConverse would reject it.
    _install_fake_chat_class(
        monkeypatch, "langchain_aws", class_name="ChatBedrockConverse"
    )
    cfg = FundConfig(llm_provider="aws", aws_region="us-east-1")

    llm = model.build_chat_model(cfg, model_name="anthropic.claude-3")

    assert "reasoning" not in llm.kwargs


def test_aws_without_region_raises_runtimeerror(monkeypatch):
    # No API key to require — the required secret is the region.
    _install_fake_chat_class(
        monkeypatch, "langchain_aws", class_name="ChatBedrockConverse"
    )
    cfg = FundConfig(llm_provider="aws", aws_region=None)

    with pytest.raises(RuntimeError, match="AWS_REGION"):
        model.build_chat_model(cfg, model_name="anthropic.claude-3")


# --- microsoft / Azure OpenAI (langchain-openai / AzureChatOpenAI) -----------
#
# Azure reuses the `openai` extra (no new package): `AzureChatOpenAI` is wired from
# four Azure env vars, not one key. The class exposes `model` / `api_key` /
# `api_version` / `azure_deployment` as aliases, keeping this consistent with the
# other builders.


def _azure_cfg(**overrides: object) -> FundConfig:
    """A `microsoft`-provider config with all four Azure vars set (override to drop)."""
    kwargs: dict[str, object] = {
        "llm_provider": "microsoft",
        "azure_openai_api_key": "sk-azure-test",
        "azure_openai_endpoint": "https://example.openai.azure.com",
        "azure_openai_api_version": "2024-06-01",
        "azure_openai_deployment_name": "my-deploy",
    }
    kwargs.update(overrides)
    return FundConfig(**kwargs)  # type: ignore[arg-type]


def test_microsoft_is_a_registered_provider():
    assert "microsoft" in model._BUILDERS


def test_microsoft_builds_azurechatopenai_from_the_four_azure_env_vars(monkeypatch):
    fake_cls = _install_fake_chat_class(
        monkeypatch, "langchain_openai", class_name="AzureChatOpenAI"
    )
    cfg = _azure_cfg()

    llm = model.build_chat_model(cfg, model_name="gpt-4o")

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["model"] == "gpt-4o"
    assert llm.kwargs["api_key"] == "sk-azure-test"
    assert llm.kwargs["azure_endpoint"] == "https://example.openai.azure.com"
    assert llm.kwargs["api_version"] == "2024-06-01"
    assert llm.kwargs["azure_deployment"] == "my-deploy"
    assert llm.kwargs["temperature"] == cfg.model_temperature


def test_microsoft_never_passes_the_ollama_only_reasoning_kwarg(monkeypatch):
    # `reasoning=` is Ollama-only; AzureChatOpenAI would reject it.
    _install_fake_chat_class(
        monkeypatch, "langchain_openai", class_name="AzureChatOpenAI"
    )

    llm = model.build_chat_model(_azure_cfg(), model_name="gpt-4o")

    assert "reasoning" not in llm.kwargs


def test_microsoft_fallback_model_targets_its_own_deployment(monkeypatch):
    # Azure routes by deployment, not by the `model` label: a fallback built on the
    # primary's deployment would just retry the deployment that already failed.
    _install_fake_chat_class(
        monkeypatch, "langchain_openai", class_name="AzureChatOpenAI"
    )
    cfg = _azure_cfg(
        primary_model="gpt-4o",
        fallback_model="gpt-4o-mini",
        azure_openai_deployment_name="dep-primary",
        azure_openai_fallback_deployment_name="dep-fallback",
    )

    primary = model.build_primary(cfg)
    fallback = model.build_fallback(cfg)

    assert primary.kwargs["azure_deployment"] == "dep-primary"
    assert fallback.kwargs["azure_deployment"] == "dep-fallback"


def test_microsoft_fallback_keeps_the_primary_deployment_when_none_is_configured(
    monkeypatch,
):
    _install_fake_chat_class(
        monkeypatch, "langchain_openai", class_name="AzureChatOpenAI"
    )
    cfg = _azure_cfg(primary_model="gpt-4o", fallback_model="gpt-4o-mini")

    fallback = model.build_fallback(cfg)

    assert fallback.kwargs["azure_deployment"] == "my-deploy"


def test_microsoft_fallback_deployment_applies_when_both_roles_use_the_same_model(
    monkeypatch,
):
    # The usual Azure resilience setup is one model on a second deployment, and the
    # setup wizard mirrors the primary model id into FUND_FALLBACK_MODEL. Which
    # deployment serves a role cannot depend on the two model labels differing.
    _install_fake_chat_class(
        monkeypatch, "langchain_openai", class_name="AzureChatOpenAI"
    )
    cfg = _azure_cfg(
        primary_model="gpt-4o",
        fallback_model="gpt-4o",
        azure_openai_fallback_deployment_name="dep-fallback",
    )

    assert model.build_primary(cfg).kwargs["azure_deployment"] == "my-deploy"
    assert model.build_fallback(cfg).kwargs["azure_deployment"] == "dep-fallback"


def test_microsoft_build_chat_model_alone_always_uses_the_primary_deployment(
    monkeypatch,
):
    # `build_chat_model` is role-agnostic: only build_fallback opts into the
    # fallback deployment.
    _install_fake_chat_class(
        monkeypatch, "langchain_openai", class_name="AzureChatOpenAI"
    )
    cfg = _azure_cfg(
        primary_model="gpt-4o",
        fallback_model="gpt-4o-mini",
        azure_openai_fallback_deployment_name="dep-fallback",
    )

    llm = model.build_chat_model(cfg, model_name="gpt-4o-mini")

    assert llm.kwargs["azure_deployment"] == "my-deploy"


@pytest.mark.parametrize(
    ("missing_field", "env_name"),
    [
        ("azure_openai_api_key", "AZURE_OPENAI_API_KEY"),
        ("azure_openai_endpoint", "AZURE_OPENAI_ENDPOINT"),
        ("azure_openai_api_version", "OPENAI_API_VERSION"),
        ("azure_openai_deployment_name", "AZURE_OPENAI_DEPLOYMENT_NAME"),
    ],
)
def test_microsoft_missing_any_azure_var_raises_naming_it(
    monkeypatch, missing_field, env_name
):
    _install_fake_chat_class(
        monkeypatch, "langchain_openai", class_name="AzureChatOpenAI"
    )
    cfg = _azure_cfg(**{missing_field: None})

    with pytest.raises(RuntimeError, match=env_name):
        model.build_chat_model(cfg, model_name="gpt-4o")


# --- nvidia (langchain-nvidia-ai-endpoints / ChatNVIDIA) ---------------------
#
# NVIDIA has two paths, branched on `nvidia_base_url`: hosted (build.nvidia.com)
# *requires* `NVIDIA_API_KEY`; a self-hosted NIM sets `NVIDIA_BASE_URL` and needs
# no key.


def _nvidia_module_name() -> str:
    return "langchain_nvidia_ai_endpoints"


def test_nvidia_is_a_registered_provider():
    assert "nvidia" in model._BUILDERS


def test_nvidia_hosted_builds_chatnvidia_threading_model_key_and_temperature(
    monkeypatch,
):
    fake_cls = _install_fake_chat_class(
        monkeypatch, _nvidia_module_name(), class_name="ChatNVIDIA"
    )
    cfg = FundConfig(llm_provider="nvidia", nvidia_api_key="nvapi-test")

    llm = model.build_chat_model(cfg, model_name="meta/llama-3.1-70b-instruct")

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["model"] == "meta/llama-3.1-70b-instruct"
    assert llm.kwargs["api_key"] == "nvapi-test"
    assert llm.kwargs["temperature"] == cfg.model_temperature
    # Hosted path: no base_url override (targets build.nvidia.com).
    assert "base_url" not in llm.kwargs


def test_nvidia_hosted_without_key_raises_runtimeerror(monkeypatch):
    _install_fake_chat_class(
        monkeypatch, _nvidia_module_name(), class_name="ChatNVIDIA"
    )
    cfg = FundConfig(llm_provider="nvidia", nvidia_api_key=None)

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        model.build_chat_model(cfg, model_name="meta/llama-3.1-70b-instruct")


def test_nvidia_self_host_base_url_builds_without_a_key(monkeypatch):
    fake_cls = _install_fake_chat_class(
        monkeypatch, _nvidia_module_name(), class_name="ChatNVIDIA"
    )
    cfg = FundConfig(
        llm_provider="nvidia",
        nvidia_api_key=None,
        nvidia_base_url="http://localhost:8000/v1",
    )

    llm = model.build_chat_model(cfg, model_name="meta/llama-3.1-8b-instruct")

    assert isinstance(llm, fake_cls)
    assert llm.kwargs["base_url"] == "http://localhost:8000/v1"
    assert "api_key" not in llm.kwargs  # self-hosted NIM needs no key
    assert llm.kwargs["temperature"] == cfg.model_temperature


def test_nvidia_never_passes_the_ollama_only_reasoning_kwarg(monkeypatch):
    # `reasoning=` is Ollama-only; ChatNVIDIA would reject it.
    _install_fake_chat_class(
        monkeypatch, _nvidia_module_name(), class_name="ChatNVIDIA"
    )
    cfg = FundConfig(llm_provider="nvidia", nvidia_api_key="nvapi-test")

    llm = model.build_chat_model(cfg, model_name="meta/llama-3.1-70b-instruct")

    assert "reasoning" not in llm.kwargs


# --- huggingface (langchain-huggingface, cloud-only) -------------------------
#
# HF is cloud-only: `ChatHuggingFace(llm=HuggingFaceEndpoint(...))`
# from HUGGINGFACEHUB_API_TOKEN. `hf_mode == "local"` raises a deferral RuntimeError
# *before* the SDK import (so it fires even without the extra installed). The
# builder imports two names from one module, so a dedicated recording fake is used.

# Stub token for the tests below. Named without a secret-like substring so ruff's
# S105/S106 (bandit hardcoded-password) don't flag it — the `hf_api_token` field
# name itself trips those checks on a bare string literal.
_FAKE_HF = "hf-test"


def _install_fake_huggingface(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type, type]:
    """Fake ``langchain_huggingface`` with recording ChatHuggingFace + endpoint.

    Returns ``(ChatHuggingFace, HuggingFaceEndpoint)`` recorder classes so a test
    can assert both the outer chat wrapper and the inner endpoint's kwargs.
    """

    class _RecordingEndpoint:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _RecordingChat:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.llm = kwargs.get("llm")

    fake = types.ModuleType("langchain_huggingface")
    fake.ChatHuggingFace = _RecordingChat  # type: ignore[attr-defined]
    fake.HuggingFaceEndpoint = _RecordingEndpoint  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake)
    return _RecordingChat, _RecordingEndpoint


def test_huggingface_is_a_registered_provider():
    assert "huggingface" in model._BUILDERS


def test_huggingface_cloud_builds_chathuggingface_wrapping_the_endpoint(monkeypatch):
    chat_cls, endpoint_cls = _install_fake_huggingface(monkeypatch)
    # hf_mode defaults to "cloud".
    cfg = FundConfig(llm_provider="huggingface", hf_api_token=_FAKE_HF)

    llm = model.build_chat_model(cfg, model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    assert isinstance(llm, chat_cls)
    endpoint = llm.llm
    assert isinstance(endpoint, endpoint_cls)
    assert endpoint.kwargs["repo_id"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert endpoint.kwargs["huggingfacehub_api_token"] == _FAKE_HF
    assert endpoint.kwargs["temperature"] == cfg.model_temperature


def test_huggingface_without_token_raises_runtimeerror(monkeypatch):
    _install_fake_huggingface(monkeypatch)
    cfg = FundConfig(llm_provider="huggingface", hf_api_token=None)

    with pytest.raises(RuntimeError, match="HUGGINGFACEHUB_API_TOKEN"):
        model.build_chat_model(cfg, model_name="m")


def test_huggingface_local_mode_raises_deferral_error():
    # Local HF is deferred; the error fires *before* the SDK import, so no fake is
    # installed here — it must raise even when the extra is absent.
    cfg = FundConfig(llm_provider="huggingface", hf_api_token=_FAKE_HF, hf_mode="local")

    with pytest.raises(RuntimeError, match="not yet supported"):
        model.build_chat_model(cfg, model_name="m")


def test_huggingface_never_passes_the_ollama_only_reasoning_kwarg(monkeypatch):
    # `reasoning=` is Ollama-only; neither HF class accepts it.
    _install_fake_huggingface(monkeypatch)
    cfg = FundConfig(llm_provider="huggingface", hf_api_token=_FAKE_HF)

    llm = model.build_chat_model(cfg, model_name="m")

    assert "reasoning" not in llm.kwargs
    assert "reasoning" not in llm.llm.kwargs


# --- temperature is sent only to models that accept it -----------------------
#
# Anthropic's Fable family and OpenAI's o-series reject a non-default sampling
# temperature. These tests use the real SDK classes and inspect the request body
# the SDK would send, because construction alone succeeds for every model id.


def _sent_temperature(llm: object) -> object:
    """Temperature in the request body, wherever the SDK places it.

    ``anthropic>=1`` no longer accepts ``temperature`` as a named argument, so
    ``langchain-anthropic`` relocates it into ``extra_body``.
    """
    from langchain_core.messages import HumanMessage

    payload = llm._get_request_payload([HumanMessage("hi")])  # type: ignore[attr-defined]
    extra_body = payload.get("extra_body") or {}
    return payload.get("temperature", extra_body.get("temperature"))


def test_anthropic_fable_request_payload_builds_without_a_temperature():
    cfg = FundConfig(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    llm = model.build_chat_model(cfg, model_name="claude-fable-5")

    assert _sent_temperature(llm) is None


def test_anthropic_sonnet_keeps_the_configured_temperature():
    cfg = FundConfig(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    llm = model.build_chat_model(cfg, model_name="claude-sonnet-4-5")

    assert _sent_temperature(llm) == cfg.model_temperature


@pytest.mark.parametrize("model_name", ["o1", "o3", "o3-mini", "o4-mini"])
def test_openai_reasoning_models_are_built_without_a_temperature(model_name):
    cfg = FundConfig(llm_provider="openai", openai_api_key="sk-openai-test")

    llm = model.build_chat_model(cfg, model_name=model_name)

    # langchain-openai itself pins o1 to the default temperature of 1.
    assert _sent_temperature(llm) in (None, 1)


@pytest.mark.parametrize(
    ("provider", "model_name", "expected"),
    [
        ("anthropic", "claude-fable-5", None),
        ("anthropic", "claude-sonnet-4-5", 0.0),
        ("openai", "o3-mini", None),
        ("openai", "gpt-4o", 0.0),
        ("microsoft", "o4-mini", None),
        ("microsoft", "gpt-4o", 0.0),
        ("groq", "o1", 0.0),
        ("ollama", "deepseek-v4-pro:cloud", 0.0),
    ],
)
def test_effective_temperature_reports_what_the_provider_is_sent(
    provider, model_name, expected
):
    # The audit record of a run must say None for a model built without a
    # temperature, and only for the families whose builder skips it.
    cfg = FundConfig(llm_provider=provider)

    assert model.effective_temperature(cfg, model_name) == expected


def test_openai_chat_model_keeps_the_configured_temperature():
    cfg = FundConfig(llm_provider="openai", openai_api_key="sk-openai-test")

    llm = model.build_chat_model(cfg, model_name="gpt-4o")

    assert _sent_temperature(llm) == cfg.model_temperature


def test_azure_reasoning_model_is_built_without_a_temperature():
    llm = model.build_chat_model(_azure_cfg(), model_name="o3-mini")

    assert _sent_temperature(llm) in (None, 1)


# --- completion budget is not left at a provider's tiny default --------------
#
# ChatNVIDIA defaults to 1024 completion tokens and the Hugging Face wrapper to
# 512; a long `task` brief or tool-call argument is cut off mid-JSON. Both are
# checked through the real SDK classes' request payloads.

# Smallest budget that still holds a full allocator report plus tool arguments.
_MIN_COMPLETION_TOKENS = 4096


@pytest.mark.parametrize(
    "extra",
    [{"nvidia_api_key": "nvapi-test"}, {"nvidia_base_url": "http://localhost:8000/v1"}],
    ids=["hosted", "self-hosted"],
)
def test_nvidia_request_payload_allows_a_long_completion(extra):
    cfg = FundConfig(llm_provider="nvidia", **extra)

    llm = model.build_chat_model(cfg, model_name="meta/llama-3.1-70b-instruct")

    payload = llm._get_payload(
        inputs=[{"role": "user", "content": "hi"}], stop=None, stream=False
    )
    assert payload["max_tokens"] >= _MIN_COMPLETION_TOKENS


def test_huggingface_chat_wrapper_sends_a_zero_temperature():
    # ChatHuggingFace inherits temperature from the endpoint only when truthy, so
    # the falsy 0.0 contract is silently dropped unless set on the wrapper itself.
    cfg = FundConfig(llm_provider="huggingface", hf_api_token=_FAKE_HF)

    llm = model.build_chat_model(cfg, model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    assert llm._default_params["temperature"] == cfg.model_temperature


def test_huggingface_chat_wrapper_allows_a_long_completion():
    cfg = FundConfig(llm_provider="huggingface", hf_api_token=_FAKE_HF)

    llm = model.build_chat_model(cfg, model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    assert llm._default_params["max_tokens"] >= _MIN_COMPLETION_TOKENS
    assert llm.llm.max_new_tokens >= _MIN_COMPLETION_TOKENS
