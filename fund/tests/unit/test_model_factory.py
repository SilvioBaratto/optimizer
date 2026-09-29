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

import sys
import types

import pytest

from fund.agents import model
from fund.config import FundConfig

# A config carrying every default plus a key, so build never hits use-time None.
_CFG = FundConfig(ollama_api_key="sk-ollama-test")


def test_build_chat_model_threads_config_onto_chatollama():
    llm = model.build_chat_model(_CFG, model_name="deepseek-v4.1-flash:cloud")

    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == "deepseek-v4.1-flash:cloud"
    assert llm.base_url == _CFG.ollama_base_url
    assert llm.temperature == _CFG.model_temperature
    assert llm.reasoning is _CFG.model_reasoning


def test_build_chat_model_sends_api_key_as_bearer_header():
    llm = model.build_chat_model(_CFG, model_name="deepseek-v4-pro:cloud")

    headers = llm.client_kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-ollama-test"


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


# --- Ollama cloud-vs-local key rule -----------------------------------------


def test_ollama_cloud_default_without_key_raises_naming_the_env_var():
    # Default base_url is the cloud host, so a missing key still raises here.
    cfg = FundConfig(ollama_api_key=None)
    assert "ollama.com" in cfg.ollama_base_url
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        model.build_chat_model(cfg, model_name="deepseek-v4.1-flash:cloud")


def test_ollama_local_base_url_builds_without_a_key():
    cfg = FundConfig(ollama_api_key=None, ollama_base_url="http://localhost:11434")

    llm = model.build_chat_model(cfg, model_name="llama3")

    assert type(llm).__name__ == "ChatOllama"
    assert "headers" not in llm.client_kwargs  # no key ⇒ no bearer header


def test_ollama_local_with_key_still_sends_the_bearer_header():
    cfg = FundConfig(
        ollama_api_key="sk-local", ollama_base_url="http://localhost:11434"
    )

    llm = model.build_chat_model(cfg, model_name="llama3")

    assert llm.client_kwargs["headers"]["Authorization"] == "Bearer sk-local"


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
