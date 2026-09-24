"""``fund.agents.model`` — the provider-registry chat-model factory (unit slice).

Construction only: the SDK class is built but never invoked, so no network is
touched. The tests assert the frozen ``FundConfig`` (D4) is threaded onto the
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

# A config carrying every D4 default plus a key, so build never hits use-time None.
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


# --- provider registry / dispatch (Task 2) ----------------------------------


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
    # A builder whose SDK is absent raises ImportError; build_chat_model catches
    # it and re-raises a RuntimeError naming the `uv sync ... --extra` command.
    def _boom(_config, _model_name):
        raise ImportError("No module named 'langchain_openai'")

    monkeypatch.setitem(model._BUILDERS, "faux", _boom)
    cfg = FundConfig(llm_provider="faux")
    with pytest.raises(RuntimeError, match="--extra faux"):
        model.build_chat_model(cfg, model_name="m")


# --- Ollama cloud-vs-local key rule (Task 2) --------------------------------


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


# --- openai / openrouter (Task 3; langchain-openai / ChatOpenAI) -------------


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
