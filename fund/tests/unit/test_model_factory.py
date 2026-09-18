"""Task 1 — ``fund.agents.model`` DeepSeek-on-Ollama-Cloud factory (unit slice).

Construction only: ``ChatOllama`` is built but never invoked, so no network is
touched. The tests assert the frozen ``FundConfig`` (D4) is threaded onto the
model — base URL, model id, ``temperature``, non-thinking ``reasoning`` route, and
the ``OLLAMA_API_KEY`` bearer header — and that a missing key raises at *build*
time (never at import, so a bare ``import fund.agents.model`` stays green in CI).
"""

from __future__ import annotations

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
