"""The Anthropic chat model must not carry its API key into any rendering of itself.

LangChain hands ``repr(model)`` and ``dumpd(model)`` to every callback and tracer
(LangSmith, debug logging), so a key that shows up in either is exported verbatim.
These tests build the real ``ChatAnthropic`` (construction makes no request) rather
than the fake class of ``test_model_factory``, because only the real class decides
which of its fields are secret.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.load import dumpd, dumps

from fund.agents import model
from fund.config import FundConfig

_SENTINEL = "S3NTINEL-anthropic-api-key"


@pytest.fixture
def chat_model() -> BaseChatModel:
    """Build the real ``ChatAnthropic`` the factory returns for the sentinel key."""
    cfg = FundConfig(llm_provider="anthropic", anthropic_api_key=_SENTINEL)
    return model.build_chat_model(cfg, model_name="claude-sonnet-4-5")


@pytest.mark.criterion("REQ-FUND-018")
def test_repr_and_str_hide_the_anthropic_api_key(chat_model: BaseChatModel) -> None:
    """Validates ``fund.agents.model._build_anthropic``."""
    assert _SENTINEL not in repr(chat_model)
    assert _SENTINEL not in str(chat_model)


@pytest.mark.criterion("REQ-FUND-018")
def test_serialised_forms_hide_the_anthropic_api_key(
    chat_model: BaseChatModel,
) -> None:
    """Validates ``fund.agents.model._build_anthropic``."""
    assert _SENTINEL not in dumps(chat_model)
    assert _SENTINEL not in json.dumps(dumpd(chat_model))
    assert _SENTINEL not in str(chat_model.model_dump())
