"""Contract: ``.env.example`` documents the D4 switchable-LLM-backend env vars.

Task 13 (SPEC "Switchable LLM Backends", success criterion 8) requires the
repo-root ``.env.example`` to carry a per-provider block: the provider selector
(``LLM_PROVIDER``), the shared model-id pair (``FUND_PRIMARY_MODEL`` /
``FUND_FALLBACK_MODEL``), and each provider's primary auth env var. This guard
fails the build if a provider is wired into ``fund.config`` but its env var is
never documented for an operator.

The env-var names below are the D4 contract sourced in
``fund/src/fund/config.py::load_config``.

``Path(__file__).resolve().parents[3]`` resolves to the repo root
(unit → tests → fund → optimizer), never ``Path.cwd()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"

# The D4 switchable-backend env-var contract (fund.config.load_config). One
# primary auth/selection var per provider plus the shared selector + model pair.
_REQUIRED_ENV_VARS = [
    "LLM_PROVIDER",
    "FUND_PRIMARY_MODEL",
    "FUND_FALLBACK_MODEL",
    "OLLAMA_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "NVIDIA_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
    "AWS_REGION",
    "AZURE_OPENAI_API_KEY",
]


def _env_example_text() -> str:
    return _ENV_EXAMPLE.read_text(encoding="utf-8")


@pytest.mark.parametrize("env_var", _REQUIRED_ENV_VARS)
def test_env_example_documents_provider_env_var(env_var: str):
    assert env_var in _env_example_text(), (
        f"{env_var} must be documented in .env.example (D4 provider contract)"
    )
