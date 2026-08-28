"""Guard: the LLM-provider allow-list is cloud-only and IDENTICAL across the
three places that enforce it — the install wizard (``wizard._CLOUD_PROVIDERS``),
the live key-validators (``validators.validate_llm``), and the daemon
``Settings.llm_provider`` validator.

This locks them in sync so a future edit to one cannot silently diverge, and
documents that Ollama/local inference is deliberately unsupported (see
baml_src/clients.baml + config.py) — the wizard being cloud-only is therefore
consistent with the pipeline, not a gap.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.setup import validators, wizard

_CLOUD = ("openai", "anthropic")


def test_wizard_offers_exactly_the_cloud_providers() -> None:
    assert tuple(wizard._CLOUD_PROVIDERS) == _CLOUD


def test_config_accepts_each_wizard_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for provider in wizard._CLOUD_PROVIDERS:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert Settings().llm_provider == provider


def test_validators_dispatch_each_wizard_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every wizard provider must route through validate_llm without a ValueError,
    # proving the wizard can never offer a provider the validators can't check.
    monkeypatch.setattr(validators, "validate_openai", lambda k: True)
    monkeypatch.setattr(validators, "validate_anthropic", lambda k: True)
    for provider in wizard._CLOUD_PROVIDERS:
        assert validators.validate_llm(provider, "key") is True


def test_ollama_rejected_by_config_validators_and_wizard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # config: pydantic validator rejects local providers
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    with pytest.raises(ValidationError):
        Settings()
    # validators: dispatch raises for an unsupported provider
    with pytest.raises(ValueError):
        validators.validate_llm("ollama", "key")
    # wizard: non-interactive setup fails loud (docker patched so the provider
    # gate is what trips, not the environment)
    monkeypatch.setattr(wizard.docker_bootstrap, "check_docker", lambda: None)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(
            passphrase="pw", llm_provider="ollama", llm_key="key"
        )
