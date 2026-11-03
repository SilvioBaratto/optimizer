"""When the fund LLM backend counts as configured (issue #1124).

`portopt start` skips the fund container unless this says yes, because the worker
refuses to build its model (and so never writes a heartbeat) otherwise. The facts
are exactly what the wizard persists: non-secret choices in config.toml, keys in
the decrypted secret store. Pure functions, no I/O.
"""

import pytest

from app.setup import llm_backend, validators


def _configured(config: dict[str, object], secrets: dict[str, str]) -> bool:
    return llm_backend.is_configured(config, secrets)


def test_nothing_persisted_is_unconfigured() -> None:
    """A data-pipeline-only install has no LLM config and no LLM key."""
    assert _configured({}, {}) is False


def test_default_provider_is_ollama_cloud_and_needs_its_key() -> None:
    """With no provider stored the default (Ollama Cloud) applies and wants its key."""
    assert _configured({}, {"ollama_api_key": "k"}) is True


def test_default_provider_needs_no_explicit_model() -> None:
    """fund.config falls back to its built-in Ollama model id."""
    assert _configured({"llm_provider": "ollama"}, {"ollama_api_key": "k"}) is True


def test_a_blank_key_is_missing() -> None:
    """compose_secrets renders an empty file for every unset secret."""
    assert _configured({}, {"ollama_api_key": ""}) is False
    assert _configured({}, {"ollama_api_key": "  \n"}) is False


def test_other_secrets_do_not_count() -> None:
    """Trading 212 / FRED keys say nothing about the LLM backend."""
    secrets = {"fred_api_key": "f", "trading_212_api_key": "t"}
    assert _configured({}, secrets) is False


def test_ollama_cloud_base_url_still_needs_the_key() -> None:
    """An explicit ollama.com base URL is the cloud path, so it is keyed."""
    config = {"llm_provider": "ollama", "llm_base_url": "https://ollama.com"}
    assert _configured(config, {}) is False
    assert _configured(config, {"ollama_api_key": "k"}) is True


def test_local_ollama_needs_no_key() -> None:
    """A local daemon is keyless, so base_url alone makes it configured."""
    config = {"llm_provider": "ollama", "llm_base_url": "http://localhost:11434"}
    assert _configured(config, {}) is True


def test_local_base_url_alone_selects_local_ollama() -> None:
    """The provider defaults to ollama when the wizard stored none."""
    assert _configured({"llm_base_url": "http://localhost:11434"}, {}) is True


def test_provider_with_model_and_key_is_configured() -> None:
    """A non-Ollama provider needs its own key and an explicit model id."""
    config = {"llm_provider": "openai", "llm_model": "gpt-4o"}
    assert _configured(config, {"openai_api_key": "k"}) is True


def test_provider_without_its_key_is_unconfigured() -> None:
    """The model id alone does not make a keyed provider usable."""
    config = {"llm_provider": "openai", "llm_model": "gpt-4o"}
    assert _configured(config, {}) is False


def test_a_key_for_another_provider_does_not_count() -> None:
    """Only the selected provider's secret is looked at."""
    config = {"llm_provider": "openai", "llm_model": "gpt-4o"}
    assert _configured(config, {"ollama_api_key": "k"}) is False


def test_non_ollama_provider_without_a_model_is_unconfigured() -> None:
    """build_primary refuses the built-in Ollama model id on another provider."""
    config = {"llm_provider": "openai"}
    assert _configured(config, {"openai_api_key": "k"}) is False


def test_blank_model_is_unconfigured() -> None:
    """Whitespace is not a model id."""
    config = {"llm_provider": "openai", "llm_model": "  "}
    assert _configured(config, {"openai_api_key": "k"}) is False


def test_keyless_provider_still_needs_a_model() -> None:
    """Bedrock needs no key but is still a non-Ollama provider."""
    assert _configured({"llm_provider": "aws"}, {}) is False
    assert _configured({"llm_provider": "aws", "llm_model": "m"}, {}) is True


def test_hosted_nvidia_needs_a_key_but_self_hosted_does_not() -> None:
    """A self-hosted NIM base_url is the keyless path."""
    hosted = {"llm_provider": "nvidia", "llm_model": "m"}
    assert _configured(hosted, {}) is False
    assert _configured(hosted, {"nvidia_api_key": "k"}) is True
    nim = {**hosted, "llm_base_url": "http://nim:8000/v1"}
    assert _configured(nim, {}) is True


def test_unknown_provider_is_unconfigured() -> None:
    """The worker would raise "Unknown LLM provider" at build time."""
    config = {"llm_provider": "nonesuch", "llm_model": "m"}
    assert _configured(config, {"nonesuch_api_key": "k"}) is False


@pytest.mark.parametrize("provider", validators.SUPPORTED_LLM_PROVIDERS)
def test_every_provider_the_wizard_offers_can_reach_configured(provider: str) -> None:
    """A provider added to the wizard must be startable here too."""
    config: dict[str, object] = {"llm_provider": provider, "llm_model": "m"}
    secret = llm_backend.PROVIDER_SECRET_NAMES.get(provider)
    secrets = {secret: "k"} if secret else {}
    assert _configured(config, secrets) is True


@pytest.mark.parametrize(
    ("provider", "base_url", "expected"),
    [
        ("aws", None, False),
        ("ollama", None, True),
        ("ollama", "https://ollama.com", True),
        ("ollama", "http://localhost:11434", False),
        ("nvidia", None, True),
        ("nvidia", "http://nim:8000/v1", False),
        ("openai", None, True),
        ("openai", "http://proxy:4000/v1", True),
    ],
)
def test_needs_key(provider: str, base_url: str | None, expected: bool) -> None:
    """Which provider/base_url combinations require an auth key."""
    assert llm_backend.needs_key(provider, base_url) is expected
