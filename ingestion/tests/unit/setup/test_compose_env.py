"""Contract: config.toml -> fund env-var translation + .env.fund rendering.

`config_to_env` is a pure map from the wizard's persisted config keys to the exact
env-var names `fund.config.load_config` reads; `render`/`cleanup` write/remove the
generated `.env.fund` env_file the fund service mounts.
"""

from __future__ import annotations

from pathlib import Path

from app.setup import compose_env
from app.setup.compose_env import config_to_env


def test_translates_provider_and_model() -> None:
    """provider/model map to LLM_PROVIDER / FUND_PRIMARY_MODEL, plus a mirrored
    fallback for a non-ollama provider."""
    assert config_to_env({"llm_provider": "openai", "llm_model": "gpt-4o"}) == {
        "LLM_PROVIDER": "openai",
        "FUND_PRIMARY_MODEL": "gpt-4o",
        "FUND_FALLBACK_MODEL": "gpt-4o",
    }


def test_fallback_mirrors_primary_for_non_ollama_provider() -> None:
    """fund.config's fallback default is an Ollama model id, invalid on another
    provider — so a non-ollama selection mirrors the primary as the fallback."""
    env = config_to_env({"llm_provider": "anthropic", "llm_model": "claude-opus-4-5"})
    assert env["FUND_FALLBACK_MODEL"] == "claude-opus-4-5"


def test_no_fallback_emitted_for_ollama() -> None:
    """On ollama the fund's own fallback default is valid, so nothing is written
    (avoids clobbering it)."""
    env = config_to_env({"llm_provider": "ollama", "llm_model": "deepseek-v4.1:cloud"})
    assert "FUND_FALLBACK_MODEL" not in env


def test_explicit_fallback_model_respected() -> None:
    """A captured llm_fallback_model wins over the mirror."""
    env = config_to_env(
        {
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "llm_fallback_model": "gpt-4o-mini",
        }
    )
    assert env["FUND_FALLBACK_MODEL"] == "gpt-4o-mini"


def test_empty_string_values_are_omitted() -> None:
    """Empty values must be dropped, not emitted as `KEY=` (which would pin the
    field to "" and shadow the fund's code default)."""
    assert (
        config_to_env({"llm_provider": "", "llm_model": "", "llm_base_url": ""}) == {}
    )


def test_azure_api_version_uses_openai_env_name() -> None:
    """fund.config reads OPENAI_API_VERSION, not AZURE_OPENAI_API_VERSION."""
    env = config_to_env(
        {
            "llm_provider": "microsoft",
            "azure_openai_endpoint": "https://x.openai.azure.com/",
            "azure_openai_api_version": "2024-02-01",
            "azure_openai_deployment_name": "gpt4o-dep",
        }
    )
    assert env["AZURE_OPENAI_ENDPOINT"] == "https://x.openai.azure.com/"
    assert env["OPENAI_API_VERSION"] == "2024-02-01"
    assert "AZURE_OPENAI_API_VERSION" not in env
    assert env["AZURE_OPENAI_DEPLOYMENT_NAME"] == "gpt4o-dep"


def test_aws_region() -> None:
    """aws region maps to AWS_REGION alongside the provider selector."""
    assert config_to_env({"llm_provider": "aws", "aws_region": "eu-west-1"}) == {
        "LLM_PROVIDER": "aws",
        "AWS_REGION": "eu-west-1",
    }


def test_base_url_routes_per_provider() -> None:
    """A generic llm_base_url lands on the provider's own *_BASE_URL var."""
    assert (
        config_to_env({"llm_provider": "ollama", "llm_base_url": "http://h:11434"})[
            "OLLAMA_BASE_URL"
        ]
        == "http://h:11434"
    )
    assert (
        config_to_env({"llm_provider": "openai", "llm_base_url": "http://proxy"})[
            "OPENAI_BASE_URL"
        ]
        == "http://proxy"
    )
    assert (
        config_to_env({"llm_provider": "nvidia", "llm_base_url": "http://nim"})[
            "NVIDIA_BASE_URL"
        ]
        == "http://nim"
    )


def test_base_url_dropped_for_provider_without_a_base_url_field() -> None:
    """anthropic/google/groq/huggingface/openrouter have no base-url env var."""
    env = config_to_env({"llm_provider": "anthropic", "llm_base_url": "http://nope"})
    assert not any("BASE_URL" in key for key in env)


def test_only_present_keys_emitted() -> None:
    """An absent setting is omitted, never emitted as "" (which would clobber the
    env.get(name, default) fallback in fund.config)."""
    env = config_to_env({"llm_provider": "ollama"})
    assert env == {"LLM_PROVIDER": "ollama"}


def test_non_llm_keys_ignored() -> None:
    """Only the known LLM keys are translated; other config keys are dropped."""
    env = config_to_env({"llm_provider": "openai", "universe_source": "yfinance"})
    assert "UNIVERSE_SOURCE" not in env
    assert "universe_source" not in env


def test_empty_config() -> None:
    """No config yields no env vars."""
    assert config_to_env({}) == {}


def test_render_writes_translated_env_file(tmp_path: Path) -> None:
    """render writes KEY=value lines under the fund's env-var names."""
    path = tmp_path / ".env.fund"
    compose_env.render({"llm_provider": "openai", "llm_model": "gpt-4o"}, path=path)
    text = path.read_text(encoding="utf-8")
    assert "LLM_PROVIDER=openai" in text
    assert "FUND_PRIMARY_MODEL=gpt-4o" in text


def test_render_writes_empty_file_when_unconfigured(tmp_path: Path) -> None:
    """render always creates the file (empty) so the env_file entry resolves."""
    path = tmp_path / ".env.fund"
    compose_env.render({}, path=path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_render_never_writes_a_secret(tmp_path: Path) -> None:
    """Provider keys flow via docker secrets — never the plaintext env file."""
    path = tmp_path / ".env.fund"
    compose_env.render({"llm_provider": "openai", "llm_model": "gpt-4o"}, path=path)
    text = path.read_text(encoding="utf-8")
    assert "KEY" not in text
    assert "TOKEN" not in text
    assert "SECRET" not in text


def test_cleanup_removes_file(tmp_path: Path) -> None:
    """cleanup deletes a rendered .env.fund."""
    path = tmp_path / ".env.fund"
    path.write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    compose_env.cleanup(path=path)
    assert not path.exists()


def test_cleanup_missing_file_is_noop(tmp_path: Path) -> None:
    """cleanup on an absent file does not raise."""
    compose_env.cleanup(path=tmp_path / "absent")
