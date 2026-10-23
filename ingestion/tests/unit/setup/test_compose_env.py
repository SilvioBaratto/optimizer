"""Contract: config.toml -> compose env-file translation and rendering.

`config_to_env` is a pure map from the wizard's persisted config keys to the exact
env-var names `fund.config.load_config` reads; `render`/`cleanup` write/remove the
generated `.env.fund` env_file the fund service mounts. The scheduler's one setting
(`TRADING_212_MODE`) lives in a separate `.env.scheduler` so that `portopt stop`,
which deletes `.env.fund`, cannot strip it from a still-running scheduler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.setup import compose_env
from app.setup.compose_env import config_to_env, config_to_scheduler_env


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


def test_trading_212_mode_is_not_a_fund_setting() -> None:
    """.env.fund is the fund's file and is wiped on stop; the scheduler's mode must
    not ride in it."""
    assert config_to_env({"trading_212_mode": "demo"}) == {}


def test_scheduler_env_carries_the_trading_212_mode() -> None:
    """The mode the keys were validated against must reach the daemon: it only
    sees env vars, never config.toml, and defaults to live without this."""
    assert config_to_scheduler_env({"trading_212_mode": "demo"}) == {
        "TRADING_212_MODE": "demo"
    }


def test_scheduler_env_omits_absent_mode_and_ignores_other_keys() -> None:
    """An unset mode stays absent (so the daemon's own default applies); LLM keys
    are the fund's business and never reach the scheduler file."""
    assert config_to_scheduler_env({}) == {}
    assert config_to_scheduler_env({"llm_provider": "openai"}) == {}


def test_scheduler_reads_its_own_env_file_after_root_env() -> None:
    """TRADING_212_MODE rides in its own file, mounted last so the wizard's choice
    overrides a stale root .env; the fund's file is not the scheduler's concern."""
    yaml = pytest.importorskip("yaml")
    compose = Path(__file__).resolve().parents[4] / "docker-compose.yml"
    scheduler = yaml.safe_load(compose.read_text(encoding="utf-8"))["services"][
        "scheduler"
    ]
    entries = [e if isinstance(e, dict) else {"path": e} for e in scheduler["env_file"]]
    paths = [e["path"] for e in entries]
    assert paths[-1] == compose_env.DEFAULT_ENV_SCHEDULER_PATH.name
    assert entries[-1].get("required") is False
    assert compose_env.DEFAULT_ENV_FUND_PATH.name not in paths


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


@pytest.mark.parametrize(
    "loopback_url",
    [
        "http://localhost:11434",
        "http://LOCALHOST:11434",
        "http://127.0.0.1:11434",
        "http://127.0.1.1:11434",
        "http://[::1]:11434",
    ],
)
def test_loopback_base_url_is_pointed_at_the_host_for_the_container(
    loopback_url: str,
) -> None:
    """Inside the fund container ``localhost`` is the container itself, so a
    loopback URL that validated fine from the host must be rendered as the compose
    host alias (the service maps it with ``extra_hosts: host-gateway``)."""
    env = config_to_env({"llm_provider": "ollama", "llm_base_url": loopback_url})
    assert env["OLLAMA_BASE_URL"] == "http://host.docker.internal:11434"


@pytest.mark.parametrize(
    ("provider", "var", "url", "expected"),
    [
        (
            "nvidia",
            "NVIDIA_BASE_URL",
            "http://localhost:8000/v1",
            "http://host.docker.internal:8000/v1",
        ),
        (
            "openai",
            "OPENAI_BASE_URL",
            "https://127.0.0.1:4000/v1?x=1",
            "https://host.docker.internal:4000/v1?x=1",
        ),
        (
            "ollama",
            "OLLAMA_BASE_URL",
            "http://user:pw@localhost/",
            "http://user:pw@host.docker.internal/",
        ),
    ],
)
def test_loopback_rewrite_keeps_scheme_port_path_and_credentials(
    provider: str, var: str, url: str, expected: str
) -> None:
    """Only the host component changes; everything else round-trips verbatim."""
    assert config_to_env({"llm_provider": provider, "llm_base_url": url})[var] == (
        expected
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://host.docker.internal:11434",
        "http://ollama.internal:11434",
        "http://192.168.1.20:11434",
        "https://ollama.com",
        "http://localhost.example.com:11434",
    ],
)
def test_non_loopback_base_url_is_rendered_verbatim(url: str) -> None:
    """Hosts that already resolve from the container are left untouched."""
    env = config_to_env({"llm_provider": "ollama", "llm_base_url": url})
    assert env["OLLAMA_BASE_URL"] == url


def test_loopback_rewrite_does_not_mutate_the_persisted_config() -> None:
    """The wizard re-validates the stored URL from the host next run, so the
    config mapping must keep the user's original loopback form."""
    config = {"llm_provider": "ollama", "llm_base_url": "http://localhost:11434"}
    config_to_env(config)
    assert config["llm_base_url"] == "http://localhost:11434"


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


def test_render_scheduler_writes_the_mode(tmp_path: Path) -> None:
    """render_scheduler writes TRADING_212_MODE=<mode> for the scheduler env_file."""
    path = tmp_path / ".env.scheduler"
    compose_env.render_scheduler({"trading_212_mode": "demo"}, path=path)
    assert path.read_text(encoding="utf-8") == "TRADING_212_MODE=demo\n"


def test_render_scheduler_empties_the_file_when_mode_is_unset(tmp_path: Path) -> None:
    """The file is always rewritten, so a mode dropped from config.toml cannot
    linger from an earlier start and keep pinning the scheduler to it."""
    path = tmp_path / ".env.scheduler"
    path.write_text("TRADING_212_MODE=demo\n", encoding="utf-8")
    compose_env.render_scheduler({}, path=path)
    assert path.read_text(encoding="utf-8") == ""


def test_cleanup_never_touches_the_scheduler_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`portopt stop` and a failed setup both call cleanup() while the scheduler
    keeps running; losing its mode file would make the next `docker compose up`
    recreate it against the live host."""
    fund = tmp_path / ".env.fund"
    scheduler = tmp_path / ".env.scheduler"
    fund.write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    scheduler.write_text("TRADING_212_MODE=demo\n", encoding="utf-8")
    compose_env.cleanup(path=fund)
    assert not fund.exists()
    assert scheduler.read_text(encoding="utf-8") == "TRADING_212_MODE=demo\n"


def test_render_and_cleanup_require_an_explicit_path_never_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default of `./.env.fund` would write next to wherever the operator stood
    while compose read the checkout's own file."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError):
        compose_env.render({"llm_provider": "openai"})  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        compose_env.cleanup()  # type: ignore[call-arg]
    assert list(tmp_path.iterdir()) == []
