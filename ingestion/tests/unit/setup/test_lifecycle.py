"""Lifecycle orchestration contract (SPEC D6/D10, task T7b).

`run_start` decrypts the store, renders compose secrets, and brings the stack up;
`run_stop` tears it down and wipes the plaintext secret files; `run_status`
reports docker + service health. All collaborators are patched.
"""

import pytest

from app.setup import lifecycle


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict:
    env_fund = tmp_path / ".env.fund"
    calls: dict = {
        "rendered": None,
        "compose": [],
        "cleaned": False,
        "env_fund_path": env_fund,
        "build_profile": None,
    }
    monkeypatch.setattr(
        lifecycle.secret_store,
        "load_secrets",
        lambda passphrase, **kw: {"fred_api_key": "fk"},
    )
    monkeypatch.setattr(
        lifecycle.compose_secrets,
        "render",
        lambda secrets, **kw: calls.update(rendered=dict(secrets)),
    )
    monkeypatch.setattr(
        lifecycle.compose_secrets, "cleanup", lambda **kw: calls.update(cleaned=True)
    )
    # Default to no persisted LLM config; real compose_env.render/cleanup run, but
    # against a tmp .env.fund so they never touch the repo root.
    monkeypatch.setattr(lifecycle.config_file, "load_config", lambda **kw: {})
    monkeypatch.setattr(lifecycle.compose_env, "DEFAULT_ENV_FUND_PATH", env_fund)
    monkeypatch.setattr(lifecycle.docker_bootstrap, "check_docker", lambda: None)

    def _fake_build_and_up(**kw: object) -> list[str]:
        calls["build_profile"] = kw.get("profile")
        calls["compose"].append("up")
        return []

    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", _fake_build_and_up)
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "compose_down",
        lambda: calls["compose"].append("down"),
    )
    return calls


def test_run_start_renders_secrets_then_brings_up(patched: dict) -> None:
    lifecycle.run_start("pw")
    assert patched["rendered"] == {"fred_api_key": "fk"}
    assert patched["compose"] == ["up"]


def test_run_start_requires_passphrase(patched: dict) -> None:
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.run_start("")
    assert patched["compose"] == []


def test_run_start_propagates_bad_passphrase(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.setup.secret_store import InvalidPassphraseError

    def _boom(passphrase: str, **kw: object) -> dict:
        raise InvalidPassphraseError("wrong")

    monkeypatch.setattr(lifecycle.secret_store, "load_secrets", _boom)
    with pytest.raises(InvalidPassphraseError):
        lifecycle.run_start("wrong")
    assert patched["compose"] == []  # nothing brought up


def test_run_stop_tears_down_and_cleans(patched: dict) -> None:
    patched["env_fund_path"].write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    lifecycle.run_stop()
    assert patched["compose"] == ["down"]
    assert patched["cleaned"] is True
    assert not patched["env_fund_path"].exists()  # .env.fund wiped alongside secrets


def test_run_start_writes_env_fund_from_config(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wizard's persisted provider/model surface in .env.fund before up."""
    monkeypatch.setattr(
        lifecycle.config_file,
        "load_config",
        lambda **kw: {"llm_provider": "openai", "llm_model": "gpt-4o"},
    )
    lifecycle.run_start("pw")
    text = patched["env_fund_path"].read_text(encoding="utf-8")
    assert "LLM_PROVIDER=openai" in text
    assert "FUND_PRIMARY_MODEL=gpt-4o" in text
    assert "KEY" not in text  # secrets never land in the plaintext env file
    assert patched["compose"] == ["up"]


def test_run_start_renders_secrets_and_env_before_bringing_up(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secrets and the .env.fund selection are rendered before `compose up` — a
    reordering that started the stack before rendering would be a regression."""
    order: list[str] = []
    monkeypatch.setattr(
        lifecycle.compose_secrets,
        "render",
        lambda secrets, **kw: order.append("secrets"),
    )
    monkeypatch.setattr(
        lifecycle.compose_env, "render", lambda config, **kw: order.append("env")
    )

    def _fake_build_and_up(**kw: object) -> list[str]:
        order.append("up")
        return []

    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", _fake_build_and_up)
    lifecycle.run_start("pw")
    assert order == ["secrets", "env", "up"]


def test_run_start_brings_up_the_fund_profile(patched: dict) -> None:
    """run_start must target the fund profile: every compose service is
    profile-gated (T6), so a bare `up` would start nothing."""
    lifecycle.run_start("pw")
    assert patched["build_profile"] == "fund"


def test_run_start_writes_empty_env_fund_when_unconfigured(patched: dict) -> None:
    """With no persisted LLM config, .env.fund is written empty (env_file resolves)."""
    lifecycle.run_start("pw")
    assert patched["env_fund_path"].read_text(encoding="utf-8") == ""
    assert patched["compose"] == ["up"]


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-opus-4-5"),
        ("ollama", "deepseek-v4.1-flash:cloud"),
    ],
)
def test_wizard_selection_surfaces_correct_fund_env(
    patched: dict, monkeypatch: pytest.MonkeyPatch, provider: str, model: str
) -> None:
    """End-to-end (real config_to_env): a selected provider surfaces under the exact
    env-var names fund.config reads."""
    monkeypatch.setattr(
        lifecycle.config_file,
        "load_config",
        lambda **kw: {"llm_provider": provider, "llm_model": model},
    )
    lifecycle.run_start("pw")
    text = patched["env_fund_path"].read_text(encoding="utf-8")
    assert f"LLM_PROVIDER={provider}" in text
    assert f"FUND_PRIMARY_MODEL={model}" in text


def test_run_status_all_up(patched: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle.docker_bootstrap, "docker_available", lambda: True)
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "running_services",
        lambda: {"db", "scheduler", "fund"},
    )
    assert lifecycle.run_status() == {
        "docker": True,
        "db": True,
        "scheduler": True,
        "fund": True,
    }


def test_run_status_reports_fund_independently(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle.docker_bootstrap, "docker_available", lambda: True)
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "running_services", lambda: {"db", "fund"}
    )
    status = lifecycle.run_status()
    assert status["fund"] is True
    assert status["scheduler"] is False


def test_run_status_docker_down(patched: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle.docker_bootstrap, "docker_available", lambda: False)
    status = lifecycle.run_status()
    assert status == {
        "docker": False,
        "db": False,
        "scheduler": False,
        "fund": False,
    }
