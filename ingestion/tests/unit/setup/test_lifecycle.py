"""Lifecycle orchestration contract (SPEC D6/D10).

`run_start` decrypts the store, renders compose secrets, and brings the full stack
up (fund + ingestion); `run_stop` stops only the fund, leaving the scheduler up;
`run_status` reports docker + service health. All collaborators are patched.
"""

from pathlib import Path

import pytest

from app.setup import lifecycle

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict:
    env_fund = tmp_path / ".env.fund"
    env_scheduler = tmp_path / ".env.scheduler"
    calls: dict = {
        "rendered": None,
        "compose": [],
        "cleaned": False,
        "env_fund_path": env_fund,
        "env_scheduler_path": env_scheduler,
        "build_profiles": None,
        "force_build": None,
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
        lifecycle.compose_secrets,
        "cleanup",
        lambda **kw: calls.update(cleaned=True, cleaned_names=kw.get("names")),
    )
    # Default to no persisted LLM config; real compose_env.render/cleanup run, but
    # against tmp env files so they never touch the repo root.
    monkeypatch.setattr(lifecycle.config_file, "load_config", lambda **kw: {})
    monkeypatch.setattr(lifecycle.compose_env, "DEFAULT_ENV_FUND_PATH", env_fund)
    monkeypatch.setattr(
        lifecycle.compose_env, "DEFAULT_ENV_SCHEDULER_PATH", env_scheduler
    )
    monkeypatch.setattr(lifecycle.docker_bootstrap, "check_docker", lambda: None)
    monkeypatch.setattr(lifecycle.docker_bootstrap, "compose_dir", lambda: tmp_path)

    def _fake_build_and_up(**kw: object) -> list[str]:
        calls["build_profiles"] = kw.get("profiles")
        calls["force_build"] = kw.get("force_build")
        calls["compose"].append("up")
        return []

    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", _fake_build_and_up)
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "stop_services",
        lambda services: calls["compose"].append(f"stop:{','.join(services)}"),
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
    assert patched["compose"] == []


def test_run_stop_stops_fund_and_keeps_scheduler(patched: dict) -> None:
    """Stopping the fund halts only that service; the scheduler's secrets and
    .env.scheduler stay, and only the fund's own .env.fund and fund-only LLM keys
    are removed."""
    patched["env_fund_path"].write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    patched["env_scheduler_path"].write_text(
        "TRADING_212_MODE=demo\n", encoding="utf-8"
    )
    lifecycle.run_stop()
    assert patched["compose"] == ["stop:fund"]
    assert patched["cleaned_names"] == lifecycle.compose_secrets.FUND_ONLY_SECRET_NAMES
    assert not patched["env_fund_path"].exists()  # fund's own env file wiped
    # Wiping it would let a later `docker compose up` recreate the scheduler
    # without the mode, so a demo key would 401 against the live host.
    assert (
        patched["env_scheduler_path"].read_text(encoding="utf-8")
        == "TRADING_212_MODE=demo\n"
    )


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
    """Secrets and both env files are rendered before `compose up` — a reordering
    that started the stack before rendering would be a regression."""
    order: list[str] = []
    monkeypatch.setattr(
        lifecycle.compose_secrets,
        "render",
        lambda secrets, **kw: order.append("secrets"),
    )
    monkeypatch.setattr(
        lifecycle.compose_env, "render", lambda config, **kw: order.append("env")
    )
    monkeypatch.setattr(
        lifecycle.compose_env,
        "render_scheduler",
        lambda config, **kw: order.append("scheduler-env"),
    )

    def _fake_build_and_up(**kw: object) -> list[str]:
        order.append("up")
        return []

    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", _fake_build_and_up)
    lifecycle.run_start("pw")
    assert order == ["secrets", "env", "scheduler-env", "up"]


def test_run_start_hands_the_trading_212_mode_to_the_scheduler(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode the wizard validated the keys against lands in .env.scheduler, not
    in the fund's file that `portopt stop` deletes."""
    monkeypatch.setattr(
        lifecycle.config_file,
        "load_config",
        lambda **kw: {"trading_212_mode": "demo", "llm_provider": "openai"},
    )
    lifecycle.run_start("pw")
    scheduler_text = patched["env_scheduler_path"].read_text(encoding="utf-8")
    assert scheduler_text == "TRADING_212_MODE=demo\n"
    assert "TRADING_212_MODE" not in patched["env_fund_path"].read_text(
        encoding="utf-8"
    )


def test_run_start_writes_empty_scheduler_env_when_unconfigured(
    patched: dict,
) -> None:
    """With no persisted mode the file is still written (empty) so the scheduler's
    env_file entry resolves and the daemon's own default applies."""
    lifecycle.run_start("pw")
    assert patched["env_scheduler_path"].read_text(encoding="utf-8") == ""


def test_run_start_brings_up_fund_and_ingestion(patched: dict) -> None:
    """run_start brings up BOTH the fund engine and the persistent ingestion daemon
    (every compose service is profile-gated, so a bare `up` would start nothing)."""
    lifecycle.run_start("pw")
    assert patched["build_profiles"] == ("fund", "ingestion")


def test_run_start_defaults_to_reusing_images(patched: dict) -> None:
    """A plain start reuses existing images, so force_build stays False."""
    lifecycle.run_start("pw")
    assert patched["force_build"] is False


def test_run_start_rebuild_forces_image_build(patched: dict) -> None:
    """`--rebuild` threads force_build=True through so a start after a code or
    migration change does not silently reuse a stale image."""
    lifecycle.run_start("pw", rebuild=True)
    assert patched["force_build"] is True


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


@pytest.fixture
def real_render(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict:
    """A checkout and an unrelated cwd, with the real secret/env renderers."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("OPTIMIZER_REPO", str(repo))
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(
        lifecycle.secret_store, "load_secrets", lambda pw, **kw: {"fred_api_key": "fk"}
    )
    monkeypatch.setattr(lifecycle.config_file, "load_config", lambda **kw: {})
    monkeypatch.setattr(lifecycle.docker_bootstrap, "check_docker", lambda: None)
    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", lambda **kw: [])
    monkeypatch.setattr(lifecycle.docker_bootstrap, "stop_services", lambda s: None)
    return {"repo": repo, "elsewhere": elsewhere}


def test_run_start_renders_plaintext_into_the_checkout_not_the_cwd(
    real_render: dict,
) -> None:
    """Compose mounts `<checkout>/secrets/*`; rendering into the cwd would leave those
    mount sources missing and scatter plaintext keys into an arbitrary directory."""
    lifecycle.run_start("pw")
    repo, elsewhere = real_render["repo"], real_render["elsewhere"]
    assert (repo / "secrets" / "fred_api_key").read_text(encoding="utf-8") == "fk"
    assert (repo / ".env.fund").is_file()
    assert not (elsewhere / "secrets").exists()
    assert not (elsewhere / ".env.fund").exists()


def test_run_start_without_a_checkout_writes_nothing(
    real_render: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No docker-compose.yml anywhere: fail before decrypting or rendering."""
    from app.setup import path_install

    elsewhere = real_render["elsewhere"]
    monkeypatch.delenv("OPTIMIZER_REPO", raising=False)
    monkeypatch.setattr(path_install.config_file, "load_config", lambda **kw: {})
    monkeypatch.setattr(path_install, "_repo_root", lambda: elsewhere)
    loaded: list[str] = []
    monkeypatch.setattr(
        lifecycle.secret_store,
        "load_secrets",
        lambda pw, **kw: loaded.append(pw) or {},
    )
    with pytest.raises(lifecycle.docker_bootstrap.DockerError):
        lifecycle.run_start("pw")
    assert loaded == []
    assert list(elsewhere.iterdir()) == []


def test_run_stop_removes_the_checkouts_env_fund(real_render: dict) -> None:
    repo, elsewhere = real_render["repo"], real_render["elsewhere"]
    (repo / ".env.fund").write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    (elsewhere / ".env.fund").write_text("UNRELATED=1\n", encoding="utf-8")
    lifecycle.run_stop()
    assert not (repo / ".env.fund").exists()
    assert (elsewhere / ".env.fund").exists()


def test_cleanup_rendered_wipes_the_checkouts_plaintext(real_render: dict) -> None:
    repo = real_render["repo"]
    lifecycle.run_start("pw")
    lifecycle.cleanup_rendered()
    assert not (repo / ".env.fund").exists()
    assert not (repo / "secrets" / "fred_api_key").exists()


def test_cleanup_rendered_is_a_noop_without_a_checkout(
    real_render: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.setup import path_install

    elsewhere = real_render["elsewhere"]
    monkeypatch.delenv("OPTIMIZER_REPO", raising=False)
    monkeypatch.setattr(path_install.config_file, "load_config", lambda **kw: {})
    monkeypatch.setattr(path_install, "_repo_root", lambda: elsewhere)
    lifecycle.cleanup_rendered()  # must not raise or touch the cwd
    assert list(elsewhere.iterdir()) == []


_DAEMON_FILES = ("trading_212_api_key", "trading_212_secret_key", "fred_api_key")
_FUND_ONLY_FILES = (
    "ollama_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "azure_openai_api_key",
)


def _render_with_every_key(real_render: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lifecycle.secret_store,
        "load_secrets",
        lambda pw, **kw: dict.fromkeys((*_DAEMON_FILES, *_FUND_ONLY_FILES), "live"),
    )
    lifecycle.run_start("pw")


def test_run_stop_wipes_the_fund_only_keys_but_keeps_the_schedulers(
    real_render: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler keeps running (and restarts on boot), so its three mounted files
    must survive a stop; the LLM keys only the stopped fund mounts must not."""
    _render_with_every_key(real_render, monkeypatch)
    secrets = real_render["repo"] / "secrets"
    lifecycle.run_stop()
    for name in _FUND_ONLY_FILES:
        assert not (secrets / name).exists(), name
    for name in _DAEMON_FILES:
        assert (secrets / name).read_text(encoding="utf-8") == "live", name


def _compose_services() -> set[str]:
    """Every service docker-compose.yml declares, whatever profile gates it."""
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text("utf-8"))
    return set(doc["services"])


def test_run_stop_all_stops_every_service_and_wipes_every_rendered_secret(
    real_render: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped: list[list[str]] = []
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "stop_services", lambda s: stopped.append(list(s))
    )
    _render_with_every_key(real_render, monkeypatch)
    repo = real_render["repo"]
    lifecycle.run_stop(all_services=True)
    assert len(stopped) == 1
    assert set(stopped[0]) == _compose_services()
    assert stopped[0][-1] == "db"
    for name in (*_DAEMON_FILES, *_FUND_ONLY_FILES):
        assert not (repo / "secrets" / name).exists(), name
    assert not (repo / ".env.fund").exists()
