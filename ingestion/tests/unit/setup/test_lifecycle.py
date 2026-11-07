"""Lifecycle orchestration contract (SPEC D6/D10).

`run_start` decrypts the store, renders compose secrets, and brings the full stack
up (fund + ingestion); `run_stop` stops only the fund, leaving the scheduler up;
`run_status` reports docker + service health. All collaborators are patched.
"""

from pathlib import Path

import pytest

from app.setup import lifecycle, path_install

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict:
    # The checkout and the operator's cwd are different directories, so a file that
    # lands next to the cwd instead of under the repo is visible to every test.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("OPTIMIZER_REPO", str(repo))
    env_fund = repo / ".env.fund"
    env_scheduler = repo / ".env.scheduler"
    calls: dict = {
        "rendered": None,
        "secrets_dir": None,
        "compose": [],
        "repo": repo,
        "env_fund_path": env_fund,
        "env_scheduler_path": env_scheduler,
        "build_profiles": None,
        "force_build": None,
        "build_repo": None,
        "stop_repo": None,
    }
    monkeypatch.setattr(
        lifecycle.secret_store,
        "load_secrets",
        lambda passphrase, **kw: {"fred_api_key": "fk"},
    )
    monkeypatch.setattr(
        lifecycle.compose_secrets,
        "render",
        lambda secrets, **kw: calls.update(
            rendered=dict(secrets), secrets_dir=kw.get("secrets_dir")
        ),
    )
    monkeypatch.setattr(
        lifecycle.compose_secrets,
        "cleanup",
        lambda **kw: calls.update(cleaned=True, cleaned_names=kw.get("names")),
    )
    # Default to no persisted LLM config; the real compose_env.render/cleanup run
    # against the tmp repo's env files.
    monkeypatch.setattr(lifecycle.config_file, "load_config", lambda **kw: {})
    monkeypatch.setattr(lifecycle.docker_bootstrap, "ensure_docker", lambda: None)

    def _fake_build_and_up(**kw: object) -> list[str]:
        calls["build_profiles"] = kw.get("profiles")
        calls["force_build"] = kw.get("force_build")
        calls["build_repo"] = kw.get("repo")
        calls["compose"].append("up")
        return []

    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", _fake_build_and_up)

    def _fake_stop_services(services: list[str], **kw: object) -> None:
        calls["compose"].append(f"stop:{','.join(services)}")
        calls["stop_repo"] = kw.get("repo")

    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "stop_services", _fake_stop_services
    )
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "fund_drivers",
        lambda **kw: calls.update(drivers_repo=kw.get("repo")) or [],
    )
    calls["running"] = set()
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "running_services",
        lambda **kw: set(calls["running"]),
    )
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "restart_services",
        lambda services, **kw: calls["compose"].append(f"restart:{','.join(services)}"),
    )
    return calls


_ROUND = "/app/.venv/bin/python /app/.venv/bin/fund run f47ac10b --asof 2026-01-30"


@pytest.mark.parametrize("all_services", [False, True])
def test_run_stop_refuses_while_a_fund_round_is_running(
    patched: dict, monkeypatch: pytest.MonkeyPatch, all_services: bool
) -> None:
    """A round exec'd into the fund container is SIGKILLed with it, mid-flight, so
    stop names it and leaves every service and secret as it was."""
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "fund_drivers", lambda **kw: [_ROUND]
    )

    with pytest.raises(lifecycle.LifecycleError, match="--force") as excinfo:
        lifecycle.run_stop(all_services=all_services)

    assert _ROUND in str(excinfo.value)
    assert patched["compose"] == []
    assert "cleaned" not in patched


def test_run_stop_force_stops_despite_a_running_round(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "fund_drivers", lambda **kw: [_ROUND]
    )

    lifecycle.run_stop(force=True)

    assert patched["compose"] == ["stop:fund"]


def test_run_stop_checks_the_repo_stack_for_running_rounds(patched: dict) -> None:
    lifecycle.run_stop()

    assert patched["drivers_repo"] == patched["repo"]


def test_run_start_renders_secrets_then_brings_up(patched: dict) -> None:
    lifecycle.run_start("pw")
    assert patched["rendered"] == {"fred_api_key": "fk"}
    assert patched["compose"] == ["up"]


def _rotate(
    patched: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_disk: dict[str, str],
    new: dict[str, str],
    running: set[str],
) -> None:
    """Leave `on_disk` rendered under the repo, then start with `new` secrets."""
    secrets_dir = patched["repo"] / "secrets"
    secrets_dir.mkdir()
    for name, value in on_disk.items():
        (secrets_dir / name).write_text(value, encoding="utf-8")
    monkeypatch.setattr(
        lifecycle.secret_store, "load_secrets", lambda passphrase, **kw: dict(new)
    )
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "running_services", lambda **kw: set(running)
    )
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "restart_services",
        lambda services, **kw: patched["compose"].append(
            f"restart:{','.join(services)}"
        ),
        raising=False,
    )


def test_run_start_restarts_a_running_fund_whose_provider_key_was_rotated(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The worker reads the key once at import and compose does not see a changed
    # secret file, so without a restart a rotated key reaches only new processes.
    _rotate(
        patched,
        monkeypatch,
        on_disk={"ollama_api_key": "revoked"},
        new={"ollama_api_key": "fresh"},
        running={"db", "fund", "scheduler"},
    )
    lifecycle.run_start("pw")
    assert patched["compose"] == ["restart:fund", "up"]


def test_run_start_restarts_the_scheduler_when_its_key_was_rotated(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rotate(
        patched,
        monkeypatch,
        on_disk={"trading_212_api_key": "old", "ollama_api_key": "same"},
        new={"trading_212_api_key": "new", "ollama_api_key": "same"},
        running={"db", "fund", "scheduler"},
    )
    lifecycle.run_start("pw")
    assert patched["compose"] == ["restart:scheduler", "up"]


def test_run_start_leaves_services_alone_when_no_secret_changed(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rotate(
        patched,
        monkeypatch,
        on_disk={"ollama_api_key": "same"},
        new={"ollama_api_key": "same"},
        running={"db", "fund", "scheduler"},
    )
    lifecycle.run_start("pw")
    assert patched["compose"] == ["up"]


def test_run_start_does_not_restart_a_service_that_is_not_running(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rotate(
        patched,
        monkeypatch,
        on_disk={"ollama_api_key": "revoked"},
        new={"ollama_api_key": "fresh"},
        running={"db", "scheduler"},
    )
    lifecycle.run_start("pw")
    assert patched["compose"] == ["up"]


def test_run_start_stops_a_fund_left_running_when_no_backend_is_configured(
    patched: dict,
) -> None:
    """`up --profile ingestion` leaves a running fund untouched, so the container
    started under an earlier config would keep serving the provider the CLI now
    reports as not started."""
    patched["running"] = {"db", "fund", "scheduler"}
    lifecycle.run_start("pw")
    assert patched["compose"] == ["up", "stop:fund"]


def test_run_start_does_not_touch_a_fund_that_is_already_stopped(
    patched: dict,
) -> None:
    patched["running"] = {"db", "scheduler"}
    lifecycle.run_start("pw")
    assert patched["compose"] == ["up"]


def test_run_start_does_not_restart_a_disabled_fund_for_a_rotated_key(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart keeps the container's old env_file values, so restarting a fund
    that is about to be left off only prolongs the stale provider."""
    _rotate(
        patched,
        monkeypatch,
        on_disk={"groq_api_key": "revoked"},
        new={"groq_api_key": "fresh"},
        running={"db", "fund", "scheduler"},
    )
    lifecycle.run_start("pw")
    assert "restart:fund" not in patched["compose"]
    assert patched["compose"] == ["up", "stop:fund"]


def test_run_start_keeps_a_disabled_fund_that_is_mid_round_and_says_so(
    patched: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stopping it would SIGKILL the round, so start leaves it and names the way
    out instead of killing client work."""
    patched["running"] = {"db", "fund", "scheduler"}
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "fund_drivers", lambda **kw: [_ROUND]
    )
    lifecycle.run_start("pw")
    out = capsys.readouterr().out
    assert patched["compose"] == ["up"]
    assert _ROUND in out
    assert "portopt stop" in out


def test_run_start_leaves_an_enabled_fund_running(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    patched["running"] = {"db", "fund", "scheduler"}
    _configure_llm(monkeypatch, config={}, secrets={"ollama_api_key": "k"})
    lifecycle.run_start("pw")
    assert patched["compose"] == ["up"]


def test_run_start_creates_the_exports_dir_as_the_invoking_user(
    patched: dict,
) -> None:
    """Compose creates a missing bind source as root, after which the fund
    container's non-root user cannot write its TUI exports there."""
    exports = patched["repo"] / "exports"
    assert not exports.exists()
    lifecycle.run_start("pw")
    assert exports.is_dir()


def test_run_start_creates_the_exports_dir_before_compose_sees_the_mount(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    exports = patched["repo"] / "exports"

    def _build_and_up(**kw: object) -> list[str]:
        seen.append(exports.is_dir())
        return []

    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", _build_and_up)
    lifecycle.run_start("pw")
    assert seen == [True]


def test_run_start_keeps_an_existing_exports_dir_and_its_files(
    patched: dict,
) -> None:
    exports = patched["repo"] / "exports"
    exports.mkdir()
    (exports / "audit.csv").write_text("kept", encoding="utf-8")
    lifecycle.run_start("pw")
    assert (exports / "audit.csv").read_text(encoding="utf-8") == "kept"


@pytest.mark.parametrize(
    ("platform", "owner_matches", "world_writable"),
    [
        ("linux", False, True),
        ("linux", True, False),
        ("darwin", False, False),
    ],
)
def test_run_start_opens_the_exports_to_the_container_user_on_linux(
    patched: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform: str,
    owner_matches: bool,
    world_writable: bool,
) -> None:
    """On a Linux host the bind mount keeps the host's numeric owner; the image's
    user (uid 1000) then needs to own the directory or find it world-writable, so a
    directory the invoking user owns is made writable (sticky, like /tmp) instead of
    leaving the export to fail. Docker Desktop and Colima present host files as the
    container user and need nothing."""
    import os
    import stat

    monkeypatch.setattr(lifecycle.sys, "platform", platform)
    monkeypatch.setattr(lifecycle.docker_bootstrap, "fund_user_env", dict)
    monkeypatch.setattr(
        lifecycle, "_CONTAINER_UID", os.getuid() if owner_matches else os.getuid() + 1
    )
    lifecycle.run_start("pw")
    mode = stat.S_IMODE((patched["repo"] / "exports").stat().st_mode)
    assert bool(mode & 0o002) is world_writable
    if world_writable:
        assert mode & stat.S_ISVTX
    assert "permission" not in capsys.readouterr().out


def test_run_start_leaves_the_exports_alone_when_the_container_runs_as_the_invoker(
    patched: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With the fund container running as the invoking user, a directory that user
    owns is already writable: widening its mode would only expose the exports."""
    import os
    import stat

    monkeypatch.setattr(lifecycle.sys, "platform", "linux")
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "fund_user_env",
        lambda: {"PORTOPT_FUND_UID": str(os.getuid())},
    )
    lifecycle.run_start("pw")
    mode = stat.S_IMODE((patched["repo"] / "exports").stat().st_mode)
    assert not mode & 0o002
    assert "permission" not in capsys.readouterr().out


def test_run_start_warns_when_it_cannot_open_the_exports_to_the_container_user(
    patched: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A directory left behind by root cannot be chmod-ed by the invoking user, so the
    operator is told the exact command instead of seeing every export fail later."""
    import os
    from pathlib import Path

    def refuse(self: Path, mode: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(lifecycle.sys, "platform", "linux")
    monkeypatch.setattr(lifecycle.docker_bootstrap, "fund_user_env", dict)
    monkeypatch.setattr(lifecycle, "_CONTAINER_UID", os.getuid() + 1)
    monkeypatch.setattr(Path, "chmod", refuse)
    lifecycle.run_start("pw")
    out = capsys.readouterr().out
    assert "exports" in out and "permission" in out and "sudo chown" in out


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


def test_run_start_renders_nothing_when_docker_cannot_be_booted(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _down() -> None:
        raise lifecycle.docker_bootstrap.DockerError("colima start failed")

    monkeypatch.setattr(lifecycle.docker_bootstrap, "ensure_docker", _down)
    with pytest.raises(lifecycle.docker_bootstrap.DockerError):
        lifecycle.run_start("pw")
    assert patched["rendered"] is None
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


def _configure_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, object],
    secrets: dict[str, str],
) -> None:
    monkeypatch.setattr(lifecycle.config_file, "load_config", lambda **kw: dict(config))
    monkeypatch.setattr(
        lifecycle.secret_store, "load_secrets", lambda passphrase, **kw: dict(secrets)
    )


def test_run_start_brings_up_fund_and_ingestion_when_an_llm_is_configured(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an LLM backend run_start brings up BOTH the fund engine and the
    persistent ingestion daemon (every compose service is profile-gated, so a bare
    `up` would start nothing)."""
    _configure_llm(monkeypatch, config={}, secrets={"ollama_api_key": "k"})
    lifecycle.run_start("pw")
    assert patched["build_profiles"] == ("fund", "ingestion")


def test_run_start_with_a_local_ollama_brings_up_the_fund_without_a_key(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keyless local backend is a configured one."""
    _configure_llm(
        monkeypatch,
        config={"llm_provider": "ollama", "llm_base_url": "http://localhost:11434"},
        secrets={},
    )
    lifecycle.run_start("pw")
    assert patched["build_profiles"] == ("fund", "ingestion")


def test_run_start_skips_the_fund_when_no_llm_is_configured(patched: dict) -> None:
    """The fund worker cannot build its model without a backend, so its container
    would never turn healthy and `up --wait` would hang on it. The data pipeline
    still comes up."""
    lifecycle.run_start("pw")
    assert patched["build_profiles"] == ("ingestion",)
    assert patched["compose"] == ["up"]


def test_run_start_skips_the_fund_when_the_provider_key_is_missing(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider chosen in config.toml whose key was never stored is unusable."""
    _configure_llm(
        monkeypatch,
        config={"llm_provider": "openai", "llm_model": "gpt-4o"},
        secrets={"fred_api_key": "fk"},
    )
    lifecycle.run_start("pw")
    assert patched["build_profiles"] == ("ingestion",)


def test_run_start_skips_the_fund_for_a_non_ollama_provider_without_a_model(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker refuses the built-in Ollama model id on any other provider."""
    _configure_llm(
        monkeypatch,
        config={"llm_provider": "openai"},
        secrets={"openai_api_key": "k"},
    )
    lifecycle.run_start("pw")
    assert patched["build_profiles"] == ("ingestion",)


def test_run_start_says_when_it_leaves_the_fund_off(
    patched: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    """The skip is announced in one line that names the way to turn it on."""
    lifecycle.run_start("pw")
    out = capsys.readouterr().out
    assert out.count(lifecycle.FUND_SKIPPED_NOTICE) == 1
    assert len(lifecycle.FUND_SKIPPED_NOTICE.splitlines()) == 1
    assert "no LLM backend" in lifecycle.FUND_SKIPPED_NOTICE
    assert "portopt setup" in lifecycle.FUND_SKIPPED_NOTICE


def test_run_start_is_quiet_about_the_fund_when_an_llm_is_configured(
    patched: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing to announce when both profiles start."""
    _configure_llm(monkeypatch, config={}, secrets={"ollama_api_key": "k"})
    lifecycle.run_start("pw")
    assert lifecycle.FUND_SKIPPED_NOTICE not in capsys.readouterr().out


def test_run_start_announces_the_skip_only_once_the_stack_is_up(
    patched: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed `up` must not be preceded by a claim that the stack is running."""

    def _boom(**kw: object) -> list[str]:
        raise lifecycle.docker_bootstrap.DockerError("up failed")

    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", _boom)
    with pytest.raises(lifecycle.docker_bootstrap.DockerError):
        lifecycle.run_start("pw")
    assert lifecycle.FUND_SKIPPED_NOTICE not in capsys.readouterr().out


def test_run_start_brings_the_fund_back_once_an_llm_is_configured(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configure later, start again: the same call now includes the fund profile."""
    lifecycle.run_start("pw")
    assert patched["build_profiles"] == ("ingestion",)
    _configure_llm(monkeypatch, config={}, secrets={"ollama_api_key": "k"})
    lifecycle.run_start("pw")
    assert patched["build_profiles"] == ("fund", "ingestion")


def test_run_start_still_renders_the_fund_env_when_the_fund_is_skipped(
    patched: dict,
) -> None:
    """The env file stays resolvable for a later plain `docker compose up`."""
    lifecycle.run_start("pw")
    assert patched["env_fund_path"].read_text(encoding="utf-8") == ""


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


def test_run_start_renders_secrets_under_the_repo_not_the_cwd(patched: dict) -> None:
    """The plaintext secret files belong in the repo's git-ignored `secrets/`, the
    directory compose mounts from, regardless of where `portopt start` was typed."""
    lifecycle.run_start("pw")
    assert patched["secrets_dir"] == patched["repo"] / "secrets"


def test_run_start_writes_env_fund_under_the_repo_not_the_cwd(patched: dict) -> None:
    lifecycle.run_start("pw")
    assert patched["env_fund_path"].is_file()
    assert not Path(".env.fund").exists()


def test_run_start_runs_compose_from_the_repo(patched: dict) -> None:
    lifecycle.run_start("pw")
    assert patched["build_repo"] == patched["repo"]


def test_run_start_from_a_repo_subdirectory_renders_nothing_there(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start from `ingestion/` must not plant a second, ungitignored `secrets/`
    that a later image build would COPY."""
    sub = patched["repo"] / "ingestion"
    sub.mkdir()
    monkeypatch.chdir(sub)
    lifecycle.run_start("pw")
    assert patched["secrets_dir"] == patched["repo"] / "secrets"
    assert sorted(p.name for p in sub.iterdir()) == []


def test_run_start_in_a_checkout_ignores_the_persisted_repo_path(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A second checkout recorded in config must not capture a start typed inside this
    one: secrets and the compose project would land in a stack the operator is not
    looking at."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    monkeypatch.delenv("OPTIMIZER_REPO")
    monkeypatch.setattr(
        lifecycle.config_file, "load_config", lambda **kw: {"repo_path": str(other)}
    )
    monkeypatch.chdir(patched["repo"])
    lifecycle.run_start("pw")
    assert patched["secrets_dir"] == patched["repo"] / "secrets"
    assert patched["build_repo"] == patched["repo"]


def test_run_start_refuses_before_rendering_when_the_checkout_is_unknown(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """With no checkout to anchor to, nothing may be written anywhere."""
    monkeypatch.delenv("OPTIMIZER_REPO")
    monkeypatch.setattr(path_install, "_repo_root", lambda: tmp_path / "gone")
    with pytest.raises(lifecycle.docker_bootstrap.DockerError, match="OPTIMIZER_REPO"):
        lifecycle.run_start("pw")
    assert patched["rendered"] is None
    assert patched["compose"] == []
    assert not Path(".env.fund").exists()


def test_run_stop_targets_the_repo_stack_from_any_directory(patched: dict) -> None:
    patched["env_fund_path"].write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    lifecycle.run_stop()
    assert patched["stop_repo"] == patched["repo"]
    assert not patched["env_fund_path"].exists()


def _states(*rows: tuple[str, str, str]) -> dict:
    return {
        name: lifecycle.docker_bootstrap.ServiceState(state, health)
        for name, state, health in rows
    }


def _status_with(
    monkeypatch: pytest.MonkeyPatch, states: dict, *, docker: bool = True
) -> dict[str, str]:
    monkeypatch.setattr(lifecycle.docker_bootstrap, "docker_available", lambda: docker)
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "service_states", lambda **kw: states
    )
    return lifecycle.run_status()


def test_run_status_all_up(patched: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _status_with(
        monkeypatch,
        _states(
            ("db", "running", "healthy"),
            ("scheduler", "running", "healthy"),
            ("fund", "running", "healthy"),
        ),
    ) == {"docker": "ok", "db": "ok", "scheduler": "ok", "fund": "ok"}


def test_run_status_reports_fund_independently(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = _status_with(
        monkeypatch,
        _states(("db", "running", "healthy"), ("fund", "running", "healthy")),
    )
    assert status["fund"] == "ok"
    assert status["scheduler"] == "down"


def test_run_status_docker_down(patched: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _status_with(monkeypatch, {}, docker=False) == {
        "docker": "down",
        "db": "down",
        "scheduler": "down",
        "fund": "down",
    }


def test_run_status_queries_the_repo_stack_from_any_directory(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    def _states_of(**kw: object) -> dict:
        seen.append(kw.get("repo"))
        return {}

    monkeypatch.setattr(lifecycle.docker_bootstrap, "docker_available", lambda: True)
    monkeypatch.setattr(lifecycle.docker_bootstrap, "service_states", _states_of)
    lifecycle.run_status()
    assert seen == [patched["repo"]]


@pytest.mark.parametrize(
    "fund_row",
    [
        None,
        ("fund", "exited", ""),
        ("fund", "created", ""),
    ],
)
def test_run_status_reports_a_stopped_or_never_started_fund_as_disabled(
    patched: dict, monkeypatch: pytest.MonkeyPatch, fund_row: tuple | None
) -> None:
    """No LLM backend, or `portopt stop`, leave the fund off on purpose; calling
    that "down" made `status` fail for a healthy data-only install."""
    rows = [("db", "running", "healthy"), ("scheduler", "running", "healthy")]
    if fund_row:
        rows.append(fund_row)
    status = _status_with(monkeypatch, _states(*rows))
    assert status["fund"] == "disabled"
    assert lifecycle.status_ok(status)


@pytest.mark.parametrize(
    ("state", "health", "expected"),
    [
        ("running", "healthy", "ok"),
        ("running", "", "ok"),
        ("running", "starting", "starting"),
        ("running", "unhealthy", "unhealthy"),
        ("restarting", "", "restarting"),
        ("exited", "", "down"),
        ("paused", "", "down"),
    ],
)
def test_run_status_reads_the_health_not_just_the_running_state(
    patched: dict,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    health: str,
    expected: str,
) -> None:
    """A crash-looping container is `running` between restarts and a hung one stays
    `running` while its healthcheck fails; neither is up."""
    status = _status_with(
        monkeypatch,
        _states(("db", "running", "healthy"), ("scheduler", state, health)),
    )
    assert status["scheduler"] == expected
    assert lifecycle.status_ok(status) is (expected == "ok")


def test_run_status_flags_a_crash_looping_fund_rather_than_disabling_it(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = _status_with(
        monkeypatch,
        _states(
            ("db", "running", "healthy"),
            ("scheduler", "running", "healthy"),
            ("fund", "restarting", ""),
        ),
    )
    assert status["fund"] == "restarting"
    assert not lifecycle.status_ok(status)


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
    monkeypatch.setattr(lifecycle.docker_bootstrap, "ensure_docker", lambda: None)
    monkeypatch.setattr(lifecycle.docker_bootstrap, "build_and_up", lambda **kw: [])
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "running_services", lambda **kw: set()
    )
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "stop_services", lambda s, **kw: None
    )
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
        lifecycle.docker_bootstrap,
        "stop_services",
        lambda s, **kw: stopped.append(list(s)),
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
