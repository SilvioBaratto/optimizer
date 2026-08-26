"""Lifecycle orchestration contract (SPEC D6/D10, task T7b).

`run_start` decrypts the store, renders compose secrets, and brings the stack up;
`run_stop` tears it down and wipes the plaintext secret files; `run_status`
reports docker + service health. All collaborators are patched.
"""

import pytest

from app.setup import lifecycle


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {"rendered": None, "compose": [], "cleaned": False}
    monkeypatch.setattr(
        lifecycle.secret_store,
        "load_secrets",
        lambda passphrase, **kw: {"openai_api_key": "sk-o"},
    )
    monkeypatch.setattr(
        lifecycle.compose_secrets,
        "render",
        lambda secrets, **kw: calls.update(rendered=dict(secrets)),
    )
    monkeypatch.setattr(
        lifecycle.compose_secrets, "cleanup", lambda **kw: calls.update(cleaned=True)
    )
    monkeypatch.setattr(lifecycle.docker_bootstrap, "check_docker", lambda: None)
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "compose_up",
        lambda: calls["compose"].append("up"),
    )
    monkeypatch.setattr(
        lifecycle.docker_bootstrap,
        "compose_down",
        lambda: calls["compose"].append("down"),
    )
    return calls


def test_run_start_renders_secrets_then_brings_up(patched: dict) -> None:
    lifecycle.run_start("pw")
    assert patched["rendered"] == {"openai_api_key": "sk-o"}
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
    lifecycle.run_stop()
    assert patched["compose"] == ["down"]
    assert patched["cleaned"] is True


def test_run_status_all_up(patched: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle.docker_bootstrap, "docker_available", lambda: True)
    monkeypatch.setattr(
        lifecycle.docker_bootstrap, "running_services", lambda: {"db", "scheduler"}
    )
    assert lifecycle.run_status() == {"docker": True, "db": True, "scheduler": True}


def test_run_status_docker_down(patched: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle.docker_bootstrap, "docker_available", lambda: False)
    status = lifecycle.run_status()
    assert status == {"docker": False, "db": False, "scheduler": False}
