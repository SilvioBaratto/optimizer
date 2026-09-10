"""Install-wizard setup-flow contract (SPEC D4/D8, task T7a).

Covers both entry points: `run_setup_noninteractive` (CI/flags) and
`run_setup_interactive` (prompt seam). Validation, encryption, and Docker calls
are all patched — nothing hits the network, disk secret store, or Docker. Key
guarantees: invalid keys fail loud with nothing persisted, and exported env
credentials are auto-detected before prompting.
"""

import pytest

from app.setup import wizard
from app.setup.prompts import NonInteractivePrompter


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch all wizard collaborators; record persistence + bootstrap calls."""
    calls: dict = {"saved_secrets": None, "saved_config": None, "bootstrapped": []}
    monkeypatch.setattr(wizard.docker_bootstrap, "check_docker", lambda: None)
    monkeypatch.setattr(
        wizard.docker_bootstrap,
        "bring_up_db",
        lambda: calls["bootstrapped"].append("db"),
    )
    monkeypatch.setattr(
        wizard.docker_bootstrap,
        "migrate",
        lambda: calls["bootstrapped"].append("migrate"),
    )
    monkeypatch.setattr(
        wizard.secret_store,
        "save_secrets",
        lambda secrets, passphrase, **kw: calls.update(saved_secrets=dict(secrets)),
    )
    monkeypatch.setattr(
        wizard.config_file,
        "save_config",
        lambda config, **kw: calls.update(saved_config=dict(config)),
    )
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s: True)
    monkeypatch.setattr(wizard.validators, "validate_fred", lambda k: True)
    # Clear every credential env var so interactive tests are deterministic:
    # env-autodetection (Rule 2) must not pick up the developer's real shell env.
    for _var in (
        "PORTOPT_PASSPHRASE",
        "TRADING_212_API_KEY",
        "TRADING_212_SECRET_KEY",
        "FRED_API_KEY",
    ):
        monkeypatch.delenv(_var, raising=False)
    return calls


def test_noninteractive_minimal_bootstraps(patched: dict) -> None:
    wizard.run_setup_noninteractive(passphrase="pw")
    assert patched["saved_secrets"] == {}
    assert patched["saved_config"] == {}
    assert patched["bootstrapped"] == ["db", "migrate"]


def test_noninteractive_full_persists_all(patched: dict) -> None:
    wizard.run_setup_noninteractive(
        passphrase="pw",
        t212_key="tk",
        t212_secret="ts",
        fred_key="fk",
    )
    assert patched["saved_secrets"] == {
        "trading_212_api_key": "tk",
        "trading_212_secret_key": "ts",
        "fred_api_key": "fk",
    }


def test_noninteractive_t212_partial_fails(patched: dict) -> None:
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase="pw", t212_key="only-key")
    assert patched["saved_secrets"] is None


def test_noninteractive_requires_passphrase(patched: dict) -> None:
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase=None)


def test_noninteractive_t212_invalid_fails(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s: False)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(
            passphrase="pw",
            t212_key="tk",
            t212_secret="ts",
        )
    assert patched["saved_secrets"] is None


def test_noninteractive_fred_invalid_fails(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.validators, "validate_fred", lambda k: False)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase="pw", fred_key="bad")
    assert patched["saved_secrets"] is None


def test_interactive_t212_no_completes(patched: dict) -> None:
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] == {}
    assert patched["saved_config"] == {}
    assert patched["bootstrapped"] == ["db", "migrate"]


def test_interactive_t212_and_fred_yes_persists_all(patched: dict) -> None:
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: True,
            wizard._MSG_T212_KEY: "tk",
            wizard._MSG_T212_SECRET: "ts",
            wizard._MSG_CONNECT_FRED: True,
            wizard._MSG_FRED_KEY: "fk",
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] == {
        "trading_212_api_key": "tk",
        "trading_212_secret_key": "ts",
        "fred_api_key": "fk",
    }


def test_interactive_t212_invalid_fails_loud(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s: False)
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: True,
            wizard._MSG_T212_KEY: "tk",
            wizard._MSG_T212_SECRET: "ts",
        }
    )
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None


def test_interactive_fred_invalid_fails_loud(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.validators, "validate_fred", lambda k: False)
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: True,
            wizard._MSG_FRED_KEY: "bad",
        }
    )
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None


def test_interactive_empty_passphrase_fails(patched: dict) -> None:
    prompter = NonInteractivePrompter({wizard._MSG_PASSPHRASE: ""})
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None


def test_interactive_t212_keys_autodetected_from_env(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Rule 2: exported T212 env vars are used without prompting. The prompt map
    # omits _MSG_T212_KEY/_SECRET; NonInteractivePrompter.password would raise if
    # the wizard prompted, so completing proves env was read first.
    monkeypatch.setenv("TRADING_212_API_KEY", "env-tk")
    monkeypatch.setenv("TRADING_212_SECRET_KEY", "env-ts")
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: True,
            wizard._MSG_CONNECT_FRED: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"]["trading_212_api_key"] == "env-tk"
    assert patched["saved_secrets"]["trading_212_secret_key"] == "env-ts"  # noqa: S105 - test fixture value, not a secret


def test_interactive_fred_key_autodetected_from_env(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "env-fk")
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: True,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"]["fred_api_key"] == "env-fk"


def test_interactive_docker_down_propagates(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        wizard.docker_bootstrap,
        "check_docker",
        lambda: (_ for _ in ()).throw(wizard.docker_bootstrap.DockerError("down")),
    )
    prompter = NonInteractivePrompter({wizard._MSG_PASSPHRASE: "pw"})
    with pytest.raises(wizard.docker_bootstrap.DockerError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None
