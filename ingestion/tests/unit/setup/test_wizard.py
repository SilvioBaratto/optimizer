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
    """Patch all wizard collaborators; record persistence + bootstrap calls.

    Defaults model a first-run with an empty store: ``load_secrets`` raises
    NotFound (so the merge starts empty), ``load_config`` is empty, and
    ``_is_first_run`` is False so no auto-launch fires unless a test opts in.
    """
    calls: dict = {
        "saved_secrets": None,
        "saved_config": None,
        "bootstrapped": [],
        "path_installed": False,
        "launched": False,
    }
    monkeypatch.setattr(wizard.docker_bootstrap, "check_docker", lambda: None)
    monkeypatch.setattr(
        wizard.path_install,
        "install_launcher",
        lambda **kw: calls.update(path_installed=True),
    )
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

    def _no_store(passphrase: str, **kw: object) -> dict:
        raise wizard.secret_store.SecretStoreNotFoundError("none")

    monkeypatch.setattr(wizard.secret_store, "load_secrets", _no_store)
    monkeypatch.setattr(
        wizard.config_file,
        "update_config",
        lambda config, **kw: calls.update(saved_config=dict(config)),
    )
    monkeypatch.setattr(wizard.config_file, "load_config", lambda **kw: {})
    monkeypatch.setattr(wizard, "_is_first_run", lambda: False)
    monkeypatch.setattr(
        wizard.lifecycle, "run_start", lambda pw: calls.update(launched=True)
    )
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s: True)
    monkeypatch.setattr(wizard.validators, "validate_fred", lambda k: True)
    monkeypatch.setattr(wizard.validators, "validate_llm", lambda *a, **k: True)
    # Clear every credential env var so interactive tests are deterministic:
    # env-autodetection (Rule 2) must not pick up the developer's real shell env.
    for _var in (
        "PORTOPT_PASSPHRASE",
        "TRADING_212_API_KEY",
        "TRADING_212_SECRET_KEY",
        "FRED_API_KEY",
        "OLLAMA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "NVIDIA_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "AWS_REGION",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ):
        monkeypatch.delenv(_var, raising=False)
    return calls


def test_noninteractive_minimal_bootstraps(patched: dict) -> None:
    wizard.run_setup_noninteractive(passphrase="pw")
    assert patched["saved_secrets"] == {}
    assert patched["saved_config"] == {}
    assert patched["bootstrapped"] == ["db", "migrate"]


def test_setup_installs_launcher_by_default(patched: dict) -> None:
    """Setup installs the `optimizer` launcher onto PATH unless told to skip."""
    wizard.run_setup_noninteractive(passphrase="pw")
    assert patched["path_installed"] is True


def test_setup_skip_path_install_leaves_path_untouched(patched: dict) -> None:
    """`skip_path_install=True` persists + bootstraps but does no PATH edit."""
    wizard.run_setup_noninteractive(passphrase="pw", skip_path_install=True)
    assert patched["bootstrapped"] == ["db", "migrate"]
    assert patched["path_installed"] is False


def test_setup_survives_path_install_failure(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PathInstallError is a warning, not a setup failure — secrets are persisted
    and the DB migrated before the launcher install runs, so it must not roll back."""

    def boom(**kw):
        raise wizard.path_install.PathInstallError("no PATH")

    monkeypatch.setattr(wizard.path_install, "install_launcher", boom)
    wizard.run_setup_noninteractive(passphrase="pw")
    assert patched["bootstrapped"] == ["db", "migrate"]
    assert patched["saved_secrets"] == {}


def test_interactive_installs_launcher_by_default(patched: dict) -> None:
    """The interactive path also installs the launcher by default."""
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: False,
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["path_installed"] is True


def test_interactive_skip_path_install(patched: dict) -> None:
    """The interactive path honours skip_path_install (forwarded to the bootstrap)."""
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: False,
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter, skip_path_install=True)
    assert patched["path_installed"] is False


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
            wizard._MSG_CONFIGURE_LLM: False,
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
            wizard._MSG_CONFIGURE_LLM: False,
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
            wizard._MSG_CONFIGURE_LLM: False,
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
            wizard._MSG_CONFIGURE_LLM: False,
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


# --- T5: LLM prompt group + --llm-* flags -------------------------------------


def _llm_prompter(answers: dict) -> NonInteractivePrompter:
    base = {
        wizard._MSG_PASSPHRASE: "pw",
        wizard._MSG_CONNECT_T212: False,
        wizard._MSG_CONNECT_FRED: False,
        wizard._MSG_CONFIGURE_LLM: True,
        wizard._MSG_LLM_BASE_URL: "",
    }
    base.update(answers)
    return NonInteractivePrompter(base)


def test_noninteractive_llm_persists_config_and_secret(patched: dict) -> None:
    wizard.run_setup_noninteractive(
        passphrase="pw",
        llm_provider="openai",
        llm_model="gpt-4o",
        llm_key="sk-openai",
    )
    assert patched["saved_config"]["llm_provider"] == "openai"
    assert patched["saved_config"]["llm_model"] == "gpt-4o"
    assert patched["saved_secrets"]["openai_api_key"] == "sk-openai"


def test_noninteractive_llm_none_skips(patched: dict) -> None:
    wizard.run_setup_noninteractive(passphrase="pw")
    assert "llm_provider" not in patched["saved_config"]


def test_noninteractive_llm_bad_key_fails_without_persist(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.validators, "validate_llm", lambda *a, **k: False)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(
            passphrase="pw", llm_provider="openai", llm_key="bad"
        )
    assert patched["saved_secrets"] is None


def test_noninteractive_llm_network_error_aborts(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a: object, **k: object) -> bool:
        raise wizard.validators.ValidationNetworkError("down")

    monkeypatch.setattr(wizard.validators, "validate_llm", _boom)
    with pytest.raises(wizard.validators.ValidationNetworkError):
        wizard.run_setup_noninteractive(
            passphrase="pw", llm_provider="openai", llm_key="k"
        )
    assert patched["saved_secrets"] is None


def test_noninteractive_llm_key_from_env(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-sk")
    wizard.run_setup_noninteractive(passphrase="pw", llm_provider="anthropic")
    assert patched["saved_secrets"]["anthropic_api_key"] == "env-sk"


def test_noninteractive_llm_missing_key_fails(patched: dict) -> None:
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase="pw", llm_provider="openai")
    assert patched["saved_secrets"] is None


def test_noninteractive_llm_unknown_provider_fails(patched: dict) -> None:
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(
            passphrase="pw", llm_provider="bogus", llm_key="k"
        )
    assert patched["saved_secrets"] is None


def test_noninteractive_llm_aws_presence_only_no_secret(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    wizard.run_setup_noninteractive(passphrase="pw", llm_provider="aws")
    assert patched["saved_config"]["llm_provider"] == "aws"
    assert patched["saved_config"]["aws_region"] == "eu-west-1"
    assert patched["saved_secrets"] == {}


def test_interactive_llm_persists_config_and_secret(patched: dict) -> None:
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_LLM_KEY: "sk-openai",
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_config"]["llm_provider"] == "openai"
    assert patched["saved_config"]["llm_model"] == "gpt-4o"
    assert patched["saved_secrets"]["openai_api_key"] == "sk-openai"


def test_interactive_llm_declined_persists_nothing(patched: dict) -> None:
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: False,
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert "llm_provider" not in patched["saved_config"]
    assert patched["saved_secrets"] == {}


def test_interactive_llm_config_never_contains_a_secret(patched: dict) -> None:
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_LLM_KEY: "sk-openai",
        }
    )
    wizard.run_setup_interactive(prompter)
    assert not any("key" in k.lower() for k in patched["saved_config"])


def test_interactive_llm_bad_key_reprompts_then_succeeds(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}

    def _validate(provider, key=None, base_url=None, **fields):
        attempts["n"] += 1
        return attempts["n"] >= 2  # first rejected, second accepted

    monkeypatch.setattr(wizard.validators, "validate_llm", _validate)
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_LLM_KEY: "sk-openai",
        }
    )
    wizard.run_setup_interactive(prompter)
    assert attempts["n"] == 2
    assert patched["saved_secrets"]["openai_api_key"] == "sk-openai"


def test_interactive_llm_network_error_aborts(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a: object, **k: object) -> bool:
        raise wizard.validators.ValidationNetworkError("down")

    monkeypatch.setattr(wizard.validators, "validate_llm", _boom)
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_LLM_KEY: "sk",
        }
    )
    with pytest.raises(wizard.validators.ValidationNetworkError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None


def test_interactive_llm_aws_presence_only(patched: dict) -> None:
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "aws",
            wizard._MSG_LLM_MODEL: "amazon.titan-text",
            wizard._MSG_AWS_REGION: "eu-west-1",
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_config"]["aws_region"] == "eu-west-1"
    assert patched["saved_secrets"] == {}


def test_interactive_llm_azure_persists_fields_and_secret(patched: dict) -> None:
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "microsoft",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_AZURE_ENDPOINT: "https://az.openai.azure.com",
            wizard._MSG_AZURE_API_VERSION: "2024-06-01",
            wizard._MSG_AZURE_DEPLOYMENT: "gpt4o-deploy",
            wizard._MSG_LLM_KEY: "az-key",
        }
    )
    wizard.run_setup_interactive(prompter)
    assert (
        patched["saved_config"]["azure_openai_endpoint"]
        == "https://az.openai.azure.com"
    )
    assert patched["saved_config"]["azure_openai_deployment_name"] == "gpt4o-deploy"
    assert patched["saved_secrets"]["azure_openai_api_key"] == "az-key"


# --- T15: idempotency / --reconfigure / --skip-validation / auto-launch -------


def test_noninteractive_idempotent_rerun_preserves_untouched_secrets(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run with no flags must NOT wipe secrets stored by an earlier run."""
    monkeypatch.setattr(
        wizard.secret_store, "load_secrets", lambda pw, **k: {"fred_api_key": "fk"}
    )
    wizard.run_setup_noninteractive(passphrase="pw")
    assert patched["saved_secrets"] == {"fred_api_key": "fk"}


def test_noninteractive_idempotent_rerun_merges_new_secret(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new secret merges with the existing store rather than replacing it."""
    monkeypatch.setattr(
        wizard.secret_store, "load_secrets", lambda pw, **k: {"fred_api_key": "fk"}
    )
    wizard.run_setup_noninteractive(passphrase="pw", t212_key="tk", t212_secret="ts")
    assert patched["saved_secrets"] == {
        "fred_api_key": "fk",
        "trading_212_api_key": "tk",
        "trading_212_secret_key": "ts",
    }


def test_noninteractive_first_run_saves_only_new_secrets(patched: dict) -> None:
    """With no existing store (NotFound), the merge is just the new secrets."""
    wizard.run_setup_noninteractive(passphrase="pw", fred_key="fk")
    assert patched["saved_secrets"] == {"fred_api_key": "fk"}


def test_noninteractive_rerun_wrong_passphrase_fails_without_wiping(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong passphrase on a re-run fails loud and never overwrites the store."""

    def _bad(pw: str, **k: object) -> dict:
        raise wizard.secret_store.InvalidPassphraseError("wrong")

    monkeypatch.setattr(wizard.secret_store, "load_secrets", _bad)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase="wrong", fred_key="fk")
    assert patched["saved_secrets"] is None


def test_noninteractive_skip_validation_bypasses_t212(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--skip-validation persists T212 without a live call, even if it would fail."""
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s: False)
    wizard.run_setup_noninteractive(
        passphrase="pw", t212_key="tk", t212_secret="ts", skip_validation=True
    )
    assert patched["saved_secrets"]["trading_212_api_key"] == "tk"


def test_noninteractive_skip_validation_bypasses_fred(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--skip-validation persists FRED without a live call."""
    monkeypatch.setattr(wizard.validators, "validate_fred", lambda k: False)
    wizard.run_setup_noninteractive(
        passphrase="pw", fred_key="fk", skip_validation=True
    )
    assert patched["saved_secrets"]["fred_api_key"] == "fk"


def test_noninteractive_skip_validation_forwarded_to_validate_llm(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skip_validation flag reaches validators.validate_llm."""
    seen: dict = {}

    def _capture(provider, key=None, base_url=None, skip_validation=False, **f):
        seen["skip"] = skip_validation
        return True

    monkeypatch.setattr(wizard.validators, "validate_llm", _capture)
    wizard.run_setup_noninteractive(
        passphrase="pw",
        llm_provider="openai",
        llm_key="sk",
        skip_validation=True,
    )
    assert seen["skip"] is True


def test_noninteractive_never_auto_launches(patched: dict) -> None:
    """Non-interactive/CI setup persists + bootstraps but never brings the stack up."""
    wizard.run_setup_noninteractive(passphrase="pw")
    assert patched["bootstrapped"] == ["db", "migrate"]
    assert patched["launched"] is False


def test_interactive_first_run_auto_launches(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first interactive run brings the stack up unless told not to."""
    monkeypatch.setattr(wizard, "_is_first_run", lambda: True)
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: False,
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["launched"] is True


def test_interactive_no_launch_suppresses_auto_launch(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-launch keeps a first interactive run from bringing the stack up."""
    monkeypatch.setattr(wizard, "_is_first_run", lambda: True)
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: False,
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter, no_launch=True)
    assert patched["launched"] is False


def test_interactive_rerun_never_auto_launches(patched: dict) -> None:
    """A re-run (not first run) never auto-launches, even interactively."""
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: False,
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)  # fixture _is_first_run -> False
    assert patched["launched"] is False


def test_interactive_auto_launch_failure_is_non_fatal(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed auto-launch warns but does not undo a completed setup."""
    monkeypatch.setattr(wizard, "_is_first_run", lambda: True)

    def _boom(pw: str) -> None:
        raise wizard.docker_bootstrap.DockerError("up failed")

    monkeypatch.setattr(wizard.lifecycle, "run_start", _boom)
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_CONNECT_T212: False,
            wizard._MSG_CONNECT_FRED: False,
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)  # must not raise
    assert patched["saved_secrets"] == {}
    assert patched["bootstrapped"] == ["db", "migrate"]


def test_interactive_reconfigure_reruns_the_llm_prompt(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--reconfigure re-prompts the LLM group even when a provider is already set."""
    monkeypatch.setattr(
        wizard.config_file,
        "load_config",
        lambda **k: {"llm_provider": "anthropic", "llm_model": "old"},
    )
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_LLM_KEY: "sk-openai",
        }
    )
    wizard.run_setup_interactive(prompter, reconfigure=True)
    assert patched["saved_config"]["llm_provider"] == "openai"
    assert patched["saved_secrets"]["openai_api_key"] == "sk-openai"
