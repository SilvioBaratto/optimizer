"""Install-wizard setup-flow contract (SPEC D4/D8).

Covers both entry points: `run_setup_noninteractive` (CI/flags) and
`run_setup_interactive` (prompt seam). Validation, encryption, and Docker calls
are all patched — nothing hits the network, disk secret store, or Docker. Key
guarantees: invalid keys fail loud with nothing persisted, and exported env
credentials are auto-detected before prompting.
"""

import pytest

from app.setup import compose_env, config_file, secret_store, wizard
from app.setup.prompts import NonInteractivePrompter

# Long enough for the new-store passphrase policy.
_PW = "correct-horse-battery"

# Captured at import, before the `patched` fixture swaps the module attributes, so
# a test can run the wizard against the real on-disk config merge.
_REAL_UPDATE_CONFIG = config_file.update_config
_REAL_LOAD_CONFIG = config_file.load_config
_REAL_SAVE_CONFIG = config_file.save_config
_REAL_SAVE_SECRETS = secret_store.save_secrets
_REAL_IS_FIRST_RUN = wizard._is_first_run
_REAL_VALIDATE_LLM = wizard.validators.validate_llm


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
    monkeypatch.setattr(wizard.secret_store, "store_exists", lambda **kw: False)
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
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s, **kw: True)
    monkeypatch.setattr(wizard.validators, "validate_fred", lambda k: True)
    monkeypatch.setattr(wizard.validators, "validate_llm", lambda *a, **k: True)
    # Clear every credential env var so interactive tests are deterministic:
    # env-autodetection (Rule 2) must not pick up the developer's real shell env.
    for _var in (
        "PORTOPT_PASSPHRASE",
        "TRADING_212_API_KEY",
        "TRADING_212_SECRET_KEY",
        "TRADING_212_MODE",
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
    wizard.run_setup_noninteractive(passphrase=_PW)
    assert patched["saved_secrets"] == {}
    assert patched["saved_config"] == {}
    assert patched["bootstrapped"] == ["db", "migrate"]


def test_setup_installs_launcher_by_default(patched: dict) -> None:
    """Setup installs the `optimizer` launcher onto PATH unless told to skip."""
    wizard.run_setup_noninteractive(passphrase=_PW)
    assert patched["path_installed"] is True


def test_setup_skip_path_install_leaves_path_untouched(patched: dict) -> None:
    """`skip_path_install=True` persists + bootstraps but does no PATH edit."""
    wizard.run_setup_noninteractive(passphrase=_PW, skip_path_install=True)
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
    wizard.run_setup_noninteractive(passphrase=_PW)
    assert patched["bootstrapped"] == ["db", "migrate"]
    assert patched["saved_secrets"] == {}


def test_interactive_installs_launcher_by_default(patched: dict) -> None:
    """The interactive path also installs the launcher by default."""
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["path_installed"] is True


def test_interactive_skip_path_install(patched: dict) -> None:
    """The interactive path honours skip_path_install (forwarded to the bootstrap)."""
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter, skip_path_install=True)
    assert patched["path_installed"] is False


def test_noninteractive_full_persists_all(patched: dict) -> None:
    wizard.run_setup_noninteractive(
        passphrase=_PW,
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
        wizard.run_setup_noninteractive(passphrase=_PW, t212_key="only-key")
    assert patched["saved_secrets"] is None


def test_noninteractive_requires_passphrase(patched: dict) -> None:
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase=None)


def test_noninteractive_t212_invalid_fails(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s, **kw: False)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(
            passphrase=_PW,
            t212_key="tk",
            t212_secret="ts",
        )
    assert patched["saved_secrets"] is None


def test_noninteractive_fred_invalid_fails(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.validators, "validate_fred", lambda k: False)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase=_PW, fred_key="bad")
    assert patched["saved_secrets"] is None


def test_interactive_t212_no_completes(patched: dict) -> None:
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
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
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "tk",
            wizard._MSG_T212_SECRET: "ts",
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
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s, **kw: False)
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
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
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
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
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"]["trading_212_api_key"] == "env-tk"
    assert patched["saved_secrets"]["trading_212_secret_key"] == "env-ts"  # noqa: S105 - test fixture value, not a secret


def _record_t212_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    seen: list[str] = []

    def _fake(key: str, secret: str, *, mode: str = "live") -> bool:
        seen.append(mode)
        return True

    monkeypatch.setattr(wizard.validators, "validate_t212", _fake)
    return seen


def test_noninteractive_t212_validated_against_env_mode(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Demo keys 401 on the live host; the wizard must honour the same
    # TRADING_212_MODE the daemon reads instead of always probing live.
    seen = _record_t212_mode(monkeypatch)
    monkeypatch.setenv("TRADING_212_MODE", "demo")
    wizard.run_setup_noninteractive(passphrase=_PW, t212_key="tk", t212_secret="ts")
    assert seen == ["demo"]


def test_noninteractive_t212_mode_defaults_to_live(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_t212_mode(monkeypatch)
    wizard.run_setup_noninteractive(passphrase=_PW, t212_key="tk", t212_secret="ts")
    assert seen == ["live"]


def test_interactive_t212_validated_against_env_mode(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_t212_mode(monkeypatch)
    monkeypatch.setenv("TRADING_212_MODE", " demo ")
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "tk",
            wizard._MSG_T212_SECRET: "ts",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert seen == ["demo"]


def test_noninteractive_persists_the_t212_mode_it_validated(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scheduler container never sees the wizard's shell env, so the mode the
    # keys were validated against must travel with them through config.toml.
    _record_t212_mode(monkeypatch)
    monkeypatch.setenv("TRADING_212_MODE", "demo")
    wizard.run_setup_noninteractive(passphrase=_PW, t212_key="tk", t212_secret="ts")
    assert patched["saved_config"]["trading_212_mode"] == "demo"


def test_noninteractive_persists_live_when_mode_unset(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Re-running with live keys after a demo setup must flip the stored mode back.
    _record_t212_mode(monkeypatch)
    wizard.run_setup_noninteractive(passphrase=_PW, t212_key="tk", t212_secret="ts")
    assert patched["saved_config"]["trading_212_mode"] == "live"


def test_noninteractive_without_t212_keys_leaves_stored_mode_alone(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADING_212_MODE", "demo")
    wizard.run_setup_noninteractive(passphrase=_PW, fred_key="fk")
    assert "trading_212_mode" not in patched["saved_config"]


def test_interactive_persists_the_t212_mode_it_validated(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_t212_mode(monkeypatch)
    monkeypatch.setenv("TRADING_212_MODE", "demo")
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "tk",
            wizard._MSG_T212_SECRET: "ts",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_config"]["trading_212_mode"] == "demo"


def test_unknown_t212_mode_fails_loud_instead_of_probing_live(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # validate_t212 silently maps an unknown mode to live; a typo such as
    # "paper" would validate (or fail) against the wrong host and be persisted.
    monkeypatch.setenv("TRADING_212_MODE", "paper")
    with pytest.raises(wizard.SetupError, match="TRADING_212_MODE"):
        wizard.run_setup_noninteractive(passphrase=_PW, t212_key="tk", t212_secret="ts")
    assert patched["saved_config"] is None


def test_t212_mode_is_case_insensitive(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_t212_mode(monkeypatch)
    monkeypatch.setenv("TRADING_212_MODE", "Demo")
    wizard.run_setup_noninteractive(passphrase=_PW, t212_key="tk", t212_secret="ts")
    assert seen == ["demo"]
    assert patched["saved_config"]["trading_212_mode"] == "demo"


def test_interactive_fred_key_autodetected_from_env(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "env-fk")
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
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
    prompter = NonInteractivePrompter(
        {wizard._MSG_PASSPHRASE: _PW, wizard._MSG_PASSPHRASE_CONFIRM: _PW}
    )
    with pytest.raises(wizard.docker_bootstrap.DockerError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None


# --- LLM prompt group + --llm-* flags -----------------------------------------


def _llm_prompter(answers: dict) -> NonInteractivePrompter:
    base = {
        wizard._MSG_PASSPHRASE: _PW,
        wizard._MSG_PASSPHRASE_CONFIRM: _PW,
        wizard._MSG_T212_KEY: "",
        wizard._MSG_FRED_KEY: "",
        wizard._MSG_CONFIGURE_LLM: True,
        wizard._MSG_LLM_BASE_URL: "",
    }
    base.update(answers)
    return NonInteractivePrompter(base)


def test_noninteractive_llm_persists_config_and_secret(patched: dict) -> None:
    wizard.run_setup_noninteractive(
        passphrase=_PW,
        llm_provider="openai",
        llm_model="gpt-4o",
        llm_key="sk-openai",
    )
    assert patched["saved_config"]["llm_provider"] == "openai"
    assert patched["saved_config"]["llm_model"] == "gpt-4o"
    assert patched["saved_secrets"]["openai_api_key"] == "sk-openai"


def test_noninteractive_llm_none_skips(patched: dict) -> None:
    wizard.run_setup_noninteractive(passphrase=_PW)
    assert "llm_provider" not in patched["saved_config"]


def test_noninteractive_llm_bad_key_fails_without_persist(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.validators, "validate_llm", lambda *a, **k: False)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(
            passphrase=_PW, llm_provider="openai", llm_key="bad"
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
            passphrase=_PW, llm_provider="openai", llm_key="k"
        )
    assert patched["saved_secrets"] is None


def test_noninteractive_llm_key_from_env(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-sk")
    wizard.run_setup_noninteractive(passphrase=_PW, llm_provider="anthropic")
    assert patched["saved_secrets"]["anthropic_api_key"] == "env-sk"


def test_noninteractive_llm_missing_key_fails(patched: dict) -> None:
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase=_PW, llm_provider="openai")
    assert patched["saved_secrets"] is None


def test_noninteractive_llm_unknown_provider_fails(patched: dict) -> None:
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(
            passphrase=_PW, llm_provider="bogus", llm_key="k"
        )
    assert patched["saved_secrets"] is None


def test_noninteractive_llm_aws_presence_only_no_secret(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    wizard.run_setup_noninteractive(passphrase=_PW, llm_provider="aws")
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
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
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


# --- idempotency / --reconfigure / --skip-validation / auto-launch -----------


def test_noninteractive_idempotent_rerun_preserves_untouched_secrets(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run with no flags must NOT wipe secrets stored by an earlier run."""
    monkeypatch.setattr(
        wizard.secret_store, "load_secrets", lambda pw, **k: {"fred_api_key": "fk"}
    )
    wizard.run_setup_noninteractive(passphrase=_PW)
    assert patched["saved_secrets"] == {"fred_api_key": "fk"}


def test_noninteractive_idempotent_rerun_merges_new_secret(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new secret merges with the existing store rather than replacing it."""
    monkeypatch.setattr(
        wizard.secret_store, "load_secrets", lambda pw, **k: {"fred_api_key": "fk"}
    )
    wizard.run_setup_noninteractive(passphrase=_PW, t212_key="tk", t212_secret="ts")
    assert patched["saved_secrets"] == {
        "fred_api_key": "fk",
        "trading_212_api_key": "tk",
        "trading_212_secret_key": "ts",
    }


def test_noninteractive_first_run_saves_only_new_secrets(patched: dict) -> None:
    """With no existing store (NotFound), the merge is just the new secrets."""
    wizard.run_setup_noninteractive(passphrase=_PW, fred_key="fk")
    assert patched["saved_secrets"] == {"fred_api_key": "fk"}


def test_noninteractive_rerun_wrong_passphrase_fails_without_wiping(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong passphrase on a re-run fails loud and never overwrites the store."""

    def _bad(pw: str, **k: object) -> dict:
        raise wizard.secret_store.InvalidPassphraseError("wrong")

    monkeypatch.setattr(wizard.secret_store, "load_secrets", _bad)
    monkeypatch.setattr(wizard.secret_store, "store_exists", lambda **kw: True)
    with pytest.raises(wizard.SetupError, match="original passphrase"):
        wizard.run_setup_noninteractive(passphrase="wrong", fred_key="fk")
    assert patched["saved_secrets"] is None


def test_noninteractive_skip_validation_bypasses_t212(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--skip-validation persists T212 without a live call, even if it would fail."""
    monkeypatch.setattr(wizard.validators, "validate_t212", lambda k, s, **kw: False)
    wizard.run_setup_noninteractive(
        passphrase=_PW, t212_key="tk", t212_secret="ts", skip_validation=True
    )
    assert patched["saved_secrets"]["trading_212_api_key"] == "tk"


def test_noninteractive_skip_validation_bypasses_fred(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--skip-validation persists FRED without a live call."""
    monkeypatch.setattr(wizard.validators, "validate_fred", lambda k: False)
    wizard.run_setup_noninteractive(passphrase=_PW, fred_key="fk", skip_validation=True)
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
        passphrase=_PW,
        llm_provider="openai",
        llm_key="sk",
        skip_validation=True,
    )
    assert seen["skip"] is True


def test_noninteractive_never_auto_launches(patched: dict) -> None:
    """Non-interactive/CI setup persists + bootstraps but never brings the stack up."""
    wizard.run_setup_noninteractive(passphrase=_PW)
    assert patched["bootstrapped"] == ["db", "migrate"]
    assert patched["launched"] is False


def test_interactive_first_run_auto_launches(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first interactive run brings the stack up unless told not to."""
    monkeypatch.setattr(wizard, "_is_first_run", lambda: True)
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
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
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )
    wizard.run_setup_interactive(prompter, no_launch=True)
    assert patched["launched"] is False


def test_interactive_rerun_never_auto_launches(patched: dict) -> None:
    """A re-run (not first run) never auto-launches, even interactively."""
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
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
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
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


def test_setup_without_a_checkout_fails_before_persisting_anything(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stack can only be brought up from a checkout; discovering that after the
    secret store and config were written leaves a half-installed machine whose re-run
    is no longer a first run (so it never auto-launches)."""

    def _no_checkout() -> None:
        raise wizard.docker_bootstrap.DockerError("no docker-compose.yml")

    monkeypatch.setattr(wizard.docker_bootstrap, "compose_dir", _no_checkout)
    with pytest.raises(wizard.docker_bootstrap.DockerError):
        wizard.run_setup_noninteractive(passphrase=_PW, skip_validation=True)
    prompter = NonInteractivePrompter(
        {wizard._MSG_PASSPHRASE: _PW, wizard._MSG_PASSPHRASE_CONFIRM: _PW}
    )
    with pytest.raises(wizard.docker_bootstrap.DockerError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None
    assert patched["saved_config"] is None
    assert patched["bootstrapped"] == []


# --- new-store passphrase policy: minimum length + confirmation ---------------


class _ConfirmingPrompter(NonInteractivePrompter):
    """Presets known prompts; any other password prompt is the confirmation.

    Confirmations are answered from ``confirmations`` in order, so a test can script a
    typo followed by a correct re-entry without depending on the prompt wording.
    """

    def __init__(self, answers: dict, confirmations: list[str]) -> None:
        super().__init__(answers)
        self._confirmations = list(confirmations)
        self.confirmations_asked = 0

    def password(self, message: str) -> str:
        if message in self._answers:
            return str(self._answers[message])
        self.confirmations_asked += 1
        return self._confirmations.pop(0)


def _first_run_answers(passphrase: str) -> dict:
    return {
        wizard._MSG_PASSPHRASE: passphrase,
        wizard._MSG_T212_KEY: "",
        wizard._MSG_FRED_KEY: "",
        wizard._MSG_CONFIGURE_LLM: False,
    }


def test_noninteractive_rejects_a_short_passphrase_for_a_new_store(
    patched: dict,
) -> None:
    with pytest.raises(wizard.SetupError, match="at least"):
        wizard.run_setup_noninteractive(passphrase="short")
    assert patched["saved_secrets"] is None


def test_interactive_env_passphrase_too_short_for_a_new_store_is_rejected(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PORTOPT_PASSPHRASE", "short")
    prompter = _ConfirmingPrompter(_first_run_answers("unused"), confirmations=[])
    with pytest.raises(wizard.SetupError, match="at least"):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None


def test_interactive_new_store_rejects_a_short_passphrase(patched: dict) -> None:
    prompter = _ConfirmingPrompter(
        _first_run_answers("short"), confirmations=["short"] * 10
    )
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None


def test_interactive_new_store_rejects_a_confirmation_mismatch(
    patched: dict,
) -> None:
    prompter = _ConfirmingPrompter(
        _first_run_answers(_PW), confirmations=["typo-horse-battery"] * 10
    )
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] is None


def test_interactive_new_store_reprompts_after_a_confirmation_typo(
    patched: dict,
) -> None:
    prompter = _ConfirmingPrompter(
        _first_run_answers(_PW), confirmations=["typo-horse-battery", _PW]
    )
    wizard.run_setup_interactive(prompter)
    assert patched["saved_secrets"] == {}
    assert prompter.confirmations_asked == 2


def test_existing_store_is_opened_with_a_short_legacy_passphrase(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The length floor is for choosing a passphrase; a store made under an older
    # policy must stay reachable with the passphrase it was made with.
    monkeypatch.setattr(wizard.secret_store, "store_exists", lambda **kw: True)
    monkeypatch.setattr(
        wizard.secret_store, "load_secrets", lambda pw, **k: {"fred_api_key": "fk"}
    )
    wizard.run_setup_noninteractive(passphrase=_PW)
    assert patched["saved_secrets"] == {"fred_api_key": "fk"}


def test_interactive_existing_store_asks_once_and_accepts_a_short_passphrase(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard.secret_store, "store_exists", lambda **kw: True)
    prompter = _ConfirmingPrompter(_first_run_answers("pw"), confirmations=[])
    wizard.run_setup_interactive(prompter)
    assert prompter.confirmations_asked == 0
    assert patched["saved_secrets"] == {}


def test_interactive_env_passphrase_is_not_asked_to_confirm(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PORTOPT_PASSPHRASE", _PW)
    prompter = _ConfirmingPrompter(_first_run_answers("unused"), confirmations=[])
    wizard.run_setup_interactive(prompter)
    assert prompter.confirmations_asked == 0
    assert patched["saved_secrets"] == {}


# --- provider switch replaces the LLM section instead of merging over it -----


def _use_real_config_file(monkeypatch: pytest.MonkeyPatch, path) -> None:
    monkeypatch.setattr(
        wizard.config_file,
        "update_config",
        lambda config, **kw: _REAL_UPDATE_CONFIG(config, path=path),
    )
    monkeypatch.setattr(
        wizard.config_file, "load_config", lambda **kw: _REAL_LOAD_CONFIG(path=path)
    )


def test_provider_switch_drops_the_previous_providers_llm_settings(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A stale model/base URL/region/Azure field must not follow a new provider.

    The merge keeps keys a run does not stage, so ``--llm-provider openai`` over an
    earlier local-ollama setup would otherwise route the OpenAI key to the old
    ollama endpoint (and ask it for an ollama model id).
    """
    path = tmp_path / "config.toml"
    _REAL_SAVE_CONFIG(
        {
            "llm_provider": "ollama",
            "llm_model": "llama3.1:8b",
            "llm_fallback_model": "llama3",
            "llm_base_url": "http://localhost:11434",
            "aws_region": "eu-west-1",
            "azure_openai_endpoint": "https://az.openai.azure.com",
            "azure_openai_api_version": "2024-06-01",
            "azure_openai_deployment_name": "gpt4o-deploy",
            "repo_path": "/home/u/optimizer",
        },
        path=path,
    )
    _use_real_config_file(monkeypatch, path)

    wizard.run_setup_noninteractive(
        passphrase=_PW, llm_provider="openai", llm_model="gpt-4o", llm_key="sk-o"
    )

    persisted = _REAL_LOAD_CONFIG(path=path)
    assert persisted == {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "repo_path": "/home/u/optimizer",
    }
    env = compose_env.config_to_env(persisted)
    assert "OPENAI_BASE_URL" not in env
    assert env["FUND_FALLBACK_MODEL"] == "gpt-4o"


def test_interactive_blank_model_and_base_url_clear_the_stored_values(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Blank answers mean "hosted default", not "keep whatever was stored"."""
    path = tmp_path / "config.toml"
    _REAL_SAVE_CONFIG(
        {
            "llm_provider": "ollama",
            "llm_model": "llama3.1:8b",
            "llm_base_url": "http://localhost:11434",
        },
        path=path,
    )
    _use_real_config_file(monkeypatch, path)
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "",
            wizard._MSG_LLM_KEY: "sk-openai",
        }
    )

    wizard.run_setup_interactive(prompter, reconfigure=True)

    assert _REAL_LOAD_CONFIG(path=path) == {"llm_provider": "openai"}


def test_declining_llm_reconfigure_keeps_the_stored_llm_settings(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Only a staged provider replaces the section; declining must not wipe it."""
    path = tmp_path / "config.toml"
    stored = {
        "llm_provider": "ollama",
        "llm_model": "llama3.1:8b",
        "llm_base_url": "http://localhost:11434",
    }
    _REAL_SAVE_CONFIG(stored, path=path)
    _use_real_config_file(monkeypatch, path)
    prompter = NonInteractivePrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: False,
        }
    )

    wizard.run_setup_interactive(prompter)

    assert _REAL_LOAD_CONFIG(path=path) == stored


# --- a same-provider re-run keeps what it was not asked about ----------------

_STORED_OPENAI_PROXY = {
    "llm_provider": "openai",
    "llm_model": "gpt-4o",
    "llm_fallback_model": "gpt-4o-mini",
    "llm_base_url": "http://proxy.internal/v1",
    "repo_path": "/home/u/optimizer",
}


def _store_config(monkeypatch: pytest.MonkeyPatch, tmp_path, stored: dict):
    path = tmp_path / "config.toml"
    _REAL_SAVE_CONFIG(dict(stored), path=path)
    _use_real_config_file(monkeypatch, path)
    return path


def _record_llm_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str | None, str | None]]:
    probes: list[tuple[str, str | None, str | None]] = []

    def _validate(provider, key=None, base_url=None, **fields):
        probes.append((provider, key, base_url))
        return True

    monkeypatch.setattr(wizard.validators, "validate_llm", _validate)
    return probes


def test_interactive_same_provider_rerun_keeps_the_hand_set_fallback_model(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The wizard never asks for ``llm_fallback_model``, so a reconfigure of the same
    provider must not drop the one the operator set by hand."""
    path = _store_config(monkeypatch, tmp_path, _STORED_OPENAI_PROXY)
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "gpt-4.1",
            wizard._MSG_LLM_BASE_URL: "http://proxy.internal/v1",
            wizard._MSG_LLM_KEY: "sk-new",
        }
    )

    wizard.run_setup_interactive(prompter, reconfigure=True)

    assert _REAL_LOAD_CONFIG(path=path) == {
        **_STORED_OPENAI_PROXY,
        "llm_model": "gpt-4.1",
    }


def test_interactive_same_provider_blank_answers_clear_model_and_base_url(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Only the fields the wizard asks about follow the answer; blank clears them."""
    path = _store_config(
        monkeypatch,
        tmp_path,
        {
            "llm_provider": "ollama",
            "llm_model": "llama3.1:8b",
            "llm_fallback_model": "llama3",
            "llm_base_url": "http://localhost:11434",
        },
    )
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "ollama",
            wizard._MSG_LLM_MODEL: "",
            wizard._MSG_LLM_BASE_URL: "",
            wizard._MSG_LLM_KEY: "ollama-cloud-key",
        }
    )

    wizard.run_setup_interactive(prompter, reconfigure=True)

    assert _REAL_LOAD_CONFIG(path=path) == {
        "llm_provider": "ollama",
        "llm_fallback_model": "llama3",
    }


def test_noninteractive_same_provider_key_rotation_keeps_model_and_base_url(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``--llm-provider openai --llm-key NEW`` must not forget the proxy or model,
    and must probe the new key against the host the config keeps pointing at."""
    path = _store_config(monkeypatch, tmp_path, _STORED_OPENAI_PROXY)
    probes = _record_llm_probes(monkeypatch)

    wizard.run_setup_noninteractive(
        passphrase=_PW, llm_provider="openai", llm_key="sk-new"
    )

    assert _REAL_LOAD_CONFIG(path=path) == _STORED_OPENAI_PROXY
    assert patched["saved_secrets"] == {"openai_api_key": "sk-new"}
    assert probes == [("openai", "sk-new", "http://proxy.internal/v1")]
    env = compose_env.config_to_env(_REAL_LOAD_CONFIG(path=path))
    assert env["FUND_FALLBACK_MODEL"] == "gpt-4o-mini"


def test_noninteractive_same_provider_flags_override_the_stored_values(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = _store_config(monkeypatch, tmp_path, _STORED_OPENAI_PROXY)
    probes = _record_llm_probes(monkeypatch)

    wizard.run_setup_noninteractive(
        passphrase=_PW,
        llm_provider="openai",
        llm_model="gpt-4.1",
        llm_base_url="http://other.internal/v1",
        llm_key="sk-new",
    )

    assert _REAL_LOAD_CONFIG(path=path) == {
        **_STORED_OPENAI_PROXY,
        "llm_model": "gpt-4.1",
        "llm_base_url": "http://other.internal/v1",
    }
    assert probes == [("openai", "sk-new", "http://other.internal/v1")]


def test_noninteractive_provider_switch_does_not_inherit_model_or_base_url(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = _store_config(monkeypatch, tmp_path, _STORED_OPENAI_PROXY)
    probes = _record_llm_probes(monkeypatch)

    wizard.run_setup_noninteractive(
        passphrase=_PW, llm_provider="anthropic", llm_key="sk-ant"
    )

    assert _REAL_LOAD_CONFIG(path=path) == {
        "llm_provider": "anthropic",
        "repo_path": "/home/u/optimizer",
    }
    assert probes == [("anthropic", "sk-ant", None)]


# --- prompt defaults come from the stored config only for the same provider ---


class _DefaultTakingPrompter(NonInteractivePrompter):
    """Mimics the real prompters: a blank answer takes the offered default."""

    def __init__(self, answers: dict) -> None:
        super().__init__(answers)
        self.defaults: dict[str, str | None] = {}

    def text(self, message: str, *, default: str | None = None) -> str:
        self.defaults[message] = default
        return super().text(message, default=default) or (default or "")


_STORED_NVIDIA_NIM = {
    "llm_provider": "nvidia",
    "llm_model": "meta/llama-3.1-70b-instruct",
    "llm_base_url": "http://10.0.0.5:8000/v1",
}


def test_interactive_provider_switch_does_not_prefill_the_old_base_url(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enter at the base-URL prompt must not probe the new key against the old host."""
    monkeypatch.setattr(
        wizard.config_file, "load_config", lambda **k: dict(_STORED_NVIDIA_NIM)
    )
    probes: list[tuple[str, str | None, str | None]] = []

    def _validate(provider, key=None, base_url=None, **fields):
        probes.append((provider, key, base_url))
        return True

    monkeypatch.setattr(wizard.validators, "validate_llm", _validate)
    prompter = _DefaultTakingPrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: True,
            wizard._MSG_LLM_PROVIDER: "anthropic",
            wizard._MSG_LLM_MODEL: "",
            wizard._MSG_LLM_BASE_URL: "",
            wizard._MSG_LLM_KEY: "sk-ant-real",
        }
    )

    wizard.run_setup_interactive(prompter, reconfigure=True)

    assert not prompter.defaults[wizard._MSG_LLM_BASE_URL]
    assert not prompter.defaults[wizard._MSG_LLM_MODEL]
    assert probes == [("anthropic", "sk-ant-real", None)]
    assert patched["saved_config"].get("llm_base_url") is None


def test_interactive_same_provider_rerun_prefills_the_stored_values(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run of the same provider still offers what the earlier run stored."""
    monkeypatch.setattr(
        wizard.config_file, "load_config", lambda **k: dict(_STORED_NVIDIA_NIM)
    )
    prompter = _DefaultTakingPrompter(
        {
            wizard._MSG_PASSPHRASE: _PW,
            wizard._MSG_PASSPHRASE_CONFIRM: _PW,
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: True,
            wizard._MSG_LLM_PROVIDER: "nvidia",
            wizard._MSG_LLM_MODEL: "",
            wizard._MSG_LLM_BASE_URL: "",
        }
    )

    wizard.run_setup_interactive(prompter, reconfigure=True)

    assert prompter.defaults[wizard._MSG_LLM_BASE_URL] == "http://10.0.0.5:8000/v1"
    assert patched["saved_config"]["llm_base_url"] == "http://10.0.0.5:8000/v1"
    assert patched["saved_config"]["llm_model"] == "meta/llama-3.1-70b-instruct"


def _model_default_after(
    patched: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stored: dict[str, object],
    provider: str,
) -> tuple[str | None, str | None]:
    """Run a reconfigure that presses Enter at the model prompt.

    Returns the model default that was offered and the model that got staged.
    """
    monkeypatch.setattr(wizard.config_file, "load_config", lambda **k: dict(stored))
    prompter = _DefaultTakingPrompter(
        {
            wizard._MSG_PASSPHRASE: "pw",
            wizard._MSG_T212_KEY: "",
            wizard._MSG_FRED_KEY: "",
            wizard._MSG_CONFIGURE_LLM: True,
            wizard._MSG_LLM_PROVIDER: provider,
            wizard._MSG_LLM_MODEL: "",
            wizard._MSG_LLM_BASE_URL: "",
            wizard._MSG_LLM_KEY: "sk-test",
        }
    )
    wizard.run_setup_interactive(prompter, reconfigure=True)
    return (
        prompter.defaults[wizard._MSG_LLM_MODEL],
        patched["saved_config"].get("llm_model"),
    )


def test_ollama_rerun_offers_the_stored_model_not_the_cloud_default(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enter at the model prompt must keep a model the user chose earlier."""
    offered, staged = _model_default_after(
        patched,
        monkeypatch,
        stored={"llm_provider": "ollama", "llm_model": "qwen3:8b"},
        provider="ollama",
    )

    assert offered == "qwen3:8b"
    assert staged == "qwen3:8b"


def test_ollama_first_setup_offers_the_cloud_default_model(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    offered, staged = _model_default_after(
        patched, monkeypatch, stored={}, provider="ollama"
    )

    assert offered == "deepseek-v4.1-flash:cloud"
    assert staged == "deepseek-v4.1-flash:cloud"


def test_switching_to_ollama_offers_the_cloud_default_not_the_old_model(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    offered, staged = _model_default_after(
        patched,
        monkeypatch,
        stored={"llm_provider": "openai", "llm_model": "gpt-4o"},
        provider="ollama",
    )

    assert offered == "deepseek-v4.1-flash:cloud"
    assert staged == "deepseek-v4.1-flash:cloud"


def test_switching_providers_never_offers_the_old_providers_model(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare Enter used to store ``gpt-4o`` as the model id for anthropic."""
    offered, staged = _model_default_after(
        patched,
        monkeypatch,
        stored={"llm_provider": "openai", "llm_model": "gpt-4o"},
        provider="anthropic",
    )

    assert not offered
    assert staged is None


# --- a stored base URL / model can be cleared on a same-provider re-run -------
#
# A blank answer cannot express "clear": the plain-stdin prompter (Git Bash) turns it
# into the offered default, and a missing flag means "not given". A sentinel does.

_STORED_LOCAL_OLLAMA = {
    "llm_provider": "ollama",
    "llm_model": "llama3.1:8b",
    "llm_base_url": "http://localhost:11434",
}


def _same_provider_rerun_answers(**overrides: object) -> dict:
    answers = {
        wizard._MSG_PASSPHRASE: _PW,
        wizard._MSG_PASSPHRASE_CONFIRM: _PW,
        wizard._MSG_T212_KEY: "",
        wizard._MSG_FRED_KEY: "",
        wizard._MSG_CONFIGURE_LLM: True,
        wizard._MSG_LLM_PROVIDER: "ollama",
        wizard._MSG_LLM_MODEL: "llama3.1:8b",
        wizard._MSG_LLM_BASE_URL: "",
        wizard._MSG_LLM_KEY: "ollama-cloud-key",
    }
    answers.update(overrides)
    return answers


def test_enter_at_the_base_url_prompt_keeps_the_stored_url(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = _store_config(monkeypatch, tmp_path, _STORED_LOCAL_OLLAMA)
    prompter = _DefaultTakingPrompter(_same_provider_rerun_answers())

    wizard.run_setup_interactive(prompter, reconfigure=True)

    assert _REAL_LOAD_CONFIG(path=path) == _STORED_LOCAL_OLLAMA


@pytest.mark.parametrize("sentinel", ["-", "none", "None", " - "])
def test_sentinel_at_the_base_url_prompt_clears_the_stored_url(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path, sentinel: str
) -> None:
    """Moving local ollama to Ollama Cloud: Enter re-offers the local URL, the
    sentinel is the way to say "back to the hosted default"."""
    path = _store_config(monkeypatch, tmp_path, _STORED_LOCAL_OLLAMA)
    probes = _record_llm_probes(monkeypatch)
    prompter = _DefaultTakingPrompter(
        _same_provider_rerun_answers(**{wizard._MSG_LLM_BASE_URL: sentinel})
    )

    wizard.run_setup_interactive(prompter, reconfigure=True)

    assert _REAL_LOAD_CONFIG(path=path) == {
        "llm_provider": "ollama",
        "llm_model": "llama3.1:8b",
    }
    assert probes == [("ollama", "ollama-cloud-key", None)]
    assert patched["saved_secrets"] == {"ollama_api_key": "ollama-cloud-key"}


def test_sentinel_at_the_model_prompt_clears_the_stored_model(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = _store_config(monkeypatch, tmp_path, _STORED_OPENAI_PROXY)
    prompter = _DefaultTakingPrompter(
        _same_provider_rerun_answers(
            **{
                wizard._MSG_LLM_PROVIDER: "openai",
                wizard._MSG_LLM_MODEL: "-",
                wizard._MSG_LLM_BASE_URL: "",
                wizard._MSG_LLM_KEY: "sk-new",
            }
        )
    )

    wizard.run_setup_interactive(prompter, reconfigure=True)

    stored = dict(_STORED_OPENAI_PROXY)
    del stored["llm_model"]
    assert _REAL_LOAD_CONFIG(path=path) == stored


def test_noninteractive_sentinel_clears_the_stored_base_url_and_model(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = _store_config(monkeypatch, tmp_path, _STORED_OPENAI_PROXY)
    probes = _record_llm_probes(monkeypatch)

    wizard.run_setup_noninteractive(
        passphrase=_PW,
        llm_provider="openai",
        llm_model="none",
        llm_base_url="-",
        llm_key="sk-new",
    )

    assert _REAL_LOAD_CONFIG(path=path) == {
        "llm_provider": "openai",
        "llm_fallback_model": "gpt-4o-mini",
        "repo_path": "/home/u/optimizer",
    }
    assert probes == [("openai", "sk-new", None)]


def test_noninteractive_blank_flag_counts_as_not_given(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An unset shell variable expanding to "" must not silently wipe the stored URL."""
    path = _store_config(monkeypatch, tmp_path, _STORED_OPENAI_PROXY)

    wizard.run_setup_noninteractive(
        passphrase=_PW, llm_provider="openai", llm_base_url="  ", llm_key="sk-new"
    )

    assert _REAL_LOAD_CONFIG(path=path) == _STORED_OPENAI_PROXY


@pytest.mark.parametrize("message", [wizard._MSG_LLM_MODEL, wizard._MSG_LLM_BASE_URL])
def test_llm_prompts_tell_the_operator_how_to_clear_a_stored_value(
    message: str,
) -> None:
    assert "clear" in message.lower()
    assert "-" in message


def test_wrong_passphrase_error_names_the_recovery_path(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The only way out of a lost passphrase is deleting the store; say so."""
    store = tmp_path / "secrets.enc"
    monkeypatch.setattr(wizard.secret_store, "DEFAULT_SECRETS_PATH", store)

    def _bad(pw: str, **k: object) -> dict:
        raise wizard.secret_store.InvalidPassphraseError("wrong")

    monkeypatch.setattr(wizard.secret_store, "load_secrets", _bad)
    monkeypatch.setattr(wizard.secret_store, "store_exists", lambda **kw: True)
    with pytest.raises(wizard.SetupError) as excinfo:
        wizard.run_setup_noninteractive(passphrase=_PW)
    message = str(excinfo.value)
    assert str(store) in message
    assert "delete" in message.lower()


# --- persistence happens only after the DB bootstrap succeeded --------------


def test_bootstrap_failure_persists_nothing(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Docker failure must not leave an encrypted store or config behind."""

    def _down() -> None:
        raise wizard.docker_bootstrap.DockerError("port 54320 busy")

    monkeypatch.setattr(wizard.docker_bootstrap, "bring_up_db", _down)
    with pytest.raises(wizard.docker_bootstrap.DockerError):
        wizard.run_setup_noninteractive(passphrase=_PW, fred_key="fk")
    assert patched["saved_secrets"] is None
    assert patched["saved_config"] is None


def test_secrets_and_config_are_written_after_the_bootstrap(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        wizard.docker_bootstrap, "bring_up_db", lambda: events.append("db")
    )
    monkeypatch.setattr(
        wizard.docker_bootstrap, "migrate", lambda: events.append("migrate")
    )
    monkeypatch.setattr(
        wizard.secret_store,
        "save_secrets",
        lambda secrets, passphrase, **kw: events.append("secrets"),
    )
    monkeypatch.setattr(
        wizard.config_file,
        "update_config",
        lambda config, **kw: events.append("config"),
    )
    wizard.run_setup_noninteractive(passphrase=_PW)
    assert events == ["db", "migrate", "secrets", "config"]


def test_wrong_passphrase_fails_before_the_db_is_touched(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _bad(pw: str, **k: object) -> dict:
        raise wizard.secret_store.InvalidPassphraseError("wrong")

    monkeypatch.setattr(wizard.secret_store, "load_secrets", _bad)
    with pytest.raises(wizard.SetupError):
        wizard.run_setup_noninteractive(passphrase="wrong")
    assert patched["bootstrapped"] == []


def test_failed_first_setup_is_still_a_first_run_on_retry(
    patched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The retry after a Docker failure must still auto-launch and take a new passphrase."""
    monkeypatch.setattr(
        wizard.secret_store, "DEFAULT_SECRETS_PATH", tmp_path / "new" / "secrets.enc"
    )
    monkeypatch.setattr(
        wizard.config_file, "DEFAULT_CONFIG_PATH", tmp_path / "new" / "config.toml"
    )
    monkeypatch.setattr(wizard.secret_store, "save_secrets", _REAL_SAVE_SECRETS)
    monkeypatch.setattr(wizard.config_file, "update_config", _REAL_UPDATE_CONFIG)
    monkeypatch.setattr(wizard, "_is_first_run", _REAL_IS_FIRST_RUN)

    def _down() -> None:
        raise wizard.docker_bootstrap.DockerError("image pull failed")

    monkeypatch.setattr(wizard.docker_bootstrap, "bring_up_db", _down)
    with pytest.raises(wizard.docker_bootstrap.DockerError):
        wizard.run_setup_noninteractive(passphrase=_PW, fred_key="fk")

    assert wizard._is_first_run() is True


class _ScriptedPrompter(NonInteractivePrompter):
    """Answers a repeated prompt from a list, in order; records error messages."""

    def __init__(self, answers: dict) -> None:
        super().__init__(answers)
        self.errors: list[str] = []

    def text(self, message: str, *, default: str | None = None) -> str:
        return self._next(message)

    def password(self, message: str) -> str:
        return self._next(message)

    def _next(self, message: str) -> str:
        answer = self._get(message)
        if isinstance(answer, list):
            return answer.pop(0)
        return str(answer)

    def error(self, message: str) -> None:
        self.errors.append(message)


def _first_setup_answers(**extra: object) -> dict:
    return {
        wizard._MSG_PASSPHRASE: _PW,
        wizard._MSG_PASSPHRASE_CONFIRM: _PW,
        wizard._MSG_T212_KEY: "",
        wizard._MSG_FRED_KEY: "",
        wizard._MSG_CONFIGURE_LLM: False,
        **extra,
    }


# --- required provider fields are re-asked, not reported as a bad key --------


def _use_real_llm_validation(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Run the real validators with a stub transport; return the probed URLs."""
    urls: list[str] = []

    def _ok(url: str, **kwargs: object):
        urls.append(url)
        return type("Resp", (), {"status_code": 200})()

    monkeypatch.setattr(wizard.validators, "validate_llm", _REAL_VALIDATE_LLM)
    monkeypatch.setattr(wizard.validators, "_llm_get", _ok)
    return urls


def _azure_answers(**extra: object) -> dict:
    return _first_setup_answers(
        **{
            wizard._MSG_CONFIGURE_LLM: True,
            wizard._MSG_LLM_PROVIDER: "microsoft",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_LLM_BASE_URL: "",
            wizard._MSG_AZURE_ENDPOINT: "https://az.openai.azure.com",
            wizard._MSG_AZURE_API_VERSION: "2024-06-01",
            wizard._MSG_AZURE_DEPLOYMENT: "gpt4o-deploy",
            wizard._MSG_LLM_KEY: "az-key",
            **extra,
        }
    )


def test_blank_azure_fields_are_reasked_instead_of_rejecting_the_key(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank API version/deployment is a missing field, not a bad key.

    The validator returns False for it, which the key loop used to read as a
    rejected key: the user retyped a valid key three times and then got a failure
    that never named the empty field.
    """
    _use_real_llm_validation(monkeypatch)
    prompter = _ScriptedPrompter(
        _azure_answers(
            **{
                wizard._MSG_AZURE_API_VERSION: ["", "2024-06-01"],
                wizard._MSG_AZURE_DEPLOYMENT: ["", "gpt4o-deploy"],
            }
        )
    )

    wizard.run_setup_interactive(prompter)

    assert not any("key" in message.lower() for message in prompter.errors)
    assert len(prompter.errors) == 2
    assert patched["saved_config"]["azure_openai_api_version"] == "2024-06-01"
    assert patched["saved_config"]["azure_openai_deployment_name"] == "gpt4o-deploy"
    assert patched["saved_secrets"]["azure_openai_api_key"] == "az-key"


def test_azure_field_left_blank_fails_naming_the_field_without_asking_for_the_key(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_llm_validation(monkeypatch)
    # No answer for the key prompt: asking for it would raise PromptUnavailableError.
    answers = _azure_answers(**{wizard._MSG_AZURE_API_VERSION: ""})
    del answers[wizard._MSG_LLM_KEY]

    with pytest.raises(wizard.SetupError, match="API version"):
        wizard.run_setup_interactive(_ScriptedPrompter(answers))
    assert patched["saved_secrets"] is None


def test_blank_aws_region_is_reasked(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_llm_validation(monkeypatch)
    prompter = _ScriptedPrompter(
        _first_setup_answers(
            **{
                wizard._MSG_CONFIGURE_LLM: True,
                wizard._MSG_LLM_PROVIDER: "aws",
                wizard._MSG_LLM_MODEL: "amazon.titan-text",
                wizard._MSG_LLM_BASE_URL: "",
                wizard._MSG_AWS_REGION: ["", "eu-west-1"],
            }
        )
    )

    wizard.run_setup_interactive(prompter)

    assert len(prompter.errors) == 1
    assert patched["saved_config"]["aws_region"] == "eu-west-1"


# --- pasted secrets are stripped before they reach a header ------------------


def _record_llm_keys(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    keys: list[str | None] = []

    def _validate(provider, key=None, base_url=None, **fields):
        keys.append(key)
        return True

    monkeypatch.setattr(wizard.validators, "validate_llm", _validate)
    return keys


@pytest.mark.parametrize("pasted", ["sk-abc ", "sk-abc\n", "  sk-abc\r\n"])
def test_interactive_llm_key_typed_at_the_prompt_is_stripped(
    patched: dict, monkeypatch: pytest.MonkeyPatch, pasted: str
) -> None:
    """A trailing space/newline made the HTTP client reject the header outright."""
    keys = _record_llm_keys(monkeypatch)
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_LLM_KEY: pasted,
        }
    )

    wizard.run_setup_interactive(prompter)

    assert keys == ["sk-abc"]
    assert patched["saved_secrets"]["openai_api_key"] == "sk-abc"


def test_interactive_llm_key_from_env_is_stripped(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CRLF env file leaves a carriage return on the exported value."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env\r")
    keys = _record_llm_keys(monkeypatch)
    prompter = _llm_prompter(
        {wizard._MSG_LLM_PROVIDER: "openai", wizard._MSG_LLM_MODEL: "gpt-4o"}
    )

    wizard.run_setup_interactive(prompter)

    assert keys == ["sk-env"]
    assert patched["saved_secrets"]["openai_api_key"] == "sk-env"


def test_interactive_blank_env_key_falls_through_to_the_prompt(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "  ")
    keys = _record_llm_keys(monkeypatch)
    prompter = _llm_prompter(
        {
            wizard._MSG_LLM_PROVIDER: "openai",
            wizard._MSG_LLM_MODEL: "gpt-4o",
            wizard._MSG_LLM_KEY: "sk-typed",
        }
    )

    wizard.run_setup_interactive(prompter)

    assert keys == ["sk-typed"]


def test_noninteractive_llm_key_flag_is_stripped(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = _record_llm_keys(monkeypatch)

    wizard.run_setup_noninteractive(
        passphrase=_PW, llm_provider="openai", llm_key="sk-flag \n"
    )

    assert keys == ["sk-flag"]
    assert patched["saved_secrets"]["openai_api_key"] == "sk-flag"


def test_noninteractive_llm_key_from_env_is_stripped(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-sk\r\n")
    keys = _record_llm_keys(monkeypatch)

    wizard.run_setup_noninteractive(passphrase=_PW, llm_provider="anthropic")

    assert keys == ["env-sk"]
    assert patched["saved_secrets"]["anthropic_api_key"] == "env-sk"


def test_noninteractive_whitespace_only_llm_key_flag_counts_as_missing(
    patched: dict,
) -> None:
    with pytest.raises(wizard.SetupError, match="needs an API key"):
        wizard.run_setup_noninteractive(
            passphrase=_PW, llm_provider="openai", llm_key="  "
        )
    assert patched["saved_secrets"] is None


def test_noninteractive_t212_and_fred_flags_are_stripped(
    patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, tuple[str, ...]] = {}
    monkeypatch.setattr(
        wizard.validators,
        "validate_t212",
        lambda k, s, **_: seen.update(t212=(k, s)) or True,
    )
    monkeypatch.setattr(
        wizard.validators, "validate_fred", lambda k: seen.update(fred=(k,)) or True
    )

    wizard.run_setup_noninteractive(
        passphrase=_PW, t212_key="id \n", t212_secret=" sec\r", fred_key="fred \n"
    )

    assert seen == {"t212": ("id", "sec"), "fred": ("fred",)}
    assert patched["saved_secrets"]["trading_212_api_key"] == "id"
    assert patched["saved_secrets"]["trading_212_secret_key"] == "sec"  # noqa: S105 - test value, not a secret
    assert patched["saved_secrets"]["fred_api_key"] == "fred"
