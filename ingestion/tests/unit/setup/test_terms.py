"""Yahoo's terms: setup asks for the acknowledgement and the daemon requires it.

``portopt setup`` shows Yahoo's terms and stores the acknowledgement in
``config.toml``; a non-interactive setup needs ``--accept-yahoo-terms``; a re-run
that finds it stored asks nothing. ``portopt start`` hands it to the scheduler
container as ``YAHOO_TERMS_ACKNOWLEDGED``, and the ingestion daemon refuses to
start without it. Docker, the secret store and the network are patched out; the
config file lives in a temporary directory.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app import worker
from app.config import Settings
from app.setup import compose_env, config_file, terms, wizard
from app.setup.prompts import NonInteractivePrompter

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_COMPOSE_VARIABLE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")

_UNLOCK_PHRASE = "correct-horse-battery"
_DAEMON_REFUSAL_EXIT_CODE = 2
_DOCKER_STEPS = ("ensure_docker", "compose_dir", "bring_up_db", "migrate")
# The answers a first interactive setup needs once the terms are out of the way.
_SETUP_ANSWERS = {
    wizard._MSG_PASSPHRASE: _UNLOCK_PHRASE,
    wizard._MSG_PASSPHRASE_CONFIRM: _UNLOCK_PHRASE,
    wizard._MSG_FRED_KEY: "",
    wizard._MSG_CONFIGURE_LLM: False,
}


@pytest.fixture
def setup_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Run the wizard against a temporary config file with every side effect stubbed.

    Returns:
        The config path and a record of the Docker and secret-store calls, so a test
        can tell whether setup got past the terms.
    """
    record: dict[str, Any] = {
        "config_path": tmp_path / "config.toml",
        "docker": [],
        "saved_secrets": None,
    }
    monkeypatch.setattr(config_file, "DEFAULT_CONFIG_PATH", record["config_path"])
    monkeypatch.setattr(
        wizard.secret_store, "DEFAULT_SECRETS_PATH", tmp_path / "secrets.enc"
    )
    for step in _DOCKER_STEPS:
        monkeypatch.setattr(
            wizard.docker_bootstrap,
            step,
            lambda step=step: record["docker"].append(step),
        )

    def _no_store(passphrase: str, **kwargs: object) -> dict[str, str]:
        raise wizard.secret_store.SecretStoreNotFoundError("none")

    monkeypatch.setattr(wizard.secret_store, "load_secrets", _no_store)
    monkeypatch.setattr(wizard.secret_store, "store_exists", lambda **kwargs: False)
    monkeypatch.setattr(
        wizard.secret_store,
        "save_secrets",
        lambda secrets, passphrase, **kwargs: record.update(
            saved_secrets=dict(secrets)
        ),
    )
    monkeypatch.delenv("PORTOPT_PASSPHRASE", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    return record


def _store_acknowledgement(record: dict[str, Any]) -> None:
    config_file.save_config(
        {terms.ACKNOWLEDGEMENT_CONFIG_KEY: True}, path=record["config_path"]
    )


@pytest.mark.criterion("REQ-ING-026")
def test_interactive_setup_aborts_when_yahoo_terms_declined(
    setup_run: dict[str, Any],
) -> None:
    """Validates wizard.run_setup_interactive: a No to the terms stops setup first.

    The prompter knows only the terms question, so reaching the passphrase prompt
    would fail the test with a missing-answer error instead.
    """
    prompter = NonInteractivePrompter({terms.MSG_ACCEPT_YAHOO_TERMS: False})

    with pytest.raises(terms.YahooTermsNotAcceptedError) as declined:
        wizard.run_setup_interactive(prompter, skip_path_install=True)

    assert str(declined.value) == terms.DECLINED_MESSAGE
    assert setup_run["docker"] == []
    assert setup_run["saved_secrets"] is None
    assert not setup_run["config_path"].exists()


@pytest.mark.criterion("REQ-ING-026")
def test_terms_prompt_shows_the_terms_and_defaults_to_no() -> None:
    """Validates terms.confirm_yahoo_terms: the question carries the text, default No."""
    prompter = MagicMock()
    prompter.confirm.return_value = False

    accepted = terms.confirm_yahoo_terms(prompter, accepted_flag=False)

    assert accepted is False
    prompter.confirm.assert_called_once_with(
        terms.MSG_ACCEPT_YAHOO_TERMS, default=False
    )
    assert terms.YAHOO_TERMS_TEXT in terms.MSG_ACCEPT_YAHOO_TERMS
    assert terms.YAHOO_TERMS_URL in terms.YAHOO_TERMS_TEXT


@pytest.mark.criterion("REQ-ING-026")
def test_interactive_setup_stores_the_acknowledgement_when_accepted(
    setup_run: dict[str, Any],
) -> None:
    """Validates wizard.run_setup_interactive: a Yes is persisted to config.toml."""
    prompter = NonInteractivePrompter(
        {terms.MSG_ACCEPT_YAHOO_TERMS: True, **_SETUP_ANSWERS}
    )

    wizard.run_setup_interactive(prompter, skip_path_install=True, no_launch=True)

    stored = config_file.load_config(path=setup_run["config_path"])
    assert stored == {terms.ACKNOWLEDGEMENT_CONFIG_KEY: True}


@pytest.mark.criterion("REQ-ING-026")
def test_noninteractive_setup_requires_accept_flag(
    setup_run: dict[str, Any],
) -> None:
    """Validates wizard.run_setup_noninteractive: no flag, no stored answer, no setup.

    The message names both ways to accept, and nothing reaches Docker or disk.
    """
    with pytest.raises(terms.YahooTermsNotAcceptedError) as refused:
        wizard.run_setup_noninteractive(passphrase=_UNLOCK_PHRASE)

    assert terms.ACCEPT_FLAG in str(refused.value)
    assert terms.ACCEPT_ENV_VAR in str(refused.value)
    assert setup_run["docker"] == []
    assert setup_run["saved_secrets"] is None
    assert not setup_run["config_path"].exists()


@pytest.mark.criterion("REQ-ING-026")
def test_accept_flag_persists_acknowledgement_to_config_toml(
    setup_run: dict[str, Any],
) -> None:
    """Validates wizard.run_setup_noninteractive: the flag stores a TOML ``true``."""
    wizard.run_setup_noninteractive(
        passphrase=_UNLOCK_PHRASE, accept_yahoo_terms=True, skip_path_install=True
    )

    config_path = setup_run["config_path"]
    assert config_file.load_config(path=config_path) == {
        terms.ACKNOWLEDGEMENT_CONFIG_KEY: True
    }
    assert (
        f"{terms.ACKNOWLEDGEMENT_CONFIG_KEY} = true"
        in config_path.read_text(encoding="utf-8").splitlines()
    )


@pytest.mark.criterion("REQ-ING-026")
def test_rerun_with_existing_acknowledgement_does_not_prompt_again(
    setup_run: dict[str, Any],
) -> None:
    """Validates wizard.run_setup_interactive: a stored acknowledgement is not re-asked.

    The prompter has no answer for the terms question, so asking it would raise.
    """
    _store_acknowledgement(setup_run)

    wizard.run_setup_interactive(
        NonInteractivePrompter(_SETUP_ANSWERS), skip_path_install=True, no_launch=True
    )

    stored = config_file.load_config(path=setup_run["config_path"])
    assert stored[terms.ACKNOWLEDGEMENT_CONFIG_KEY] is True


@pytest.mark.criterion("REQ-ING-026")
def test_noninteractive_rerun_with_existing_acknowledgement_needs_no_flag(
    setup_run: dict[str, Any],
) -> None:
    """Validates wizard.run_setup_noninteractive: a stored acknowledgement suffices."""
    _store_acknowledgement(setup_run)

    wizard.run_setup_noninteractive(passphrase=_UNLOCK_PHRASE, skip_path_install=True)

    assert setup_run["docker"] == list(_DOCKER_STEPS)
    stored = config_file.load_config(path=setup_run["config_path"])
    assert stored[terms.ACKNOWLEDGEMENT_CONFIG_KEY] is True


@pytest.mark.criterion("REQ-ING-026")
@pytest.mark.parametrize("stored_value", ["true", "yes", 1, False])
def test_only_a_stored_boolean_true_counts_as_acknowledged(
    stored_value: object,
) -> None:
    """Validates terms.is_acknowledged: a hand-edited look-alike is not an answer."""
    assert not terms.is_acknowledged({terms.ACKNOWLEDGEMENT_CONFIG_KEY: stored_value})
    assert terms.is_acknowledged({terms.ACKNOWLEDGEMENT_CONFIG_KEY: True})


@pytest.mark.criterion("REQ-ING-028")
def test_acknowledgement_maps_to_scheduler_env_key() -> None:
    """Validates compose_env.config_to_scheduler_env: only a stored true is passed on.

    An absent answer passes nothing, so a value the operator exported for a plain
    ``docker compose up`` is not overridden; the fund's env file never carries it.
    """
    acknowledged = {terms.ACKNOWLEDGEMENT_CONFIG_KEY: True}
    declined = {terms.ACKNOWLEDGEMENT_CONFIG_KEY: False}

    assert compose_env.config_to_scheduler_env(acknowledged) == {
        terms.ACKNOWLEDGEMENT_ENV_VAR: "true"
    }
    assert compose_env.config_to_scheduler_env({}) == {}
    assert compose_env.config_to_scheduler_env(declined) == {}
    assert terms.ACKNOWLEDGEMENT_ENV_VAR not in compose_env.config_to_env(acknowledged)


def _interpolate(text: str, environment: dict[str, str]) -> str:
    """Resolve compose ``${VAR}`` and ``${VAR:-default}`` as ``docker compose`` does."""
    return _COMPOSE_VARIABLE.sub(
        lambda found: environment.get(found.group(1)) or (found.group(2) or ""), text
    )


def _parsed_flag(raw: str, monkeypatch: pytest.MonkeyPatch) -> bool:
    monkeypatch.setenv(terms.ACKNOWLEDGEMENT_ENV_VAR, raw)
    return Settings(_env_file=None).yahoo_terms_acknowledged


@pytest.mark.criterion("REQ-ING-028")
def test_compose_passes_the_acknowledgement_to_the_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validates docker-compose.yml: the scheduler gets the value compose is given.

    ``portopt start`` and CI put it in the compose environment; without it the
    container receives a value the daemon reads as not acknowledged.
    """
    services = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]
    passed = services["scheduler"]["environment"][terms.ACKNOWLEDGEMENT_ENV_VAR]

    given = _interpolate(passed, {terms.ACKNOWLEDGEMENT_ENV_VAR: "true"})
    unset = _interpolate(passed, {})

    assert given == "true"
    assert _parsed_flag(unset, monkeypatch) is False


@pytest.mark.criterion("REQ-ING-027")
def test_settings_default_is_unacknowledged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates config.Settings.yahoo_terms_acknowledged: off unless the env says so."""
    monkeypatch.delenv(terms.ACKNOWLEDGEMENT_ENV_VAR, raising=False)

    assert Settings(_env_file=None).yahoo_terms_acknowledged is False
    assert _parsed_flag("true", monkeypatch) is True


@contextmanager
def _booting_daemon(*, acknowledged: bool) -> Iterator[dict[str, MagicMock]]:
    """Patch every side effect of ``worker.main`` and set the acknowledgement.

    ``_configure_logging`` runs for real, since it is the start-up step that checks
    the terms; the root logger keeps pytest's handlers, so ``basicConfig`` adds none.
    """
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(patch.object(worker.settings, "yahoo_terms_acknowledged", acknowledged))
        enter(patch.object(worker, "_install_signal_handlers"))
        enter(patch.object(worker, "_reconcile_orphans"))
        enter(patch.object(worker, "close_db"))
        enter(patch.object(worker.database_manager, "health_check", return_value=True))
        enter(
            patch.object(
                worker, "_wait_for_shutdown_or_scheduler_death", return_value=False
            )
        )
        yield {
            "configure_yfinance": enter(
                patch("app.services.market_data.yfinance.configure_yfinance")
            ),
            "serve": enter(patch.object(worker, "start_http_server")),
            "init_db": enter(patch.object(worker, "init_db")),
            "create_scheduler": enter(patch.object(worker, "create_scheduler")),
        }


@pytest.mark.criterion("REQ-ING-027")
def test_daemon_refuses_to_start_without_acknowledgement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validates worker.main via terms.require_yahoo_terms: exit 2, before any work.

    Nothing is served, opened or scheduled, and the log tells the operator to run
    ``portopt setup``.
    """
    caplog.set_level(logging.CRITICAL, logger=terms.__name__)

    with (
        _booting_daemon(acknowledged=False) as calls,
        pytest.raises(SystemExit) as refused,
    ):
        worker.main()

    assert refused.value.code == _DAEMON_REFUSAL_EXIT_CODE
    assert "portopt setup" in caplog.text
    for call in calls.values():
        call.assert_not_called()


@pytest.mark.criterion("REQ-ING-027")
def test_daemon_starts_when_acknowledged() -> None:
    """Validates worker.main via terms.require_yahoo_terms: an acknowledged boot runs."""
    with _booting_daemon(acknowledged=True) as calls:
        worker.main()

    calls["configure_yfinance"].assert_called_once()
    calls["init_db"].assert_called_once()
    calls["create_scheduler"].return_value.start.assert_called_once()
