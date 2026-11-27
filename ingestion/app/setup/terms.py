"""Yahoo's terms: the acknowledgement ``portopt setup`` asks for and the daemon needs.

Market data comes through yfinance, an unofficial client of Yahoo Finance, and
Yahoo's terms restrict automated access and commercial use. Setup shows the terms
and stores the operator's acknowledgement in ``~/.portopt/config.toml``;
``portopt start`` hands it to the scheduler container as ``YAHOO_TERMS_ACKNOWLEDGED``
(``compose_env.config_to_scheduler_env``), and the long-running ingestion daemon
refuses to start without it. Manual one-shot CLI commands do not check it.

The daemon imports this module at start-up, so it imports nothing beyond the
standard library at run time.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings
    from app.setup.prompts import Prompter

logger = logging.getLogger(__name__)

YAHOO_TERMS_URL = "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html"
YAHOO_TERMS_TEXT = (
    "Yahoo Finance data is fetched with the unofficial yfinance library. Yahoo's "
    f"terms ({YAHOO_TERMS_URL}) restrict automated access and commercial use, and "
    "yfinance is intended for personal research and education. By continuing you "
    "confirm that you have read them, that you will use this software only for "
    "personal, non-commercial research, and that you, not the project's author, are "
    "responsible for complying with them. This project is not affiliated with Yahoo."
)
# The Prompter seam has no plain "show text" call, so the terms ride in the question.
MSG_ACCEPT_YAHOO_TERMS = f"{YAHOO_TERMS_TEXT}\n\nDo you accept Yahoo's terms?"

ACKNOWLEDGEMENT_CONFIG_KEY = "yahoo_terms_acknowledged"
ACKNOWLEDGEMENT_ENV_VAR = "YAHOO_TERMS_ACKNOWLEDGED"
ACCEPT_FLAG = "--accept-yahoo-terms"
ACCEPT_ENV_VAR = "PORTOPT_ACCEPT_YAHOO_TERMS"
# 2, the usual status of a configuration error, keeps the refusal apart from a
# crashed daemon (1) in `docker inspect` and in exit-code alerting.
REFUSAL_EXIT_CODE = 2

DECLINED_MESSAGE = (
    "Yahoo's terms were not accepted, so setup stopped and saved nothing. The "
    "ingestion daemon cannot run without them."
)
FLAG_REQUIRED_MESSAGE = (
    f"A first non-interactive setup must accept Yahoo's terms ({YAHOO_TERMS_URL}): "
    f"once you have read them, pass {ACCEPT_FLAG} or set {ACCEPT_ENV_VAR}=1."
)
DAEMON_REFUSAL_MESSAGE = (
    f"Refusing to start: Yahoo's terms ({YAHOO_TERMS_URL}) have not been "
    "acknowledged. Run `portopt setup` to read and accept them, then "
    f"`portopt start`; a daemon started by hand needs {ACKNOWLEDGEMENT_ENV_VAR}=true."
)


class YahooTermsNotAcceptedError(RuntimeError):
    """Raised when setup cannot record an acknowledgement of Yahoo's terms."""


def is_acknowledged(stored: Mapping[str, object]) -> bool:
    """Return whether a persisted config already records the acknowledgement.

    Only a TOML ``true`` counts; a hand-edited ``"yes"`` or ``1`` is not one.
    """
    return stored.get(ACKNOWLEDGEMENT_CONFIG_KEY) is True


def confirm_yahoo_terms(prompter: Prompter, *, accepted_flag: bool) -> bool:
    """Show Yahoo's terms and return whether the operator accepts them.

    Args:
        prompter: Shows the terms in its question; the answer defaults to No.
        accepted_flag: The operator already accepted with ``ACCEPT_FLAG``, so
            nothing is asked.

    Returns:
        True when the operator accepts, by flag or by answering yes.
    """
    if accepted_flag:
        return True
    return prompter.confirm(MSG_ACCEPT_YAHOO_TERMS, default=False)


def acknowledge(
    prompter: Prompter | None,
    stored: Mapping[str, object],
    *,
    accepted_flag: bool,
) -> dict[str, object]:
    """Return the config entry a setup run must persist for Yahoo's terms.

    Args:
        prompter: Asks the operator. ``None`` for a non-interactive setup, which can
            accept only through ``accepted_flag``.
        stored: The config an earlier setup persisted. One that already holds the
            acknowledgement means nothing is asked again.
        accepted_flag: The operator passed ``ACCEPT_FLAG`` (or ``ACCEPT_ENV_VAR``).

    Returns:
        A new dict the wizard stages the rest of its config into: the
        acknowledgement when it was given now, empty when it was already stored.

    Raises:
        YahooTermsNotAcceptedError: The operator declined, or a non-interactive
            setup has neither the flag nor a stored acknowledgement.
    """
    if is_acknowledged(stored):
        return {}
    if prompter is None:
        if not accepted_flag:
            raise YahooTermsNotAcceptedError(FLAG_REQUIRED_MESSAGE)
    elif not confirm_yahoo_terms(prompter, accepted_flag=accepted_flag):
        raise YahooTermsNotAcceptedError(DECLINED_MESSAGE)
    return {ACKNOWLEDGEMENT_CONFIG_KEY: True}


def require_yahoo_terms(settings: Settings) -> None:
    """Stop the daemon unless the operator acknowledged Yahoo's terms.

    Args:
        settings: The daemon settings; ``yahoo_terms_acknowledged`` comes from
            ``ACKNOWLEDGEMENT_ENV_VAR``.

    Raises:
        SystemExit: With ``REFUSAL_EXIT_CODE``, after logging
            ``DAEMON_REFUSAL_MESSAGE`` at critical level.
    """
    if settings.yahoo_terms_acknowledged:
        return
    logger.critical(DAEMON_REFUSAL_MESSAGE)
    raise SystemExit(REFUSAL_EXIT_CODE)
