"""Keep credentials that travel in URL query strings out of logs and errors."""

from __future__ import annotations

import logging
import re

# `key`/`token` are only treated as credentials inside a query string (`?`/`&`
# prefix): as bare words they are common in unrelated log text such as cache keys.
_QUERY_CREDENTIAL = re.compile(
    r"""(?ix)
    (
        [?&](?:key|token|access_token)
        | \bapi[_-]?key
    )
    =[^&\s"')]+
    """
)


def redact_secrets(text: str) -> str:
    """Mask the value of credential-bearing query parameters in *text*.

    Providers such as FRED accept their API key only as ``?api_key=``, and
    ``requests``/``urllib3`` embed the full URL in connection errors and debug
    records, so the key would otherwise be echoed to the terminal or a log
    shipper.

    Args:
        text: An error message or log line that may contain a URL.

    Returns:
        *text* with each ``api_key``/``key``/``token``/``access_token`` query
        value replaced by ``***``; text without such parameters is unchanged.
    """
    return _QUERY_CREDENTIAL.sub(r"\1=***", text)


class SecretRedactingFilter(logging.Filter):
    """Logging filter that masks credential query parameters in each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite *record* in place so its rendered message carries no key.

        Always returns ``True``: this filter redacts, it never drops records.
        A record whose ``%`` arguments do not match its format string is left
        for the logging machinery to report, as it would be without the filter.
        """
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = None
        return True


def install_secret_redaction() -> None:
    """Attach :class:`SecretRedactingFilter` to every root-logger handler.

    The filter sits on handlers, not loggers, because a logger's filters are
    skipped for records propagated from child loggers such as
    ``urllib3.connectionpool``. Call it after ``logging.basicConfig`` has
    created the handlers; calling it again is harmless.
    """
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(SecretRedactingFilter())
