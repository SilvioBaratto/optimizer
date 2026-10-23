"""Keep credentials that travel in URL query strings out of logs and errors."""

from __future__ import annotations

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
