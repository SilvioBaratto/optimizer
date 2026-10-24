"""Stand-ins for the exceptions curl_cffi raises when a request fails.

yfinance 1.x reaches Yahoo through curl_cffi, whose errors carry their own
class names and libcurl messages rather than the ``requests`` ones. The real
classes are rebuilt by name instead of imported because curl_cffi's native
extension fails to load on some hosts that run the unit suite.
"""

from __future__ import annotations

_DETAILS = "See https://curl.se/libcurl/c/libcurl-errors.html first for more details."


def curl_error(name: str, message: str) -> OSError:
    """Build an ``OSError`` subclass instance named like a curl_cffi exception.

    Args:
        name: Class name curl_cffi uses for the failure (``Timeout``).
        message: libcurl text, as curl_cffi formats it after ``Failed to perform``.

    Returns:
        An exception whose ``type(...).__name__`` and text match the real one.
    """
    return type(name, (OSError,), {})(f"Failed to perform, curl: {message}. {_DETAILS}")


def curl_timeout() -> OSError:
    """Read timeout (libcurl error 28), raised as ``curl_cffi`` ``Timeout``."""
    return curl_error(
        "Timeout",
        "(28) Operation timed out after 30002 milliseconds with 0 bytes received",
    )


def curl_dns_failure() -> OSError:
    """Unresolvable host (libcurl error 6), raised as ``DNSError``."""
    return curl_error(
        "DNSError", "(6) Could not resolve host: query2.finance.yahoo.com"
    )


def curl_connect_failure() -> OSError:
    """Refused or unreachable server (libcurl error 7), raised as ``ConnectionError``."""
    return curl_error(
        "ConnectionError",
        "(7) Failed to connect to query2.finance.yahoo.com port 443 after 8 ms: "
        "Couldn't connect to server",
    )


def curl_http_404() -> OSError:
    """HTTP 404 answer (libcurl error 22), raised as ``HTTPError``."""
    return curl_error("HTTPError", "(22) The requested URL returned error: 404")
