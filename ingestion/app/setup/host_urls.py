"""Translate a user-supplied base URL between the host's and the container's view.

The wizard probes a local/self-hosted LLM endpoint from the host process, but the
same URL is later read by the fund container, where ``localhost`` is the container
itself. The compose file maps ``host.docker.internal`` to the host gateway for the
fund services, so that alias is the one name that reaches the host from inside the
container; on the host side it generally does not resolve (Linux, Colima), so the
probe goes through loopback instead.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit, urlunsplit

CONTAINER_HOST_ALIAS = "host.docker.internal"
_LOOPBACK_NAME = "localhost"


def _is_loopback(host: str) -> bool:
    if host == _LOOPBACK_NAME:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _with_host(url: str, new_host: str) -> str:
    parts = urlsplit(url)
    userinfo, at, _ = parts.netloc.rpartition("@")
    port = f":{parts.port}" if parts.port is not None else ""
    return urlunsplit(parts._replace(netloc=f"{userinfo}{at}{new_host}{port}"))


def to_container_url(url: str) -> str:
    """Rewrite a loopback host to the compose host alias for the fund container.

    Args:
        url: Base URL as the user typed it (and as validated from the host).

    Returns:
        ``url`` with a ``localhost``/``127.x.x.x``/``::1`` host replaced by
        ``host.docker.internal``; scheme, credentials, port, path and query are kept.
        Any other host, or a string that does not parse as a URL, is returned
        unchanged.
    """
    try:
        host = urlsplit(url).hostname
        if host is None or not _is_loopback(host.lower()):
            return url
        return _with_host(url, CONTAINER_HOST_ALIAS)
    except ValueError:
        return url


def to_host_probe_url(url: str) -> str:
    """Rewrite the container host alias to loopback for a probe run on the host.

    Args:
        url: Base URL as the user typed it, possibly using the container alias.

    Returns:
        ``url`` with a ``host.docker.internal`` host replaced by ``localhost``;
        everything else is kept. Any other host, or a string that does not parse as
        a URL, is returned unchanged.
    """
    try:
        host = urlsplit(url).hostname
        if host is None or host.lower() != CONTAINER_HOST_ALIAS:
            return url
        return _with_host(url, _LOOPBACK_NAME)
    except ValueError:
        return url
