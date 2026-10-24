"""Contract: base-URL translation between the host's and the container's view."""

from __future__ import annotations

import pytest

from app.setup.host_urls import to_container_url, to_host_probe_url


@pytest.mark.parametrize(
    "url", ["http://localhost:bad", "not a url", "localhost:11434", "eu-west-1", ""]
)
def test_unparseable_or_hostless_input_is_returned_unchanged(url: str) -> None:
    """A region name or a malformed URL must never be mangled or raise."""
    assert to_container_url(url) == url
    assert to_host_probe_url(url) == url


def test_probe_url_keeps_credentials_port_and_path() -> None:
    assert (
        to_host_probe_url("https://u:p@Host.Docker.Internal:8443/v1/x?y=1")
        == "https://u:p@localhost:8443/v1/x?y=1"
    )


def test_probe_url_leaves_a_real_loopback_host_alone() -> None:
    assert to_host_probe_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
