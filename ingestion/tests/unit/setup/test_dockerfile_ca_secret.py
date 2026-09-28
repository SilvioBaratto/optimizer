"""Build-time CA-secret parity across the two service Dockerfiles (task T9).

Both images resolve their Python deps with `uv sync` over HTTPS to PyPI, which
fails behind a TLS-inspecting proxy when the slim image lacks the corporate root.
Each Dockerfile mounts an OPTIONAL BuildKit secret (`id=ca_bundle`) on that step
and exports SSL_CERT_FILE from it only when the mount is non-empty, so a build
that passes the secret trusts the proxy while a build without it falls back to
certifi. The fund image additionally sets TERM for the Textual TUI's colors.
"""

from __future__ import annotations

import re
from pathlib import Path

# unit -> setup -> tests -> ingestion -> repo root (both Dockerfiles live here).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_FUND_DOCKERFILE = _REPO_ROOT / "fund" / "Dockerfile"
_INGESTION_DOCKERFILE = _REPO_ROOT / "ingestion" / "Dockerfile"


def _uv_sync_line(dockerfile: Path) -> str:
    """Return the `uv sync` RUN instruction with line-continuations collapsed.

    Newlines are normalised and backslash-continuations joined so a multi-line
    RUN reads as one logical line, letting a caller assert the secret mount sits
    on the same instruction as `uv sync` (not some unrelated step).
    """
    text = dockerfile.read_text(encoding="utf-8").replace("\r\n", "\n")
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if "uv sync" in line:
            return line
    return ""


def test_fund_uv_sync_mounts_optional_ca_secret() -> None:
    """The fund image's dep-resolution step honours the optional ca_bundle secret."""
    line = _uv_sync_line(_FUND_DOCKERFILE)
    assert "--mount=type=secret,id=ca_bundle" in line
    assert "/run/secrets/ca_bundle" in line
    assert "SSL_CERT_FILE" in line


def test_ingestion_uv_sync_mounts_optional_ca_secret() -> None:
    """The ingestion image mirrors the same build-time CA-secret handling."""
    line = _uv_sync_line(_INGESTION_DOCKERFILE)
    assert "--mount=type=secret,id=ca_bundle" in line
    assert "/run/secrets/ca_bundle" in line
    assert "SSL_CERT_FILE" in line


def test_fund_image_sets_term_for_textual() -> None:
    """The fund runtime image sets TERM so the Textual cockpit renders in colour."""
    text = _FUND_DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"ENV\s+TERM=xterm-256color", text)
