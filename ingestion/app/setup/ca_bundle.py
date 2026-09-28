"""Generate a CA bundle that merges certifi's roots with the machine's own roots.

Behind a TLS-inspecting proxy (e.g. Zscaler) the corporate root CA that re-signs
outbound certificates is trusted by the OS trust store but absent from certifi's
bundle, so Python HTTPS (requests/httpx) and `uv sync` fail to verify. `generate`
writes ``.certs/ca-bundle.pem`` = certifi's roots (seeded first) followed by the
OS/corporate roots, which callers point ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE``
at. Roots are collected per-OS: PowerShell over ``Cert:\\LocalMachine\\Root`` on
Windows, ``security find-certificate`` on macOS, the system bundle on Linux.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import certifi

DEFAULT_BUNDLE_PATH = Path(".certs/ca-bundle.pem")

_PEM_BEGIN = "-----BEGIN CERTIFICATE-----"
_PEM_END = "-----END CERTIFICATE-----"

# Distro-varying system trust bundles, most-common first. On Linux the corporate
# root is installed into the system store (update-ca-certificates), so the merged
# bundle already carries it.
_LINUX_BUNDLE_CANDIDATES = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/etc/ssl/cert.pem",
)

# macOS keeps admin/MDM-added roots in the System keychain and the built-in roots
# in SystemRootCertificates; a corporate root lands in the former.
_MACOS_KEYCHAINS = (
    "/Library/Keychains/System.keychain",
    "/System/Library/Keychains/SystemRootCertificates.keychain",
)


class CABundleError(RuntimeError):
    """Raised when the OS/corporate root certificates cannot be collected."""


def _run_capture(cmd: list[str], source: str) -> str:
    """Run a static trust-store export command and return its stdout.

    Args:
        cmd: The argv to execute (never shell-interpreted).
        source: Human-readable name of the store, used in error messages.

    Returns:
        The command's stdout (expected to be PEM text).

    Raises:
        CABundleError: If the executable is missing or the command exits non-zero.
    """
    try:
        result = subprocess.run(  # noqa: S603 - static, trusted argv; never shell=True
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise CABundleError(f"cannot read {source}: {cmd[0]} not found") from exc
    if result.returncode != 0:
        raise CABundleError(f"cannot read {source}: {result.stderr.strip()}")
    return result.stdout


def _certifi_roots() -> str:
    """Return the certifi CA bundle as PEM text."""
    return Path(certifi.where()).read_text(encoding="utf-8")


def _windows_roots() -> str:
    """Export the Windows ``LocalMachine\\Root`` store as PEM via PowerShell.

    Reads the store through the .NET ``X509Store`` API rather than the ``Cert:``
    PSDrive: the drive's provider (``Microsoft.PowerShell.Security``) is not
    autoloaded when PowerShell is launched from a stripped environment (e.g. Git
    Bash with an empty ``PSModulePath``), whereas the .NET type is always present.
    """
    script = (
        "$ErrorActionPreference='Stop'; "
        "$store = New-Object System.Security.Cryptography.X509Certificates.X509Store "
        "-ArgumentList 'Root','LocalMachine'; "
        "$store.Open('ReadOnly'); "
        "foreach ($c in $store.Certificates) { "
        "'-----BEGIN CERTIFICATE-----'; "
        "[System.Convert]::ToBase64String($c.RawData, 'InsertLineBreaks'); "
        "'-----END CERTIFICATE-----' }; "
        "$store.Close()"
    )
    return _run_capture(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        "Windows LocalMachine\\Root store",
    )


def _macos_roots() -> str:
    """Export the macOS System and SystemRoot keychains as PEM via ``security``."""
    parts = [
        _run_capture(
            ["security", "find-certificate", "-a", "-p", keychain],
            f"macOS keychain {keychain}",
        )
        for keychain in _MACOS_KEYCHAINS
    ]
    return "\n".join(parts)


def _linux_roots() -> str:
    """Return the first present Linux system trust bundle as PEM text.

    Raises:
        CABundleError: If none of the well-known bundle paths exist.
    """
    for candidate in _LINUX_BUNDLE_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise CABundleError("no system CA bundle found on this Linux host")


def _os_roots() -> str:
    """Collect the machine's trust-store roots for the current platform."""
    if sys.platform == "win32":
        return _windows_roots()
    if sys.platform == "darwin":
        return _macos_roots()
    return _linux_roots()


def _pem_blocks(text: str) -> list[str]:
    """Extract complete ``BEGIN/END CERTIFICATE`` blocks from PEM text.

    Lines outside a certificate block (store metadata, blank lines) are dropped, so
    the output is safe to concatenate regardless of the source tool's framing.
    """
    blocks: list[str] = []
    current: list[str] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == _PEM_BEGIN:
            inside = True
            current = [stripped]
        elif stripped == _PEM_END and inside:
            current.append(stripped)
            blocks.append("\n".join(current))
            inside = False
        elif inside:
            current.append(stripped)
    return blocks


def generate(*, path: Path | None = None) -> Path:
    """Write the merged certifi ⊕ OS/corporate CA bundle and return its path.

    certifi's roots are written first, then the OS roots; duplicate certificates
    are dropped while preserving that order, so the result is a clean union and
    re-running yields byte-identical output (idempotent).

    Args:
        path: Destination bundle path; defaults to ``.certs/ca-bundle.pem``.

    Returns:
        The path the bundle was written to.

    Raises:
        CABundleError: If no certificates could be collected (e.g. the OS export
            failed), leaving no trustworthy bundle to write.
    """
    target = path or DEFAULT_BUNDLE_PATH
    blocks = _pem_blocks(_certifi_roots()) + _pem_blocks(_os_roots())

    seen: set[str] = set()
    unique: list[str] = []
    for block in blocks:
        key = "".join(block.split())
        if key not in seen:
            seen.add(key)
            unique.append(block)

    if not unique:
        raise CABundleError("no certificates collected for the CA bundle")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(unique) + "\n", encoding="utf-8")
    return target
