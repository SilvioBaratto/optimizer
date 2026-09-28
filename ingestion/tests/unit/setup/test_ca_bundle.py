"""Contract: merge certifi roots with the machine trust store into one PEM bundle.

`generate` seeds certifi's roots first, appends the OS/corporate roots, de-dupes,
and writes ``.certs/ca-bundle.pem``; `_os_roots` dispatches to the right per-OS
export. OS calls are stubbed so the suite never touches a real trust store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.setup import ca_bundle
from app.setup.ca_bundle import _PEM_BEGIN, CABundleError

_CERTIFI_A = "-----BEGIN CERTIFICATE-----\nCERTIFI_A_BODY\n-----END CERTIFICATE-----"
_CERTIFI_B = "-----BEGIN CERTIFICATE-----\nCERTIFI_B_BODY\n-----END CERTIFICATE-----"
_CORP = "-----BEGIN CERTIFICATE-----\nCORP_ROOT_BODY\n-----END CERTIFICATE-----"


def _stub_roots(monkeypatch: pytest.MonkeyPatch, certifi_pem: str, os_pem: str) -> None:
    monkeypatch.setattr(ca_bundle, "_certifi_roots", lambda: certifi_pem)
    monkeypatch.setattr(ca_bundle, "_os_roots", lambda: os_pem)


def test_bundle_is_certifi_seeded_superset_of_os_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merged bundle carries every certifi root AND the corporate root, with
    certifi seeded first."""
    _stub_roots(monkeypatch, f"{_CERTIFI_A}\n{_CERTIFI_B}\n", f"{_CORP}\n")
    text = ca_bundle.generate(path=tmp_path / "ca.pem").read_text(encoding="utf-8")
    assert "CERTIFI_A_BODY" in text
    assert "CERTIFI_B_BODY" in text
    assert "CORP_ROOT_BODY" in text
    assert text.count(_PEM_BEGIN) == 3
    assert text.index("CERTIFI_A_BODY") < text.index("CORP_ROOT_BODY")


def test_generate_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running against a stable trust store yields byte-identical output."""
    _stub_roots(monkeypatch, f"{_CERTIFI_A}\n", f"{_CORP}\n")
    path = tmp_path / "ca.pem"
    first = ca_bundle.generate(path=path).read_text(encoding="utf-8")
    second = ca_bundle.generate(path=path).read_text(encoding="utf-8")
    assert first == second


def test_generate_dedupes_roots_present_in_both_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root that appears in both certifi and the OS store is written once."""
    _stub_roots(monkeypatch, f"{_CERTIFI_A}\n", f"{_CERTIFI_A}\n{_CORP}\n")
    text = ca_bundle.generate(path=tmp_path / "ca.pem").read_text(encoding="utf-8")
    assert text.count("CERTIFI_A_BODY") == 1
    assert text.count(_PEM_BEGIN) == 2


def test_generate_creates_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `.certs/` parent is created on demand."""
    _stub_roots(monkeypatch, f"{_CERTIFI_A}\n", f"{_CORP}\n")
    out = ca_bundle.generate(path=tmp_path / "nested" / "ca.pem")
    assert out.is_file()


def test_generate_raises_when_no_certificates_collected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty result must fail loudly rather than write an empty trust bundle."""
    _stub_roots(monkeypatch, "", "")
    with pytest.raises(CABundleError):
        ca_bundle.generate(path=tmp_path / "ca.pem")


def test_os_roots_uses_windows_branch_on_win32(monkeypatch: pytest.MonkeyPatch) -> None:
    """win32 collects roots from the Windows store."""
    monkeypatch.setattr(ca_bundle.sys, "platform", "win32")
    monkeypatch.setattr(ca_bundle, "_windows_roots", lambda: "WIN")
    monkeypatch.setattr(ca_bundle, "_macos_roots", lambda: "MAC")
    monkeypatch.setattr(ca_bundle, "_linux_roots", lambda: "LNX")
    assert ca_bundle._os_roots() == "WIN"


def test_os_roots_uses_macos_branch_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """darwin collects roots from the macOS keychains."""
    monkeypatch.setattr(ca_bundle.sys, "platform", "darwin")
    monkeypatch.setattr(ca_bundle, "_windows_roots", lambda: "WIN")
    monkeypatch.setattr(ca_bundle, "_macos_roots", lambda: "MAC")
    monkeypatch.setattr(ca_bundle, "_linux_roots", lambda: "LNX")
    assert ca_bundle._os_roots() == "MAC"


def test_os_roots_falls_back_to_linux_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any non-Windows/non-macOS platform reads the Linux system bundle."""
    monkeypatch.setattr(ca_bundle.sys, "platform", "linux")
    monkeypatch.setattr(ca_bundle, "_windows_roots", lambda: "WIN")
    monkeypatch.setattr(ca_bundle, "_macos_roots", lambda: "MAC")
    monkeypatch.setattr(ca_bundle, "_linux_roots", lambda: "LNX")
    assert ca_bundle._os_roots() == "LNX"


def test_windows_roots_queries_localmachine_root_via_powershell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Windows export shells PowerShell into the LocalMachine `Root` X509 store
    (via .NET, not the Cert: PSDrive that a stripped env fails to autoload)."""
    captured: dict[str, list[str]] = {}

    def fake_capture(cmd: list[str], _source: str) -> str:
        captured["cmd"] = cmd
        return _CORP

    monkeypatch.setattr(ca_bundle, "_run_capture", fake_capture)
    assert ca_bundle._windows_roots() == _CORP
    assert "powershell" in captured["cmd"][0]
    joined = " ".join(captured["cmd"])
    assert "X509Store" in joined
    assert "'Root','LocalMachine'" in joined


def test_macos_roots_reads_both_keychains_with_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The macOS export runs `security find-certificate -a -p` per keychain."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        ca_bundle, "_run_capture", lambda cmd, source: calls.append(cmd) or ""
    )
    ca_bundle._macos_roots()
    assert len(calls) == len(ca_bundle._MACOS_KEYCHAINS)
    for cmd in calls:
        assert cmd[0] == "security"
        assert "find-certificate" in cmd


def test_linux_roots_reads_first_present_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first existing well-known bundle path wins."""
    present = tmp_path / "ca-certificates.crt"
    present.write_text("LINUX_BUNDLE", encoding="utf-8")
    monkeypatch.setattr(
        ca_bundle,
        "_LINUX_BUNDLE_CANDIDATES",
        (str(tmp_path / "missing"), str(present)),
    )
    assert ca_bundle._linux_roots() == "LINUX_BUNDLE"


def test_linux_roots_raises_when_no_bundle_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No system bundle ⇒ a clear failure rather than a silent empty result."""
    monkeypatch.setattr(
        ca_bundle, "_LINUX_BUNDLE_CANDIDATES", (str(tmp_path / "nope"),)
    )
    with pytest.raises(CABundleError):
        ca_bundle._linux_roots()


def test_run_capture_returns_stdout_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-exit export command yields its stdout (the PEM text)."""

    class _Result:
        returncode = 0
        stderr = ""
        stdout = "PEM"

    monkeypatch.setattr(ca_bundle.subprocess, "run", lambda *_a, **_k: _Result())
    assert ca_bundle._run_capture(["whatever"], "the store") == "PEM"


def test_run_capture_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing export command surfaces as CABundleError."""

    class _Result:
        returncode = 1
        stderr = "boom"
        stdout = ""

    monkeypatch.setattr(ca_bundle.subprocess, "run", lambda *_a, **_k: _Result())
    with pytest.raises(CABundleError):
        ca_bundle._run_capture(["whatever"], "the store")


def test_run_capture_raises_when_executable_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing export tool (no PowerShell/security) surfaces as CABundleError."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(ca_bundle.subprocess, "run", _boom)
    with pytest.raises(CABundleError):
        ca_bundle._run_capture(["whatever"], "the store")


def test_pem_blocks_extracts_blocks_and_drops_metadata() -> None:
    """Store-metadata and blank lines between blocks are discarded."""
    text = f"junk header\n{_CERTIFI_A}\nkeychain: foo\n{_CORP}\n"
    blocks = ca_bundle._pem_blocks(text)
    assert len(blocks) == 2
    assert "CERTIFI_A_BODY" in blocks[0]
    assert "CORP_ROOT_BODY" in blocks[1]


def test_certifi_roots_returns_real_pem() -> None:
    """The certifi seed is real PEM (guards the certifi dependency staying present)."""
    assert _PEM_BEGIN in ca_bundle._certifi_roots()
