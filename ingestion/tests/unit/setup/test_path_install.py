"""PATH-install for the `optimizer` launcher (task T12).

POSIX installs a *symlink* (so a `git pull` updates the launcher in place); Windows
writes a ``.cmd`` with the repo path baked in. Both put ``~/.local/bin`` on the User
PATH via ``userpath`` and are idempotent. The branch helpers are exercised directly so
the one suite runs on any OS without patching the global ``os.name`` (which would break
pathlib's WindowsPath/PosixPath dispatch); the dispatch itself is checked with the
filesystem work stubbed. The symlink tests skip where the filesystem grants no symlink
privilege (Windows without Developer Mode).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.setup import path_install


@pytest.fixture
def stub_userpath(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub userpath so no real PATH edit runs; record append() locations."""
    calls: dict = {"appended": []}
    monkeypatch.setattr(path_install.userpath, "in_current_path", lambda loc: False)
    monkeypatch.setattr(path_install.userpath, "in_new_path", lambda loc: False)
    monkeypatch.setattr(
        path_install.userpath,
        "append",
        lambda loc, app_name=None: bool(calls["appended"].append(loc)) or True,
    )
    return calls


def _symlinks_supported(tmp_path: Path) -> bool:
    probe = tmp_path / "_probe"
    try:
        probe.symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


def _expected_launcher() -> Path:
    return (path_install._repo_root() / "scripts" / "optimizer").resolve()


def test_posix_symlinks_to_repo_launcher(tmp_path: Path) -> None:
    """POSIX install creates a symlink resolving to the repo's scripts/optimizer."""
    if not _symlinks_supported(tmp_path):
        pytest.skip("filesystem has no symlink privilege")
    link = path_install._install_posix(tmp_path, path_install._repo_root())
    assert link.is_symlink()
    assert link.resolve() == _expected_launcher()


def test_posix_repoints_stale_symlink(tmp_path: Path) -> None:
    """A stale optimizer symlink is repointed at the repo launcher, not errored on."""
    if not _symlinks_supported(tmp_path):
        pytest.skip("filesystem has no symlink privilege")
    (tmp_path / "optimizer").symlink_to(tmp_path / "gone")
    link = path_install._install_posix(tmp_path, path_install._repo_root())
    assert link.is_symlink()
    assert link.resolve() == _expected_launcher()


def test_posix_idempotent_second_run(tmp_path: Path) -> None:
    """Running the POSIX install twice succeeds and leaves the symlink intact."""
    if not _symlinks_supported(tmp_path):
        pytest.skip("filesystem has no symlink privilege")
    path_install._install_posix(tmp_path, path_install._repo_root())
    link = path_install._install_posix(tmp_path, path_install._repo_root())
    assert link.resolve() == _expected_launcher()


def test_windows_writes_cmd_with_embedded_repo(tmp_path: Path) -> None:
    """Windows install writes optimizer.cmd with the repo path baked in (no %~dp0)."""
    launcher = path_install._install_windows(tmp_path, path_install._repo_root())
    assert launcher.name == "optimizer.cmd"
    text = launcher.read_text(encoding="utf-8")
    assert str(path_install._repo_root()) in text
    assert "%~dp0.." not in text


@pytest.mark.parametrize(
    ("os_name", "expected_leaf"),
    [("nt", "optimizer.cmd"), ("posix", "optimizer")],
)
def test_install_launcher_routes_by_os_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_userpath: dict,
    os_name: str,
    expected_leaf: str,
) -> None:
    """install_launcher dispatches to the Windows/.cmd or POSIX/symlink branch."""
    monkeypatch.setattr(path_install, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        path_install, "_install_windows", lambda b, r: b / "optimizer.cmd"
    )
    monkeypatch.setattr(path_install, "_install_posix", lambda b, r: b / "optimizer")
    monkeypatch.setattr(path_install.os, "name", os_name)
    installed = path_install.install_launcher(bin_dir=tmp_path)
    assert installed.name == expected_leaf
    assert stub_userpath["appended"] == [str(tmp_path)]


def test_ensure_on_path_appends_when_absent(
    tmp_path: Path, stub_userpath: dict
) -> None:
    """userpath.append runs with the bin dir when it is not already on PATH."""
    path_install._ensure_on_path(tmp_path)
    assert stub_userpath["appended"] == [str(tmp_path)]


def test_ensure_on_path_skips_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_userpath: dict
) -> None:
    """An already-resolvable bin dir short-circuits without touching PATH."""
    monkeypatch.setattr(path_install.userpath, "in_current_path", lambda loc: True)
    path_install._ensure_on_path(tmp_path)
    assert stub_userpath["appended"] == []


def test_install_launcher_wraps_oserror_as_pathinstallerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_userpath: dict
) -> None:
    """A filesystem/PATH failure surfaces as PathInstallError, not a raw OSError."""

    def boom(loc, app_name=None):
        raise OSError("cannot write PATH")

    monkeypatch.setattr(path_install.userpath, "append", boom)
    with pytest.raises(path_install.PathInstallError):
        path_install.install_launcher(bin_dir=tmp_path)


def test_install_launcher_wraps_non_oserror_as_pathinstallerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_userpath: dict
) -> None:
    """A non-OSError from userpath (e.g. a UnicodeDecodeError from a malformed Windows
    PATH) is still wrapped, so the wizard's best-effort catch keeps setup non-fatal."""

    def boom(loc):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad PATH")

    monkeypatch.setattr(path_install.userpath, "in_current_path", boom)
    with pytest.raises(path_install.PathInstallError):
        path_install.install_launcher(bin_dir=tmp_path)


def test_ensure_on_path_warns_when_append_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_userpath: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A userpath.append that returns False warns instead of failing silently."""
    monkeypatch.setattr(
        path_install.userpath, "append", lambda loc, app_name=None: False
    )
    with caplog.at_level("WARNING"):
        path_install._ensure_on_path(tmp_path)
    assert "PATH" in caplog.text
