"""Structural checks for the `optimizer` daily launcher.

The launcher is Docker-driven and only smoke-tested by hand, so
these tests pin the load-bearing structure a manual run can't guard on every commit:
the POSIX shim must stay LF (a CRLF ``#!/usr/bin/env bash`` shebang fails in Git Bash),
resolve its repo, bring the fund stack up gated on the alembic-at-head healthcheck
(``--wait``), and attach the Textual cockpit *with* a TTY — winpty only under mintty,
and never ``-T`` (which disables the pseudo-TTY the cockpit needs). The Windows ``.cmd``
wrapper must locate ``bash`` and know the repo.

Tests that execute the launcher run a copy inside a throwaway repo with ``portopt`` and
``docker`` stubbed on a scrubbed PATH: the real launcher decrypts the secret store and
brings the live compose stack up, which a unit test must never do.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

# unit -> setup -> tests -> ingestion -> repo root (scripts/ lives at the root).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_LAUNCHER = _REPO_ROOT / "scripts" / "optimizer"
_LAUNCHER_CMD = _REPO_ROOT / "scripts" / "optimizer.cmd"


def _launcher_text() -> str:
    return _LAUNCHER.read_text(encoding="utf-8")


def _attach_lines() -> list[str]:
    """Return the launcher's executable cockpit-attach lines (comments excluded)."""
    return [
        line
        for line in _launcher_text().splitlines()
        if "fund-tui" in line and not line.lstrip().startswith("#")
    ]


def _bash() -> str | None:
    return shutil.which("bash")


def _run_launcher_with_stubs(
    tmp_path: Path, *args: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run a copy of the launcher in a fake repo with `portopt`/`docker` stubbed.

    The environment is built from scratch instead of inherited, so an exported
    ``PORTOPT_PASSPHRASE``, a real ``portopt``/``docker`` on the developer's PATH or a
    ``~/.portopt`` config can never decrypt secrets or touch the live compose stack.

    Returns:
        The completed process and the stub invocations in call order, one
        ``"<tool> <args>"`` line each.
    """
    bash = _bash()
    if bash is None:
        pytest.skip("bash not available")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "docker-compose.yml").write_text("", encoding="utf-8")
    launcher = repo / "scripts" / "optimizer"
    launcher.write_text(_launcher_text(), encoding="utf-8")

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    calls = tmp_path / "calls.log"
    for tool in ("portopt", "docker"):
        stub = stubs / tool
        stub.write_text(
            f'#!/bin/sh\necho "{tool} $*" >> {shlex.quote(str(calls))}\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(  # noqa: S603 - fixed bash path + a copy of our launcher
        [bash, str(launcher), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env={"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(home)},
    )
    recorded = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    return result, recorded


def test_launcher_exists() -> None:
    """The POSIX launcher is present at scripts/optimizer."""
    assert _LAUNCHER.is_file()


def test_launcher_is_lf_only() -> None:
    """The launcher carries no CR bytes (a CRLF shebang breaks Git Bash)."""
    assert b"\r" not in _LAUNCHER.read_bytes()


def test_launcher_has_bash_shebang() -> None:
    """The first line is the portable bash shebang."""
    assert _launcher_text().splitlines()[0] == "#!/usr/bin/env bash"


def test_launcher_starts_stack_via_portopt_start() -> None:
    """The launcher renders secrets + brings the stack up by delegating to `portopt
    start` (which decrypts→renders→`up --wait`), not a secret-less bare compose up."""
    text = _launcher_text()
    assert "portopt start" in text
    assert "app.cli start" in text  # uv-run fallback when portopt is not on PATH
    # The launcher itself must NOT run a bare compose up — that would skip the
    # secret rendering the fund service's `file:` secrets require.
    assert "compose up" not in text
    assert "up -d" not in text


def test_launcher_resolves_repo() -> None:
    """Repo resolution honours OPTIMIZER_REPO and follows the symlink via readlink."""
    text = _launcher_text()
    assert "OPTIMIZER_REPO" in text
    assert "readlink" in text


def test_launcher_attaches_the_tui() -> None:
    """The launcher invokes fund-tui and forwards its args (the portfolio id)."""
    assert _attach_lines()
    assert '"$@"' in _launcher_text()


def test_launcher_attaches_under_the_fund_profile() -> None:
    """Every attach line targets the fund profile, so `exec` finds the running service."""
    for line in _attach_lines():
        assert "--profile fund" in line


def test_launcher_winpty_gated_under_mintty() -> None:
    """winpty wraps the attach only under mintty (MSYSTEM signal, per prompts.py)."""
    text = _launcher_text()
    assert "winpty" in text
    assert "MSYSTEM" in text


def test_launcher_winpty_gated_on_a_real_tty() -> None:
    """The winpty branch also requires a real bash tty (`-t 0`/`-t 1`), not MSYSTEM
    alone — dropping the tty guard would break non-interactive/no-tty invocations."""
    text = _launcher_text()
    assert "-t 0" in text
    assert "-t 1" in text


def test_launcher_never_disables_tty_on_attach() -> None:
    """No fund-tui attach passes `-T`, which would kill the cockpit's TTY."""
    for line in _attach_lines():
        assert " -T " not in line
        assert "exec -T" not in line


def test_launcher_bare_invocation_opens_the_fund_wide_shell(tmp_path: Path) -> None:
    """With no portfolio id the launcher starts the stack through `portopt start` and
    then attaches fund-tui with no id, which opens the fund-wide shell."""
    result, calls = _run_launcher_with_stubs(tmp_path)
    assert result.returncode == 0, result.stderr
    assert calls == [
        "portopt start",
        "docker compose --profile fund exec fund fund-tui",
    ]


def test_launcher_forwards_the_portfolio_id_to_the_tui(tmp_path: Path) -> None:
    """A portfolio id is passed to fund-tui verbatim so the shell opens focused on it."""
    result, calls = _run_launcher_with_stubs(tmp_path, "some-portfolio-id")
    assert result.returncode == 0, result.stderr
    assert calls == [
        "portopt start",
        "docker compose --profile fund exec fund fund-tui some-portfolio-id",
    ]


def test_launcher_outside_a_repo_reports_helpful_error(tmp_path: Path) -> None:
    """Run from a non-repo location with no OPTIMIZER_REPO and no config, resolve_repo
    surfaces the guided 'cannot locate the repo' message and exits 1 — it falls through
    the resolution chain cleanly instead of crashing or reaching `portopt start`."""
    bash = _bash()
    if bash is None:
        pytest.skip("bash not available")
    stray = tmp_path / "bin" / "optimizer"
    stray.parent.mkdir(parents=True)
    stray.write_text(_launcher_text(), encoding="utf-8")  # a copy, no repo around it
    home = tmp_path / "home"  # no ~/.portopt/config.toml here
    home.mkdir()
    env = {**os.environ, "HOME": str(home)}
    env.pop("OPTIMIZER_REPO", None)
    result = subprocess.run(  # noqa: S603 - fixed bash path + our own launcher script
        [bash, str(stray), "some-portfolio-id"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 1
    assert "cannot locate the repo" in result.stderr.lower()


def test_launcher_pins_the_resolved_repo_for_portopt_start(tmp_path: Path) -> None:
    """The launcher resolves its repo from its own location, but `portopt start` ranks
    the persisted config `repo_path` above the cwd. With a second checkout recorded
    there, start would otherwise render into and bring up the other stack while the
    launcher's `exec fund` attaches here. Real bash, stubbed portopt and docker."""
    bash = _bash()
    if bash is None:
        pytest.skip("bash not available")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    launcher = repo / "scripts" / "optimizer"
    launcher.write_text(_launcher_text(), encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    home = tmp_path / "home"
    (home / ".portopt").mkdir(parents=True)
    (home / ".portopt" / "config.toml").write_text(
        f'repo_path = "{other}"\n', encoding="utf-8"
    )
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    seen = tmp_path / "seen"
    portopt = stubs / "portopt"
    portopt.write_text(
        f'#!/bin/sh\nprintf "%s\\n%s\\n" "${{OPTIMIZER_REPO:-unset}}" "$PWD" > "{seen}"\n',
        encoding="utf-8",
    )
    docker = stubs / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    portopt.chmod(0o755)
    docker.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{stubs}:{os.environ['PATH']}"}
    env.pop("OPTIMIZER_REPO", None)
    env.pop("MSYSTEM", None)
    result = subprocess.run(  # noqa: S603 - fixed bash path + our own launcher copy
        [bash, str(launcher), "some-portfolio-id"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    seen_repo, seen_cwd = seen.read_text(encoding="utf-8").splitlines()
    assert Path(seen_repo).resolve() == repo.resolve()
    assert Path(seen_cwd).resolve() == repo.resolve()


def test_cmd_wrapper_exists() -> None:
    """The Windows launcher is present at scripts/optimizer.cmd."""
    assert _LAUNCHER_CMD.is_file()


def test_cmd_wrapper_discovers_bash_and_delegates() -> None:
    """The .cmd locates bash and delegates to the POSIX optimizer launcher."""
    text = _LAUNCHER_CMD.read_text(encoding="utf-8")
    assert "bash" in text.lower()
    assert "optimizer" in text.lower()
