"""Structural checks for the `optimizer` daily launcher (task T11).

The launcher is Docker-driven and only smoke-tested by hand (see tasks/todo.md), so
these tests pin the load-bearing structure a manual run can't guard on every commit:
the POSIX shim must stay LF (a CRLF ``#!/usr/bin/env bash`` shebang fails in Git Bash),
resolve its repo, bring the fund stack up gated on the alembic-at-head healthcheck
(``--wait``), and attach the Textual cockpit *with* a TTY — winpty only under mintty,
and never ``-T`` (which disables the pseudo-TTY the cockpit needs). The Windows ``.cmd``
wrapper must locate ``bash`` and know the repo.
"""

from __future__ import annotations

from pathlib import Path

# unit -> setup -> tests -> ingestion -> repo root (scripts/ lives at the root).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_LAUNCHER = _REPO_ROOT / "scripts" / "optimizer"
_LAUNCHER_CMD = _REPO_ROOT / "scripts" / "optimizer.cmd"


def _launcher_text() -> str:
    return _LAUNCHER.read_text(encoding="utf-8")


def _attach_lines() -> list[str]:
    """Return the launcher lines that invoke the cockpit entrypoint."""
    return [line for line in _launcher_text().splitlines() if "fund-tui" in line]


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


def test_launcher_requires_a_portfolio_id() -> None:
    """A bare `optimizer` (no portfolio id) fails fast with a usage message, since
    fund-tui requires the id and the launcher forwards it verbatim."""
    text = _launcher_text()
    assert "$# -eq 0" in text
    assert "usage" in text.lower()


def test_launcher_resolves_repo() -> None:
    """Repo resolution honours OPTIMIZER_REPO and follows the symlink via readlink."""
    text = _launcher_text()
    assert "OPTIMIZER_REPO" in text
    assert "readlink" in text


def test_launcher_attaches_the_tui() -> None:
    """The launcher invokes fund-tui and forwards its args (the portfolio id)."""
    assert _attach_lines()
    assert '"$@"' in _launcher_text()


def test_launcher_winpty_gated_under_mintty() -> None:
    """winpty wraps the attach only under mintty (MSYSTEM signal, per prompts.py)."""
    text = _launcher_text()
    assert "winpty" in text
    assert "MSYSTEM" in text


def test_launcher_never_disables_tty_on_attach() -> None:
    """No fund-tui attach passes `-T`, which would kill the cockpit's TTY."""
    for line in _attach_lines():
        assert " -T " not in line
        assert "exec -T" not in line


def test_cmd_wrapper_exists() -> None:
    """The Windows launcher is present at scripts/optimizer.cmd."""
    assert _LAUNCHER_CMD.is_file()


def test_cmd_wrapper_discovers_bash_and_delegates() -> None:
    """The .cmd locates bash and delegates to the POSIX optimizer launcher."""
    text = _LAUNCHER_CMD.read_text(encoding="utf-8")
    assert "bash" in text.lower()
    assert "optimizer" in text.lower()
