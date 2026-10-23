"""Clone-free installer contract.

``install.sh`` is the ``curl … | bash`` path: piping the script into bash binds the
shell's stdin to the pipe, so an interactive ``portopt setup`` would read EOF and
abort. The fix reconnects stdin to ``/dev/tty`` when a controlling terminal exists
and runs as-is when it does not (headless / ``--non-interactive``). ``install.ps1``
mirrors it (no ``/dev/tty`` on Windows — PowerShell children inherit the console
stdin directly — so parity there is arg-forwarding + matching structure).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit.setup._shell_stubs import run_front_door

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(name: str) -> str:
    return (_REPO_ROOT / name).read_text(encoding="utf-8")


_DEFAULT_REPO = "https://github.com/SilvioBaratto/optimizer.git"


def _git(repo: Path, *args: str) -> str:
    """Run real git inside ``repo`` with the user's own git config out of the way."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    result = subprocess.run(  # noqa: S603 - resolved git path + test-built repos
        [
            git,
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.test",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    )
    return result.stdout.strip()


def _make_origin(path: Path) -> tuple[str, str]:
    """Build a repo with two commits on ``development``, tag ``v1`` on the first.

    Returns:
        The full SHAs of the first and the second commit.
    """
    path.mkdir(parents=True)
    _git(path, "init", "--quiet", "--initial-branch=development")
    (path / "docker-compose.yml").write_text("services: {}\n")
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "one")
    first = _git(path, "rev-parse", "HEAD")
    _git(path, "tag", "v1")
    (path / "marker").write_text("two\n")
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "two")
    return first, _git(path, "rev-parse", "HEAD")


class TestInstallSh:
    """The POSIX clone-free installer reaches interactive prompts under curl|bash."""

    def test_exists(self) -> None:
        """install.sh is present at the repo root."""
        assert (_REPO_ROOT / "install.sh").is_file()

    def test_is_lf_only(self) -> None:
        """A CRLF shebang breaks bash (.gitattributes forces LF)."""
        assert b"\r" not in (_REPO_ROOT / "install.sh").read_bytes()

    def test_guards_the_dev_tty_read(self) -> None:
        """It reconnects stdin to /dev/tty only when the device can be opened.

        A ``-r /dev/tty`` mode-bit test is not a guard: it passes with no
        controlling terminal.
        """
        text = _read("install.sh")
        assert "< /dev/tty" in text
        assert "-r /dev/tty" not in text

    def test_has_a_headless_fallback(self) -> None:
        """A no-tty branch still runs setup (so --non-interactive/CI survives)."""
        text = _read("install.sh")
        # Two invocations: one redirected from /dev/tty, one plain fallback.
        assert text.count("portopt setup") >= 2

    def test_forwards_args_to_setup(self) -> None:
        """It forwards args so `curl … | bash -s -- --non-interactive …` works."""
        assert '"$@"' in _read("install.sh")

    def test_installs_the_cli_via_uv_tool(self) -> None:
        """The clone-free path installs the published `portopt` via uv tool."""
        assert "uv tool install portopt" in _read("install.sh")

    def test_ensures_uv(self) -> None:
        """It bootstraps uv when missing."""
        assert "astral.sh/uv/install.sh" in _read("install.sh")

    def test_bash_syntax_is_valid(self) -> None:
        """``bash -n`` parses the script without error."""
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not available")
        result = subprocess.run(  # noqa: S603 - resolved bash path + our own script
            [bash, "-n", str(_REPO_ROOT / "install.sh")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_headless_run_without_a_controlling_terminal_still_runs_setup(
        self, tmp_path: Path
    ) -> None:
        """No controlling terminal (CI, cron, ``ssh -T``) takes the plain fallback.

        ``/dev/tty`` stays world-readable by mode even when it cannot be opened, so
        a mode-bit check sends the script into a ``< /dev/tty`` redirect that fails
        and ``set -e`` aborts before ``portopt setup`` ever runs.
        """
        run = run_front_door("install.sh", tmp_path, "--non-interactive")

        assert run.result.returncode == 0, run.result.stderr
        setups = run.calls_to("portopt")
        assert len(setups) == 1, run.result.stderr
        assert setups[0][3] == "setup --non-interactive"

    def test_runs_setup_from_a_checkout_that_has_the_compose_file(
        self, tmp_path: Path
    ) -> None:
        """The wizard starts Docker from a clone, not from wherever curl was run.

        ``portopt setup`` runs ``docker compose`` against the cwd and the stack's
        Dockerfiles live only in the repo, so a bare CLI install dies with
        ``no configuration file provided`` after it already persisted secrets.
        """
        run = run_front_door("install.sh", tmp_path, "--non-interactive")

        assert run.result.returncode == 0, run.result.stderr
        checkout = (run.home / "optimizer").resolve()
        (setup,) = run.calls_to("portopt")
        assert Path(setup[1]).resolve() == checkout
        assert (checkout / "docker-compose.yml").is_file()
        assert Path(setup[2]).resolve() == checkout

    def test_clones_the_requested_repo_and_ref_into_portopt_dir(
        self, tmp_path: Path
    ) -> None:
        """PORTOPT_REPO / PORTOPT_REF pick what is cloned, PORTOPT_DIR where."""
        target = tmp_path / "elsewhere" / "checkout"
        run = run_front_door(
            "install.sh",
            tmp_path,
            env={
                "PORTOPT_REPO": "https://example.test/fork.git",
                "PORTOPT_REF": "v9",
                "PORTOPT_DIR": str(target),
            },
        )

        assert run.result.returncode == 0, run.result.stderr
        (clone,) = [c for c in run.calls_to("git") if c[3].startswith("clone")]
        assert " --branch v9 " in f" {clone[3]} "
        assert clone[3].endswith(f" https://example.test/fork.git {target}")
        (setup,) = run.calls_to("portopt")
        assert Path(setup[1]).resolve() == target.resolve()

    def test_reuses_an_existing_checkout_instead_of_recloning(
        self, tmp_path: Path
    ) -> None:
        """A re-run refreshes the checkout; ``git clone`` would fail on a non-empty dir."""
        target = tmp_path / "checkout"
        (target / ".git").mkdir(parents=True)
        (target / "docker-compose.yml").write_text("services: {}\n")

        run = run_front_door(
            "install.sh",
            tmp_path,
            env={"PORTOPT_DIR": str(target), "STUB_GIT_ORIGIN": _DEFAULT_REPO},
        )

        assert run.result.returncode == 0, run.result.stderr
        git_calls = [c[3] for c in run.calls_to("git")]
        assert not any(call.startswith("clone") for call in git_calls)
        assert any("fetch" in call for call in git_calls)
        (setup,) = run.calls_to("portopt")
        assert Path(setup[1]).resolve() == target.resolve()

    @pytest.mark.parametrize(
        "origin",
        [
            "https://github.com/SilvioBaratto/optimizer",
            "https://github.com/SilvioBaratto/optimizer/",
            "git@github.com:SilvioBaratto/optimizer.git",
            "ssh://git@github.com/SilvioBaratto/optimizer.git",
        ],
    )
    def test_reuses_a_checkout_whose_origin_is_the_same_repo_spelled_differently(
        self, tmp_path: Path, origin: str
    ) -> None:
        """SSH vs HTTPS and a trailing ``.git`` or slash do not make it a stranger."""
        target = tmp_path / "checkout"
        (target / ".git").mkdir(parents=True)
        (target / "docker-compose.yml").write_text("services: {}\n")

        run = run_front_door(
            "install.sh",
            tmp_path,
            env={"PORTOPT_DIR": str(target), "STUB_GIT_ORIGIN": origin},
        )

        assert run.result.returncode == 0, run.result.stderr
        assert any("fetch" in c[3] for c in run.calls_to("git"))
        assert len(run.calls_to("portopt")) == 1

    @pytest.mark.parametrize("ref_kind", ["branch", "tag", "sha"])
    def test_first_run_checks_out_a_branch_tag_or_commit_sha(
        self, tmp_path: Path, ref_kind: str
    ) -> None:
        """PORTOPT_REF may be a commit SHA, not only something ``--branch`` accepts.

        ``git clone --branch <sha>`` aborts with ``Remote branch ... not found``, so
        a pinned-SHA install used to die under ``set -e`` before installing anything.
        """
        origin = tmp_path / "origin"
        first, second = _make_origin(origin)
        ref, expected = {
            "branch": ("development", second),
            "tag": ("v1", first),
            "sha": (first, first),
        }[ref_kind]
        target = tmp_path / "checkout"

        run = run_front_door(
            "install.sh",
            tmp_path,
            env={
                "PORTOPT_REPO": origin.as_uri(),
                "PORTOPT_REF": ref,
                "PORTOPT_DIR": str(target),
            },
            real_git=True,
        )

        assert run.result.returncode == 0, run.result.stderr
        assert _git(target, "rev-parse", "HEAD") == expected
        assert len(run.calls_to("portopt")) == 1

    def test_a_rerun_moves_an_existing_checkout_to_a_pinned_commit_sha(
        self, tmp_path: Path
    ) -> None:
        """The update path agrees with the first-run path about what a ref may be."""
        origin = tmp_path / "origin"
        first, second = _make_origin(origin)
        target = tmp_path / "checkout"
        env = {"PORTOPT_REPO": origin.as_uri(), "PORTOPT_DIR": str(target)}
        run_front_door(
            "install.sh",
            tmp_path,
            env={**env, "PORTOPT_REF": "development"},
            real_git=True,
        )
        assert _git(target, "rev-parse", "HEAD") == second

        rerun = run_front_door(
            "install.sh", tmp_path, env={**env, "PORTOPT_REF": first}, real_git=True
        )

        assert rerun.result.returncode == 0, rerun.result.stderr
        assert _git(target, "rev-parse", "HEAD") == first

    def test_refuses_to_touch_a_checkout_of_a_different_repo(
        self, tmp_path: Path
    ) -> None:
        """An unrelated clone in PORTOPT_DIR is left on its own branch, untouched.

        Fetch + checkout + ``merge --ff-only`` inside someone's own working copy
        would silently switch their feature branch to PORTOPT_REF.
        """
        theirs = tmp_path / "theirs"
        _make_origin(theirs)
        target = tmp_path / "checkout"
        _git(tmp_path, "clone", "--quiet", theirs.as_uri(), str(target))
        _git(target, "checkout", "--quiet", "-b", "feature")
        ours = tmp_path / "ours"
        _make_origin(ours)

        run = run_front_door(
            "install.sh",
            tmp_path,
            env={"PORTOPT_REPO": ours.as_uri(), "PORTOPT_DIR": str(target)},
            real_git=True,
        )

        assert run.result.returncode != 0
        assert "PORTOPT_DIR" in run.result.stderr
        assert _git(target, "rev-parse", "--abbrev-ref", "HEAD") == "feature"
        assert run.calls_to("portopt") == []
        assert run.calls_to("uv") == []

    def test_refuses_a_git_directory_with_no_origin_remote(
        self, tmp_path: Path
    ) -> None:
        """No ``origin`` means it cannot be shown to be this repo, so it is not reused."""
        target = tmp_path / "checkout"
        (target / ".git").mkdir(parents=True)

        run = run_front_door("install.sh", tmp_path, env={"PORTOPT_DIR": str(target)})

        assert run.result.returncode != 0
        assert "PORTOPT_DIR" in run.result.stderr
        assert [c[3] for c in run.calls_to("git") if "fetch" in c[3]] == []
        assert run.calls_to("portopt") == []


class TestInstallPs1:
    """The Windows clone-free installer mirrors install.sh behaviour."""

    def test_exists(self) -> None:
        """install.ps1 is present at the repo root."""
        assert (_REPO_ROOT / "install.ps1").is_file()

    def test_forwards_args_to_setup(self) -> None:
        """It forwards @args to `portopt setup` (parity with install.sh)."""
        text = _read("install.ps1")
        assert "portopt setup" in text
        assert "@args" in text

    def test_installs_the_cli_via_uv_tool(self) -> None:
        """It installs the published `portopt` via uv tool."""
        assert "uv tool install portopt" in _read("install.ps1")

    def test_ensures_uv(self) -> None:
        """It bootstraps uv when missing."""
        assert "astral.sh/uv/install.ps1" in _read("install.ps1")
