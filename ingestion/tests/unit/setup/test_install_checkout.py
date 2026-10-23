"""The one-line installers must leave a checkout for ``portopt`` to run compose in.

``portopt setup`` / ``start`` run ``docker compose`` against the checkout that owns
``docker-compose.yml`` and fail fast without one. ``uv tool install git+...`` only
installs the CLI into a tool venv, so a bootstrap that stops there dies at
"Cannot locate the optimizer checkout". The installers therefore clone (or reuse) a
checkout and export ``OPTIMIZER_REPO`` before handing off to ``portopt setup``.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(name: str) -> str:
    return (_REPO_ROOT / name).read_text(encoding="utf-8")


def _first_setup_invocation(text: str) -> int:
    """Offset of the first ``portopt setup`` command line, skipping comments."""
    match = re.search(r"^[ \t]*portopt setup", text, re.MULTILINE)
    assert match, "no `portopt setup` invocation"
    return match.start()


class TestInstallShCheckout:
    """``install.sh`` provides the checkout the CLI needs."""

    def test_clones_the_repo(self) -> None:
        """It clones PORTOPT_REPO and pins the requested ref."""
        text = _read("install.sh")
        assert 'git clone "${PORTOPT_REPO}"' in text
        assert "checkout" in text
        assert "${PORTOPT_REF}" in text

    def test_checkout_location_is_overridable(self) -> None:
        """PORTOPT_HOME picks the clone destination."""
        assert "PORTOPT_HOME" in _read("install.sh")

    def test_reuses_an_existing_checkout(self) -> None:
        """A re-run must not re-clone over (or fail on) an existing checkout."""
        assert "docker-compose.yml" in _read("install.sh")

    def test_requires_git_up_front(self) -> None:
        """Missing git is reported before any install work starts."""
        text = _read("install.sh")
        assert "command -v git" in text
        assert text.index("command -v git") < text.index("uv tool install")

    def test_exports_optimizer_repo_before_setup(self) -> None:
        """The wizard resolves the checkout from OPTIMIZER_REPO."""
        text = _read("install.sh")
        assert "export OPTIMIZER_REPO=" in text
        assert text.index("export OPTIMIZER_REPO=") < _first_setup_invocation(text)


class TestInstallPs1Checkout:
    """``install.ps1`` mirrors ``install.sh``."""

    def test_clones_the_repo(self) -> None:
        """It clones PORTOPT_REPO and pins the requested ref."""
        text = _read("install.ps1")
        assert "git clone $PortoptRepo" in text
        assert "checkout" in text
        assert "$PortoptRef" in text

    def test_checkout_location_is_overridable(self) -> None:
        """PORTOPT_HOME picks the clone destination."""
        assert "PORTOPT_HOME" in _read("install.ps1")

    def test_reuses_an_existing_checkout(self) -> None:
        """A re-run must not re-clone over (or fail on) an existing checkout."""
        assert "docker-compose.yml" in _read("install.ps1")

    def test_requires_git_up_front(self) -> None:
        """Missing git is reported before any install work starts."""
        text = _read("install.ps1")
        assert "Get-Command git" in text
        assert text.index("Get-Command git") < text.index("uv tool install")

    def test_checks_native_exit_codes(self) -> None:
        """``$ErrorActionPreference = "Stop"`` ignores a failing native git, so the
        clone and checkout exit codes must be tested explicitly."""
        assert "$LASTEXITCODE" in _read("install.ps1")

    def test_exports_optimizer_repo_before_setup(self) -> None:
        """The wizard resolves the checkout from OPTIMIZER_REPO."""
        text = _read("install.ps1")
        assert "$env:OPTIMIZER_REPO =" in text
        assert text.index("$env:OPTIMIZER_REPO =") < _first_setup_invocation(text)


class TestReadmeDocumentsTheCheckout:
    """The README must not present the bootstrap as checkout-free."""

    def test_install_section_mentions_the_checkout(self) -> None:
        """The full-platform section says a checkout is needed and where it goes."""
        text = _read("README.md")
        start = text.index("### 2. The full platform")
        end = text.index("### 3. Developer setup")
        section = text[start:end]
        assert "checkout" in section
        assert "PORTOPT_HOME" in section
        assert "OPTIMIZER_REPO" in section
