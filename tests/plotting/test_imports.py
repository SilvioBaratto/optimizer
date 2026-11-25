"""Guard: plotly is confined to ``optimizer.plotting`` and loaded only on demand.

``import optimizer`` already loads plotly, but not through this library: skfolio
imports it at module level (``skfolio/typing.py``) and ``optimizer`` imports
skfolio. A ``"plotly" not in sys.modules`` check after ``import optimizer`` can
therefore never pass. What this library controls, and what is checked here, is
that none of its own modules outside ``optimizer.plotting`` imports plotly and
that the plotting package is not loaded by ``import optimizer``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import optimizer

from .._import_scan import imported_module_names

_OPTIMIZER_SRC = Path(optimizer.__path__[0]).resolve()
_PLOTTING_SRC = _OPTIMIZER_SRC / "plotting"


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter that resolves the same ``optimizer``."""
    env = dict(os.environ)
    source_root = str(_OPTIMIZER_SRC.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [source_root, *filter(None, [env.get("PYTHONPATH")])]
    )
    # The interpreter is sys.executable and the code is a literal from this file,
    # so no untrusted input reaches the subprocess.
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _imports_plotly(path: Path) -> bool:
    """Return whether the module at ``path`` imports plotly."""
    names = imported_module_names(path.read_text(encoding="utf-8"), str(path))
    return any(name.split(".", 1)[0] == "plotly" for name in names)


def test_importing_optimizer_does_not_load_the_plotting_package() -> None:
    """The plotting package is loaded by ``import optimizer.plotting`` only."""
    result = _run_python(
        "import sys, optimizer\n"
        "print('optimizer.plotting' in sys.modules)\n"
        "print(hasattr(optimizer, 'plotting'))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False"]


def test_importing_the_plotting_package_loads_plotly_but_not_kaleido() -> None:
    """Charts need plotly; static image export (kaleido) is never pulled in."""
    result = _run_python(
        "import sys, optimizer.plotting\n"
        "print('plotly' in sys.modules)\n"
        "print('kaleido' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "False"]


def test_no_module_outside_the_plotting_package_imports_plotly() -> None:
    """Plotly is imported by ``optimizer.plotting`` and by nothing else here."""
    sources = [p for p in _OPTIMIZER_SRC.rglob("*.py") if "__pycache__" not in p.parts]
    assert sources, f"no optimizer source found at {_OPTIMIZER_SRC}"
    outside = [p for p in sources if _PLOTTING_SRC not in p.parents]
    offenders = [p.relative_to(_OPTIMIZER_SRC) for p in outside if _imports_plotly(p)]
    assert offenders == []


def test_the_plotting_package_does_import_plotly() -> None:
    """Keeps the scan above honest: it must see the imports it is looking for."""
    inside = [p for p in _PLOTTING_SRC.rglob("*.py") if "__pycache__" not in p.parts]
    assert any(_imports_plotly(p) for p in inside)


@pytest.mark.parametrize(
    "name",
    [
        "ChartTheme",
        "correlation_heatmap",
        "efficient_frontier_figure",
        "frontier_from_weights",
        "returns_volatility_figure",
        "save_figure_html",
        "security_market_line_figure",
    ],
)
def test_the_public_names_are_exported(name: str) -> None:
    """Every documented name is importable from ``optimizer.plotting``."""
    import optimizer.plotting as plotting

    assert name in plotting.__all__
    assert hasattr(plotting, name)
