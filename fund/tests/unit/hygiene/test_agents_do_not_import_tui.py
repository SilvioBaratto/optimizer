"""Guard: no module under ``fund.agents`` imports ``fund.tui``.

``fund.tui`` is the Textual cockpit that drives the agents. An agent module that
imports it makes the agent layer depend on the presentation layer above it. The
code the two layers took from ``fund.tui`` now lives in neutral ``fund`` modules
(``fund.stream_events``, ``fund.text_safety``, ``fund.ask_context``).

The scan parses the syntax tree of each ``.py`` file under ``fund/agents`` and
reports an ``import`` or ``from ... import`` statement that names ``fund.tui`` or
one of its submodules, wherever it sits (inside a function or under
``if TYPE_CHECKING:``), whether absolute, relative (``from ..tui import ask``) or
``from fund import tui``. It also reports an ``import_module`` or ``__import__``
call whose first positional argument is a string literal naming such a module. It
does not detect a computed name, a name passed as a keyword argument, a relative
string name such as ``import_module(".tui", "fund")``, or the ``fromlist`` form
``__import__("fund", fromlist=["tui"])``. A docstring that only mentions
``fund.tui`` is not a hit. The scan imports no implementation module.

The neutral modules are then imported, each in a fresh interpreter because the
test session may already have loaded the agent stack: none may load a ``fund.tui``
module or any of ``deepagents``, ``langchain``, ``langchain_core``, ``langgraph``,
``optimizer``, ``skfolio`` and ``textual``, and ``fund.stream_events`` and
``fund.text_safety`` may load no package outside the standard library and
``fund``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

# hygiene -> unit -> tests -> fund, then the src layout's import root.
_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
_AGENTS_DIR = _SRC_ROOT / "fund" / "agents"
_FORBIDDEN = "fund.tui"
_DYNAMIC_IMPORT_CALLEES = frozenset({"import_module", "__import__"})

# Top-level packages a neutral module must not load: the agent stack, the optimizer
# and the cockpit's UI toolkit.
_HEAVY_PACKAGES = frozenset(
    {
        "deepagents",
        "langchain",
        "langchain_core",
        "langgraph",
        "optimizer",
        "skfolio",
        "textual",
    }
)
# Prints, as JSON, every module the import under test added to ``sys.modules``.
_IMPORT_PROBE = (
    "import json, sys\n"
    "before = set(sys.modules)\n"
    "import {module}\n"
    "print(json.dumps(sorted(set(sys.modules) - before)))\n"
)


def _package_of(path: Path, src_root: Path) -> str:
    """Return the dotted package that relative imports in ``path`` resolve against.

    Args:
        path: A module file under ``src_root``.
        src_root: The directory that holds the top-level ``fund`` package.

    Returns:
        ``fund.agents`` for both ``fund/agents/graph.py`` and
        ``fund/agents/__init__.py``.
    """
    return ".".join(path.relative_to(src_root).parts[:-1])


def _resolve_from(node: ast.ImportFrom, package: str) -> str:
    """Return the absolute module a ``from ... import`` statement reads from."""
    if node.level == 0:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - node.level + 1]
    return ".".join([*base, *([node.module] if node.module else [])])


def _dynamic_import_target(node: ast.Call) -> str | None:
    """Return the string literal a dynamic-import call passes first, else ``None``.

    Only an ``import_module`` or ``__import__`` call whose first positional argument
    is a string literal yields a name.
    """
    callee = node.func
    name = callee.attr if isinstance(callee, ast.Attribute) else None
    if isinstance(callee, ast.Name):
        name = callee.id
    if name not in _DYNAMIC_IMPORT_CALLEES or not node.args:
        return None
    target = node.args[0]
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value
    return None


def _imported_names(node: ast.AST, package: str) -> list[str]:
    """Return the dotted names an import node names, relative ones resolved.

    ``from a import b`` yields both ``a`` and ``a.b``, because ``b`` may be a
    submodule: ``from fund import tui`` loads ``fund.tui``.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        module = _resolve_from(node, package)
        return [module, *(f"{module}.{alias.name}" for alias in node.names)]
    if isinstance(node, ast.Call):
        target = _dynamic_import_target(node)
        return [target] if target else []
    return []


def _is_tui(module: str) -> bool:
    """Return whether ``module`` is ``fund.tui`` or one of its submodules."""
    return module == _FORBIDDEN or module.startswith(f"{_FORBIDDEN}.")


def find_tui_imports(src_root: Path, package_dir: Path) -> list[str]:
    """Return one ``path:line: module`` entry per detected ``fund.tui`` import.

    Args:
        src_root: The directory that holds the top-level ``fund`` package; paths
            in the result are relative to it.
        package_dir: The package tree to scan, every ``.py`` file at any depth.

    Returns:
        The offending import statements ordered by file and line, empty when
        the tree is clean.
    """
    hits: list[tuple[str, int, str]] = []
    for path in package_dir.rglob("*.py"):
        package = _package_of(path, src_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(src_root).as_posix()
        for node in ast.walk(tree):
            tui = sorted(m for m in _imported_names(node, package) if _is_tui(m))
            if tui:
                hits.append((relative, getattr(node, "lineno", 0), tui[0]))
    return [f"{path}:{line}: {module}" for path, line, module in sorted(hits)]


def test_the_agent_package_is_where_the_guard_looks() -> None:
    """A moved or renamed package must fail here, not pass the scan with no files."""
    assert (_AGENTS_DIR / "__init__.py").is_file()
    assert len(list(_AGENTS_DIR.rglob("*.py"))) > 1


def test_no_agent_module_imports_the_tui() -> None:
    """Each detected ``fund.tui`` import under ``fund/agents`` fails the test, listed
    as ``path:line: module``."""
    hits = find_tui_imports(_SRC_ROOT, _AGENTS_DIR)
    assert not hits, (
        "fund.agents must not import fund.tui; move what both layers need into a "
        "neutral fund module. Offending imports:\n" + "\n".join(hits)
    )


def _modules_loaded_by(module: str) -> list[str]:
    """Import ``module`` in a fresh interpreter and return the modules it added.

    Args:
        module: The dotted module to import.

    Returns:
        Every name the import added to ``sys.modules``, its own included.
    """
    result = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
        [sys.executable, "-c", _IMPORT_PROBE.format(module=module)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "module", ["fund.stream_events", "fund.text_safety", "fund.ask_context"]
)
def test_a_neutral_module_loads_neither_the_tui_nor_the_agent_stack(
    module: str,
) -> None:
    """Loading the cockpit would bring the inversion back; loading the agent stack,
    the optimizer or Textual would make the cockpit's light imports heavy."""
    loaded = _modules_loaded_by(module)
    assert [name for name in loaded if _is_tui(name)] == []
    assert sorted({name.split(".")[0] for name in loaded} & _HEAVY_PACKAGES) == []


@pytest.mark.parametrize("module", ["fund.stream_events", "fund.text_safety"])
def test_a_stdlib_only_neutral_module_loads_nothing_outside_the_stdlib(
    module: str,
) -> None:
    """Both module docstrings promise a standard-library-only import."""
    roots = {name.split(".")[0] for name in _modules_loaded_by(module)}
    assert sorted(roots - set(sys.stdlib_module_names) - {"fund"}) == []


def test_the_import_probe_reports_what_an_import_pulls_in() -> None:
    """Fail-injection: the cockpit's entry module loads Textual and ``fund.tui``, so
    the probe is shown to observe the imports the two tests above forbid."""
    loaded = _modules_loaded_by("fund.tui.app")
    assert "textual" in {name.split(".")[0] for name in loaded}
    assert any(_is_tui(name) for name in loaded)


def _seed(tmp_path: Path, source: str) -> Path:
    """Write ``source`` as ``fund/agents/offender.py`` under ``tmp_path``."""
    package = tmp_path / "fund" / "agents"
    package.mkdir(parents=True)
    (package / "offender.py").write_text(source, encoding="utf-8")
    return package


@pytest.mark.parametrize(
    ("source", "line", "module"),
    [
        ("import fund.tui.app\n", 1, "fund.tui.app"),
        ("from fund.tui.stream import StreamEvent\n", 1, "fund.tui.stream"),
        ("from fund import tui\n", 1, "fund.tui"),
        ("from fund import observe, tui as cockpit\n", 1, "fund.tui"),
        ("from ..tui import ask\n", 1, "fund.tui"),
        ("from .. import tui\n", 1, "fund.tui"),
        ("from ..tui.safe_text import clip\n", 1, "fund.tui.safe_text"),
        ("def build():\n    from fund.tui import ask\n", 2, "fund.tui"),
        (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from fund.tui.stream import StreamEvent\n",
            3,
            "fund.tui.stream",
        ),
        (
            "import importlib\nimportlib.import_module('fund.tui.ask')\n",
            2,
            "fund.tui.ask",
        ),
        ("__import__('fund.tui')\n", 1, "fund.tui"),
    ],
    ids=[
        "import-submodule",
        "from-submodule",
        "from-fund-import-tui",
        "from-fund-import-list",
        "relative-module",
        "relative-package",
        "relative-submodule",
        "function-local",
        "type-checking",
        "import_module",
        "dunder-import",
    ],
)
def test_an_injected_tui_import_is_reported_with_its_line(
    tmp_path: Path, source: str, line: int, module: str
) -> None:
    """Fail-injection: each import form the guard claims to see is reported."""
    package = _seed(tmp_path, source)
    assert find_tui_imports(tmp_path, package) == [
        f"fund/agents/offender.py:{line}: {module}"
    ]


@pytest.mark.parametrize(
    "source",
    [
        '"""Streams events to ``fund.tui.stream`` for the cockpit."""\n',
        "from fund import observe\n",
        "from fund.stream_events import StreamEvent\n",
        "from . import backend\n",
        "from .. import tools\n",
        "import fund.tuition\n",
    ],
    ids=[
        "docstring",
        "sibling-module",
        "neutral-module",
        "relative-sibling",
        "relative-parent-sibling",
        "lookalike-name",
    ],
)
def test_source_that_only_resembles_a_tui_import_is_allowed(
    tmp_path: Path, source: str
) -> None:
    """A mention, a neutral module or a lookalike name is not a ``fund.tui`` import."""
    package = _seed(tmp_path, source)
    assert find_tui_imports(tmp_path, package) == []
