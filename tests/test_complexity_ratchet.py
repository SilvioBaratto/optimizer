"""The complexity and length ratchet: no new offender, no offender grows.

Standard 2.8 asks for methods of at most 30 lines and a cyclomatic complexity below
10. The code base predates the rule, so ``scripts/ci/check_complexity_ratchet.py``
records today's offenders in two baselines that may only shrink and fails the lint
job for any function that is new or larger than its recorded value. The tests build
synthetic trees in a temporary directory and never touch the committed baselines,
except for the two that prove those baselines accept the tree as it stands.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "check_complexity_ratchet.py"
_BASELINES = _REPO_ROOT / "scripts" / "ci" / "baselines"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"


@functools.cache
def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_complexity_ratchet", _SCRIPT)
    assert spec and spec.loader, _SCRIPT
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves string annotations through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _long_function(name: str, lines: int) -> str:
    """Source of a module-level function whose body is exactly ``lines`` lines."""
    body = "".join(f"    value_{i} = {i}\n" for i in range(lines))
    return f"def {name}():\n{body}"


def _branchy_function(name: str, branches: int) -> str:
    """Source of a function with ``branches`` ifs, so complexity ``branches + 1``."""
    body = "".join(f"    if flag_{i}:\n        return {i}\n" for i in range(branches))
    args = ", ".join(f"flag_{i}" for i in range(branches))
    return f"def {name}({args}):\n{body}    return -1\n"


class _Tree:
    """A synthetic repository: one source root and one baseline directory."""

    def __init__(self, tmp_path: Path) -> None:
        self.base = tmp_path
        self.source = tmp_path / "pkg" / "mod.py"
        self.source.parent.mkdir()
        self.long_baseline = tmp_path / "baselines" / "long_functions.json"
        self.complex_baseline = tmp_path / "baselines" / "complex_functions.json"
        self.layout = _load().Layout(
            base=tmp_path,
            roots=(tmp_path / "pkg",),
            baseline_dir=tmp_path / "baselines",
        )

    def write_source(self, text: str) -> None:
        """Replace the single source file of the synthetic package."""
        self.source.write_text(text, encoding="utf-8")

    def write_baselines(
        self, long: dict[str, int] | None = None, complex_: dict[str, int] | None = None
    ) -> None:
        """Write both baseline files; an omitted one is an empty baseline."""
        self.long_baseline.parent.mkdir(exist_ok=True)
        self.long_baseline.write_text(json.dumps(long or {}), encoding="utf-8")
        self.complex_baseline.write_text(json.dumps(complex_ or {}), encoding="utf-8")

    def stored(self, baseline: Path) -> dict[str, int]:
        """Read back what a baseline file holds."""
        return json.loads(baseline.read_text(encoding="utf-8"))

    def run(self, *argv: str) -> int:
        """Run the script's ``main`` against this tree and return its exit code."""
        return _load().main(list(argv), layout=self.layout)

    def long_functions(self, limit: int = 30) -> dict[str, int]:
        """Measure the synthetic package with ``find_long_functions``."""
        return _load().find_long_functions([self.base / "pkg"], limit, base=self.base)


@pytest.fixture()
def tree(tmp_path: Path) -> _Tree:
    """An empty synthetic repository."""
    return _Tree(tmp_path)


def test_a_new_function_of_31_lines_is_a_violation(
    tree: _Tree, capsys: pytest.CaptureFixture[str]
) -> None:
    """A function over the limit that the baseline does not know fails the check."""
    tree.write_source(_long_function("too_long", 31))
    tree.write_baselines()

    assert tree.run() == 1

    assert "pkg/mod.py::too_long" in capsys.readouterr().err


def test_a_new_function_of_exactly_30_lines_is_accepted(tree: _Tree) -> None:
    """The standard allows 30 lines, so only 31 and above counts as too long."""
    tree.write_source(_long_function("at_the_limit", 30))
    tree.write_baselines()

    assert tree.run() == 0


def test_a_baselined_function_that_grew_by_one_line_is_a_violation(
    tree: _Tree, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ratchet only turns one way: a known offender may not get longer."""
    tree.write_source(_long_function("known", 41))
    tree.write_baselines(long={"pkg/mod.py::known": 40})

    assert tree.run() == 1

    report = capsys.readouterr().err
    assert "pkg/mod.py::known" in report
    assert "41" in report
    assert "40" in report


def test_a_baselined_function_at_its_recorded_size_is_accepted(tree: _Tree) -> None:
    """An unchanged offender passes: the debt is recorded, not forgiven."""
    tree.write_source(_long_function("known", 40))
    tree.write_baselines(long={"pkg/mod.py::known": 40})

    assert tree.run() == 0


def test_update_lowers_the_stored_value_of_a_function_that_shrank(tree: _Tree) -> None:
    """After a refactor ``--update`` records the smaller size as the new ceiling."""
    tree.write_source(_long_function("known", 35))
    tree.write_baselines(long={"pkg/mod.py::known": 40})

    assert tree.run("--update") == 0

    assert tree.stored(tree.long_baseline) == {"pkg/mod.py::known": 35}


def test_update_never_raises_a_stored_value(tree: _Tree) -> None:
    """A function that grew keeps its old ceiling, so the check still fails it."""
    tree.write_source(_long_function("known", 45))
    tree.write_baselines(long={"pkg/mod.py::known": 40})

    assert tree.run("--update") == 1

    assert tree.stored(tree.long_baseline) == {"pkg/mod.py::known": 40}


def test_update_never_adds_an_entry(tree: _Tree) -> None:
    """A new offender cannot be waved through by regenerating the baseline."""
    tree.write_source(_long_function("known", 40) + "\n\n" + _long_function("new", 33))
    tree.write_baselines(long={"pkg/mod.py::known": 40})

    assert tree.run("--update") == 1

    assert tree.stored(tree.long_baseline) == {"pkg/mod.py::known": 40}


def test_update_removes_an_entry_that_is_now_under_the_limit(tree: _Tree) -> None:
    """A function that shrank below 31 lines leaves the baseline for good."""
    tree.write_source(_long_function("known", 12))
    tree.write_baselines(long={"pkg/mod.py::known": 40, "pkg/mod.py::gone": 50})

    assert tree.run("--update") == 0

    assert tree.stored(tree.long_baseline) == {}


def test_a_stale_baseline_entry_does_not_fail_the_check(tree: _Tree) -> None:
    """An entry for a deleted or fixed function is only noise until ``--update``."""
    tree.write_source(_long_function("short", 5))
    tree.write_baselines(long={"pkg/mod.py::gone": 50}, complex_={"pkg/mod.py::x": 12})

    assert tree.run() == 0


def test_the_first_update_creates_both_baselines_from_the_measurement(
    tree: _Tree,
) -> None:
    """Without a baseline file the initial write records every current offender."""
    tree.write_source(
        _long_function("big", 33) + "\n\n" + _branchy_function("bushy", 10)
    )

    assert tree.run("--update") == 0

    assert tree.stored(tree.long_baseline) == {"pkg/mod.py::big": 33}
    assert tree.stored(tree.complex_baseline) == {"pkg/mod.py::bushy": 11}
    assert tree.run() == 0


def test_baselines_are_written_sorted_with_two_space_indent(tree: _Tree) -> None:
    """Stable formatting keeps the baseline diff down to the changed entries."""
    tree.write_source(_long_function("zeta", 31) + "\n\n" + _long_function("alpha", 32))

    tree.run("--update")

    assert tree.long_baseline.read_text(encoding="utf-8") == (
        '{\n  "pkg/mod.py::alpha": 32,\n  "pkg/mod.py::zeta": 31\n}\n'
    )


def test_a_new_function_of_complexity_10_is_a_violation(
    tree: _Tree, capsys: pytest.CaptureFixture[str]
) -> None:
    """The standard wants complexity below 10, so 10 already counts as complex."""
    tree.write_source(_branchy_function("bushy", 9))
    tree.write_baselines()

    assert tree.run() == 1

    report = capsys.readouterr().err
    assert "pkg/mod.py::bushy" in report
    assert "10" in report


def test_a_new_function_of_complexity_9_is_accepted(tree: _Tree) -> None:
    """Complexity 9 is below the standard's limit of 10."""
    tree.write_source(_branchy_function("calm", 8))
    tree.write_baselines()

    assert tree.run() == 0


def test_a_baselined_function_whose_complexity_grew_is_a_violation(tree: _Tree) -> None:
    """Complexity is ratcheted the same way as length."""
    tree.write_source(_branchy_function("bushy", 11))
    tree.write_baselines(complex_={"pkg/mod.py::bushy": 11})

    assert tree.run() == 1


def test_update_lowers_the_stored_complexity_of_a_function_that_improved(
    tree: _Tree,
) -> None:
    """``--update`` also shrinks the complexity baseline."""
    tree.write_source(_branchy_function("bushy", 10))
    tree.write_baselines(complex_={"pkg/mod.py::bushy": 14})

    assert tree.run("--update") == 0

    assert tree.stored(tree.complex_baseline) == {"pkg/mod.py::bushy": 11}


def test_find_long_functions_excludes_the_docstring_from_the_count(
    tree: _Tree,
) -> None:
    """A long docstring must not make a short function an offender."""
    docstring = "".join(f"    line {i}\n" for i in range(40))
    tree.write_source(
        f'def documented():\n    """Summary.\n\n{docstring}    """\n    return 1\n'
    )

    assert tree.long_functions(limit=0) == {"pkg/mod.py::documented": 1}


def test_find_long_functions_names_methods_and_nested_functions(tree: _Tree) -> None:
    """Keys carry the qualified name so they survive edits elsewhere in the file."""
    method = textwrap.indent(_long_function("method", 32), "    ")
    inner = textwrap.indent(_long_function("inner", 31), "    ")
    tree.write_source(
        f"class Service:\n{method}\n\ndef outer():\n{inner}    return inner\n"
    )

    found = tree.long_functions()

    assert found["pkg/mod.py::Service.method"] == 32
    assert found["pkg/mod.py::outer.inner"] == 31
    assert found["pkg/mod.py::outer"] == 33


def test_find_long_functions_counts_async_functions(tree: _Tree) -> None:
    """``async def`` is a function like any other."""
    tree.write_source(_long_function("job", 31).replace("def job", "async def job"))

    assert tree.long_functions() == {"pkg/mod.py::job": 31}


def test_find_long_functions_keeps_the_longest_of_two_same_named_functions(
    tree: _Tree,
) -> None:
    """A property getter and setter share a qualified name; the larger one counts."""
    getter = textwrap.indent(_long_function("size", 33), "    ")
    setter = textwrap.indent(_long_function("size", 40), "    ")
    tree.write_source(f"class Box:\n{getter}\n{setter}")

    assert tree.long_functions() == {"pkg/mod.py::Box.size": 40}


def test_find_complex_functions_reports_ruff_c901_by_qualified_name(
    tree: _Tree,
) -> None:
    """Ruff gives the row; the AST turns it into a ``Class.method`` key."""
    method = textwrap.indent(_branchy_function("decide", 11), "    ")
    tree.write_source(f"class Policy:\n{method}\n{_branchy_function('calm', 8)}")

    found = _load().find_complex_functions([tree.base / "pkg"], base=tree.base)

    assert found == {"pkg/mod.py::Policy.decide": 12}


def test_an_unparsable_source_file_stops_the_run(tree: _Tree) -> None:
    """Silently skipping a file would let its offenders through."""
    tree.write_source("def broken(:\n    pass\n")
    tree.write_baselines()

    with pytest.raises(SystemExit, match=r"pkg[/\\]mod\.py"):
        tree.run()


def test_the_committed_baselines_are_sorted_json_of_real_offenders() -> None:
    """Both baselines exist, hold entries and are formatted as ``--update`` writes."""
    ratchet = _load()
    assert ratchet.MAX_LINES == 30
    assert ratchet.MAX_COMPLEXITY == 9

    for name, floor in (("long_functions", 30), ("complex_functions", 9)):
        text = (_BASELINES / f"{name}.json").read_text(encoding="utf-8")
        stored = json.loads(text)

        assert stored, name
        assert text == json.dumps(stored, indent=2, sort_keys=True) + "\n"
        assert all(isinstance(v, int) and v > floor for v in stored.values()), name


def test_the_committed_baselines_accept_the_tree_as_it_stands() -> None:
    """The check CI runs passes on this checkout, so the lint job starts green."""
    assert _load().main([]) == 0


def test_ci_lint_job_runs_the_ratchet_after_the_format_check() -> None:
    """The step belongs to ``lint`` and follows ``ruff format --check``."""
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    commands = [s["run"] for s in workflow["jobs"]["lint"]["steps"] if "run" in s]
    ratchet = "uv run python scripts/ci/check_complexity_ratchet.py"

    assert commands.count(ratchet) == 1
    format_step = next(i for i, c in enumerate(commands) if "ruff format" in c)
    assert commands.index(ratchet) == format_step + 1
