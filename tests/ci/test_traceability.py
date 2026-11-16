"""The requirement traceability check: tests cite registered ids, and every id has one.

Validates ``scripts/ci/traceability.py``: ``collect`` and the pytest plugin that
writes one CSV row per (``criterion`` id, collected test), ``parse_registry`` over the
table in ``docs/requirements.md``, ``find_holes`` and the ``check`` command the lint
job runs. The unit tests use hand-built items, registries and CSV files in a
temporary directory; the committed registry and the four suite configs are read
only by the tests that say so.
"""

from __future__ import annotations

import configparser
import csv
import functools
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import tomllib
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "traceability.py"
_REGISTRY = _REPO_ROOT / "docs" / "requirements.md"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_HEADER = "| id | statement | module | standard |\n|---|---|---|---|\n"
_CSV_HEADER = "req_id,nodeid,module,defects\n"
_SUITE_DIRS = ("", "ingestion", "packages/portopt-db", "fund")
_CRITERION_HELP = (
    "criterion(id): cite requirement <id> of docs/requirements.md (a REQ- id)"
    " or an acceptance-criterion label"
)
_DEFECT_HELP = "defect(issue): link this regression test to GitHub issue <issue>"


@functools.cache
def _load() -> ModuleType:
    # A name of its own, so the copy under test never replaces the plugin that the
    # root suite's own config already loaded as ``traceability``.
    name = "traceability_under_test"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec and spec.loader, _SCRIPT
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves string annotations through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Item:
    """The slice of a collected ``pytest.Item`` that ``collect`` reads."""

    def __init__(self, nodeid: str, path: Path, *marks: Any) -> None:
        self.nodeid = nodeid
        self.path = path
        self._marks = [mark.mark for mark in marks]

    def iter_markers(self, name: str | None = None) -> Iterator[Any]:
        """Yield the item's marks named ``name``, in the order they were given."""
        return (mark for mark in self._marks if name is None or mark.name == name)


def _item(nodeid: str, *marks: Any) -> _Item:
    return _Item(nodeid, _REPO_ROOT / "fund" / nodeid.split("::")[0], *marks)


def _registry(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "requirements.md"
    path.write_text(f"# Requirements\n\n{_HEADER}{rows}\nTrailing prose.\n")
    return path


def _row(req_id: str) -> str:
    return f"| {req_id} | A statement. | `fund/src/fund/tools/risk.py` | 6.2 |\n"


def _trace(tmp_path: Path, rows: str, name: str = "trace.csv") -> Path:
    path = tmp_path / name
    path.write_text(_CSV_HEADER + rows)
    return path


_NO_TABLE = "# Requirements\n\n| id | text |\n|---|---|\n"
_BROKEN_TABLES = [
    ("", "no requirement rows"),
    ("| REQ-FUND-001 | A statement. | `m.py` |\n", "4 cells"),
    ("| REQ-FUND-1 | A statement. | `m.py` | 6.2 |\n", "REQ-FUND-1"),
    ("| REQ-WEB-001 | A statement. | `m.py` | 6.2 |\n", "REQ-WEB-001"),
    ("| REQ-FUND-001 |  | `m.py` | 6.2 |\n", "empty"),
    (_row("REQ-FUND-001") + _row("REQ-FUND-001"), "twice"),
]
_BROKEN_IDS = ["empty", "short-row", "short-number", "unknown-area", "blank", "dup"]


def _check(registry: Path, *csvs: Path) -> int:
    argv = ["check", "--requirements", str(registry)]
    for path in csvs:
        argv += ["--csv", str(path)]
    return _load().main(argv)


@pytest.mark.criterion("REQ-OPS-001")
class TestCollect:
    """``collect`` turns the markers of collected items into trace rows."""

    def test_one_row_per_id_and_test(self) -> None:
        """Each (id, test) pair is one row, in item then marker order."""
        mark = pytest.mark
        items = [
            _item("tests/test_a.py::test_one", mark.criterion("REQ-FUND-001")),
            _item(
                "tests/test_a.py::test_two",
                mark.criterion("REQ-FUND-002"),
                mark.criterion("REQ-FUND-001"),
            ),
            _item("tests/test_a.py::test_unbound"),
        ]

        rows = _load().collect(items, root=_REPO_ROOT)

        assert [(row.req_id, row.nodeid) for row in rows] == [
            ("REQ-FUND-001", "tests/test_a.py::test_one"),
            ("REQ-FUND-002", "tests/test_a.py::test_two"),
            ("REQ-FUND-001", "tests/test_a.py::test_two"),
        ]

    def test_an_id_on_both_the_class_and_the_test_is_one_row(self) -> None:
        """The same id reached twice through the node tree is not repeated."""
        item = _item(
            "tests/test_a.py::TestA::test_one",
            pytest.mark.criterion("REQ-FUND-001"),
            pytest.mark.criterion("REQ-FUND-001"),
        )

        rows = _load().collect([item], root=_REPO_ROOT)

        assert [row.req_id for row in rows] == ["REQ-FUND-001"]

    def test_module_is_the_test_file_relative_to_the_root(self) -> None:
        """Rows from the four suites stay apart although their node ids may clash."""
        item = _item("tests/unit/test_a.py::test_one", pytest.mark.criterion("x-1"))

        (row,) = _load().collect([item], root=_REPO_ROOT)

        assert row.module == "fund/tests/unit/test_a.py"
        assert row.nodeid == "tests/unit/test_a.py::test_one"

    def test_a_defect_rides_on_every_row_of_its_test(self) -> None:
        """The issue numbers are carried, sorted, on each requirement row."""
        item = _item(
            "tests/test_a.py::test_one",
            pytest.mark.criterion("REQ-FUND-001"),
            pytest.mark.criterion("scope-1"),
            pytest.mark.defect(1200),
            pytest.mark.defect(1125),
        )

        rows = _load().collect([item], root=_REPO_ROOT)

        assert [(row.req_id, row.defects) for row in rows] == [
            ("REQ-FUND-001", (1125, 1200)),
            ("scope-1", (1125, 1200)),
        ]

    def test_a_defect_without_a_criterion_gets_a_row_with_no_id(self) -> None:
        """The row exists so the check can refuse a defect linked to no requirement."""
        item = _item("tests/test_a.py::test_one", pytest.mark.defect(1125))

        (row,) = _load().collect([item], root=_REPO_ROOT)

        assert (row.req_id, row.defects) == ("", (1125,))

    @pytest.mark.parametrize(
        "mark",
        [
            pytest.mark.criterion(),
            pytest.mark.criterion("REQ-FUND-001", "REQ-FUND-002"),
            pytest.mark.criterion(""),
            pytest.mark.defect("1125"),
            pytest.mark.defect(0),
            pytest.mark.defect(True),
        ],
        ids=["no-id", "two-ids", "blank-id", "text-issue", "zero-issue", "bool-issue"],
    )
    def test_a_marker_without_one_valid_argument_is_refused(self, mark: Any) -> None:
        """A malformed marker fails the collection pass instead of vanishing."""
        with pytest.raises(ValueError, match=re.escape("tests/test_a.py::test_one")):
            _load().collect([_item("tests/test_a.py::test_one", mark)], root=_REPO_ROOT)


@pytest.mark.criterion("REQ-OPS-001")
class TestCsv:
    """The CSV the plugin writes is the CSV the check reads."""

    def test_round_trip_keeps_every_row(self, tmp_path: Path) -> None:
        """Writing then reading gives the same rows, defects included."""
        module = _load()
        rows = [
            module.TraceRow("REQ-FUND-001", "tests/a.py::t", "fund/tests/a.py"),
            module.TraceRow("", "tests/a.py::u", "fund/tests/a.py", (1125, 1200)),
        ]
        path = tmp_path / "trace.csv"

        module.write_csv(rows, path)

        assert module.read_csv(path) == rows

    def test_the_file_has_one_line_per_row_under_a_header(self, tmp_path: Path) -> None:
        """The header names the four columns; an empty ``defects`` cell means none."""
        module = _load()
        path = tmp_path / "trace.csv"

        module.write_csv(
            [module.TraceRow("REQ-FUND-001", "tests/a.py::t", "fund/tests/a.py")], path
        )

        with path.open(newline="", encoding="utf-8") as handle:
            assert list(csv.reader(handle)) == [
                ["req_id", "nodeid", "module", "defects"],
                ["REQ-FUND-001", "tests/a.py::t", "fund/tests/a.py", ""],
            ]

    def test_a_file_with_other_columns_is_refused(self, tmp_path: Path) -> None:
        """A CSV from another tool is not read as an empty matrix."""
        path = tmp_path / "trace.csv"
        path.write_text("id,test\nREQ-FUND-001,tests/a.py::t\n")

        with pytest.raises(ValueError, match="columns"):
            _load().read_csv(path)


@pytest.mark.criterion("REQ-OPS-001")
class TestParseRegistry:
    """``parse_registry`` reads the requirements table and refuses a broken one."""

    def test_reads_each_row(self, tmp_path: Path) -> None:
        """Every cell is kept, with the backticks of the module cell dropped."""
        text = _registry(tmp_path, _row("REQ-FUND-001") + _row("REQ-DB-002"))

        requirements = _load().parse_registry(text.read_text())

        assert [r.req_id for r in requirements] == ["REQ-FUND-001", "REQ-DB-002"]
        assert requirements[0].statement == "A statement."
        assert requirements[0].module == "fund/src/fund/tools/risk.py"
        assert requirements[0].standard == "6.2"

    @pytest.mark.parametrize(("rows", "message"), _BROKEN_TABLES, ids=_BROKEN_IDS)
    def test_a_broken_table_is_refused(
        self, tmp_path: Path, rows: str, message: str
    ) -> None:
        """An empty table or a short, misnumbered, blank or repeated row is refused."""
        module = _load()
        text = _registry(tmp_path, rows).read_text()

        with pytest.raises(module.RegistryError, match=message):
            module.parse_registry(text)

    def test_a_document_without_the_table_is_refused(self) -> None:
        """A renamed header must not read as a registry with nothing in it."""
        module = _load()

        with pytest.raises(module.RegistryError, match="no requirements table"):
            module.parse_registry(_NO_TABLE)


@pytest.mark.criterion("REQ-OPS-001")
class TestFindHoles:
    """``find_holes`` names each way the matrix can be incomplete."""

    @staticmethod
    def _requirements(*ids: str) -> list[Any]:
        module = _load()
        return [module.Requirement(i, "s", "m.py", "6.2") for i in ids]

    @staticmethod
    def _rows(*pairs: tuple[str, tuple[int, ...]]) -> list[Any]:
        module = _load()
        return [
            module.TraceRow(req_id, f"tests/a.py::t{n}", "fund/tests/a.py", defects)
            for n, (req_id, defects) in enumerate(pairs)
        ]

    def test_a_complete_matrix_has_no_hole(self) -> None:
        """Every id cited, every citation registered, every defect linked."""
        holes = _load().find_holes(
            self._requirements("REQ-FUND-001"),
            self._rows(("REQ-FUND-001", (1125,))),
        )

        assert holes == []

    def test_a_registered_id_without_a_test_is_a_hole(self) -> None:
        """The message names the id nobody cites."""
        holes = _load().find_holes(
            self._requirements("REQ-FUND-001", "REQ-FUND-002"),
            self._rows(("REQ-FUND-001", ())),
        )

        assert len(holes) == 1
        assert "REQ-FUND-002" in holes[0]

    def test_an_unregistered_requirement_id_is_a_hole(self) -> None:
        """The message names the test and the id it cites."""
        holes = _load().find_holes(
            self._requirements("REQ-FUND-001"),
            self._rows(("REQ-FUND-001", ()), ("REQ-FUND-009", ())),
        )

        assert holes == [
            "fund/tests/a.py::t1 cites REQ-FUND-009, which the registry does not list"
        ]

    def test_a_label_that_is_not_a_requirement_id_is_not_looked_up(self) -> None:
        """Acceptance-criterion labels such as ``scope-3`` share the marker."""
        holes = _load().find_holes(
            self._requirements("REQ-FUND-001"),
            self._rows(("REQ-FUND-001", ()), ("scope-3", ()), ("global-6", ())),
        )

        assert holes == []

    def test_a_defect_on_a_test_with_no_requirement_id_is_a_hole(self) -> None:
        """A defect must be linked to a registered requirement through its test."""
        holes = _load().find_holes(
            self._requirements("REQ-FUND-001"),
            self._rows(("REQ-FUND-001", ()), ("scope-1", (1125,)), ("", (1200,))),
        )

        assert holes == [
            "fund/tests/a.py::t1 carries defect 1125 but cites no requirement id",
            "fund/tests/a.py::t2 carries defect 1200 but cites no requirement id",
        ]


@pytest.mark.criterion("REQ-OPS-001")
class TestCheckCommand:
    """``traceability.py check`` exits 1 on any hole and 0 on a complete matrix."""

    def test_a_complete_matrix_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A one-line summary goes to standard output and nothing to standard error."""
        registry = _registry(tmp_path, _row("REQ-FUND-001"))
        trace = _trace(tmp_path, "REQ-FUND-001,tests/a.py::t,fund/tests/a.py,\n")

        assert _check(registry, trace) == 0
        captured = capsys.readouterr()
        assert "no holes" in captured.out
        assert captured.err == ""

    def test_the_csv_files_of_several_suites_are_merged(self, tmp_path: Path) -> None:
        """Each suite writes its own file; the check reads them as one matrix."""
        registry = _registry(tmp_path, _row("REQ-FUND-001") + _row("REQ-DB-001"))
        fund = _trace(
            tmp_path, "REQ-FUND-001,tests/a.py::t,fund/tests/a.py,\n", "f.csv"
        )
        db = _trace(
            tmp_path, "REQ-DB-001,tests/a.py::t,packages/db/tests/a.py,\n", "d.csv"
        )

        assert _check(registry, fund) == 1
        assert _check(registry, fund, db) == 0

    @pytest.mark.parametrize(
        "rows",
        [
            "REQ-FUND-001,tests/a.py::t,fund/tests/a.py,\n"
            "REQ-FUND-009,tests/a.py::u,fund/tests/a.py,\n",
            "scope-1,tests/a.py::t,fund/tests/a.py,\n",
            "REQ-FUND-001,tests/a.py::t,fund/tests/a.py,\n,tests/a.py::u,x.py,1125\n",
        ],
        ids=["unknown-id", "id-without-test", "unlinked-defect"],
    )
    def test_a_hole_exits_one_and_is_named(
        self, tmp_path: Path, rows: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The hole is printed on standard error."""
        registry = _registry(tmp_path, _row("REQ-FUND-001"))

        assert _check(registry, _trace(tmp_path, rows)) == 1
        assert capsys.readouterr().err.strip()

    @pytest.mark.parametrize(("rows", "message"), _BROKEN_TABLES, ids=_BROKEN_IDS)
    def test_a_broken_registry_exits_one(
        self,
        tmp_path: Path,
        rows: str,
        message: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A broken table fails the check with the parser's reason, never vacuously."""
        trace = _trace(tmp_path, "REQ-FUND-001,tests/a.py::t,fund/tests/a.py,\n")

        assert _check(_registry(tmp_path, rows), trace) == 1
        assert message in capsys.readouterr().err

    def test_a_registry_without_the_table_exits_one(self, tmp_path: Path) -> None:
        """A renamed table header fails the check instead of reading as empty."""
        registry = tmp_path / "requirements.md"
        registry.write_text(_NO_TABLE)
        trace = _trace(tmp_path, "REQ-FUND-001,tests/a.py::t,fund/tests/a.py,\n")

        assert _check(registry, trace) == 1

    def test_a_missing_csv_exits_one(self, tmp_path: Path) -> None:
        """A suite whose collection pass wrote nothing is not an empty suite."""
        registry = _registry(tmp_path, _row("REQ-FUND-001"))

        assert _check(registry, tmp_path / "absent.csv") == 1

    def test_a_csv_with_other_columns_exits_one(self, tmp_path: Path) -> None:
        """A file another tool wrote is refused, not read as an empty matrix."""
        registry = _registry(tmp_path, _row("REQ-FUND-001"))
        other = tmp_path / "other.csv"
        other.write_text("id,test\nREQ-FUND-001,tests/a.py::t\n")

        assert _check(registry, other) == 1

    def test_a_malformed_csv_exits_one_with_a_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file the csv reader rejects ends in a ``traceability:`` line."""
        registry = _registry(tmp_path, _row("REQ-FUND-001"))
        oversized = "x" * (csv.field_size_limit() + 1)
        trace = _trace(tmp_path, f"REQ-FUND-001,{oversized},fund/tests/a.py,\n")

        assert _check(registry, trace) == 1
        assert capsys.readouterr().err.startswith("traceability: ")

    def test_a_missing_argument_is_a_usage_error(self, tmp_path: Path) -> None:
        """``--csv`` is required; argparse exits 2."""
        registry = _registry(tmp_path, _row("REQ-FUND-001"))

        with pytest.raises(SystemExit) as raised:
            _load().main(["check", "--requirements", str(registry)])

        assert raised.value.code == 2


@pytest.mark.criterion("REQ-OPS-001")
class TestPlugin:
    """The plugin writes the CSV from a real collection pass."""

    def _collect(self, tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        (tmp_path / "pytest.ini").write_text(
            "[pytest]\nmarkers =\n    criterion(id): x\n    defect(issue): x\n"
        )
        (tmp_path / "test_sample.py").write_text(
            textwrap.dedent(
                """
                import pytest

                @pytest.mark.criterion("REQ-OPS-001")
                class TestA:
                    \"\"\"A class whose marker reaches each test.\"\"\"

                    @pytest.mark.defect(1125)
                    @pytest.mark.parametrize("n", [1, 2])
                    def test_one(self, n):
                        \"\"\"A parametrized test with an issue.\"\"\"

                def test_unbound():
                    \"\"\"A test that cites nothing.\"\"\"
                """
            )
        )
        env = {**os.environ, "PYTHONPATH": str(_SCRIPT.parent)}
        return subprocess.run(  # noqa: S603 - fixed argv, no user input
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "traceability",
                "-p",
                "no:cacheprovider",
                *extra,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_the_option_writes_one_row_per_id_and_test(self, tmp_path: Path) -> None:
        """Class markers reach each parametrized test; unbound tests get no row."""
        out = tmp_path / "out" / "trace.csv"

        result = self._collect(tmp_path, f"--traceability-out={out}")

        assert result.returncode == 0, result.stdout + result.stderr
        rows = _load().read_csv(out)
        assert [(row.req_id, row.nodeid, row.defects) for row in rows] == [
            ("REQ-OPS-001", "test_sample.py::TestA::test_one[1]", (1125,)),
            ("REQ-OPS-001", "test_sample.py::TestA::test_one[2]", (1125,)),
        ]

    def test_without_the_option_nothing_is_written(self, tmp_path: Path) -> None:
        """A normal run leaves no file behind."""
        result = self._collect(tmp_path)

        assert result.returncode == 0, result.stdout + result.stderr
        assert list(tmp_path.rglob("*.csv")) == []


def _suite_options(suite: str) -> dict[str, str]:
    """``addopts``, ``pythonpath`` and ``markers`` of a suite's pytest config."""
    directory = _REPO_ROOT / suite
    ini = directory / "pytest.ini"
    if ini.is_file():
        parser = configparser.ConfigParser()
        parser.read(ini, encoding="utf-8")
        return {
            key: parser.get("pytest", key)
            for key in ("addopts", "pythonpath", "markers")
        }
    options = tomllib.loads((directory / "pyproject.toml").read_text(encoding="utf-8"))
    ini_options = options["tool"]["pytest"]["ini_options"]
    return {
        "addopts": ini_options.get("addopts", ""),
        "pythonpath": " ".join(ini_options.get("pythonpath", [])),
        "markers": "\n".join(ini_options.get("markers", [])),
    }


@pytest.mark.criterion("REQ-OPS-002")
@pytest.mark.parametrize("suite", _SUITE_DIRS, ids=lambda s: s or "portopt-core")
def test_every_suite_config_loads_the_plugin_and_registers_both_markers(
    suite: str,
) -> None:
    """The four suites load the one plugin file and describe the markers alike."""
    options = _suite_options(suite)

    assert "-p traceability" in options["addopts"]
    (path,) = options["pythonpath"].split()
    assert (_REPO_ROOT / suite / path).resolve() == _SCRIPT.parent
    assert _CRITERION_HELP in options["markers"]
    assert _DEFECT_HELP in options["markers"]


@pytest.mark.criterion("REQ-OPS-002")
def test_the_committed_registry_parses_and_names_existing_modules() -> None:
    """Every registered module path exists, so moving or deleting one fails here."""
    requirements = _load().parse_registry(_REGISTRY.read_text(encoding="utf-8"))

    missing = [r.module for r in requirements if not (_REPO_ROOT / r.module).exists()]

    assert requirements
    assert missing == []


@pytest.mark.criterion("REQ-OPS-002")
def test_the_lint_job_collects_each_suite_and_checks_the_merged_trace() -> None:
    """A hole fails ``lint``, a required check, and the four files are uploaded."""
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["lint"]["steps"]
    (step,) = [s for s in steps if "traceability.py check" in s.get("run", "")]
    lines = step["run"].splitlines()
    trace_names = {
        "": "core",
        "ingestion": "ingestion",
        "packages/portopt-db": "portopt-db",
        "fund": "fund",
    }

    for suite, name in trace_names.items():
        output = f'--traceability-out="$TRACE/{name}.csv"'
        (collect,) = [line for line in lines if output in line]
        assert "--collect-only" in collect
        assert (f"cd {suite} &&" in collect) is bool(suite)
        assert f'--csv "$TRACE/{name}.csv"' in step["run"]
    (upload,) = [
        s for s in steps if s.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert step["env"]["TRACE"] == "${{ runner.temp }}/traceability"
    assert upload["with"]["path"].rstrip("/") == step["env"]["TRACE"]
