"""Guard: no new ORM or SQL statement call outside the ``fund`` repositories.

Data access belongs in a repository, so a ``fund`` module that builds or runs a
statement itself bypasses that layer. ``orm_baseline.txt`` lists the calls that still
do so. The test fails on a call the baseline does not list and on a baseline entry the
tree no longer matches, so moving a call into a repository forces its entry out of
the file, or its ``xN`` count down, and a new call fails the test. The baseline's
header says never to add a line.

The scan parses the syntax tree of every ``.py`` file under ``fund/src/fund`` except
the repository modules: the files directly under ``fund/audit/`` whose name ends in
``repository.py``. It counts these calls, at module level or inside a class, method,
function, nested function or lambda:

* ``select(``, ``insert(``, ``update(``, ``delete(`` or ``text(`` called through a
  name that a ``from sqlalchemy... import`` statement in the same module binds (any
  ``sqlalchemy`` submodule, ``as`` aliases followed, the import at module level or
  inside a function), or called as ``<alias>.select(`` and so on, where ``<alias>`` is
  bound by ``import sqlalchemy [as alias]``. The bound name is matched across the
  whole module, without scope analysis.
* ``.execute(``, ``.query(`` and ``.add_all(`` on any receiver, so a renamed session
  (``own``, ``repo.session``) and a psycopg connection count too.
* ``.scalar(`` and ``.scalars(`` with at least one positional argument (the
  ``Session.scalar(stmt)`` form). The argument-free ``Result.scalar()`` and
  ``Result.scalars()`` that follow an ``.execute(`` are not counted.
* ``.get(Model, ...)`` and ``.add(Model(...))`` on any receiver, where ``Model`` is a
  name that a ``from portopt_db.models... import`` statement in the same module binds.

It does not detect: a constructor reached any other way (a re-bound name such as
``sel = select``, ``getattr``, a parameter, ``sa.sql.select`` through a submodule
attribute); ``.get`` or ``.add`` whose argument is a variable, an attribute such as
``models.AgentRun``, a model class not imported by name, or passed by keyword
(``.get(entity=Model, ...)``); ``session.merge``, ``session.delete``,
``session.refresh``, ``session.flush``, ``session.commit``, ``session.rollback`` and
``session.bulk_save_objects``; ``cursor.executemany``; ``exec_driver_sql``,
``pandas.read_sql`` and ``DataFrame.to_sql``; SQL a lazy-loaded
relationship emits; a call made by code outside ``fund/src/fund``, such as a
``portopt_db`` repository. When a statement is built in one function and run in
another, each detected call is keyed under the function it sits in. A docstring or
comment that mentions a call is not a hit. The scan imports no implementation module.

Each hit is keyed ``<module>::<qualified name>::<kind>``: the dotted module, the
dotted chain of enclosing classes and functions (``Class.method``, ``outer.inner``;
``<module>`` at module level), and the kind (``select``, ``.execute``, ``.get`` ...).
The key holds no line number, so an edit that moves a call within its function does
not break the baseline; moving it to another function or class changes its key and
fails the test.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

# hygiene -> unit -> tests -> fund, then the src layout's import root.
_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
_BASELINE = Path(__file__).with_name("orm_baseline.txt")

_CONSTRUCTORS = frozenset({"select", "insert", "update", "delete", "text"})
_ANY_ARGUMENT_METHODS = frozenset({"execute", "query", "add_all"})
_STATEMENT_ARGUMENT_METHODS = frozenset({"scalar", "scalars"})
_MODULE_LEVEL = "<module>"
# A floor well below the number of modules ``fund`` has.
_MIN_SCANNED_MODULES = 50

_ScopeNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def is_repository_module(path: Path, src_root: Path) -> bool:
    """Return whether ``path`` is a ``fund/audit/*repository.py`` module.

    Args:
        path: A ``.py`` file under ``src_root``.
        src_root: The directory that holds the top-level ``fund`` package.
    """
    return path.parent == src_root / "fund" / "audit" and path.name.endswith(
        "repository.py"
    )


def _dotted_module(path: Path, src_root: Path) -> str:
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_package(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


class _Bindings:
    """The import-bound names the call rules look up, collected over one module."""

    def __init__(self, tree: ast.AST) -> None:
        self.constructors: dict[str, str] = {}
        self.sqlalchemy_aliases: set[str] = set()
        self.models: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                self._bind_from(node.module, node.names)
            elif isinstance(node, ast.Import):
                self.sqlalchemy_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "sqlalchemy"
                )

    def _bind_from(self, module: str, names: list[ast.alias]) -> None:
        if _is_package(module, "sqlalchemy"):
            for alias in names:
                if alias.name in _CONSTRUCTORS:
                    self.constructors[alias.asname or alias.name] = alias.name
        elif _is_package(module, "portopt_db.models"):
            self.models.update(alias.asname or alias.name for alias in names)


def _model_argument_kind(
    method: str, first: ast.expr, bindings: _Bindings
) -> str | None:
    """Return ``.get`` for ``.get(Model, ...)``, ``.add`` for ``.add(Model(...))``."""
    if method == "get" and isinstance(first, ast.Name):
        return ".get" if first.id in bindings.models else None
    if (
        method == "add"
        and isinstance(first, ast.Call)
        and isinstance(first.func, ast.Name)
    ):
        return ".add" if first.func.id in bindings.models else None
    return None


def _method_kind(
    node: ast.Call, callee: ast.Attribute, bindings: _Bindings
) -> str | None:
    method = callee.attr
    if method in _CONSTRUCTORS and isinstance(callee.value, ast.Name):
        return method if callee.value.id in bindings.sqlalchemy_aliases else None
    if method in _ANY_ARGUMENT_METHODS:
        return f".{method}"
    if not node.args:
        return None
    if method in _STATEMENT_ARGUMENT_METHODS:
        return f".{method}"
    return _model_argument_kind(method, node.args[0], bindings)


def _call_kind(node: ast.Call, bindings: _Bindings) -> str | None:
    """Return the kind of statement call ``node`` is, or ``None`` when it is none."""
    callee = node.func
    if isinstance(callee, ast.Name):
        return bindings.constructors.get(callee.id)
    if isinstance(callee, ast.Attribute):
        return _method_kind(node, callee, bindings)
    return None


def _collect(
    node: ast.AST, scope: tuple[str, ...], bindings: _Bindings, sites: list[str]
) -> None:
    """Append ``qualified name::kind`` for each call under ``node`` that
    ``_call_kind`` recognises."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _ScopeNode):
            _collect(child, (*scope, child.name), bindings, sites)
            continue
        if isinstance(child, ast.Call):
            kind = _call_kind(child, bindings)
            if kind is not None:
                sites.append(f"{'.'.join(scope) or _MODULE_LEVEL}::{kind}")
        _collect(child, scope, bindings, sites)


def find_orm_sites(src_root: Path) -> Counter[str]:
    """Count the statement calls in the non-repository modules of ``src_root/fund``.

    Args:
        src_root: The directory that holds the top-level ``fund`` package; each
            ``.py`` file under ``src_root/fund`` at any depth is scanned, except the
            files directly under ``src_root/fund/audit`` whose name ends in
            ``repository.py``.

    Returns:
        ``<module>::<qualified name>::<kind>`` keys with how often each occurs;
        empty when no scanned module makes such a call.
    """
    found: Counter[str] = Counter()
    for path in sorted((src_root / "fund").rglob("*.py")):
        if is_repository_module(path, src_root):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        sites: list[str] = []
        _collect(tree, (), _Bindings(tree), sites)
        module = _dotted_module(path, src_root)
        found.update(f"{module}::{site}" for site in sites)
    return found


def parse_baseline(text: str) -> Counter[str]:
    """Read the baseline: one ``key`` or ``key xN`` per line, ``#`` comments skipped.

    Args:
        text: The baseline file's content.

    Returns:
        Each key with its count, 1 for a line without ``xN``.

    Raises:
        ValueError: A key appears twice, or a count is not ``x`` and a whole number
            of at least 2.
    """
    entries: Counter[str] = Counter()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, suffix = line.partition(" ")
        count = 1
        if suffix:
            digits = suffix.strip().removeprefix("x")
            if digits == suffix.strip() or not digits.isdigit() or int(digits) < 2:
                raise ValueError(f"bad count in baseline line {line!r}")
            count = int(digits)
        if key in entries:
            raise ValueError(f"duplicate baseline key {key!r}")
        entries[key] = count
    return entries


def _format(entries: Counter[str]) -> str:
    return "\n".join(
        key if count == 1 else f"{key} x{count}"
        for key, count in sorted(entries.items())
    )


def test_the_fund_package_is_where_the_guard_looks() -> None:
    """A moved package must fail here, not pass a scan that read no file."""
    assert (_SRC_ROOT / "fund" / "__init__.py").is_file()
    assert len(list((_SRC_ROOT / "fund").rglob("*.py"))) >= _MIN_SCANNED_MODULES


def test_the_real_repository_modules_are_exempt() -> None:
    """The repositories build statements, and none of them reaches the result."""
    repositories = [
        path
        for path in (_SRC_ROOT / "fund" / "audit").glob("*.py")
        if is_repository_module(path, _SRC_ROOT)
    ]
    assert any("select(" in path.read_text(encoding="utf-8") for path in repositories)
    exempt = {_dotted_module(path, _SRC_ROOT) for path in repositories}
    assert not {key.split("::")[0] for key in find_orm_sites(_SRC_ROOT)} & exempt


def test_statement_calls_outside_the_repositories_match_the_baseline() -> None:
    """A call the baseline lacks is new; an entry the tree lacks is stale."""
    found = find_orm_sites(_SRC_ROOT)
    allowed = parse_baseline(_BASELINE.read_text(encoding="utf-8"))
    new = found - allowed
    stale = allowed - found
    assert not new, (
        "move these ORM/SQL calls into a fund/audit or portopt_db repository "
        "(the count is the excess over orm_baseline.txt):\n" + _format(new)
    )
    assert not stale, (
        "these orm_baseline.txt entries no longer match the tree; lower or delete "
        "them (the count is the surplus):\n" + _format(stale)
    )


def _scan(
    tmp_path: Path, source: str, relative: str = "fund/offender.py"
) -> Counter[str]:
    """Write ``source`` at ``relative`` under ``tmp_path`` and scan that tree."""
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return find_orm_sites(tmp_path)


def test_an_empty_tree_reports_nothing(tmp_path: Path) -> None:
    """A tree with no module yields no key."""
    (tmp_path / "fund").mkdir()
    assert find_orm_sites(tmp_path) == Counter()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from sqlalchemy import select\n"
            "def read(session):\n"
            "    return session.execute(select(1)).scalars().all()\n",
            ["fund.offender::read::select", "fund.offender::read::.execute"],
        ),
        (
            "def read(session):\n"
            "    from sqlalchemy import update as sa_update\n"
            "    session.execute(sa_update(T))\n",
            ["fund.offender::read::update", "fund.offender::read::.execute"],
        ),
        (
            "from sqlalchemy.dialects.postgresql import insert\nstmt = insert(T)\n",
            ["fund.offender::<module>::insert"],
        ),
        (
            "import sqlalchemy as sa\nclass Repo:\n"
            "    def drop(self):\n        return sa.delete(T)\n",
            ["fund.offender::Repo.drop::delete"],
        ),
        (
            "from sqlalchemy import text\n"
            "def mode(s):\n    s.execute(text('SET TRANSACTION READ ONLY'))\n",
            ["fund.offender::mode::text", "fund.offender::mode::.execute"],
        ),
        (
            "def outer(repo):\n    def inner():\n"
            "        repo.session.execute(stmt)\n    return inner\n",
            ["fund.offender::outer.inner::.execute"],
        ),
        (
            "async def read(conn):\n    await conn.execute('SELECT 1')\n",
            ["fund.offender::read::.execute"],
        ),
        (
            "def read(session):\n    return session.scalars(stmt)\n",
            ["fund.offender::read::.scalars"],
        ),
        (
            "def read(session):\n    return session.scalar(stmt)\n",
            ["fund.offender::read::.scalar"],
        ),
        (
            "def read(session):\n    return session.query(T).all()\n",
            ["fund.offender::read::.query"],
        ),
        (
            "def write(own, rows):\n    own.add_all(rows)\n",
            ["fund.offender::write::.add_all"],
        ),
        (
            "from portopt_db.models import AgentRun\n"
            "def write(own, rid):\n"
            "    if own.get(AgentRun, rid) is None:\n"
            "        own.add(AgentRun(id=rid))\n",
            ["fund.offender::write::.get", "fund.offender::write::.add"],
        ),
        (
            "def write(own, rid):\n"
            "    from portopt_db.models.jobs.background_job import BackgroundJob as J\n"
            "    own.add(J(id=rid))\n",
            ["fund.offender::write::.add"],
        ),
        (
            "fetch = lambda session: session.execute(stmt)\n",
            ["fund.offender::<module>::.execute"],
        ),
    ],
    ids=[
        "select-and-execute",
        "function-local-alias",
        "dialect-insert-module-level",
        "module-alias-method",
        "raw-text",
        "nested-renamed-receiver",
        "async-psycopg",
        "session-scalars",
        "session-scalar",
        "session-query",
        "add-all",
        "model-get-and-add",
        "aliased-model-submodule",
        "module-lambda",
    ],
)
def test_each_detected_call_form_is_reported(
    tmp_path: Path, source: str, expected: list[str]
) -> None:
    """Fail-injection: each probe of a detected form is counted under its exact key."""
    assert _scan(tmp_path, source) == Counter(expected)


@pytest.mark.parametrize(
    "source",
    [
        '"""Runs ``session.execute(select(Instrument))`` in a repository."""\n',
        "def f(items, cache, key):\n    items.add(key)\n    return cache.get(key, 0)\n",
        "def f(session):\n    return session.execute\n",
        "def f(result):\n    return result.scalars().all(), result.scalar()\n",
        "def f(store, ns):\n    store.delete(ns, 'k')\n    store.merge(ns)\n",
        "from textual.widgets import Select as select\nselect()\n",
        "def f(select):\n    return select(1)\n",
        "import sqlalchemy as sa\nstmt = sa.sql.select(1)\n",
        "from portopt_db.models import AgentRun\n"
        "def f(own, row, cls):\n    own.add(row)\n    own.get(cls, 1)\n",
        "import portopt_db.models as m\ndef f(own):\n    own.get(m.AgentRun, 1)\n",
        "def f(session):\n    session.flush()\n    session.commit()\n"
        "    session.rollback()\n",
    ],
    ids=[
        "docstring",
        "set-and-dict",
        "attribute-not-called",
        "result-accessors",
        "store-delete-merge",
        "non-sqlalchemy-select",
        "parameter-named-select",
        "submodule-attribute",
        "variable-model",
        "attribute-model",
        "transaction-control",
    ],
)
def test_source_outside_the_detected_forms_is_not_counted(
    tmp_path: Path, source: str
) -> None:
    """A mention, a lookalike call or a form the docstring excludes is not a hit."""
    assert _scan(tmp_path, source) == Counter()


def test_a_repository_module_is_exempt_and_its_neighbours_are_not(
    tmp_path: Path,
) -> None:
    """The exemption is the file name directly under ``fund/audit``, nothing wider."""
    source = "from sqlalchemy import select\nstmt = select(1)\n"
    _scan(tmp_path, source, "fund/audit/widget_repository.py")
    _scan(tmp_path, source, "fund/audit/persistence.py")
    _scan(tmp_path, source, "fund/tools/widget_repository.py")

    assert find_orm_sites(tmp_path) == Counter(
        {
            "fund.audit.persistence::<module>::select": 1,
            "fund.tools.widget_repository::<module>::select": 1,
        }
    )


def test_the_baseline_reader_counts_and_rejects_malformed_lines() -> None:
    """``xN`` sets a count; a duplicate key or a malformed count is refused."""
    assert parse_baseline("# header\n\na::f::select\nb::g::.execute x3\n") == Counter(
        {"a::f::select": 1, "b::g::.execute": 3}
    )
    for bad in ("a::f::select\na::f::select\n", "a::f::select 3\n", "a::f::x x1\n"):
        with pytest.raises(ValueError, match="baseline"):
            parse_baseline(bad)
