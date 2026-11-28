"""Guard: neither the ingestion daemon nor the fund bridge asks yfinance for an ISIN.

``Ticker.isin`` and ``Ticker.get_isin()`` do not ask Yahoo: yfinance scrapes a
third-party site for them, whose terms ban scrapers, so the ingestion stopped calling
them and now stores an ISIN only when the operator passes one.

A Ticker reaches code through ``yf.Ticker(...)``, through the repo's own factories
(``_get_ticker``, ``get_ticker``) and through any name, attribute or container it is
kept in, so the scan does not try to tell a Ticker apart from other objects. It parses
every ``.py`` file under ``ingestion/app`` and ``fund/src`` and reports each read of an
attribute named ``isin`` that is not a method call, each ``get_isin`` call and each
``getattr(..., "isin")``. The one exception is a read on a bare name listed in
``_ISIN_RECEIVERS``, the instruments model and row whose ``isin`` is a database
column. A new database read under another name needs its own entry and reason, and a
listed name that no file reads ``isin`` on any more fails the guard, so the list
cannot outlive its reason. A method named ``isin``, such as pandas'
``Series.isin(...)``, and an assignment to an ``isin`` attribute pass.

What it cannot see: a Ticker bound to one of the listed names, the attribute fetched
under a computed name (``getattr(t, name)``, ``operator.attrgetter``), and code outside
the two scanned trees.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCANNED_TREES = (_REPO_ROOT / "ingestion" / "app", _REPO_ROOT / "fund" / "src")
_ISIN = "isin"
_GET_ISIN = "get_isin"

# Bare names whose ``.isin`` is the ``instruments.isin`` column, never a Ticker's.
_ISIN_RECEIVERS: dict[str, str] = {
    "Instrument": "the ORM model, filtered on its isin column in add_instrument",
    "row": "the instruments row add_instrument stores the operator's ISIN on",
}


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _isin_reads(tree: ast.AST) -> Iterator[ast.Attribute]:
    """Each ``.isin`` in ``tree`` loaded as a value rather than called as a method."""
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == _ISIN
            and isinstance(node.ctx, ast.Load)
            and id(node) not in called
        ):
            yield node


def _listed_receiver(read: ast.Attribute) -> str | None:
    receiver = read.value
    if isinstance(receiver, ast.Name) and receiver.id in _ISIN_RECEIVERS:
        return receiver.id
    return None


def _call_lookup_at(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if _terminal_name(node.func) == _GET_ISIN:
        return "a get_isin() call"
    named = [arg for arg in node.args[1:2] if isinstance(arg, ast.Constant)]
    if _terminal_name(node.func) == "getattr" and any(
        arg.value == _ISIN for arg in named
    ):
        return 'getattr(..., "isin")'
    return None


def ticker_isin_lookups(source: str, label: str) -> list[str]:
    """Return ``label:line: what`` for each possible yfinance ISIN lookup in ``source``.

    Args:
        source: Python source text.
        label: The file's name, used to prefix each finding.

    Returns:
        One finding per ``.isin`` read on an unlisted receiver, ``get_isin`` call or
        ``getattr(..., "isin")``, in source order; empty when there is none.
    """
    tree = ast.parse(source, filename=label)
    found = [
        (read.lineno, f"an .isin read on {ast.unparse(read.value)}")
        for read in _isin_reads(tree)
        if _listed_receiver(read) is None
    ]
    found += [
        (node.lineno, what)
        for node in ast.walk(tree)
        if (what := _call_lookup_at(node)) is not None
    ]
    return [f"{label}:{line}: {what}" for line, what in sorted(found)]


def listed_receivers_read(source: str) -> set[str]:
    """Return the names in ``_ISIN_RECEIVERS`` that ``source`` reads ``.isin`` on."""
    return {
        name
        for read in _isin_reads(ast.parse(source))
        if (name := _listed_receiver(read)) is not None
    }


def _scanned_files() -> list[Path]:
    return sorted(path for tree in _SCANNED_TREES for path in tree.rglob("*.py"))


@pytest.mark.criterion("REQ-OPS-005")
def test_ingestion_and_fund_never_look_an_isin_up_through_yfinance() -> None:
    """Validates every module under ingestion/app and fund/src: none reads ``isin``
    off anything but a listed database receiver, calls ``get_isin`` or fetches
    ``isin`` with ``getattr``."""
    files = _scanned_files()
    assert all(tree.is_dir() for tree in _SCANNED_TREES)
    assert files

    findings = [
        finding
        for path in files
        for finding in ticker_isin_lookups(
            path.read_text(encoding="utf-8"), path.relative_to(_REPO_ROOT).as_posix()
        )
    ]

    assert findings == []


@pytest.mark.criterion("REQ-OPS-005")
def test_every_listed_receiver_still_reads_an_isin_column() -> None:
    """Validates ``_ISIN_RECEIVERS``: each listed name is still read ``.isin`` on
    under ingestion/app or fund/src, so no stale entry lets a new read through."""
    read = set().union(
        *(listed_receivers_read(path.read_text("utf-8")) for path in _scanned_files())
    )

    assert read == set(_ISIN_RECEIVERS)


@pytest.mark.criterion("REQ-OPS-005")
@pytest.mark.parametrize(
    "lookup",
    [
        'import yfinance as yf\nyf.Ticker("X").isin\n',
        'import yfinance as yf\nt = yf.Ticker("X")\nprint(t.isin)\n',
        "from yfinance import Ticker\ndef f(t: Ticker):\n    return t.isin\n",
        'def f(t: "yf.Ticker | None"):\n    return t.isin\n',
        "ticker: yf.Ticker = make()\nticker.isin\n",
        'yf.Ticker("X").get_isin()\n',
        "ticker.get_isin(proxy=None)\n",
        'getattr(yf.Ticker("X"), "isin")\n',
        # v0.2.0's MetadataClient.fetch_isin, verbatim: the repo's own Ticker factory.
        "def _action() -> str | None:\n    return self._get_ticker(symbol).isin\n",
        "def f(self, sym):\n    return self.get_ticker(sym).isin\n",
        "def f(self, sym):\n    t = self._get_ticker(sym)\n    return t.isin\n",
        "class C:\n"
        "    def __init__(self):\n"
        "        self.t = yf.Ticker('X')\n"
        "    def code(self):\n"
        "        return self.t.isin\n",
        'tickers["X"].isin\n',
    ],
)
def test_the_scan_flags_each_isin_lookup(lookup: str) -> None:
    """Validates ``ticker_isin_lookups``: each way of reaching yfinance's ISIN
    lookup is reported."""
    assert ticker_isin_lookups(lookup, "synthetic") != []


@pytest.mark.criterion("REQ-OPS-005")
@pytest.mark.parametrize(
    "source",
    [
        "row.isin = isin or row.isin\n",
        "select(Instrument).where(Instrument.isin == isin)\n",
        "mask = pd.Series(dates).isin(common).values\n",
        'isin = inst.get("isin")\n',
        'ticker = yf.Ticker("X")\nrow.isin = ticker.info.get("symbol")\n',
    ],
)
def test_the_scan_passes_isin_columns_and_pandas_isin(source: str) -> None:
    """Validates ``ticker_isin_lookups``: an ``isin`` column, a row field and
    ``Series.isin`` are not reported."""
    assert ticker_isin_lookups(source, "synthetic") == []


@pytest.mark.criterion("REQ-OPS-005")
def test_the_scan_flags_a_lookup_added_to_a_real_module() -> None:
    """Validates ``ticker_isin_lookups`` on ingestion's add_instrument module: the
    module passes as it is, and fails once a Ticker ISIN read is appended to a copy."""
    module = _REPO_ROOT / "ingestion/app/services/universe/add_instrument.py"
    original = module.read_text(encoding="utf-8")
    injected = f'{original}\n\nimport yfinance as yf\nyf.Ticker("X").isin\n'

    assert ticker_isin_lookups(original, module.name) == []
    assert ticker_isin_lookups(injected, module.name) != []
