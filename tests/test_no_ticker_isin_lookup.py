"""Guard: neither the ingestion daemon nor the fund bridge asks yfinance for an ISIN.

``Ticker.isin`` and ``Ticker.get_isin()`` do not ask Yahoo: yfinance scrapes a
third-party site for them, whose terms ban scrapers, so the ingestion stopped calling
them and now stores an ISIN only when the operator passes one. The scan parses every
``.py`` file under ``ingestion/app`` and ``fund/src`` and reports each ``get_isin``
call, each ``getattr(..., "isin")`` and each ``.isin`` read on a Ticker: a
``Ticker(...)`` call, a name bound to one, or a name annotated as one. An ``isin``
column or field on a model or a row, and pandas' ``Series.isin(...)``, are not Ticker
reads and pass.

What it cannot see: a Ticker reached through a container (``tickers["X"].isin``) or a
name it is passed to without an annotation.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCANNED_TREES = (_REPO_ROOT / "ingestion" / "app", _REPO_ROOT / "fund" / "src")
_TICKER = "Ticker"
_ISIN = "isin"
_GET_ISIN = "get_isin"
_WORD = re.compile(r"\w+")


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_ticker_call(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Call) and _terminal_name(node.func) == _TICKER


def _is_ticker_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _TICKER in _WORD.findall(node.value)
    return any(_terminal_name(part) == _TICKER for part in ast.walk(node))


def _bound_names(node: ast.AST) -> Iterator[str]:
    """The names ``node`` binds to a Ticker, by its value or its annotation."""
    if isinstance(node, ast.Assign) and _is_ticker_call(node.value):
        yield from (t.id for t in node.targets if isinstance(t, ast.Name))
    elif isinstance(node, ast.AnnAssign | ast.NamedExpr):
        annotation = getattr(node, "annotation", None)
        if isinstance(node.target, ast.Name) and (
            _is_ticker_call(node.value) or _is_ticker_annotation(annotation)
        ):
            yield node.target.id
    elif isinstance(node, ast.arg) and _is_ticker_annotation(node.annotation):
        yield node.arg


def _lookup_at(node: ast.AST, tickers: set[str]) -> str | None:
    if isinstance(node, ast.Attribute) and node.attr == _ISIN:
        receiver = node.value
        if _is_ticker_call(receiver) or _terminal_name(receiver) in tickers:
            return "a Ticker's .isin"
    if isinstance(node, ast.Call):
        if _terminal_name(node.func) == _GET_ISIN:
            return "a get_isin() call"
        named = [arg for arg in node.args[1:2] if isinstance(arg, ast.Constant)]
        if _terminal_name(node.func) == "getattr" and any(
            arg.value == _ISIN for arg in named
        ):
            return 'getattr(..., "isin")'
    return None


def ticker_isin_lookups(source: str, label: str) -> list[str]:
    """Return ``label:line: what`` for each ISIN lookup through yfinance in ``source``.

    Args:
        source: Python source text.
        label: The file's name, used to prefix each finding.

    Returns:
        One finding per lookup, in source order; empty when there is none.
    """
    tree = ast.parse(source, filename=label)
    tickers = {name for node in ast.walk(tree) for name in _bound_names(node)}
    found = [
        (node.lineno, what)
        for node in ast.walk(tree)
        if (what := _lookup_at(node, tickers)) is not None
    ]
    return [f"{label}:{line}: {what}" for line, what in sorted(found)]


def _scanned_files() -> list[Path]:
    return sorted(path for tree in _SCANNED_TREES for path in tree.rglob("*.py"))


@pytest.mark.criterion("REQ-OPS-005")
def test_ingestion_and_fund_never_look_an_isin_up_through_yfinance() -> None:
    """Validates every module under ingestion/app and fund/src: none reads a Ticker's
    ``isin``, calls ``get_isin`` or fetches ``isin`` with ``getattr``."""
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
