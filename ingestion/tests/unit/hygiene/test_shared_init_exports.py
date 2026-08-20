"""Guard: ``app/services/_shared/__init__.py`` never re-exports
``bootstrap_benchmarks`` (issue #8).

Load-bearing, not stylistic: ``_benchmark_bootstrap`` imports
``market_data.reference_index_seeder``, which imports back into ``_shared``
for ``ProgressCallback``. Re-exporting the symbol from ``_shared/__init__``
makes that import cycle load-bearing on import order, and it fails at import
time in a way that looks unrelated to the change that caused it.

Source-blind by construction: reads the tracked file's raw text and checks
for the absence of the re-export, stripping prose first so a docstring
explaining the omission does not read as a violation. No implementation
module is imported.

Anchoring reuses the shape established by ``test_no_http_surface.py`` (issue
#5, the checkpoint guard): ``Path(__file__).resolve().parents[3]`` resolves
directly to the ``ingestion/`` directory from this file's location, never
``Path.cwd()``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_INGESTION_ROOT = Path(__file__).resolve().parents[3]
_SHARED_INIT = _INGESTION_ROOT / "app" / "services" / "_shared" / "__init__.py"

_FORBIDDEN_SYMBOL = "bootstrap_benchmarks"
_DOCSTRING_PATTERN = re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL)
_COMMENT_PATTERN = re.compile(r"#.*")


def find_reexport_violation(source_text: str) -> bool:
    """Return whether ``source_text`` re-exports ``bootstrap_benchmarks``.

    Docstrings and comments are stripped first: a module may legitimately
    *name* the symbol in prose to explain why it is deliberately absent —
    both ``from ... import`` and ``__all__`` re-export forms are caught by
    checking what survives that strip.

    Args:
        source_text: Raw Python source of a module's ``__init__.py``.

    Returns:
        True if the symbol survives outside prose (an actual import or
        ``__all__`` re-export), False otherwise.
    """
    code_only = _DOCSTRING_PATTERN.sub("", source_text)
    code_only = _COMMENT_PATTERN.sub("", code_only)
    return _FORBIDDEN_SYMBOL in code_only


@pytest.mark.criterion("global-7")
def test_when_shared_services_init_is_read_then_bootstrap_benchmarks_is_not_reexported():
    content = _SHARED_INIT.read_text(encoding="utf-8")
    assert find_reexport_violation(content) is False


def test_when_ingestion_root_is_resolved_then_it_points_at_the_ingestion_directory():
    assert _INGESTION_ROOT.name == "ingestion"


def test_when_bootstrap_benchmarks_is_reexported_via_import_then_the_guard_fails():
    reexported_source = "from ._benchmark_bootstrap import bootstrap_benchmarks\n"
    assert find_reexport_violation(reexported_source) is True


def test_when_bootstrap_benchmarks_is_reexported_via_dunder_all_then_the_guard_fails():
    reexported_source = (
        "from ._benchmark_bootstrap import bootstrap_benchmarks\n\n"
        '__all__ = ["bootstrap_benchmarks"]\n'
    )
    assert find_reexport_violation(reexported_source) is True


def test_when_bootstrap_benchmarks_appears_only_in_a_docstring_then_the_guard_passes():
    # Prose naming the symbol to explain its deliberate absence must not
    # self-match — this is the exclusion the real _shared/__init__.py relies
    # on for its own explanatory docstring.
    prose_only = '"""Deliberately does not export bootstrap_benchmarks."""\n'
    assert find_reexport_violation(prose_only) is False
