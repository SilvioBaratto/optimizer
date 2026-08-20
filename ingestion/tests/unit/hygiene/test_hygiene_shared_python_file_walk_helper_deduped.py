"""Example test for issue #15, [scope-3]: ``_iter_python_files`` and
``_EXCLUDE_DIR_PARTS`` exist in exactly one place, shared by
``test_synchronous_only.py`` and ``test_no_optimizer_import.py``.

Source-blind by construction: scans the raw text of the two named consumer
modules plus their shared-helper module for the literal definition sites.
No implementation module is imported.

Ambiguity resolution: the criterion does not name where the single shared
definition should live, so this test does not assume a location — it scans
whichever hygiene module the two named consumers actually import
``_iter_python_files`` from. It also does not assume every other hygiene
file is barred from ever defining a same-named local for an unrelated
purpose (``test_no_http_surface.py`` has its own broader, differently-typed
variant scanning more than ``.py`` files; issue #15's Context section marks
that one as deliberately staying local) — only that the two *named*
consumers collapse to a single shared definition between them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_HYGIENE_DIR = Path(__file__).resolve().parent

_ITER_PYTHON_FILES_DEF = re.compile(r"^def _iter_python_files\(", re.MULTILINE)
_EXCLUDE_DIR_PARTS_DEF = re.compile(r"^_EXCLUDE_DIR_PARTS\s*=", re.MULTILINE)
_SHARED_IMPORT_LINE = re.compile(
    r"^from tests\.unit\.hygiene\.(\S+) import .*_iter_python_files", re.MULTILINE
)

_CONSUMER_MODULES = ("test_synchronous_only.py", "test_no_optimizer_import.py")


def _read(name: str) -> str:
    return (_HYGIENE_DIR / name).read_text(encoding="utf-8")


def _shared_module_names() -> set[str]:
    """Return the module(s) the two named consumers import the helper from."""
    modules = set()
    for consumer in _CONSUMER_MODULES:
        match = _SHARED_IMPORT_LINE.search(_read(consumer))
        if match:
            modules.add(f"{match.group(1)}.py")
    return modules


@pytest.mark.criterion("scope-3")
def test_when_both_consumers_are_read_then_they_import_the_helper_from_the_same_module():
    assert len(_shared_module_names()) == 1


@pytest.mark.criterion("scope-3")
def test_when_the_shared_module_is_read_then_iter_python_files_is_defined_exactly_once():
    shared_modules = _shared_module_names()
    assert shared_modules, "expected both consumers to import from a shared module"

    definition_sites = [
        name for name in shared_modules if _ITER_PYTHON_FILES_DEF.search(_read(name))
    ]
    assert definition_sites == list(shared_modules)


@pytest.mark.criterion("scope-3")
def test_when_the_shared_module_is_read_then_exclude_dir_parts_is_defined_exactly_once():
    shared_modules = _shared_module_names()
    assert shared_modules, "expected both consumers to import from a shared module"

    definition_sites = [
        name for name in shared_modules if _EXCLUDE_DIR_PARTS_DEF.search(_read(name))
    ]
    assert definition_sites == list(shared_modules)


@pytest.mark.criterion("scope-3")
@pytest.mark.parametrize("consumer", _CONSUMER_MODULES)
def test_when_a_consumer_module_is_read_then_it_does_not_locally_redefine_iter_python_files(
    consumer,
):
    text = (_HYGIENE_DIR / consumer).read_text(encoding="utf-8")
    assert _ITER_PYTHON_FILES_DEF.search(text) is None
