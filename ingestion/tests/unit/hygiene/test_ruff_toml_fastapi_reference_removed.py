"""Example test for issue #3, criterion scope-2.

``ingestion/ruff.toml`` must carry no FastAPI reference, while the ``B008``
and ``E402`` ignore entries it already carries for other reasons must be
retained.

Ambiguity resolved in this test: the criterion does not name the exact ignore
list the two codes live in (global ``ignore`` vs. a per-file-ignores table),
so this test asserts the literal tokens ``"B008"`` and ``"E402"`` are present
somewhere in the file rather than pinning a specific list shape.

Also covers the issue-comment addendum: the ``E501``/``RUF001`` rationale
comments said "api code" — reworded to "ingestion code" so no residual
FastAPI-era naming survives even where the word "FastAPI" itself never
appeared.

Source-blind by construction: reads the raw text of a tracked config file, no
implementation module is imported or exercised.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root:
from ``ingestion/tests/unit/hygiene/test_ruff_toml_fastapi_reference_removed.py``,
parents[0]=hygiene, [1]=unit, [2]=tests, [3]=ingestion, [4]=repository root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUFF_TOML = _REPO_ROOT / "ingestion" / "ruff.toml"


@pytest.mark.criterion("scope-2")
def test_when_ruff_toml_is_read_then_no_fastapi_reference_remains():
    text = _RUFF_TOML.read_text(encoding="utf-8")

    assert "fastapi" not in text.lower()


@pytest.mark.criterion("scope-2")
def test_when_ruff_toml_is_read_then_b008_and_e402_ignores_are_retained():
    text = _RUFF_TOML.read_text(encoding="utf-8")

    assert "B008" in text
    assert "E402" in text


@pytest.mark.criterion("scope-2")
def test_when_ruff_toml_is_read_then_api_code_wording_is_replaced():
    text = _RUFF_TOML.read_text(encoding="utf-8")

    assert "api code" not in text.lower()
    assert "ingestion code" in text.lower()
