"""Unit tests for ``app.utils.currency`` sub-unit normalisation."""

from __future__ import annotations

import pytest

from app.utils.currency import to_major_currency


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("GBp", "GBP"),
        ("GBX", "GBP"),
        ("GBx", "GBP"),
        ("ZAc", "ZAR"),
        ("ZAC", "ZAR"),
        ("ILA", "ILS"),
    ],
)
def test_when_code_is_a_minor_unit_in_any_yahoo_casing_then_major_is_returned(
    code: str, expected: str
) -> None:
    assert to_major_currency(code) == expected


@pytest.mark.parametrize("code", ["GBP", "USD", "ZAR", "ILS", "TWD"])
def test_when_code_is_already_major_then_it_is_unchanged(code: str) -> None:
    assert to_major_currency(code) == code


def test_when_code_is_none_then_none_is_returned() -> None:
    assert to_major_currency(None) is None
