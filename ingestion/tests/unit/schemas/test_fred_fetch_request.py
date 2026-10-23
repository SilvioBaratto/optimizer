"""``FredFetchRequest.incremental`` documents the trailing-window refetch."""

from __future__ import annotations

from app.schemas.macro.macro_regime import FredFetchRequest


def test_incremental_description_names_the_trailing_window() -> None:
    description = FredFetchRequest.model_fields["incremental"].description or ""

    assert "trailing" in description
    assert "newer than" not in description
