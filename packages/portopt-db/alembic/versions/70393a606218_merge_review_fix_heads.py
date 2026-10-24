"""Merge the parallel review-fix migration branches into a single head.

The review fixes were developed on independent branches, each extending
d8f0b2c4e6a1, so the chain gained seven heads. They touch disjoint objects, so
this revision only joins them; it changes no schema.

Revision ID: 70393a606218
Revises: 3aa457e38da0, 77f236075b2e, a5ac0fa9b0ec, a6f1c8d3e2b7, b4d6f8a0c2e5,
    c8e0a2d4f6b1, f5c7e9a1b3d2
Create Date: 2026-10-24 19:14:09.788326
"""

from collections.abc import Sequence

revision: str = "70393a606218"
down_revision: str | Sequence[str] | None = (
    "3aa457e38da0",
    "77f236075b2e",
    "a5ac0fa9b0ec",
    "a6f1c8d3e2b7",
    "b4d6f8a0c2e5",
    "c8e0a2d4f6b1",
    "f5c7e9a1b3d2",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Join the branches; no schema change."""


def downgrade() -> None:
    """Split back into the joined branches; no schema change."""
