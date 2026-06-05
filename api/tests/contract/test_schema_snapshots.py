"""Contract-parity test: committed snapshots must match regenerated schemas.

Reimports each domain's response models, regenerates their JSON schema, and
asserts string-equality with the committed snapshot file. A field rename on any
schema fails exactly that domain's case with a clear diff (no DB, no network).
"""

from __future__ import annotations

import pytest

from .emit_schemas import (
    DOMAIN_SCHEMAS,
    SNAPSHOT_DIR,
    render,
    snapshot_for_domain,
)


@pytest.mark.parametrize("domain", sorted(DOMAIN_SCHEMAS))
def test_committed_snapshot_matches_regenerated_schema(domain: str) -> None:
    """when a schema drifts from its committed snapshot, the parity test fails."""
    regenerated = render(snapshot_for_domain(DOMAIN_SCHEMAS[domain]))
    committed = (SNAPSHOT_DIR / f"{domain}.json").read_text()
    assert committed == regenerated, (
        f"Snapshot drift in '{domain}'. Run `make contract-snapshots` to "
        f"regenerate after an intended schema change."
    )


def test_every_domain_has_a_committed_snapshot() -> None:
    """when the emitter is configured, all 16 domains have a snapshot file."""
    missing = [d for d in DOMAIN_SCHEMAS if not (SNAPSHOT_DIR / f"{d}.json").exists()]
    assert missing == [], f"Domains without a committed snapshot: {missing}"
