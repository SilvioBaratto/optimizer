"""Smoke test: the fund package imports and exposes its version.

Beyond asserting the version, this test's real job in Phase 1 is to make
``import fund`` happen under coverage measurement — without it, ``--cov=fund``
collects no data on the scaffold-only package and the coverage gate reports 0%.
"""

from __future__ import annotations

import fund


def test_fund_exposes_its_version():
    assert fund.__version__ == "0.1.0"
