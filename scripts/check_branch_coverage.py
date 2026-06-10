#!/usr/bin/env python3
"""Branch-aware, per-metric coverage gate for the optimizer library.

Parses a Cobertura ``coverage.xml`` and fails when the root ``line-rate`` or
``branch-rate`` falls below the threshold (default ``0.80``).  This complements
the blended ``--cov-fail-under`` gate by enforcing the branch rate
*independently* so a branch regression cannot hide behind high statement
coverage.

coverage.py has no native function-coverage metric (it tracks statements /
branches / lines).  "Functions" is enforced structurally by the
parametrize-every-enum-member dispatch convention in ``tests/*/test_factory.py``.

Usage::

    python scripts/check_branch_coverage.py coverage.xml [threshold]
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

_DEFAULT_THRESHOLD = 0.80


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_branch_coverage.py <coverage.xml> [threshold]")
        return 2
    threshold = float(argv[1]) if len(argv) > 1 else _DEFAULT_THRESHOLD
    # coverage.xml is produced by our own test run, not untrusted input.
    root = ET.parse(argv[0]).getroot()  # noqa: S314
    failures: list[str] = []
    for metric in ("line-rate", "branch-rate"):
        actual = float(root.get(metric, "0.0"))
        status = "OK" if actual >= threshold else "FAIL"
        print(f"{metric:11} {actual:.4f} (floor {threshold:.2f}) {status}")
        if actual < threshold:
            failures.append(f"{metric} {actual:.4f} < floor {threshold:.2f}")
    if failures:
        print("\nCoverage gate failed:\n  " + "\n  ".join(failures))
        return 1
    print("\nCoverage gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
