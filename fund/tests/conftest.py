"""Pytest configuration for the fund test suite.

``fund`` is installed editable into the shared workspace venv (src-layout), so
``import fund`` resolves with no ``sys.path`` surgery. The Phase 1 hygiene guard
is source-blind (it scans tracked file text, imports no implementation module),
so no fixtures are needed here yet.
"""

from __future__ import annotations
