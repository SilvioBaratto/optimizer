"""Hygiene guard tests for the ingestion daemon.

Each module in this package asserts the *absence* of a stale pattern
(a dead dependency, a dead path, a dead import) rather than testing a
positive feature. Guards are source-blind by design — they assert on the
raw text of tracked files, not on any implementation behaviour.
"""
