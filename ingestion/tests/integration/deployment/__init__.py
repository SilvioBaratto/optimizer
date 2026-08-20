"""Deployment-verification tests for the headless ingestion daemon.

Each module here drives a real collaborator named by an acceptance
criterion of issue #4 (a fresh virtualenv, the Docker Compose stack) rather
than a mock of it — these are the tests that go red when the wiring between
the daemon and its deployment environment breaks, not when a unit's own
logic breaks.
"""
