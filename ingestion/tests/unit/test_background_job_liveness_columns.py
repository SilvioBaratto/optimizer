"""Verify worker liveness columns + composite index on background_jobs (issue #585)."""

from __future__ import annotations

from sqlalchemy import inspect

EXPECTED_NEW_COLUMNS = {"worker_pid", "worker_host", "last_heartbeat_at"}


class TestBackgroundJobLivenessColumns:
    def test_when_schema_inspected_then_worker_pid_column_present(self, test_engine):
        cols = {c["name"] for c in inspect(test_engine).get_columns("background_jobs")}
        assert "worker_pid" in cols

    def test_when_schema_inspected_then_worker_host_column_present(self, test_engine):
        cols = {c["name"] for c in inspect(test_engine).get_columns("background_jobs")}
        assert "worker_host" in cols

    def test_when_schema_inspected_then_last_heartbeat_at_column_present(
        self, test_engine
    ):
        cols = {c["name"] for c in inspect(test_engine).get_columns("background_jobs")}
        assert "last_heartbeat_at" in cols

    def test_when_schema_inspected_then_new_columns_are_nullable(self, test_engine):
        cols = {
            c["name"]: c for c in inspect(test_engine).get_columns("background_jobs")
        }
        for name in EXPECTED_NEW_COLUMNS:
            assert cols[name]["nullable"] is True, name

    def test_when_indexes_inspected_then_status_heartbeat_composite_index_present(
        self, test_engine
    ):
        indexes = inspect(test_engine).get_indexes("background_jobs")
        match = next(
            (i for i in indexes if i["name"] == "ix_background_jobs_status_heartbeat"),
            None,
        )
        assert match is not None, [i["name"] for i in indexes]
        assert match["column_names"] == ["status", "last_heartbeat_at"]


class TestBackgroundJobModelMappedAttributes:
    def test_when_model_inspected_then_worker_pid_attribute_present(self):
        from portopt_db.models.jobs.background_job import BackgroundJob

        assert hasattr(BackgroundJob, "worker_pid")

    def test_when_model_inspected_then_worker_host_attribute_present(self):
        from portopt_db.models.jobs.background_job import BackgroundJob

        assert hasattr(BackgroundJob, "worker_host")

    def test_when_model_inspected_then_last_heartbeat_at_attribute_present(self):
        from portopt_db.models.jobs.background_job import BackgroundJob

        assert hasattr(BackgroundJob, "last_heartbeat_at")
