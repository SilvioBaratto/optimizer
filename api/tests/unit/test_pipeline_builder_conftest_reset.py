"""Tests proving the autouse conftest reset for ``pipeline_builder`` (issue #694).

Each test must observe an empty registry on entry — proving the previous
test's mutation was reset by the shared conftest fixture.
"""

from __future__ import annotations

from app.services.pipeline_builder import session_store as ss
from app.services.pipeline_builder.session_store import create_session, get_job_service


class TestConftestResetsBetweenTests:
    def test_first_run_inserts_session_and_starts_empty(self):
        assert ss._sessions == {}
        assert ss._job_services == {}

        sid = create_session({})
        get_job_service(sid)

        assert len(ss._sessions) == 1
        assert len(ss._job_services) == 1

    def test_second_run_starts_empty_proving_reset(self):
        assert ss._sessions == {}
        assert ss._job_services == {}

        sid = create_session({})
        get_job_service(sid)

        assert len(ss._sessions) == 1
        assert len(ss._job_services) == 1

    def test_daemon_flag_reset_between_tests(self):
        assert ss._daemon_started is False

        ss._daemon_started = True

    def test_daemon_flag_observed_false_again(self):
        assert ss._daemon_started is False
