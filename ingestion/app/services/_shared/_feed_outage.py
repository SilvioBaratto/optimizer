"""Failure signal for feed steps that obtained no usable upstream data."""


class FeedOutageError(RuntimeError):
    """A feed step could not obtain any usable data from its upstream.

    ``_run_step`` flips a job that returns without ``status="completed"`` to
    completed, so a step that finds its whole feed dead must raise rather than
    set ``status="failed"`` itself. The step records its error list through
    ``on_progress`` first; the scheduler's failed update leaves it in place.
    """
