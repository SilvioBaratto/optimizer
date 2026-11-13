"""Guard: every ``_upsert(..., constraint_name=...)`` names a real unique constraint.

SQLite drops the conflict target, so only this static check (and the PostgreSQL
integration suite) can catch a misspelled or renamed constraint before runtime.
"""

from __future__ import annotations

from dataclasses import replace

from _upsert_sites import collect_upsert_sites, constraint_problems


def test_scan_finds_every_known_upsert_site() -> None:
    """The count pins the scan itself: a refactor that hides sites must fail here.

    Thirty-eight source lines mention ``constraint_name=``; one of them is the
    wrapper ``_upsert_estimates`` forwarding its parameter, which is not a site.
    """
    assert len(collect_upsert_sites()) == 37


def test_every_conflict_constraint_is_a_unique_or_primary_key_constraint() -> None:
    """No site names a constraint its model does not declare."""
    assert constraint_problems(collect_upsert_sites()) == []


def test_a_misspelled_constraint_name_is_reported() -> None:
    """The check can fail: a site with a wrong name yields a problem line."""
    site = collect_upsert_sites()[0]
    wrong = replace(site, constraint_name=site.constraint_name + "_typo")

    problems = constraint_problems([wrong])

    assert len(problems) == 1
    assert wrong.constraint_name in problems[0]
