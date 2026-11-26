"""A bare ``postgresql://`` URL must keep resolving to psycopg2.

SQLAlchemy 2.1 resolves a Postgres URL without a driver to psycopg (version 3).
Every deployment, template and CI job passes the bare form, and this package
declares and was verified against psycopg2 only, so the driver is pinned where
the engine and the Alembic environment are built.
"""

from __future__ import annotations

import pytest
from sqlalchemy import make_url

from portopt_db.config import DbConfig, pin_postgres_driver


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgresql://u:p@localhost:5432/db",
            "postgresql+psycopg2://u:p@localhost:5432/db",
        ),
        (
            "postgresql://u:p%40ss@h:1/db?sslmode=require",
            "postgresql+psycopg2://u:p%40ss@h:1/db?sslmode=require",
        ),
    ],
)
def test_when_url_has_no_driver_then_psycopg2_is_named(url: str, expected: str) -> None:
    """A driverless Postgres URL gains the psycopg2 driver, nothing else changes."""
    assert pin_postgres_driver(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg2://u:p@h/db",
        "postgresql+psycopg://u:p@h/db",
        "postgresql+asyncpg://u:p@h/db",
        "sqlite:///file.db",
        "sqlite://",
        "",
    ],
)
def test_when_url_names_a_driver_or_another_backend_then_it_is_unchanged(
    url: str,
) -> None:
    """An explicit driver choice, another backend or an empty URL passes through."""
    assert pin_postgres_driver(url) == url


def test_when_pinned_url_is_parsed_then_the_dialect_is_psycopg2() -> None:
    """The rewritten URL selects the psycopg2 dialect, not psycopg 3."""
    pinned = make_url(pin_postgres_driver("postgresql://u:p@h:5432/db"))

    assert pinned.get_driver_name() == "psycopg2"


def test_when_config_holds_a_bare_url_then_the_url_is_kept_as_given() -> None:
    """The pin is applied where the engine is built, so the config stays as given."""
    cfg = DbConfig(url="postgresql://u:p@h/db")

    assert cfg.url == "postgresql://u:p@h/db"
