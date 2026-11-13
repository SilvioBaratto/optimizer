"""The database URL carries the password, so DbConfig must not print it."""

from __future__ import annotations

from portopt_db.config import DbConfig

_SENTINEL = "S3NTINEL"


def test_db_config_repr_hides_the_password() -> None:
    """Neither ``repr`` nor ``str`` of a config shows the URL's password."""
    config = DbConfig(url="postgresql://user:" + _SENTINEL + "@host/db")

    assert _SENTINEL not in repr(config)
    assert _SENTINEL not in str(config)
