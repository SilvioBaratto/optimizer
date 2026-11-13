"""FundConfig carries provider keys and the database URL; none may reach a repr.

A traceback that renders locals, a crash screen or a stray ``logger.info(config)``
would otherwise print every key the run holds.
"""

from __future__ import annotations

import dataclasses
import re

from fund.config import FundConfig

_SECRET_FIELD = re.compile(r"(api_key|api_token|secret|password)$|^database_url$")


def _secret_names() -> list[str]:
    return [
        field.name
        for field in dataclasses.fields(FundConfig)
        if _SECRET_FIELD.search(field.name)
    ]


_EXPECTED_SECRET_FIELDS = frozenset(
    {
        "database_url",
        "ollama_api_key",
        "fred_api_key",
        "openai_api_key",
        "openrouter_api_key",
        "anthropic_api_key",
        "google_api_key",
        "groq_api_key",
        "nvidia_api_key",
        "hf_api_token",
        "azure_openai_api_key",
    }
)


def test_secret_field_regex_matches_exactly_the_known_secrets() -> None:
    """Pin the secret field set the repr test covers; add new secret fields here."""
    assert set(_secret_names()) == _EXPECTED_SECRET_FIELDS


def test_repr_hides_every_secret_field() -> None:
    """No secret field value appears in ``repr`` or ``str`` of a populated config."""
    names = _secret_names()
    config = FundConfig(**{name: f"S3NTINEL-{name}" for name in names})

    leaked = [
        name
        for name in names
        if f"S3NTINEL-{name}" in repr(config) or f"S3NTINEL-{name}" in str(config)
    ]

    assert leaked == []
