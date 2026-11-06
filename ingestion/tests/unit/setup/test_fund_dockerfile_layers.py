"""Layer order of the fund image's build stage.

The fund closure is the heaviest in the workspace (skfolio, scipy, pandas, the
deepagents/langgraph stack and every provider SDK). A source edit under ``fund/`` or
``packages/`` that shares a layer with the dependency resolve downloads all of it
again on the next rebuild, and ``portopt start`` rebuilds whenever the checkout
changed. Dependencies are therefore installed from the manifests alone, before the
source is copied.
"""

from __future__ import annotations

from pathlib import Path

# unit -> setup -> tests -> ingestion -> repo root (the Dockerfile's owner).
_DOCKERFILE = Path(__file__).resolve().parents[4] / "fund" / "Dockerfile"
_SOURCE_COPY = "COPY fund/ fund/"


def _builder_instructions() -> list[str]:
    """Instructions of the build stage, continuations joined, comments dropped."""
    text = _DOCKERFILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = [
        line.strip()
        for line in text.replace("\\\n", " ").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    stage_starts = [i for i, line in enumerate(lines) if line.startswith("FROM ")]
    return lines[: stage_starts[1]]


def _uv_syncs(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if "uv sync" in line]


def test_dependencies_are_installed_before_the_source_is_copied() -> None:
    """A source edit must leave the dependency layer cached."""
    lines = _builder_instructions()
    dependency_syncs = [
        i for i in _uv_syncs(lines) if "--no-install-workspace" in lines[i]
    ]

    assert dependency_syncs, "no dependency-only `uv sync --no-install-workspace`"
    assert _SOURCE_COPY in lines
    assert dependency_syncs[0] < lines.index(_SOURCE_COPY)


def test_only_manifests_are_copied_ahead_of_the_dependency_layer() -> None:
    """Any other file copied first would invalidate the dependency layer with it."""
    lines = _builder_instructions()
    first_sync = _uv_syncs(lines)[0]

    # `COPY --from=` brings the uv binary from its image, not from the context.
    copied = [
        line
        for line in lines[:first_sync]
        if line.startswith("COPY ") and "--from=" not in line
    ]

    assert copied, "the manifests are not copied before the dependency layer"
    sources = [source for line in copied for source in line.split()[1:-1]]
    assert all(source.endswith(("pyproject.toml", "uv.lock")) for source in sources)


def test_members_are_installed_after_the_source_is_copied() -> None:
    """The final sync is what installs portopt-core, portopt-db and fund editable
    from the copied source; it must come after that source is in place."""
    lines = _builder_instructions()
    members = [i for i in _uv_syncs(lines) if "--no-install-workspace" not in lines[i]]

    assert members
    assert members[-1] > lines.index(_SOURCE_COPY)


def test_every_uv_sync_installs_the_same_closure() -> None:
    """The dependency layer and the member install must resolve the same package
    and extras, or the second sync would add or drop distributions."""
    lines = _builder_instructions()
    for i in _uv_syncs(lines):
        assert "--package portopt-fund" in lines[i]
        assert "--extra all-llm" in lines[i]
        assert "--frozen" in lines[i]
