"""Layer order of the ingestion image's build stage.

The runtime packages are 100+ MB of wheels (scipy, pandas, scikit-learn, numpy,
lxml). When the dependency resolve shares a layer with the project install, any
edit under ``ingestion/`` (code, tests or docs) invalidates it and the next
rebuild downloads every wheel again, which also fails outright offline or behind
a TLS-inspecting proxy. Dependencies are therefore installed from the manifests
alone, before the source is copied, with the uv cache kept between builds.
"""

from __future__ import annotations

from pathlib import Path

# unit -> setup -> tests -> ingestion -> repo root (the Dockerfile's owner).
_DOCKERFILE = Path(__file__).resolve().parents[3] / "Dockerfile"
_SOURCE_COPY = "COPY ingestion/ ingestion/"
_UV_CACHE_MOUNT = "--mount=type=cache,target=/root/.cache/uv"


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


def test_project_is_installed_after_the_source_is_copied() -> None:
    """The final sync is what builds portopt and portopt-db from the copied source."""
    lines = _builder_instructions()
    installs = [i for i in _uv_syncs(lines) if "--no-editable" in lines[i]]

    assert installs
    assert installs[-1] > lines.index(_SOURCE_COPY)


def test_every_uv_sync_keeps_the_download_cache_between_builds() -> None:
    """A changed lock should re-download only the wheels that changed."""
    lines = _builder_instructions()

    assert all(_UV_CACHE_MOUNT in lines[i] for i in _uv_syncs(lines))
