"""Tests for ``save_figure_html``: files are written only under ``tmp_path``."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.plotting import efficient_frontier_figure, save_figure_html


@pytest.fixture()
def figure() -> go.Figure:
    """Return a small figure to export."""
    return efficient_frontier_figure([0.1, 0.2, 0.3], [0.04, 0.07, 0.08])


class TestWrite:
    """A standalone HTML page is written and its path returned."""

    def test_a_page_is_written_and_the_resolved_path_returned(
        self, figure: go.Figure, tmp_path: Path
    ) -> None:
        """The file exists, mentions plotly and is the returned path."""
        target = tmp_path / "frontier.html"
        result = save_figure_html(figure, target)
        assert result == target.resolve()
        assert result.is_file()
        text = result.read_text(encoding="utf-8")
        assert "plotly" in text.lower()
        assert "Efficient frontier" in text

    def test_a_string_path_is_accepted(self, figure: go.Figure, tmp_path: Path) -> None:
        """Paths may be given as plain strings."""
        result = save_figure_html(figure, str(tmp_path / "frontier.html"))
        assert result.is_file()

    def test_an_uppercase_suffix_is_accepted(
        self, figure: go.Figure, tmp_path: Path
    ) -> None:
        """``.HTML`` is the same format as ``.html``."""
        assert save_figure_html(figure, tmp_path / "frontier.HTML").is_file()

    def test_an_existing_file_is_overwritten(
        self, figure: go.Figure, tmp_path: Path
    ) -> None:
        """Saving twice leaves the newest page."""
        target = tmp_path / "frontier.html"
        target.write_text("old", encoding="utf-8")
        save_figure_html(figure, target)
        assert target.read_text(encoding="utf-8") != "old"

    def test_the_default_loads_plotly_from_the_cdn(
        self, figure: go.Figure, tmp_path: Path
    ) -> None:
        """The CDN page is small and points at the plotly CDN."""
        result = save_figure_html(figure, tmp_path / "cdn.html")
        assert "cdn.plot.ly" in result.read_text(encoding="utf-8")

    def test_false_leaves_the_plotly_script_out(
        self, figure: go.Figure, tmp_path: Path
    ) -> None:
        """With ``False`` the page expects the caller to provide plotly."""
        result = save_figure_html(
            figure, tmp_path / "bare.html", include_plotlyjs=False
        )
        assert "cdn.plot.ly" not in result.read_text(encoding="utf-8")

    def test_true_embeds_the_whole_library(
        self, figure: go.Figure, tmp_path: Path
    ) -> None:
        """With ``True`` the page works offline and is therefore much larger."""
        cdn = save_figure_html(figure, tmp_path / "cdn.html")
        offline = save_figure_html(
            figure, tmp_path / "offline.html", include_plotlyjs=True
        )
        assert offline.stat().st_size > 100 * cdn.stat().st_size


class TestValidation:
    """A bad destination or mode is rejected before anything is written."""

    @pytest.mark.parametrize("name", ["frontier.txt", "frontier.htm", "frontier"])
    def test_a_wrong_suffix_is_rejected(
        self, figure: go.Figure, tmp_path: Path, name: str
    ) -> None:
        """Only ``.html`` files are written, and nothing is created."""
        with pytest.raises(ConfigurationError, match=r"\.html"):
            save_figure_html(figure, tmp_path / name)
        assert list(tmp_path.iterdir()) == []

    def test_a_missing_folder_is_rejected(
        self, figure: go.Figure, tmp_path: Path
    ) -> None:
        """The folder is not created for the caller."""
        with pytest.raises(ConfigurationError, match="folder"):
            save_figure_html(figure, tmp_path / "missing" / "frontier.html")
        assert not (tmp_path / "missing").exists()

    def test_a_file_in_place_of_the_folder_is_rejected(
        self, figure: go.Figure, tmp_path: Path
    ) -> None:
        """A regular file cannot hold the page."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="folder"):
            save_figure_html(figure, blocker / "frontier.html")

    @pytest.mark.parametrize("mode", ["directory", "CDN", 1, 0, None])
    def test_an_unknown_plotlyjs_mode_is_rejected(
        self, figure: go.Figure, tmp_path: Path, mode: object
    ) -> None:
        """Only ``"cdn"``, ``True`` and ``False`` are valid; ``1`` is not ``True``."""
        with pytest.raises(ConfigurationError, match="include_plotlyjs"):
            save_figure_html(
                figure,
                tmp_path / "frontier.html",
                include_plotlyjs=mode,  # type: ignore[arg-type]
            )
        assert list(tmp_path.iterdir()) == []

    def test_something_that_is_not_a_figure_is_rejected(self, tmp_path: Path) -> None:
        """A dict or a string would write a meaningless page."""
        with pytest.raises(DataError, match="Figure"):
            save_figure_html({"data": []}, tmp_path / "frontier.html")  # type: ignore[arg-type]
