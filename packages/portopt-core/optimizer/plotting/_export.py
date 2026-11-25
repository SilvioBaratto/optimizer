"""Write a figure to a standalone HTML page."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import plotly.graph_objects as go

from optimizer.exceptions import ConfigurationError, DataError

_CDN: Final = "cdn"
_HTML_SUFFIX: Final = ".html"


def save_figure_html(
    figure: go.Figure,
    path: str | os.PathLike[str],
    *,
    include_plotlyjs: str | bool = _CDN,
) -> Path:
    """Write ``figure`` as a standalone HTML page.

    An existing file at ``path`` is overwritten.

    Args:
        figure: Figure to save.
        path: Destination file. It must end in ``.html`` and its folder must exist;
            the folder is not created.
        include_plotlyjs: How the page gets plotly.js: ``"cdn"`` loads it from the
            plotly CDN (small file, needs internet to view), ``True`` embeds the
            whole library (about 4 MB, works offline), ``False`` leaves it out for a
            page that already provides it.

    Returns:
        The absolute path of the file written.

    Raises:
        ConfigurationError: If ``include_plotlyjs`` is not ``"cdn"``, ``True`` or
            ``False``, the suffix is not ``.html``, or the folder does not exist.
        DataError: If ``figure`` is not a plotly figure.
    """
    if not isinstance(figure, go.Figure):
        msg = f"figure must be a plotly Figure, got {type(figure).__name__}"
        raise DataError(msg)
    # `1 == True` in Python, so a membership test on a set of modes would let the
    # integers 0 and 1 through.
    if not (isinstance(include_plotlyjs, bool) or include_plotlyjs == _CDN):
        msg = f"include_plotlyjs must be 'cdn', True or False, got {include_plotlyjs!r}"
        raise ConfigurationError(msg)
    target = Path(path)
    if target.suffix.lower() != _HTML_SUFFIX:
        msg = f"path must end in {_HTML_SUFFIX}, got {target.name!r}"
        raise ConfigurationError(msg)
    if not target.parent.is_dir():
        msg = f"the folder of path must exist, {str(target.parent)!r} does not"
        raise ConfigurationError(msg)
    figure.write_html(str(target), include_plotlyjs=include_plotlyjs, full_html=True)
    return target.resolve()
