"""Region-level linear-constraint helper for skfolio :class:`MeanRisk`.

Region is not a native skfolio constraint family.  This helper composes a
``ticker -> region`` group map (from a ``ticker -> country`` map and a
``country -> region`` map) and emits ``"region <= cap"`` constraint rows
in the same shape as :func:`build_sector_constraints`.

skfolio's ``equations_to_matrix`` tokenises constraint strings using
arithmetic operators including ``-`` (minus).  Region labels that contain a
literal hyphen (e.g. ``"Asia-Pacific"``) are therefore mis-parsed:
``"Asia-Pacific <= 0.6"`` is tokenised as *group "Asia" minus group
"Pacific"*, which triggers a ``UserWarning: … missing from the groups``.
``sanitize_group_token`` replaces every ``-`` with a space so that
``"Asia-Pacific"`` becomes the DSL-safe token ``"Asia Pacific"``.  Other
characters (spaces, ``&``, ``/``) are valid in skfolio group labels and
are left unchanged.
"""

from __future__ import annotations

_OTHER_REGION = "Other"


def sanitize_group_token(label: str) -> str:
    """Return a skfolio DSL-safe group token by replacing hyphens with spaces.

    skfolio's linear-constraints parser treats ``-`` as arithmetic minus, so
    a group label containing a literal hyphen (e.g. ``"Asia-Pacific"``) would
    be split into two tokens and trigger a ``UserWarning: … missing from the
    groups`` at fit time.  Replacing ``-`` with a space (``"Asia Pacific"``)
    produces a valid multi-word token that the parser handles correctly.

    Spaces, ``&``, ``/`` and other punctuation are DSL-safe and are left
    unchanged.  The transformation is idempotent.

    Parameters
    ----------
    label : str
        Arbitrary region or group label string.

    Returns
    -------
    str
        Label with every ``"-"`` replaced by ``" "``.

    Examples
    --------
    >>> sanitize_group_token("Asia-Pacific")
    'Asia Pacific'
    >>> sanitize_group_token("Middle East & Africa")
    'Middle East & Africa'
    >>> sanitize_group_token("Americas")
    'Americas'
    """
    return label.replace("-", " ")


def build_region_linear_constraints(
    country_map: dict[str, str],
    region_map: dict[str, str],
    max_region_weight: float = 0.60,
) -> tuple[dict[str, str], list[str]]:
    """Build skfolio ``groups`` and ``linear_constraints`` for region caps.

    Region labels are sanitized via :func:`sanitize_group_token` before use:
    hyphens in labels such as ``"Asia-Pacific"`` are replaced with spaces so
    that the skfolio DSL parser does not mis-interpret them as arithmetic
    subtraction operators.  The ``"Other"`` fallback bucket (tickers with no
    matching country entry) is intentionally excluded from
    ``linear_constraints`` — constraining an uncategorised catch-all is
    semantically meaningless and would trigger a skfolio warning when no
    assets resolve to that label.

    Parameters
    ----------
    country_map : dict[str, str]
        Mapping from ticker to country name
        (e.g. ``{"AAPL": "United States"}``).
    region_map : dict[str, str]
        Mapping from country name to region label
        (e.g. ``{"United States": "Americas"}``).  Tickers whose country is
        absent from ``region_map`` fall back to ``"Other"``.
    max_region_weight : float, default=0.60
        Maximum total weight for any single region.  Must lie in ``(0, 1]``.

    Returns
    -------
    groups : dict[str, str]
        Ticker -> sanitized region label, suitable for ``MeanRisk.groups``.
    linear_constraints : list[str]
        One sorted ``"<region> <= <cap>"`` row per unique non-Other region.
    """
    if not 0.0 < max_region_weight <= 1.0:
        raise ValueError(f"max_region_weight={max_region_weight!r} must be in (0, 1].")

    groups = {
        ticker: sanitize_group_token(region_map.get(country, _OTHER_REGION))
        for ticker, country in country_map.items()
    }
    cap = round(max_region_weight, 6)
    # Exclude the "Other" catch-all from constraint emission: no member assets
    # are guaranteed for it and skfolio warns when a constrained group label is
    # absent from the fitted universe.
    regions = sorted(r for r in set(groups.values()) if r != _OTHER_REGION)
    linear_constraints = [f"{region} <= {cap}" for region in regions]
    return groups, linear_constraints
