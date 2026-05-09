"""Region-level linear-constraint helper for skfolio :class:`MeanRisk`.

Region is not a native skfolio constraint family.  This helper composes a
``ticker -> region`` group map (from a ``ticker -> country`` map and a
``country -> region`` map) and emits ``"region <= cap"`` constraint rows
in the same shape as :func:`build_sector_constraints`.
"""

from __future__ import annotations

_OTHER_REGION = "Other"


def build_region_linear_constraints(
    country_map: dict[str, str],
    region_map: dict[str, str],
    max_region_weight: float = 0.60,
) -> tuple[dict[str, str], list[str]]:
    """Build skfolio ``groups`` and ``linear_constraints`` for region caps.

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
        Ticker -> region label, suitable for ``MeanRisk.groups``.
    linear_constraints : list[str]
        One sorted ``"<region> <= <cap>"`` row per unique region.
    """
    if not 0.0 < max_region_weight <= 1.0:
        raise ValueError(
            f"max_region_weight={max_region_weight!r} must be in (0, 1]."
        )

    groups = {
        ticker: region_map.get(country, _OTHER_REGION)
        for ticker, country in country_map.items()
    }
    cap = round(max_region_weight, 6)
    regions = sorted(set(groups.values()))
    linear_constraints = [f"{region} <= {cap}" for region in regions]
    return groups, linear_constraints
