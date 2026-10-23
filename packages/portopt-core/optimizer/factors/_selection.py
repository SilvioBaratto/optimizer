"""Stock selection from composite scores."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import cast

import pandas as pd

from optimizer.factors._config import SelectionConfig, SelectionMethod

logger = logging.getLogger(__name__)


def select_fixed_count(
    scores: pd.Series,
    target_count: int,
    buffer_fraction: float = 0.1,
    current_members: pd.Index | None = None,
) -> pd.Index:
    """Select top N stocks by composite score with buffer.

    Args:
        scores: Composite scores indexed by ticker.
        target_count: Target number of stocks.
        buffer_fraction: Buffer as a fraction of target_count. Current members
            within the buffer zone are retained in preference to the
            lowest-ranked direct entrants that are not current members
            (held names inside the top ``target_count`` are never displaced),
            but the returned index always contains exactly
            ``min(len(valid_scores), target_count)`` tickers.
        current_members: Tickers currently selected.

    Returns:
        Selected tickers. Length is always
        ``min(len(scores.dropna()), target_count)``.
    """
    ranked = scores.dropna().sort_values(ascending=False)

    if len(ranked) <= target_count:
        return ranked.index

    direct = cast(pd.Index, ranked.index[:target_count])

    if current_members is None or len(current_members) == 0:
        return direct

    buffer_size = max(1, int(target_count * buffer_fraction))
    extended_idx = min(target_count + buffer_size, len(ranked))
    buffer_zone = cast(pd.Index, ranked.index[target_count:extended_idx])

    held = set(current_members)
    retained = [t for t in buffer_zone if t in held]
    # Only direct entrants that are not already held can be displaced; evicting
    # a held name inside the target set to keep a lower-ranked held buffer name
    # would invert the hysteresis. Both lists are in rank order, best first.
    displaceable = [t for t in direct if t not in held]

    n_swap = min(len(retained), len(displaceable))
    if n_swap == 0:
        return direct

    admitted = pd.Index(retained[:n_swap])
    to_remove = pd.Index(displaceable[-n_swap:])
    return direct.difference(to_remove).union(admitted)


def select_quantile(
    scores: pd.Series,
    target_quantile: float = 0.8,
    exit_quantile: float | None = None,
    current_members: pd.Index | None = None,
) -> pd.Index:
    """Select stocks above a quantile threshold.

    Args:
        scores: Composite scores indexed by ticker.
        target_quantile: Quantile threshold for entry (0-1).
        exit_quantile: Quantile threshold for exit (hysteresis). If ``None``,
            uses ``target_quantile``.
        current_members: Currently selected tickers.

    Returns:
        Selected tickers.
    """
    if exit_quantile is None:
        exit_quantile = target_quantile

    valid = scores.dropna()
    if len(valid) == 0:
        return pd.Index([])

    entry_threshold = valid.quantile(target_quantile)
    new_entrants = cast(pd.Index, valid.index[valid >= entry_threshold])

    if current_members is None or len(current_members) == 0:
        return new_entrants

    exit_threshold = valid.quantile(exit_quantile)
    surviving = current_members.intersection(valid.index)
    surviving = surviving[valid.loc[surviving] >= exit_threshold]

    return surviving.union(new_entrants)


def _cap_per_sector(
    selected_set: set[str],
    scores: pd.Series,
    sector_labels: pd.Series,
    max_per_sector: int,
) -> set[str]:
    """Evict lowest-scoring excess members from any over-quota sector."""
    if max_per_sector <= 0 or not selected_set:
        return selected_set
    current_index = pd.Index(list(selected_set))
    sectors = sector_labels.reindex(current_index).dropna()
    capped = set(selected_set)
    for sector in sectors.unique():
        members = sectors[sectors == sector].index
        if len(members) <= max_per_sector:
            continue
        ranked = scores.reindex(members).dropna().sort_values(ascending=False)
        capped -= set(cast(pd.Index, ranked.index[max_per_sector:]))
    return capped


def _restore_count(
    selected_set: set[str],
    target_count: int,
    scores: pd.Series,
    sector_labels: pd.Series,
    bands: dict[Hashable, tuple[int, int]],
    max_per_sector: int,
) -> set[str]:
    """Refill or trim a balanced selection back to ``target_count`` names.

    Refilling adds the best-scoring unselected names from sectors still below
    their upper band (and below ``max_per_sector``); trimming drops the
    lowest-scoring labeled names from sectors still above their lower band.
    Neither direction can push a sector out of its band, so a count that
    cannot be reached (no eligible candidates) is left short or long rather
    than traded against the bands.
    """
    result = set(selected_set)
    sector_of = cast("dict[str, Hashable]", sector_labels.dropna().to_dict())
    counts: dict[Hashable, int] = {}
    for ticker in result:
        if ticker in sector_of:
            counts[sector_of[ticker]] = counts.get(sector_of[ticker], 0) + 1

    if len(result) < target_count:
        banded = [t for t, sector in sector_of.items() if sector in bands]
        candidates = scores.reindex(banded).dropna().sort_values(ascending=False)
        for ticker in cast("list[str]", candidates.index.tolist()):
            if len(result) >= target_count:
                break
            if ticker in result:
                continue
            sector = sector_of[ticker]
            limit = bands[sector][1]
            if max_per_sector > 0:
                limit = min(limit, max_per_sector)
            if counts.get(sector, 0) >= limit:
                continue
            result.add(ticker)
            counts[sector] = counts.get(sector, 0) + 1
    elif len(result) > target_count:
        labeled = [t for t in sorted(result) if t in sector_of]
        members = scores.reindex(labeled).dropna().sort_values()
        for ticker in cast("list[str]", members.index.tolist()):
            if len(result) <= target_count:
                break
            sector = sector_of[ticker]
            floor = bands[sector][0] if sector in bands else 0
            if counts[sector] <= floor:
                continue
            result.discard(ticker)
            counts[sector] -= 1

    if len(result) != target_count:
        logger.warning(
            "apply_sector_balance could not reach the requested %d names "
            "within the sector bands; returning %d.",
            target_count,
            len(result),
        )
    return result


def apply_sector_balance(
    selected: pd.Index,
    scores: pd.Series,
    sector_labels: pd.Series,
    parent_universe: pd.Index,
    tolerance: float = 0.05,
    max_per_sector: int = 0,
) -> pd.Index:
    """Adjust selection for sector-proportional representation.

    The per-sector band limits are sized once on the incoming selection and
    stay fixed, so trading names between sectors never changes how many are
    held: after bringing every sector into its band and applying
    ``max_per_sector``, the selection is refilled with the best-scoring names
    from sectors that still have room (or trimmed of the weakest names from
    sectors above their lower band) to return to the incoming size.

    Args:
        selected: Initially selected tickers.
        scores: Composite scores for all candidates.
        sector_labels: Sector label per ticker.
        parent_universe: Full universe for computing target sector weights.
        tolerance: Maximum deviation from parent sector weights.
        max_per_sector: Hard cap on members per sector; 0 means no cap.

    Returns:
        Sector-balanced selection of the same size as *selected*, or smaller
        when the bands and ``max_per_sector`` leave too few eligible names.
    """
    parent_sectors = sector_labels.reindex(parent_universe).dropna()
    target_weights = parent_sectors.value_counts(normalize=True)

    result_set: set[str] = set(selected)
    n_target = len(result_set)
    bands = {
        sector: (
            max(0, round((target_w - tolerance) * n_target)),
            round((target_w + tolerance) * n_target),
        )
        for sector, target_w in target_weights.items()
    }

    selected_sectors = sector_labels.reindex(pd.Index(list(result_set))).dropna()
    for sector, (min_n, max_n) in bands.items():
        current_n = int((selected_sectors == sector).sum())

        if current_n < min_n:
            candidates = sector_labels.index[
                (sector_labels == sector) & (~sector_labels.index.isin(result_set))
            ]
            candidate_scores = (
                scores.reindex(candidates).dropna().sort_values(ascending=False)
            )
            result_set.update(
                cast(pd.Index, candidate_scores.index[: min_n - current_n])
            )

        elif current_n > max_n:
            sector_members = selected_sectors[selected_sectors == sector].index
            sector_scores = scores.reindex(sector_members).dropna().sort_values()
            result_set -= set(cast(pd.Index, sector_scores.index[: current_n - max_n]))

    result_set = _cap_per_sector(result_set, scores, sector_labels, max_per_sector)
    result_set = _restore_count(
        result_set, n_target, scores, sector_labels, bands, max_per_sector
    )
    return pd.Index(sorted(result_set))


def compute_selection_turnover(
    current: pd.Index,
    new: pd.Index,
    universe: pd.Index,
) -> float:
    """Compute selection turnover as fraction of universe changed.

    Args:
        current: Currently selected tickers.
        new: Newly selected tickers.
        universe: Full investable universe.

    Returns:
        ``len(added | removed) / len(universe)``, or 0.0 if universe
        is empty.
    """
    if len(universe) == 0:
        return 0.0
    added = new.difference(current)
    removed = current.difference(new)
    return len(added.union(removed)) / len(universe)


def select_stocks(
    scores: pd.Series,
    config: SelectionConfig | None = None,
    current_members: pd.Index | None = None,
    sector_labels: pd.Series | None = None,
    parent_universe: pd.Index | None = None,
    return_turnover: bool = False,
) -> pd.Index | tuple[pd.Index, float]:
    """Select stocks from scored universe.

    Args:
        scores: Composite scores indexed by ticker.
        config: Selection configuration.
        current_members: Currently selected tickers for buffer/hysteresis.
        sector_labels: Sector labels for sector balancing.
        parent_universe: Full universe for sector weight targets.
        return_turnover: When ``True``, return ``(selected, turnover)`` tuple.

    Returns:
        Selected tickers, optionally with turnover.
    """
    if config is None:
        config = SelectionConfig()

    if config.method == SelectionMethod.FIXED_COUNT:
        selected = select_fixed_count(
            scores,
            target_count=config.target_count,
            buffer_fraction=config.buffer_fraction,
            current_members=current_members,
        )
    else:
        selected = select_quantile(
            scores,
            target_quantile=config.target_quantile,
            exit_quantile=config.exit_quantile,
            current_members=current_members,
        )

    if config.sector_balance and sector_labels is not None:
        universe = parent_universe if parent_universe is not None else scores.index
        selected = apply_sector_balance(
            selected,
            scores,
            sector_labels,
            parent_universe=universe,
            tolerance=config.sector_tolerance,
            max_per_sector=config.max_per_sector,
        )

    if return_turnover:
        prev = current_members if current_members is not None else pd.Index([])
        univ = parent_universe if parent_universe is not None else scores.index
        turnover = compute_selection_turnover(prev, selected, univ)
        return selected, turnover

    return selected
