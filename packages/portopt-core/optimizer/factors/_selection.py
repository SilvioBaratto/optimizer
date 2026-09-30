"""Stock selection from composite scores."""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from optimizer.factors._config import SelectionConfig, SelectionMethod

logger = logging.getLogger(__name__)

_MAX_BALANCE_ITERATIONS: int = 10


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
            lowest-ranked direct entrants, but the returned index always
            contains exactly ``min(len(valid_scores), target_count)`` tickers.
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
    buffer_zone = ranked.index[target_count:extended_idx]

    retained = current_members.intersection(buffer_zone)

    # Retained members outside `direct` consume slots; evict an equal number
    # of the lowest-ranked direct entrants to keep total == target_count.
    overflow = retained.difference(direct)
    if len(overflow) == 0:
        return direct

    non_retained_direct_set = set(direct) - set(retained)
    ranked_non_retained = [t for t in ranked.index if t in non_retained_direct_set]
    to_remove = pd.Index(ranked_non_retained[-len(overflow) :])
    return direct.difference(to_remove).union(overflow)


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


def apply_sector_balance(
    selected: pd.Index,
    scores: pd.Series,
    sector_labels: pd.Series,
    parent_universe: pd.Index,
    tolerance: float = 0.05,
    max_per_sector: int = 0,
) -> pd.Index:
    """Adjust selection for sector-proportional representation.

    Iterates the balance pass until convergence (no further adds or
    removes are needed) or until ``_MAX_BALANCE_ITERATIONS`` is reached.
    A warning is logged if the cap is hit before convergence.

    Args:
        selected: Initially selected tickers.
        scores: Composite scores for all candidates.
        sector_labels: Sector label per ticker.
        parent_universe: Full universe for computing target sector weights.
        tolerance: Maximum deviation from parent sector weights.
        max_per_sector: Hard cap on members per sector; 0 means no cap.

    Returns:
        Sector-balanced selection.
    """
    parent_sectors = sector_labels.reindex(parent_universe).dropna()
    target_weights = parent_sectors.value_counts(normalize=True)

    result_set: set[str] = set(selected)

    for iteration in range(_MAX_BALANCE_ITERATIONS):
        current_index = pd.Index(list(result_set))
        selected_sectors = sector_labels.reindex(current_index).dropna()
        n_target = len(result_set)  # snapshot — do not mutate during inner loop

        changed = False

        for sector, target_w in target_weights.items():
            min_n = max(0, round((target_w - tolerance) * n_target))
            max_n = round((target_w + tolerance) * n_target)

            current_n = int((selected_sectors == sector).sum())

            if current_n < min_n:
                candidates = sector_labels.index[
                    (sector_labels == sector) & (~sector_labels.index.isin(result_set))
                ]
                candidate_scores = (
                    scores.reindex(candidates).dropna().sort_values(ascending=False)
                )
                to_add = cast(pd.Index, candidate_scores.index[: min_n - current_n])
                if len(to_add) > 0:
                    result_set.update(to_add)
                    selected_sectors = sector_labels.reindex(
                        pd.Index(list(result_set))
                    ).dropna()
                    changed = True

            elif current_n > max_n:
                sector_members = selected_sectors[selected_sectors == sector].index
                sector_scores = scores.reindex(sector_members).dropna().sort_values()
                to_remove = set(
                    cast(pd.Index, sector_scores.index[: current_n - max_n])
                )
                if to_remove:
                    result_set -= to_remove
                    selected_sectors = sector_labels.reindex(
                        pd.Index(list(result_set))
                    ).dropna()
                    changed = True

        if not changed:
            logger.debug(
                "apply_sector_balance converged after %d iteration(s).",
                iteration + 1,
            )
            break
    else:
        logger.warning(
            "apply_sector_balance did not converge within %d iterations. "
            "Returning best-effort result.",
            _MAX_BALANCE_ITERATIONS,
        )

    result_set = _cap_per_sector(result_set, scores, sector_labels, max_per_sector)
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
