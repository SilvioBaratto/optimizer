"""Stock selection from composite scores."""

from __future__ import annotations

import logging

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

    Parameters
    ----------
    scores : pd.Series
        Composite scores indexed by ticker.
    target_count : int
        Target number of stocks.
    buffer_fraction : float
        Buffer as a fraction of target_count.  Current members
        within the buffer zone are retained.
    current_members : pd.Index or None
        Tickers currently selected.

    Returns
    -------
    pd.Index
        Selected tickers.
    """
    ranked = scores.dropna().sort_values(ascending=False)

    if len(ranked) <= target_count:
        return ranked.index

    # Direct entrants: top target_count
    direct = ranked.index[:target_count]

    if current_members is None or len(current_members) == 0:
        return direct

    # Buffer zone: extend selection by buffer_fraction
    buffer_size = max(1, int(target_count * buffer_fraction))
    extended_idx = min(target_count + buffer_size, len(ranked))
    buffer_zone = ranked.index[target_count:extended_idx]

    # Retain current members that fall in the buffer zone
    retained = current_members.intersection(buffer_zone)

    return direct.union(retained)


def select_quantile(
    scores: pd.Series,
    target_quantile: float = 0.8,
    exit_quantile: float | None = None,
    current_members: pd.Index | None = None,
) -> pd.Index:
    """Select stocks above a quantile threshold.

    Parameters
    ----------
    scores : pd.Series
        Composite scores indexed by ticker.
    target_quantile : float
        Quantile threshold for entry (0-1).
    exit_quantile : float or None
        Quantile threshold for exit (hysteresis).  If ``None``,
        uses ``target_quantile``.
    current_members : pd.Index or None
        Currently selected tickers.

    Returns
    -------
    pd.Index
        Selected tickers.
    """
    if exit_quantile is None:
        exit_quantile = target_quantile

    valid = scores.dropna()
    if len(valid) == 0:
        return pd.Index([])

    entry_threshold = valid.quantile(target_quantile)
    new_entrants = valid.index[valid >= entry_threshold]

    if current_members is None or len(current_members) == 0:
        return new_entrants

    exit_threshold = valid.quantile(exit_quantile)
    surviving = current_members.intersection(valid.index)
    surviving = surviving[valid.loc[surviving] >= exit_threshold]

    return surviving.union(new_entrants)


def apply_sector_balance(
    selected: pd.Index,
    scores: pd.Series,
    sector_labels: pd.Series,
    parent_universe: pd.Index,
    tolerance: float = 0.05,
) -> pd.Index:
    """Adjust selection for sector-proportional representation.

    Iterates the balance pass until convergence (no further adds or
    removes are needed) or until ``_MAX_BALANCE_ITERATIONS`` is reached.
    A warning is logged if the cap is hit before convergence.

    Parameters
    ----------
    selected : pd.Index
        Initially selected tickers.
    scores : pd.Series
        Composite scores for all candidates.
    sector_labels : pd.Series
        Sector label per ticker.
    parent_universe : pd.Index
        Full universe for computing target sector weights.
    tolerance : float
        Maximum deviation from parent sector weights.

    Returns
    -------
    pd.Index
        Sector-balanced selection.
    """
    parent_sectors = sector_labels.reindex(parent_universe).dropna()
    target_weights = parent_sectors.value_counts(normalize=True)

    result_set: set[str] = set(selected)

    for iteration in range(_MAX_BALANCE_ITERATIONS):
        current_index = pd.Index(list(result_set))
        selected_sectors = sector_labels.reindex(current_index).dropna()
        n_selected = len(result_set)

        changed = False

        for sector, target_w in target_weights.items():
            min_n = max(0, round((target_w - tolerance) * n_selected))
            max_n = round((target_w + tolerance) * n_selected)

            current_n = int((selected_sectors == sector).sum())

            if current_n < min_n:
                candidates = sector_labels.index[
                    (sector_labels == sector)
                    & (~sector_labels.index.isin(result_set))
                ]
                candidate_scores = (
                    scores.reindex(candidates).dropna().sort_values(ascending=False)
                )
                to_add = candidate_scores.index[: min_n - current_n]
                if len(to_add) > 0:
                    result_set.update(to_add)
                    selected_sectors = sector_labels.reindex(
                        pd.Index(list(result_set))
                    ).dropna()
                    n_selected = len(result_set)
                    changed = True

            elif current_n > max_n:
                sector_members = selected_sectors[
                    selected_sectors == sector
                ].index
                sector_scores = (
                    scores.reindex(sector_members).dropna().sort_values()
                )
                to_remove = set(sector_scores.index[: current_n - max_n])
                if to_remove:
                    result_set -= to_remove
                    selected_sectors = sector_labels.reindex(
                        pd.Index(list(result_set))
                    ).dropna()
                    n_selected = len(result_set)
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

    return pd.Index(sorted(result_set))


def compute_selection_turnover(
    current: pd.Index,
    new: pd.Index,
    universe: pd.Index,
) -> float:
    """Compute selection turnover as fraction of universe changed.

    Parameters
    ----------
    current : pd.Index
        Currently selected tickers.
    new : pd.Index
        Newly selected tickers.
    universe : pd.Index
        Full investable universe.

    Returns
    -------
    float
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

    Parameters
    ----------
    scores : pd.Series
        Composite scores indexed by ticker.
    config : SelectionConfig or None
        Selection configuration.
    current_members : pd.Index or None
        Currently selected tickers for buffer/hysteresis.
    sector_labels : pd.Series or None
        Sector labels for sector balancing.
    parent_universe : pd.Index or None
        Full universe for sector weight targets.
    return_turnover : bool
        When ``True``, return ``(selected, turnover)`` tuple.

    Returns
    -------
    pd.Index or tuple[pd.Index, float]
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
        )

    if return_turnover:
        prev = current_members if current_members is not None else pd.Index([])
        univ = parent_universe if parent_universe is not None else scores.index
        turnover = compute_selection_turnover(prev, selected, univ)
        return selected, turnover

    return selected
