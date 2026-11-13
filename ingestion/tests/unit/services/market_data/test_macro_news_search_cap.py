"""The search share of the macro news budget, budget by budget.

Validates ``app/services/market_data/yfinance/news/macro_news.py``
(``_search_cap``): naming the 40 % share and the 10-article floor kept the cap
the bare literals gave, ``min(max(budget * 4 // 10, 10), budget)``, for every
budget. No requirement id is registered for macro news yet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.news.macro_news import _search_cap


def _cap_with_the_old_literals(max_articles: int) -> int:
    return min(max(max_articles * 4 // 10, 10), max_articles)


@pytest.mark.parametrize("max_articles", [0, 1, 9, 10, 24, 25, 60])
def test_the_cap_is_unchanged_at_each_boundary(max_articles: int) -> None:
    """Empty, below the floor, at it, where the share passes it, and above."""
    assert _search_cap(max_articles) == _cap_with_the_old_literals(max_articles)


def test_the_cap_is_unchanged_for_every_budget_up_to_a_thousand() -> None:
    """Integer division by 10 and by 100 of the scaled budget agree everywhere."""
    budgets = range(1001)

    assert [_search_cap(n) for n in budgets] == [
        _cap_with_the_old_literals(n) for n in budgets
    ]


@pytest.mark.parametrize(
    ("max_articles", "cap"), [(0, 0), (1, 1), (9, 9), (10, 10), (25, 10), (60, 24)]
)
def test_the_cap_takes_the_share_above_the_floor_within_the_budget(
    max_articles: int, cap: int
) -> None:
    """The budget bounds the floor, and the share wins once it passes the floor."""
    assert _search_cap(max_articles) == cap
