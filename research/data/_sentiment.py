"""News sentiment data assembly — sentiment scores from MacroNewsSummary.

Extracted from ``data_assembly.py``.
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

logger = logging.getLogger(__name__)


def assemble_sentiment(
    session: Session,
    start_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Build a dates x country DataFrame of news sentiment scores.

    Queries ``macro_news_summaries`` for the ``sentiment_score`` field
    (continuous [-1, 1]) produced by the daily news pipeline.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    start_date : datetime.date | None
        Optional lower bound on summary_date.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex (summary_date), columns = country strings.
        Values are sentiment scores in [-1, 1].
    """
    from app.models.macro.macro_regime import MacroNewsSummary

    stmt = select(
        MacroNewsSummary.summary_date,
        MacroNewsSummary.country,
        MacroNewsSummary.sentiment_score,
    )
    if start_date is not None:
        stmt = stmt.where(MacroNewsSummary.summary_date >= start_date)
    stmt = stmt.order_by(MacroNewsSummary.summary_date)
    rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame()

    records = [
        {"date": pd.Timestamp(d), "country": country, "sentiment_score": float(score)}
        for d, country, score in rows
        if score is not None
    ]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivoted = df.pivot_table(
        index="date",
        columns="country",
        values="sentiment_score",
        aggfunc="first",
    )
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted.columns.name = None
    return pivoted.sort_index()
