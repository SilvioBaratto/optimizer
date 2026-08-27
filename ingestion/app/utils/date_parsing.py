"""Compatibility shim — moved to ``portopt_db.coerce`` (P1 of the portopt-db
extraction). Re-exported so existing ``app.utils.date_parsing`` imports keep
working; removed in P4.2.
"""

from portopt_db.coerce import parse_reference_date

__all__ = ["parse_reference_date"]
