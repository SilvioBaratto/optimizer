"""Domain exceptions for web scraper modules."""


class ParseStructureError(Exception):
    """Raised when a scraper parses fewer rows than the minimum threshold.

    Indicates the remote website's HTML structure has changed rather than
    the data being legitimately empty.
    """

    def __init__(self, message: str, url: str, rows_found: int) -> None:
        super().__init__(message)
        self.url = url
        self.rows_found = rows_found
