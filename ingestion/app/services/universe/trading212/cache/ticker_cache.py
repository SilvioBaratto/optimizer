from dataclasses import dataclass, field


@dataclass
class TickerMappingCache:
    """In-memory cache mapping (symbol, exchange) → yfinance ticker.

    One instance is created per universe build job and lives for the
    duration of that job, avoiding redundant yfinance lookups within
    the same build.
    """

    _store: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)
    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)

    def get_mapping(self, symbol: str, exchange_name: str) -> str | None:
        key = (symbol, exchange_name)
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def save_mapping(self, symbol: str, exchange_name: str, yf_ticker: str) -> None:
        self._store[(symbol, exchange_name)] = yf_ticker

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
