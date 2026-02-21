from dataclasses import dataclass, field
from typing import Any

from app.services.trading212.protocols import InstrumentFilter


@dataclass
class FilterPipelineImpl:
    _filters: list[InstrumentFilter] = field(default_factory=list)
    _stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def add_filter(self, filter: InstrumentFilter) -> "FilterPipelineImpl":
        self._filters.append(filter)
        self._stats[filter.name] = {"passed": 0, "failed": 0}
        return self

    def apply(self, data: dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
        if not self._filters:
            return True, "No filters configured"

        for f in self._filters:
            passed, reason = f.filter(data, yf_ticker)

            if not passed:
                self._stats[f.name]["failed"] += 1
                return False, f"[{f.name}] {reason}"

            self._stats[f.name]["passed"] += 1

        return True, "Passed all filters"

    def get_summary(self) -> dict[str, dict[str, int]]:
        return self._stats.copy()

    def reset_stats(self) -> None:
        for name in self._stats:
            self._stats[name] = {"passed": 0, "failed": 0}

    def get_filters(self) -> list[InstrumentFilter]:
        return self._filters.copy()

    def get_pipeline_summary(self) -> str:
        lines = ["Filter Pipeline Summary:", "=" * 50]

        for f in self._filters:
            stats = self._stats.get(f.name, {"passed": 0, "failed": 0})
            total = stats["passed"] + stats["failed"]
            pass_rate = (stats["passed"] / total * 100) if total > 0 else 0
            lines.append(
                f"  {f.name}: {stats['passed']} passed, "
                f"{stats['failed']} failed ({pass_rate:.1f}% pass rate)"
            )

        total_passed = min(
            (self._stats.get(f.name, {}).get("passed", 0) for f in self._filters),
            default=0,
        )
        lines.append("=" * 50)
        lines.append(f"Final: {total_passed} stocks passed all filters")

        return "\n".join(lines)
