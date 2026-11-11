from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .cache.hard_lru import HardLRUCache


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    requests: int
    hits: int
    misses: int
    weighted_miss_cost: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.requests if self.requests else 0.0


def simulate_lru(
    requests: Iterable[Iterable[int]],
    capacity: int,
    expert_costs: Mapping[int, float] | None = None,
) -> TraceMetrics:
    """Simulate an ordered sequence of simultaneous Top-K requests."""

    if expert_costs is not None:
        for expert, cost in expert_costs.items():
            if expert < 0:
                raise ValueError("expert cost ids must be non-negative")
            if float(cost) < 0:
                raise ValueError("expert costs must be non-negative")

    cache = HardLRUCache(capacity=capacity)
    request_count = 0
    hit_count = 0
    miss_count = 0
    weighted_cost = 0.0

    for request in requests:
        result = cache.access(request)
        request_count += len(result.requested)
        hit_count += result.hit_count
        miss_count += result.miss_count
        for expert in result.misses:
            cost = 1.0 if expert_costs is None else float(expert_costs.get(expert, 1.0))
            weighted_cost += cost

    return TraceMetrics(
        requests=request_count,
        hits=hit_count,
        misses=miss_count,
        weighted_miss_cost=weighted_cost,
    )
