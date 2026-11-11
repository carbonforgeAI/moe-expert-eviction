from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AccessResult:
    """Result of one simultaneous Top-K request against a pre-step cache snapshot."""

    requested: tuple[int, ...]
    resident_before: tuple[int, ...]
    hits: tuple[int, ...]
    misses: tuple[int, ...]
    resident_after: tuple[int, ...]

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    @property
    def miss_count(self) -> int:
        return len(self.misses)


class HardLRUCache:
    """Deterministic capacity-limited LRU cache for expert ids.

    Cache order is least-recently-used to most-recently-used.

    A request tuple represents one token/layer Top-K decision. Hit/miss status is
    computed from a snapshot before the step. The request order is then applied to
    update recency and perform evictions deterministically.
    """

    def __init__(self, capacity: int, initial: Iterable[int] = ()) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[int, None] = OrderedDict()
        initial_tuple = tuple(initial)
        if len(set(initial_tuple)) != len(initial_tuple):
            raise ValueError("initial expert ids must be unique")
        if len(initial_tuple) > capacity:
            raise ValueError("initial cache cannot exceed capacity")
        for expert in initial_tuple:
            if expert < 0:
                raise ValueError("expert ids must be non-negative")
            self._items[expert] = None

    @property
    def resident(self) -> tuple[int, ...]:
        """Resident experts ordered from LRU to MRU."""

        return tuple(self._items.keys())

    def access(self, requested: Iterable[int]) -> AccessResult:
        request = tuple(requested)
        if not request:
            raise ValueError("requested experts must not be empty")
        if any(expert < 0 for expert in request):
            raise ValueError("expert ids must be non-negative")
        if len(set(request)) != len(request):
            raise ValueError("requested experts must be unique")

        resident_before = self.resident
        before_set = set(resident_before)
        hits = tuple(expert for expert in request if expert in before_set)
        misses = tuple(expert for expert in request if expert not in before_set)

        for expert in request:
            if expert in self._items:
                self._items.move_to_end(expert, last=True)
            else:
                self._items[expert] = None
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

        return AccessResult(
            requested=request,
            resident_before=resident_before,
            hits=hits,
            misses=misses,
            resident_after=self.resident,
        )
