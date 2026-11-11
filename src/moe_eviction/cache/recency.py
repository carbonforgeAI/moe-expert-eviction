from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class NormalizedExponentialRecency:
    """Generic normalized exponential recency signal.

    This is a research utility, not a paper-faithful MELINOE implementation.
    State is updated as `gamma * state + request`, then scaled down only when
    total mass exceeds `capacity`.
    """

    num_experts: int
    capacity: float
    gamma: float
    state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        self.state = np.zeros(self.num_experts, dtype=np.float64)

    def update(self, requested: Iterable[int]) -> np.ndarray:
        request = tuple(requested)
        if len(set(request)) != len(request):
            raise ValueError("requested experts must be unique")
        if any(expert < 0 or expert >= self.num_experts for expert in request):
            raise ValueError("requested expert out of range")

        self.state *= self.gamma
        self.state[list(request)] += 1.0
        total = float(self.state.sum())
        if total > self.capacity:
            self.state *= self.capacity / total
        return self.state.copy()
