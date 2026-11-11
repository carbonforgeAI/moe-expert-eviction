from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestStep:
    """One simultaneous Top-K expert request for one layer and token position."""

    token_index: int
    layer_index: int
    experts: tuple[int, ...]

    @classmethod
    def from_iterable(
        cls,
        token_index: int,
        layer_index: int,
        experts: Iterable[int],
    ) -> RequestStep:
        expert_tuple = tuple(experts)
        if token_index < 0 or layer_index < 0:
            raise ValueError("token_index and layer_index must be non-negative")
        if not expert_tuple:
            raise ValueError("experts must not be empty")
        if any(expert < 0 for expert in expert_tuple):
            raise ValueError("expert ids must be non-negative")
        if len(set(expert_tuple)) != len(expert_tuple):
            raise ValueError("experts in a Top-K request must be unique")
        return cls(token_index=token_index, layer_index=layer_index, experts=expert_tuple)


@dataclass(frozen=True, slots=True)
class RequestTrace:
    """Ordered request steps, potentially spanning several MoE layers."""

    steps: tuple[RequestStep, ...]

    @classmethod
    def from_steps(cls, steps: Iterable[RequestStep]) -> RequestTrace:
        return cls(steps=tuple(steps))
