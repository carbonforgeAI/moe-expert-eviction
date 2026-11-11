from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

MelinoeCacheInitialization = Literal["zero_fill", "uniform"]


@dataclass(slots=True)
class MelinoeSoftCache:
    """MELINOE soft cache state from the paper's cache-simulation loss.

    The state stores one layer/sequence cache vector. Use `zero_fill` for the
    main-method initialization and `uniform` for the paper's no-fill alternative.
    """

    num_experts: int
    capacity: float
    gamma: float = 0.9
    initialization: MelinoeCacheInitialization = "zero_fill"
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32
    state: torch.Tensor = field(init=False)
    _normalizer: torch.Tensor = field(init=False, repr=False)
    _filled: bool = field(init=False, repr=False)
    _device: torch.device | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.capacity > self.num_experts:
            raise ValueError("capacity cannot exceed num_experts")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.initialization not in ("zero_fill", "uniform"):
            raise ValueError("initialization must be 'zero_fill' or 'uniform'")

        self._device = torch.device(self.device) if self.device is not None else None
        if self.initialization == "uniform":
            self.state = torch.full(
                (self.num_experts,),
                float(self.capacity) / float(self.num_experts),
                dtype=self.dtype,
                device=self._device,
            )
            self._normalizer = torch.ones((), dtype=self.dtype, device=self._device)
            self._filled = True
        else:
            self.state = torch.zeros(self.num_experts, dtype=self.dtype, device=self._device)
            self._normalizer = torch.ones((), dtype=self.dtype, device=self._device)
            self._filled = False

    def update(self, request: torch.Tensor) -> torch.Tensor:
        """Apply one Top-K binary request vector and return the post-step state."""

        request = request.to(device=self.state.device, dtype=self.state.dtype)
        _validate_request_vector(request, self.num_experts)

        if self.initialization == "uniform" or self._filled:
            k_over_c = request.sum() / float(self.capacity)
            next_normalizer = self.gamma * self._normalizer + k_over_c
            self.state = (self.gamma * self._normalizer * self.state + request) / next_normalizer
            self._normalizer = next_normalizer
            self._filled = True
            return self.state.clone()

        raw_state = self.gamma * self.state + request
        total = raw_state.sum()
        if bool(total >= float(self.capacity)):
            self.state = raw_state * (float(self.capacity) / total)
            self._normalizer = torch.ones((), dtype=self.state.dtype, device=self.state.device)
            self._filled = True
        else:
            self.state = raw_state
        return self.state.clone()


def melinoe_soft_cache_states(
    requests: torch.Tensor,
    *,
    capacity: float,
    gamma: float = 0.9,
    initialization: MelinoeCacheInitialization = "zero_fill",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pre-step and post-step MELINOE soft-cache states.

    `requests` is shaped `(..., T, E)`, where `T` is the token dimension and
    `E` is the expert dimension. Entries should be binary Top-K request vectors.
    """

    _validate_requests(requests, capacity, gamma, initialization)
    original_shape = requests.shape
    prefix_shape = original_shape[:-2]
    num_rows = int(torch.tensor(prefix_shape).prod().item()) if prefix_shape else 1
    tokens = original_shape[-2]
    experts = original_shape[-1]
    flat_requests = requests.reshape(num_rows, tokens, experts)

    pre_states = torch.empty_like(flat_requests, dtype=requests.dtype)
    post_states = torch.empty_like(flat_requests, dtype=requests.dtype)
    for row_index in range(num_rows):
        cache = MelinoeSoftCache(
            num_experts=experts,
            capacity=capacity,
            gamma=gamma,
            initialization=initialization,
            device=requests.device,
            dtype=requests.dtype,
        )
        for token_index in range(tokens):
            pre_states[row_index, token_index] = cache.state
            post_states[row_index, token_index] = cache.update(
                flat_requests[row_index, token_index]
            )

    return pre_states.reshape(original_shape), post_states.reshape(original_shape)


def melinoe_cache_simulation_loss(
    requests: torch.Tensor,
    *,
    capacity: float,
    gamma: float = 0.9,
    initialization: MelinoeCacheInitialization = "zero_fill",
) -> torch.Tensor:
    """MELINOE cache simulation loss `mean_{prefix,T} <r_t, 1 - c_t>`."""

    pre_states, _post_states = melinoe_soft_cache_states(
        requests,
        capacity=capacity,
        gamma=gamma,
        initialization=initialization,
    )
    miss_proxy = (requests * (1.0 - pre_states)).sum(dim=-1)
    return miss_proxy.mean()


def melinoe_rank_matching_loss(
    base_probs: torch.Tensor,
    finetuned_probs: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """MELINOE rank matching loss averaged over all non-expert dimensions."""

    if base_probs.shape != finetuned_probs.shape:
        raise ValueError("base_probs and finetuned_probs must have the same shape")
    if base_probs.ndim < 1:
        raise ValueError("router probability tensors must have an expert dimension")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    base = base_probs.float()
    finetuned = finetuned_probs.float()
    base_preference = base.unsqueeze(-1) > base.unsqueeze(-2)
    finetuned_margin = margin - (finetuned.unsqueeze(-1) - finetuned.unsqueeze(-2))
    mistakes = torch.where(
        base_preference,
        F.relu(finetuned_margin),
        torch.zeros_like(finetuned_margin),
    )
    return mistakes.sum(dim=(-1, -2)).mean()


def topk_request_vectors(
    router_probs: torch.Tensor,
    *,
    top_k: int,
    straight_through: bool = False,
) -> torch.Tensor:
    """Convert router probabilities shaped `(..., E)` to Top-K request vectors.

    With `straight_through=True`, the forward value remains the hard binary
    Top-K request, while the backward pass uses the router-probability gradient.
    """

    if router_probs.ndim < 1:
        raise ValueError("router_probs must have an expert dimension")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > router_probs.shape[-1]:
        raise ValueError("top_k cannot exceed number of experts")

    indices = torch.topk(router_probs, k=top_k, dim=-1).indices
    hard_requests = torch.zeros_like(router_probs).scatter_(-1, indices, 1.0)
    if straight_through:
        return hard_requests + router_probs - router_probs.detach()
    return hard_requests


def _validate_requests(
    requests: torch.Tensor,
    capacity: float,
    gamma: float,
    initialization: MelinoeCacheInitialization,
) -> None:
    if requests.ndim < 2:
        raise ValueError("requests must be shaped (..., tokens, experts)")
    if not requests.is_floating_point():
        raise ValueError("requests must be a floating-point tensor")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if capacity > requests.shape[-1]:
        raise ValueError("capacity cannot exceed number of experts")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if initialization not in ("zero_fill", "uniform"):
        raise ValueError("initialization must be 'zero_fill' or 'uniform'")
    if bool((requests < 0.0).any() or (requests > 1.0).any()):
        raise ValueError("requests must contain binary values in [0, 1]")


def _validate_request_vector(request: torch.Tensor, num_experts: int) -> None:
    if request.shape != (num_experts,):
        raise ValueError("request must be shaped (num_experts,)")
    if bool((request < 0.0).any() or (request > 1.0).any()):
        raise ValueError("request must contain binary values in [0, 1]")
