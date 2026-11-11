"""Working-set MoE routing: quality vs. transfer cost at a resident-memory budget.

Motivation: the frozen oracle and joint co-adaptation both forced the *exact* previous top-8
across all layers and paid a huge NLL. The per-layer probe showed the median per-layer reuse
cost is tiny (0.01-0.18 nats) — the headroom was hidden by all-layers compounding. This module
tests the gentler, hardware-relevant mechanism: at a resident budget ``C`` experts per layer,

- ``vanilla``: route freely; a non-resident pick is a real host->device transfer (miss against
  an LRU of size C); NLL = base.
- ``static_hot``: hard-restrict per-layer routing to the ``C`` most-popular experts (precomputed
  from the model's own routing); the router still picks its best top-k *within* the resident
  pool; transfers are a one-time load; NLL = base + cost(C).

The win is a budget ``C`` where ``static_hot`` keeps NLL near base while ``vanilla`` still pays
many transfers — matched quality at far lower latency in the offloaded regime. A
``layer_selective`` variant leaves the reuse-intolerant early layers unrestricted.

Pure helpers (``per_layer_popularity``, ``hot_sets``, ``restrict_to_set``) are unit-tested with
synthetic tensors; ``run``/``main`` load OLMoE and need a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .collect_trace import _dtype, _load_prompts, _resolved_model_revision
from .evaluate_cache_rerank import (
    _evaluate,
    _gate_modules,
    _simulate_topk_with_costs,
    _update_resident,
)
from .evaluate_router_state import _load_router_state
from .learned_beta import (
    assemble_policy_features,
    load_learned_beta_checkpoint,
    load_policy_head,
    update_pool_demand,
)

RESTRICTED_FILL = -1.0e4


# ---------------------------------------------------------------------------
# Pure, model-free helpers
# ---------------------------------------------------------------------------
def per_layer_popularity(
    router_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Per-layer expert selection counts over attended tokens.

    ``router_logits`` is ``[batch, num_layers, seq, num_experts]``; returns a long tensor
    ``[num_layers, num_experts]`` counting how often each expert lands in the top-k.
    """
    if router_logits.ndim != 4:
        raise ValueError(
            f"router_logits must be [batch, layers, seq, experts], got {tuple(router_logits.shape)}"
        )
    batch, num_layers, seq, num_experts = router_logits.shape
    active = attention_mask.bool()
    counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
    selected = torch.topk(router_logits, k=top_k, dim=-1).indices  # [b, L, seq, k]
    for layer in range(num_layers):
        idx = selected[:, layer][active]  # [num_active, k]
        flat = idx.reshape(-1)
        counts[layer].index_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
    return counts


def hot_sets(popularity: torch.Tensor, capacity: int) -> dict[int, torch.Tensor]:
    """Top-``capacity`` most-popular experts per layer (the resident pool)."""
    num_layers, num_experts = popularity.shape
    capacity = min(capacity, num_experts)
    return {
        layer: torch.topk(popularity[layer], k=capacity).indices.sort().values
        for layer in range(num_layers)
    }


def restrict_to_set(logits: torch.Tensor, resident: torch.Tensor, *, fill: float = RESTRICTED_FILL):
    """Keep the resident experts' logits along the last dim, set all others to ``fill``."""
    out = torch.full_like(logits, fill)
    resident = resident.to(device=logits.device, dtype=torch.long)
    out[..., resident] = logits[..., resident]
    return out


def select_cheap_layers(profile_rows: list[dict[str, Any]], k: int) -> list[int]:
    """The ``k`` layers with the lowest best-of-pool ``delta_nll`` (ties broken by layer index).

    ``profile_rows`` are the per-layer profile entries (each ``{"layer": int, "delta_nll": float}``)
    written by :func:`_run_per_layer_best_of_pool`. Returns the chosen layers sorted ascending —
    the restricted-layer set for the dynamic-selective policy.
    """
    ranked = sorted(profile_rows, key=lambda r: (float(r["delta_nll"]), int(r["layer"])))
    chosen = [int(r["layer"]) for r in ranked[: max(0, k)]]
    return sorted(chosen)


def _recompute_tuple(
    original: torch.Tensor,
    masked: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selection from the restricted (masked) logits, but weights from the router's *real* full
    softmax over the selected experts. Restriction changes which experts run, not the router's
    confidence — so an admit-all / unrestricted token reproduces vanilla weights exactly.
    """
    indices = torch.topk(masked, top_k, dim=-1).indices
    original_probs = torch.softmax(original, dtype=torch.float, dim=-1)
    scores = torch.gather(original_probs, -1, indices)
    if norm_topk_prob:
        scores = scores / scores.sum(dim=-1, keepdim=True)
    return masked, scores.to(masked.dtype), indices


@dataclass(frozen=True, slots=True)
class WorkingSetRow:
    policy: str
    capacity: int
    layer_policy: str
    nll: float
    perplexity: float
    # Steady-state per-inference transfers (what drives serving latency): for vanilla these are
    # the ongoing LRU misses; for a static resident pool they are 0 (the pool never swaps).
    transfers: int
    transfer_rate: float
    requests: int
    restricted_layers: int
    # One-time cost of loading the resident pool at startup (amortizes to ~0 over a long serve).
    one_time_load: int
    # Admission confidence threshold for the dynamic policy (None for vanilla/static).
    admit_threshold: float | None = None
    # Frozen-pool option length in tokens for the freeze_window policy (None otherwise).
    window_size: int | None = None
    # Learned-beta eval: the beta termination threshold and measured mean dwell (None otherwise).
    beta_threshold: float | None = None
    mean_dwell: float | None = None


# ---------------------------------------------------------------------------
# Gate-hook controller (static hot-set restriction)
# ---------------------------------------------------------------------------
class WorkingSetController:
    """Hard-restricts per-layer routing to a fixed resident hot-set (no per-token LRU update)."""

    def __init__(
        self,
        *,
        hot_sets: dict[int, torch.Tensor],
        restricted_layers: frozenset[int],
        top_k: int,
        batch_size: int,
        seq_len: int,
    ) -> None:
        self.hot = hot_sets
        self.restricted_layers = restricted_layers
        self.top_k = top_k
        self.batch_size = batch_size
        self.seq_len = seq_len

    def adjust(
        self,
        layer_index: int,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm_topk_prob: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_index not in self.restricted_layers:
            return output
        is_tuple = isinstance(output, tuple)
        logits = output[0] if is_tuple else output
        adjusted = restrict_to_set(logits, self.hot[layer_index])
        if not is_tuple:
            return adjusted
        return _recompute_tuple(logits, adjusted, self.top_k, norm_topk_prob)


def dynamic_admit(
    logits_row: torch.Tensor,
    resident: list[int],
    top_k: int,
    threshold: float,
) -> tuple[torch.Tensor, list[int], int]:
    """One token's dynamic-admission decision.

    The router proposes its true top-k; resident experts are free; a missed expert is *admitted*
    (a real transfer) only if its router probability exceeds ``threshold``, otherwise the slot
    falls back to the best resident expert. Returns ``(masked_logits, selected, transfers)``
    where ``transfers`` is the number of selected experts that were not resident.
    """
    probs = torch.softmax(logits_row.float(), dim=-1)
    true_topk = torch.topk(logits_row, top_k, dim=-1).indices.tolist()
    resident_set = set(resident)
    admitted = [e for e in true_topk if e not in resident_set and float(probs[e]) > threshold]
    allowed = sorted(resident_set | set(admitted))
    allowed_t = torch.tensor(allowed, device=logits_row.device, dtype=torch.long)
    masked = torch.full_like(logits_row, RESTRICTED_FILL)
    masked[allowed_t] = logits_row[allowed_t]
    selected = torch.topk(masked, top_k, dim=-1).indices.tolist()
    transfers = sum(1 for e in selected if e not in resident_set)
    return masked, selected, transfers


class DynamicWorkingSetController:
    """LRU resident pool with confidence-thresholded admission, applied at every layer.

    Interpolates between vanilla (threshold 0 -> admit every miss) and the static pool
    (threshold >= 1 -> admit nothing). Counts the real transfers (admitted misses) it incurs.
    """

    def __init__(
        self,
        *,
        attention_mask: torch.Tensor,
        num_layers: int,
        capacity: int,
        top_k: int,
        threshold: float,
        restricted_layers: frozenset[int] | None = None,
    ) -> None:
        self.attention_mask = attention_mask.detach().bool().cpu()
        self.batch_size, self.seq_len = self.attention_mask.shape
        self.capacity = capacity
        self.top_k = top_k
        self.threshold = threshold
        # Layers the resident pool applies to. Unrestricted layers route freely with all experts
        # resident (no transfers) — used to leave the reuse-intolerant early layers full.
        self.restricted_layers = (
            restricted_layers if restricted_layers is not None else frozenset(range(num_layers))
        )
        self.caches: list[list[list[int]]] = [
            [[] for _ in range(num_layers)] for _ in range(self.batch_size)
        ]
        self.transfers = 0

    def adjust(
        self,
        layer_index: int,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm_topk_prob: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_tuple = isinstance(output, tuple)
        logits = output[0] if is_tuple else output
        shape = logits.shape
        if logits.ndim == 2 and logits.shape[0] == self.batch_size * self.seq_len:
            grid = logits.reshape(self.batch_size, self.seq_len, logits.shape[-1]).clone()
            flatten = True
        elif logits.ndim == 3 and logits.shape[:2] == (self.batch_size, self.seq_len):
            grid = logits.clone()
            flatten = False
        else:
            raise RuntimeError(f"unsupported gate output shape: {shape}")

        if layer_index not in self.restricted_layers:
            return output  # unrestricted layer: full routing, all experts resident, no transfers

        for token in range(self.seq_len):
            for batch in range(self.batch_size):
                if not self.attention_mask[batch, token].item():
                    continue
                resident = self.caches[batch][layer_index]
                masked, selected, transfers = dynamic_admit(
                    grid[batch, token], resident, self.top_k, self.threshold
                )
                grid[batch, token] = masked
                self.transfers += transfers
                _update_resident(resident, selected, self.capacity)

        adjusted = grid.reshape(shape) if flatten else grid
        if not is_tuple:
            return adjusted
        return _recompute_tuple(logits, adjusted, self.top_k, norm_topk_prob)


class FreezePoolWindowController:
    """Oracle look-ahead, whole-pool re-selection every ``window_size`` tokens (M0 dwell probe).

    An *option* is the resident pool held frozen for a window of ``window_size`` attended tokens.
    At each window boundary the pool is re-selected (oracle look-ahead) to the ``capacity`` experts
    most demanded over the next window, read straight from the gate's own logit grid — no second
    pass. Between boundaries the pool is frozen and every token routes best-of-pool within it (zero
    transfers). Counts ``transfers = |new_pool \\ old_pool|`` at each boundary, aggregated over
    ``(batch, restricted layer)``. This is the *ceiling* on the policy-over-options: a clairvoyant
    pool with a blind fixed-``k`` termination clock, isolating the mechanism question from learning.

    ``window_size == 1`` re-selects every token (⇒ vanilla NLL, since ``top_k <= capacity``);
    ``window_size >= seq_len`` selects once (⇒ static pool, one-time load only).
    """

    def __init__(
        self,
        *,
        attention_mask: torch.Tensor,
        num_layers: int,
        capacity: int,
        top_k: int,
        window_size: int,
        restricted_layers: frozenset[int] | None = None,
        pool_selection: str = "oracle",
    ) -> None:
        valid_modes = {
            "oracle", "causal_recency", "current_logit_topc", "trailing_logit_topc",
        }
        if pool_selection not in valid_modes:
            raise ValueError(f"pool_selection must be one of {sorted(valid_modes)}")
        self.attention_mask = attention_mask.detach().bool().cpu()
        self.batch_size, self.seq_len = self.attention_mask.shape
        self.capacity = capacity
        self.top_k = top_k
        self.window_size = max(1, int(window_size))
        # 'oracle': re-select from the *next* window (look-ahead, M0 ceiling). 'causal_recency':
        # re-select from the *trailing* window (past + current) -- the fair blind clock.
        self.pool_selection = pool_selection
        self.restricted_layers = (
            restricted_layers if restricted_layers is not None else frozenset(range(num_layers))
        )
        # Attended token positions per batch; window boundaries are ranks within this list, so
        # padded positions never start or pad out a window.
        self.attended: list[list[int]] = [
            [t for t in range(self.seq_len) if bool(self.attention_mask[batch, t])]
            for batch in range(self.batch_size)
        ]
        self.transfers = 0

    def _select_pool(self, window_logits: torch.Tensor) -> list[int]:
        """The ``capacity`` most top-k-popular experts over a ``[w, experts]`` window.

        ``oracle`` / ``causal_recency`` rank experts by *realized top-k demand count* over the
        window (how often each expert would actually be routed to).
        """
        num_experts = window_logits.shape[-1]
        picks = torch.topk(window_logits, self.top_k, dim=-1).indices.reshape(-1).cpu()
        counts = torch.zeros(num_experts, dtype=torch.long)
        counts.index_add_(0, picks, torch.ones_like(picks, dtype=torch.long))
        capacity = min(self.capacity, num_experts)
        return sorted(torch.topk(counts, capacity).indices.tolist())

    def _select_pool_logit(self, window_logits: torch.Tensor) -> list[int]:
        """The ``capacity`` experts with the highest *mean gate logit* over the window.

        The trivial-mu predictors (comparison 2): rank by the router's own preference score, not
        by realized demand count. ``current_logit_topc`` passes the single boundary token (dwell-1
        optimum, since ``capacity >= top_k`` ⇒ covers its true top-k); ``trailing_logit_topc``
        passes a trailing window (a short-horizon causal forecast). No learning, no look-ahead.
        """
        num_experts = window_logits.shape[-1]
        mean_logit = window_logits.float().mean(dim=0)
        capacity = min(self.capacity, num_experts)
        return sorted(torch.topk(mean_logit, capacity).indices.tolist())

    def adjust(
        self,
        layer_index: int,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm_topk_prob: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_tuple = isinstance(output, tuple)
        logits = output[0] if is_tuple else output
        shape = logits.shape
        if logits.ndim == 2 and logits.shape[0] == self.batch_size * self.seq_len:
            grid = logits.reshape(self.batch_size, self.seq_len, logits.shape[-1]).clone()
            flatten = True
        elif logits.ndim == 3 and logits.shape[:2] == (self.batch_size, self.seq_len):
            grid = logits.clone()
            flatten = False
        else:
            raise RuntimeError(f"unsupported gate output shape: {shape}")

        if layer_index not in self.restricted_layers:
            return output  # unrestricted layer: full routing, all experts resident, no transfers

        # Pool selection must read the ORIGINAL (unrestricted) logits: `grid` is mutated in place
        # below (each token is masked to its resident pool), so a trailing/past window read off
        # `grid` would see experts recently *served*, not recently *demanded* -- circular, and for
        # mean-logit selection the -inf masking collapses the pool to the prior one (static).
        select_src = grid.clone()
        for batch in range(self.batch_size):
            attended = self.attended[batch]
            pool: list[int] = []
            for rank, token in enumerate(attended):
                if rank % self.window_size == 0:  # window boundary: re-select the whole pool
                    if self.pool_selection == "oracle":
                        window = attended[rank : rank + self.window_size]  # look-ahead (oracle)
                        new_pool = self._select_pool(select_src[batch, window])
                    elif self.pool_selection == "causal_recency":
                        # trailing window: past + current token only (no look-ahead).
                        window = attended[max(0, rank - self.window_size + 1) : rank + 1]
                        new_pool = self._select_pool(select_src[batch, window])
                    elif self.pool_selection == "current_logit_topc":
                        new_pool = self._select_pool_logit(select_src[batch, [token]])  # current
                    else:  # trailing_logit_topc: short-horizon causal forecast by mean logit
                        window = attended[max(0, rank - self.window_size + 1) : rank + 1]
                        new_pool = self._select_pool_logit(select_src[batch, window])
                    self.transfers += len(set(new_pool) - set(pool))
                    pool = new_pool
                grid[batch, token] = restrict_to_set(
                    grid[batch, token], torch.tensor(pool, dtype=torch.long)
                )

        adjusted = grid.reshape(shape) if flatten else grid
        if not is_tuple:
            return adjusted
        return _recompute_tuple(logits, adjusted, self.top_k, norm_topk_prob)


def _attach(
    model: torch.nn.Module,
    controller: WorkingSetController | DynamicWorkingSetController | FreezePoolWindowController,
) -> list[Any]:
    return [
        module.register_forward_hook(
            lambda module, _inputs, output, layer_index=layer_index: controller.adjust(
                layer_index,
                output,
                norm_topk_prob=bool(getattr(module, "norm_topk_prob", False)),
            )
        )
        for layer_index, module in _gate_modules(model)
    ]


class LearnedBetaPoolController:
    """Hard eval of the learned joint termination (beta) + policy-over-options (mu) on best-of-pool.

    A gate-hook sibling of ``FreezePoolWindowController``: the fixed-k boundary becomes
    ``beta > threshold`` (trained beta head) and the oracle look-ahead pool becomes the trained
    *causal* mu head (running popularity/recency, as at training). Reuses the same transfer
    accounting and ``restrict_to_set`` masking, so its rows overlay on the M0 blind frontier.
    """

    def __init__(
        self,
        *,
        attention_mask: torch.Tensor,
        num_layers: int,
        num_experts: int,
        capacity: int,
        top_k: int,
        beta_head: Any,
        policy_head: Any,
        threshold: float,
        restricted_layers: frozenset[int],
        pop_decay: float = 0.9,
    ) -> None:
        self.attention_mask = attention_mask.detach().bool().cpu()
        self.batch_size, self.seq_len = self.attention_mask.shape
        self.num_experts = int(num_experts)
        self.capacity = min(int(capacity), int(num_experts))
        self.top_k = int(top_k)
        self.beta_head = beta_head
        self.policy_head = policy_head
        self.threshold = float(threshold)
        self.restricted_layers = restricted_layers
        self.pop_decay = float(pop_decay)
        self.pools: list[list[tuple[int, ...] | None]] = [
            [None for _ in range(num_layers)] for _ in range(self.batch_size)
        ]
        self.caches: list[list[list[int]]] = [
            [[] for _ in range(num_layers)] for _ in range(self.batch_size)
        ]
        self.pop_ema = [
            [torch.zeros(num_experts) for _ in range(num_layers)] for _ in range(self.batch_size)
        ]
        self.last_seen = [
            [torch.full((num_experts,), 1.0e9) for _ in range(num_layers)]
            for _ in range(self.batch_size)
        ]
        self.ages = [[0 for _ in range(num_layers)] for _ in range(self.batch_size)]
        self.transfers = 0
        self.option_lengths: list[int] = []

    @torch.no_grad()
    def _beta(self, layer_index: int, hidden: torch.Tensor, pool: tuple[int, ...]) -> float:
        option = torch.tensor(list(pool), dtype=torch.long, device=hidden.device).unsqueeze(0)
        set_features = self.beta_head.set_embedding(option).squeeze(0)
        features = torch.cat((hidden.float(), set_features.float())).unsqueeze(0)
        if self.beta_head.config.shared_across_layers:
            beta = self.beta_head(features)
        else:
            layer_indices = torch.tensor([layer_index], dtype=torch.long, device=hidden.device)
            beta = self.beta_head(features, layer_indices=layer_indices)
        return float(beta.squeeze(0))

    @torch.no_grad()
    def _select_pool(
        self, batch_index: int, layer_index: int, hidden: torch.Tensor, logits: torch.Tensor
    ) -> tuple[int, ...]:
        features = assemble_policy_features(
            hidden, logits, self.pop_ema[batch_index][layer_index],
            self.last_seen[batch_index][layer_index], self.caches[batch_index][layer_index],
            self.num_experts, self.seq_len,
        )
        scores = self.policy_head(features, layer_index)
        indices = torch.topk(scores, self.capacity, dim=-1).indices
        return tuple(sorted(int(expert) for expert in indices.tolist()))

    def adjust(
        self,
        layer_index: int,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm_topk_prob: bool,
        hidden_grid: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_tuple = isinstance(output, tuple)
        logits = output[0] if is_tuple else output
        shape = logits.shape
        if logits.ndim == 2 and logits.shape[0] == self.batch_size * self.seq_len:
            grid = logits.reshape(self.batch_size, self.seq_len, logits.shape[-1]).clone()
            hidden = hidden_grid.reshape(self.batch_size, self.seq_len, hidden_grid.shape[-1])
            flatten = True
        elif logits.ndim == 3 and logits.shape[:2] == (self.batch_size, self.seq_len):
            grid = logits.clone()
            hidden = hidden_grid
            flatten = False
        else:
            raise RuntimeError(f"unsupported gate output shape: {shape}")

        if layer_index not in self.restricted_layers:
            return output

        for batch in range(self.batch_size):
            for token in range(self.seq_len):
                if not self.attention_mask[batch, token].item():
                    continue
                row_logits = grid[batch, token]
                hid = hidden[batch, token]
                pool = self.pools[batch][layer_index]
                fire = pool is None or self._beta(layer_index, hid, pool) > self.threshold
                if fire:
                    new_pool = self._select_pool(batch, layer_index, hid, row_logits)
                    before = set(self.caches[batch][layer_index])
                    self.transfers += sum(1 for expert in new_pool if expert not in before)
                    if pool is not None:
                        self.option_lengths.append(self.ages[batch][layer_index])
                    self.pools[batch][layer_index] = new_pool
                    self.ages[batch][layer_index] = 0
                    _update_resident(self.caches[batch][layer_index], new_pool, self.capacity)
                    pool = new_pool
                self.ages[batch][layer_index] += 1
                grid[batch, token] = restrict_to_set(
                    row_logits, torch.tensor(pool, dtype=torch.long)
                )
                update_pool_demand(
                    self.pop_ema[batch][layer_index], self.last_seen[batch][layer_index],
                    row_logits, self.top_k, self.pop_decay,
                )

        adjusted = grid.reshape(shape) if flatten else grid
        if not is_tuple:
            return adjusted
        return _recompute_tuple(logits, adjusted, self.top_k, norm_topk_prob)

    def finalize(self) -> None:
        for batch in range(self.batch_size):
            for layer_index in self.restricted_layers:
                if self.pools[batch][layer_index] is not None:
                    self.option_lengths.append(self.ages[batch][layer_index])

    @property
    def mean_dwell(self) -> float:
        return sum(self.option_lengths) / len(self.option_lengths) if self.option_lengths else 0.0


def _attach_learned(model: torch.nn.Module, controller: LearnedBetaPoolController) -> list[Any]:
    def make_hook(layer_index: int):
        def hook(module, inputs, output):
            return controller.adjust(
                layer_index,
                output,
                norm_topk_prob=bool(getattr(module, "norm_topk_prob", False)),
                hidden_grid=inputs[0],
            )

        return hook

    return [
        module.register_forward_hook(make_hook(layer_index))
        for layer_index, module in _gate_modules(model)
    ]


def _evaluate_learned_beta_pool(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    num_experts: int,
    capacity: int,
    restricted_layers: frozenset[int],
    threshold: float,
    num_layers: int,
    top_k: int,
    beta_head: Any,
    policy_head: Any,
) -> WorkingSetRow:
    controller = LearnedBetaPoolController(
        attention_mask=encoded["attention_mask"],
        num_layers=num_layers,
        num_experts=num_experts,
        capacity=capacity,
        top_k=top_k,
        beta_head=beta_head,
        policy_head=policy_head,
        threshold=threshold,
        restricted_layers=restricted_layers,
    )
    handles = _attach_learned(model, controller)
    try:
        row, _logits = _evaluate(
            f"learned_beta_t{threshold:g}", model, encoded, labels,
            vanilla_router_logits=None, capacity=capacity, lambda_cache=0.0,
        )
    finally:
        for handle in handles:
            handle.remove()
    controller.finalize()
    return WorkingSetRow(
        policy="learned_beta_pool",
        capacity=capacity,
        layer_policy="selective",
        nll=row.nll,
        perplexity=row.perplexity,
        transfers=controller.transfers,
        transfer_rate=controller.transfers / row.requests if row.requests else 0.0,
        requests=row.requests,
        restricted_layers=len(restricted_layers),
        one_time_load=capacity * len(restricted_layers),
        beta_threshold=threshold,
        mean_dwell=controller.mean_dwell,
    )


def _evaluate_static_hot(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    popularity: torch.Tensor,
    capacity: int,
    restricted_layers: frozenset[int],
    layer_policy: str,
    top_k: int,
) -> WorkingSetRow:
    controller = WorkingSetController(
        hot_sets=hot_sets(popularity, capacity),
        restricted_layers=restricted_layers,
        top_k=top_k,
        batch_size=int(encoded["input_ids"].shape[0]),
        seq_len=int(encoded["input_ids"].shape[1]),
    )
    handles = _attach(model, controller)
    try:
        row, _logits = _evaluate(
            f"static_hot_{layer_policy}",
            model,
            encoded,
            labels,
            vanilla_router_logits=None,
            capacity=capacity,
            lambda_cache=0.0,
        )
    finally:
        for handle in handles:
            handle.remove()
    # A static resident pool never swaps during inference -> zero steady-state transfers.
    # ``_evaluate``'s miss count is derived from outputs.router_logits, which does not reliably
    # reflect the gate-hook masking, so we account the transfers analytically instead.
    return WorkingSetRow(
        policy="static_hot",
        capacity=capacity,
        layer_policy=layer_policy,
        nll=row.nll,
        perplexity=row.perplexity,
        transfers=0,
        transfer_rate=0.0,
        requests=row.requests,
        restricted_layers=len(restricted_layers),
        one_time_load=capacity * len(restricted_layers),
    )


def _evaluate_dynamic(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    capacity: int,
    threshold: float,
    num_layers: int,
    top_k: int,
    restricted_layers: frozenset[int],
    layer_policy: str,
) -> WorkingSetRow:
    controller = DynamicWorkingSetController(
        attention_mask=encoded["attention_mask"],
        num_layers=num_layers,
        capacity=capacity,
        top_k=top_k,
        threshold=threshold,
        restricted_layers=restricted_layers,
    )
    handles = _attach(model, controller)
    try:
        row, _logits = _evaluate(
            f"dynamic_{layer_policy}_t{threshold:g}",
            model,
            encoded,
            labels,
            vanilla_router_logits=None,
            capacity=capacity,
            lambda_cache=0.0,
        )
    finally:
        for handle in handles:
            handle.remove()
    return WorkingSetRow(
        policy="dynamic",
        capacity=capacity,
        layer_policy=layer_policy,
        nll=row.nll,
        perplexity=row.perplexity,
        transfers=controller.transfers,
        transfer_rate=controller.transfers / row.requests if row.requests else 0.0,
        requests=row.requests,
        restricted_layers=len(restricted_layers),
        one_time_load=capacity * len(restricted_layers),
        admit_threshold=threshold,
    )


def _run_per_layer_best_of_pool(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    popularity: torch.Tensor,
    base_nll: float,
    capacity: int,
    num_layers: int,
    top_k: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Best-of-pool analog of the oracle per-layer probe: restrict *one* layer at a time to its
    ``capacity`` most-popular experts (router still picks its best top-k within that pool) and
    measure the resulting whole-sequence ``delta_nll`` vs base. Low-cost layers are the ones the
    dynamic-selective policy should restrict. Writes ``per_layer_best_of_pool_profile.json``.
    """
    rows: list[dict[str, Any]] = []
    for layer in range(num_layers):
        row = _evaluate_static_hot(
            model, encoded, labels, popularity, capacity,
            frozenset({layer}), f"layer_{layer}", top_k,
        )
        rows.append({"layer": layer, "nll": row.nll, "delta_nll": row.nll - base_nll})
    profile = {
        "base_nll": base_nll,
        "capacity": capacity,
        "num_layers": num_layers,
        "top_k": top_k,
        "per_layer": rows,
        "note": (
            "Per-layer best-of-pool cost: layer L is hard-restricted (alone) to its C most-popular "
            "experts; delta_nll = restricted NLL - base. The gentler analog of the exact-reuse "
            "per-layer probe (oracle_reuse_nll_frontier). Feed the cheapest layers into "
            "select_cheap_layers for the dynamic-selective restrict_layers set."
        ),
    }
    (output_dir / "per_layer_best_of_pool_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    return rows


def _evaluate_freeze_window(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    capacity: int,
    restricted_layers: frozenset[int],
    layer_policy: str,
    window_size: int,
    num_layers: int,
    top_k: int,
    pool_selection: str = "oracle",
) -> WorkingSetRow:
    """Freeze each restricted layer's pool for ``window_size`` tokens, re-select at each boundary
    (oracle look-ahead or causal recency), and measure whole-sequence NLL + boundary transfers."""
    controller = FreezePoolWindowController(
        attention_mask=encoded["attention_mask"],
        num_layers=num_layers,
        capacity=capacity,
        top_k=top_k,
        window_size=window_size,
        restricted_layers=restricted_layers,
        pool_selection=pool_selection,
    )
    handles = _attach(model, controller)
    try:
        row, _logits = _evaluate(
            f"freeze_{layer_policy}_w{window_size}",
            model,
            encoded,
            labels,
            vanilla_router_logits=None,
            capacity=capacity,
            lambda_cache=0.0,
        )
    finally:
        for handle in handles:
            handle.remove()
    return WorkingSetRow(
        policy="freeze_window",
        capacity=capacity,
        layer_policy=layer_policy,
        nll=row.nll,
        perplexity=row.perplexity,
        transfers=controller.transfers,
        transfer_rate=controller.transfers / row.requests if row.requests else 0.0,
        requests=row.requests,
        restricted_layers=len(restricted_layers),
        one_time_load=capacity * len(restricted_layers),
        window_size=window_size,
    )


def _run_per_layer_freeze_window(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    base_nll: float,
    capacity: int,
    window_sizes: tuple[int, ...],
    restricted_layers: frozenset[int],
    num_layers: int,
    top_k: int,
    output_dir: Path,
    pool_selection: str = "oracle",
) -> list[dict[str, Any]]:
    """M0 per-layer dwell probe: for each restricted layer (alone) and each window size, freeze the
    layer's pool for that many tokens and record ``(nll, delta_nll, transfers)``. Reveals *which*
    layers tolerate long dwell at matched quality. Writes ``per_layer_freeze_window_profile.json``.
    """
    rows: list[dict[str, Any]] = []
    for layer in sorted(restricted_layers):
        for window_size in window_sizes:
            row = _evaluate_freeze_window(
                model, encoded, labels, capacity, frozenset({layer}),
                f"layer_{layer}", window_size, num_layers, top_k, pool_selection,
            )
            rows.append({
                "layer": layer,
                "window_size": window_size,
                "nll": row.nll,
                "delta_nll": row.nll - base_nll,
                "transfers": row.transfers,
            })
    profile = {
        "base_nll": base_nll,
        "capacity": capacity,
        "num_layers": num_layers,
        "top_k": top_k,
        "window_sizes": list(window_sizes),
        "restrict_layers": sorted(restricted_layers),
        "per_layer": rows,
        "note": (
            "M0 mechanism-ceiling probe: layer L (alone) freezes its C-expert pool for "
            "window_size tokens, re-selecting by oracle look-ahead at each boundary; "
            "delta_nll = frozen NLL - base. dwell tolerance = the largest window_size holding "
            "delta_nll <= budget. Gates the learned per-layer termination "
            "(DELIBERATION_COST_OPTIONS_EXECPLAN.md)."
        ),
    }
    (output_dir / "per_layer_freeze_window_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    return rows


def run(
    config: dict[str, Any],
    output_dir: Path,
    router_state_path: Path | None = None,
    learned_beta_state: Path | None = None,
    policy_head_state: Path | None = None,
) -> None:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_config = config["model"]
    eval_config = config.get("eval", {})
    model_id = model_config["id"]
    revision = model_config.get("revision")
    device = torch.device(model_config.get("device", "cuda"))
    dtype = _dtype(model_config.get("dtype", "bfloat16"))
    capacities = tuple(int(c) for c in eval_config.get("capacities", [8, 12, 16, 24, 32, 48]))
    early_layers = int(eval_config.get("layer_selective_loose_below", 3))
    # Layers restricted by the "selective" policy. Default = the reuse-tolerant mid-deep layers
    # from the per-layer probe (cheap to constrain); early layers stay full.
    restrict_layers_cfg = eval_config.get("restrict_layers")
    dump_trace = bool(eval_config.get("dump_trace", False))
    per_layer_profile = bool(eval_config.get("per_layer_profile", False))
    profile_capacity = int(eval_config.get("profile_capacity", 32))
    # M0 dwell probe: freeze the resident pool for each of these token windows (oracle re-select).
    freeze_durations = tuple(int(w) for w in eval_config.get("freeze_durations", []))
    freeze_capacity = int(eval_config.get("freeze_capacity", 32))
    # 'oracle' (M0 look-ahead ceiling) or 'causal_recency' (the fair causal blind-clock baseline).
    freeze_pool_selection = str(eval_config.get("freeze_pool_selection", "oracle"))
    # M2 learned-beta eval: sweep hard termination thresholds for the trained beta+mu heads.
    learned_beta_thresholds = tuple(
        float(t) for t in eval_config.get("learned_beta_thresholds", [])
    )
    learned_beta_capacity = int(eval_config.get("learned_beta_capacity", 32))
    preload_router_state = bool(
        eval_config.get("preload_router_state", eval_config.get("router_state_preload", False))
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = _load_prompts({"trace": eval_config})[: int(eval_config.get("max_sequences", 8))]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(eval_config.get("max_length", 256)),
    ).to(device)
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    loaded_names: tuple[str, ...] = ()
    if router_state_path is not None and preload_router_state:
        loaded_names = _load_router_state(model, router_state_path)

    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    num_layers = len(_gate_modules(model))
    attention_mask = encoded["attention_mask"].detach().cpu().bool()

    # Vanilla pass: base NLL + routing (for popularity and per-capacity miss counts).
    vanilla_row, vanilla_router_logits = _evaluate(
        "vanilla", model, encoded, labels, vanilla_router_logits=None,
        capacity=capacities[0], lambda_cache=0.0,
    )
    popularity = per_layer_popularity(vanilla_router_logits, attention_mask, top_k)
    all_layers = frozenset(range(num_layers))
    if restrict_layers_cfg is not None:
        selective_layers = frozenset(int(layer) for layer in restrict_layers_cfg)
    else:
        selective_layers = frozenset(range(early_layers, num_layers))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Exp 1: dump the vanilla-pass routing (ReMoE state loaded) as a replay trace for the
    # cache-serving stall benchmark — schema matches ``cache_serving_stall_benchmark._load_trace``.
    if dump_trace:
        stacked = vanilla_router_logits.detach().cpu().float()
        selected = torch.topk(stacked, k=top_k, dim=-1).indices
        np.savez_compressed(
            output_dir / "router_trace.npz",
            input_ids=encoded["input_ids"].detach().cpu().numpy(),
            attention_mask=encoded["attention_mask"].detach().cpu().numpy(),
            router_logits=stacked.numpy(),
            selected_experts=selected.numpy().astype(np.int64),
        )

    # Exp 2: per-layer best-of-pool cost profile (restrict one layer at a time to its hot-C pool).
    if per_layer_profile:
        _run_per_layer_best_of_pool(
            model, encoded, labels, popularity, vanilla_row.nll,
            profile_capacity, num_layers, top_k, output_dir,
        )

    # M0: per-layer dwell-tolerance profile (freeze one layer at a time across the window sweep).
    if freeze_durations:
        _run_per_layer_freeze_window(
            model, encoded, labels, vanilla_row.nll, freeze_capacity, freeze_durations,
            selective_layers, num_layers, top_k, output_dir, freeze_pool_selection,
        )

    rows: list[WorkingSetRow] = []
    for capacity in capacities:
        misses, _wcost, requests = _simulate_topk_with_costs(
            vanilla_router_logits, attention_mask, top_k=top_k, capacity=capacity,
            expert_costs=None,
        )
        rows.append(
            WorkingSetRow(
                policy="vanilla", capacity=capacity, layer_policy="none",
                nll=vanilla_row.nll, perplexity=vanilla_row.perplexity,
                transfers=misses, transfer_rate=misses / requests if requests else 0.0,
                requests=requests, restricted_layers=0, one_time_load=0,
            )
        )
        rows.append(
            _evaluate_static_hot(
                model, encoded, labels, popularity, capacity, all_layers, "all_layers", top_k
            )
        )
        rows.append(
            _evaluate_static_hot(
                model, encoded, labels, popularity, capacity, selective_layers,
                "selective", top_k,
            )
        )

    # Dynamic admission sweep: interpolate vanilla <-> static via a confidence threshold.
    dynamic_capacities = tuple(int(c) for c in eval_config.get("dynamic_capacities", []))
    dynamic_thresholds = tuple(float(t) for t in eval_config.get("dynamic_thresholds", []))
    # Which layer policies to run the dynamic sweep under: "all_layers" and/or "selective".
    dynamic_layer_policies = tuple(eval_config.get("dynamic_layer_policies", ["all_layers"]))
    policy_layers = {"all_layers": all_layers, "selective": selective_layers}
    for capacity in dynamic_capacities:
        for layer_policy in dynamic_layer_policies:
            restricted = policy_layers[str(layer_policy)]
            for threshold in dynamic_thresholds:
                rows.append(
                    _evaluate_dynamic(
                        model, encoded, labels, capacity, threshold, num_layers, top_k,
                        restricted, str(layer_policy),
                    )
                )

    # M0 joint frontier: freeze all restricted layers together across the window sweep. Each row is
    # an aggregate (dwell=window_size, NLL, transfers) point + the blind-clock baseline for M2.
    for window_size in freeze_durations:
        rows.append(
            _evaluate_freeze_window(
                model, encoded, labels, freeze_capacity, selective_layers,
                "selective", window_size, num_layers, top_k, freeze_pool_selection,
            )
        )

    # M2: learned joint beta+mu, hard eval across termination thresholds -> (dwell, NLL, transfers)
    # points that overlay on the M0 blind fixed-k frontier.
    if learned_beta_thresholds and learned_beta_state is not None and policy_head_state is not None:
        beta_head = load_learned_beta_checkpoint(Path(learned_beta_state), device)
        policy_head = load_policy_head(Path(policy_head_state), device)
        for threshold in learned_beta_thresholds:
            rows.append(
                _evaluate_learned_beta_pool(
                    model, encoded, labels, beta_head.num_experts, learned_beta_capacity,
                    selective_layers, threshold, num_layers, top_k, beta_head, policy_head,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "working_set_eval.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")
    metadata = {
        "seed": seed,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": _resolved_model_revision(model),
        "dtype": str(dtype).removeprefix("torch."),
        "capacities": capacities,
        "num_layers": num_layers,
        "top_k": top_k,
        "layer_selective_loose_below": early_layers,
        "base_nll": vanilla_row.nll,
        "base_perplexity": math.exp(vanilla_row.nll),
        "num_prompts": len(prompts),
        "sequence_length": int(encoded["input_ids"].shape[1]),
        "router_state_path": str(router_state_path) if router_state_path is not None else None,
        "loaded_router_parameter_count": len(loaded_names),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "note": (
            "Working-set routing: static_hot hard-restricts per-layer top-k to the C most-popular "
            "experts (one-time load); vanilla counts misses against an LRU of size C. Pair with "
            "gpu_transfer_benchmark ms/transfer for a quality-vs-latency frontier."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Working-set MoE routing quality/transfer eval.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--router-state", type=Path)
    parser.add_argument("--learned-beta-state", type=Path)
    parser.add_argument("--policy-head-state", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(
        config, args.output_dir, args.router_state,
        args.learned_beta_state, args.policy_head_state,
    )


if __name__ == "__main__":
    main()
