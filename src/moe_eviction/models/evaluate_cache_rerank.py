from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_eviction.experiments.call_return_frontier import (
    _edit_distance,
    _forced_option_kl,
    _option_logit_regret,
    _topk,
)
from moe_eviction.experiments.heterogeneous_cost_frontier import _cost_profile
from moe_eviction.experiments.trace_frontier import _simulate_topk

from .collect_trace import _dtype, _load_prompts, _resolved_model_revision
from .evaluate_router_state import _load_router_state, _router_kl, _router_stack
from .learned_beta import (
    LearnedBetaSparseMoeController,
    LearnedBetaSparseMoeWrapper,
    LearnedTerminationHead,
    _restricted_gate_scores,
    learned_beta_metadata,
    load_learned_beta_checkpoint,
)
from .lora_experts import LoRAExpertsConfig, attach_lora_experts, load_lora_state


@dataclass(frozen=True, slots=True)
class RerankEvalRow:
    method: str
    cache_capacity: int
    lambda_cache: float
    nll: float
    perplexity: float
    hard_misses: int
    hard_miss_rate: float
    weighted_miss_cost: float
    requests: int
    router_kl_to_vanilla: float
    controller_calls: int = 0
    controller_call_rate: float = 0.0
    option_switches: int = 0
    switch_edit_distance: int = 0
    gate_evaluated_tokens: int = 0
    gate_evaluation_rate: float = 0.0
    mean_option_length: float = 0.0
    router_logit_regret: float = 0.0
    deliberation_cost: float = 0.0
    switch_cost: float = 0.0
    total_cost: float = 0.0
    controller_mode: str | None = None
    call_return_runtime: str | None = None
    continuation_weighting: str | None = None
    option_length: int | None = None
    termination_threshold: float | None = None
    beta_calls: int = 0
    forward_elapsed_ms: float = 0.0


def _update_resident(resident: list[int], selected: list[int], capacity: int) -> None:
    for expert in selected:
        if expert in resident:
            resident.remove(expert)
        resident.append(expert)
        while len(resident) > capacity:
            resident.pop(0)


class CacheRerankController:
    def __init__(
        self,
        attention_mask: torch.Tensor,
        num_layers: int,
        capacity: int,
        top_k: int,
        lambda_cache: float,
        expert_costs: torch.Tensor | None = None,
        weighted_penalty: bool = False,
    ) -> None:
        self.attention_mask = attention_mask.detach().bool().cpu()
        self.batch_size, self.seq_len = self.attention_mask.shape
        self.capacity = capacity
        self.top_k = top_k
        self.lambda_cache = lambda_cache
        self.expert_costs = (
            expert_costs.detach().float().cpu() if expert_costs is not None else None
        )
        self.weighted_penalty = weighted_penalty
        self.caches: list[list[list[int]]] = []
        self.reset(num_layers)

    def reset(self, num_layers: int | None = None) -> None:
        layers = num_layers if num_layers is not None else len(self.caches[0])
        self.caches = [[[] for _layer in range(layers)] for _batch in range(self.batch_size)]

    def adjust(
        self,
        layer_index: int,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm_topk_prob: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(output, tuple):
            adjusted_logits = self._adjust_logits(layer_index, output[0])
            router_probs = torch.softmax(adjusted_logits, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
            if norm_topk_prob:
                router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
            router_scores = router_top_value.to(adjusted_logits.dtype)
            return adjusted_logits, router_scores, router_indices
        return self._adjust_logits(layer_index, output)

    def _adjust_logits(self, layer_index: int, output: torch.Tensor) -> torch.Tensor:
        if self.lambda_cache == 0.0:
            return output
        original_shape = output.shape
        if output.ndim == 2 and output.shape[0] == self.batch_size * self.seq_len:
            logits = output.reshape(self.batch_size, self.seq_len, output.shape[-1]).clone()
            flatten = True
        elif output.ndim == 3 and output.shape[:2] == (self.batch_size, self.seq_len):
            logits = output.clone()
            flatten = False
        else:
            raise RuntimeError(f"unsupported gate output shape for cache rerank: {original_shape}")

        for token_index in range(self.seq_len):
            for batch_index in range(self.batch_size):
                if not self.attention_mask[batch_index, token_index].item():
                    continue

                resident = self.caches[batch_index][layer_index]
                cold = torch.ones(logits.shape[-1], dtype=logits.dtype, device=logits.device)
                if resident:
                    cold[torch.tensor(resident, device=logits.device)] = 0
                penalty = cold
                if self.weighted_penalty:
                    if self.expert_costs is None:
                        raise RuntimeError("weighted cache rerank requires expert costs")
                    penalty = penalty * self.expert_costs.to(
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                adjusted = logits[batch_index, token_index] - self.lambda_cache * penalty
                logits[batch_index, token_index] = adjusted

                selected = torch.topk(adjusted.detach(), k=self.top_k, dim=-1).indices.tolist()
                _update_resident(resident, selected, self.capacity)

        return logits.reshape(original_shape) if flatten else logits


class MoccEController:
    def __init__(
        self,
        attention_mask: torch.Tensor,
        num_layers: int,
        capacity: int,
        top_k: int,
        mode: str,
        max_rank: int,
        min_rank: int,
        probability_threshold: float,
        prior_lambda: float = 0.0,
    ) -> None:
        self.attention_mask = attention_mask.detach().bool().cpu()
        self.batch_size, self.seq_len = self.attention_mask.shape
        self.capacity = capacity
        self.top_k = top_k
        self.mode = mode
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.probability_threshold = probability_threshold
        self.prior_lambda = prior_lambda
        self.caches: list[list[list[int]]] = []
        self.range_averages: list[float] = []
        self.range_counts: list[int] = []
        self._current_range_average = 0.0
        self.reset(num_layers)

    def reset(self, num_layers: int | None = None) -> None:
        layers = num_layers if num_layers is not None else len(self.caches[0])
        self.caches = [[[] for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.range_averages = [0.0 for _layer in range(layers)]
        self.range_counts = [0 for _layer in range(layers)]

    def adjust(
        self,
        layer_index: int,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm_topk_prob: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(output, tuple):
            raise RuntimeError("MoCCE-style reranking requires gate output indices")

        logits = output[0]
        original_shape = logits.shape
        adjusted_logits = logits.clone()
        if adjusted_logits.ndim == 2 and adjusted_logits.shape[0] == self.batch_size * self.seq_len:
            shaped_logits = adjusted_logits.reshape(
                self.batch_size,
                self.seq_len,
                adjusted_logits.shape[-1],
            )
            flatten = True
        elif adjusted_logits.ndim == 3 and adjusted_logits.shape[:2] == (
            self.batch_size,
            self.seq_len,
        ):
            shaped_logits = adjusted_logits
            flatten = False
        else:
            raise RuntimeError(f"unsupported gate output shape for MoCCE rerank: {original_shape}")

        router_indices = output[2].clone()
        if flatten:
            router_indices = router_indices.reshape(self.batch_size, self.seq_len, self.top_k)
        router_scores = output[1].clone()
        if flatten:
            router_scores = router_scores.reshape(self.batch_size, self.seq_len, self.top_k)

        for token_index in range(self.seq_len):
            for batch_index in range(self.batch_size):
                if not self.attention_mask[batch_index, token_index].item():
                    continue

                resident = self.caches[batch_index][layer_index]
                token_logits = shaped_logits[batch_index, token_index]
                self._current_range_average = self._update_range_average(
                    layer_index,
                    token_logits,
                )
                selected = self._select(token_logits, resident)
                selected_tensor = torch.tensor(
                    selected,
                    dtype=router_indices.dtype,
                    device=logits.device,
                )
                token_probs = torch.softmax(token_logits.float(), dim=-1)
                selected_scores = token_probs[selected_tensor].to(router_scores.dtype)
                if norm_topk_prob:
                    selected_scores /= selected_scores.sum(dim=-1, keepdim=True)
                high_score = token_logits.max().to(adjusted_logits.dtype)
                for offset, expert in enumerate(selected):
                    shaped_logits[batch_index, token_index, expert] = high_score + float(
                        self.top_k - offset
                    )
                router_indices[batch_index, token_index] = selected_tensor
                router_scores[batch_index, token_index] = selected_scores
                _update_resident(resident, selected, self.capacity)

        if flatten:
            router_indices = router_indices.reshape(output[2].shape)
            router_scores = router_scores.reshape(output[1].shape)
        return adjusted_logits, router_scores, router_indices

    def _update_range_average(self, layer_index: int, token_logits: torch.Tensor) -> float:
        current_range = float((token_logits.max() - token_logits.min()).detach().cpu())
        count = self.range_counts[layer_index] + 1
        previous = self.range_averages[layer_index]
        updated = previous + (current_range - previous) / count
        self.range_counts[layer_index] = count
        self.range_averages[layer_index] = updated
        return updated

    def _select(self, token_logits: torch.Tensor, resident: list[int]) -> list[int]:
        if self.mode == "cache_prior":
            return self._select_cache_prior(token_logits, resident)

        ranking = torch.argsort(token_logits, descending=True).detach().cpu().tolist()
        allowed_rank = self._allowed_rank(token_logits)
        protected = ranking[: self.min_rank]
        allowed = set(ranking[:allowed_rank])
        protected_set = set(protected)
        promoted = [
            expert
            for expert in ranking
            if expert in allowed and expert in resident and expert not in protected_set
        ]
        promoted_set = set(promoted)
        reranked = protected + [
            expert for expert in promoted if expert not in protected_set
        ] + [
            expert
            for expert in ranking
            if expert not in protected_set and expert not in promoted_set
        ]
        return [int(expert) for expert in reranked[: self.top_k]]

    def _select_cache_prior(self, token_logits: torch.Tensor, resident: list[int]) -> list[int]:
        ranking = torch.argsort(token_logits, descending=True).detach().cpu().tolist()
        protected = ranking[: self.min_rank]
        protected_set = set(protected)
        range_average = self._current_range_average
        adjusted = token_logits.clone()
        boosted = set(resident) | protected_set
        if boosted:
            indices = torch.tensor(sorted(boosted), dtype=torch.long, device=token_logits.device)
            adjusted[indices] += self.prior_lambda * range_average
        prior_ranking = torch.argsort(adjusted, descending=True).detach().cpu().tolist()
        protected_prior_set = set(protected)
        reranked = protected + [
            expert for expert in prior_ranking if expert not in protected_prior_set
        ]
        return [int(expert) for expert in reranked[: self.top_k]]

    def _allowed_rank(self, token_logits: torch.Tensor) -> int:
        if self.mode == "max_rank":
            return max(self.top_k, self.min_rank, min(self.max_rank, token_logits.numel()))
        if self.mode == "cumsum_threshold":
            ranking = torch.argsort(token_logits, descending=True)
            probabilities = torch.softmax(token_logits.float(), dim=-1)
            sorted_probabilities = probabilities[ranking]
            cumulative = torch.cumsum(sorted_probabilities, dim=0)
            count = int(torch.searchsorted(cumulative, self.probability_threshold).item()) + 1
            return max(self.top_k, self.min_rank, min(count, token_logits.numel()))
        if self.mode == "cache_prior":
            return token_logits.numel()
        raise ValueError(f"unsupported MoCCE mode: {self.mode}")


class CallReturnController:
    def __init__(
        self,
        attention_mask: torch.Tensor,
        num_layers: int,
        capacity: int,
        top_k: int,
        mode: str,
        option_length: int | None = None,
        termination_threshold: float | None = None,
        expert_costs: torch.Tensor | None = None,
        deliberation_cost_unit: float = 0.0,
        switch_cost_unit: float = 0.0,
        load_cost_weight: float = 1.0,
    ) -> None:
        self.attention_mask = attention_mask.detach().bool().cpu()
        self.batch_size, self.seq_len = self.attention_mask.shape
        self.capacity = capacity
        self.top_k = top_k
        self.mode = mode
        self.option_length = option_length
        self.termination_threshold = termination_threshold
        self.expert_costs = (
            expert_costs.detach().float().cpu() if expert_costs is not None else None
        )
        self.deliberation_cost_unit = deliberation_cost_unit
        self.switch_cost_unit = switch_cost_unit
        self.load_cost_weight = load_cost_weight
        self.caches: list[list[list[int]]] = []
        self.options: list[list[tuple[int, ...] | None]] = []
        self.ages: list[list[int]] = []
        self.option_lengths: list[int] = []
        self.events = 0
        self.controller_calls = 0
        self.option_switches = 0
        self.switch_edit_distance = 0
        self.router_logit_regret_sum = 0.0
        self.router_kl_sum = 0.0
        self.reset(num_layers)

    def reset(self, num_layers: int | None = None) -> None:
        layers = num_layers if num_layers is not None else len(self.caches[0])
        self.caches = [[[] for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.options = [[None for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.ages = [[0 for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.option_lengths = []
        self.events = 0
        self.controller_calls = 0
        self.option_switches = 0
        self.switch_edit_distance = 0
        self.router_logit_regret_sum = 0.0
        self.router_kl_sum = 0.0

    def adjust(
        self,
        layer_index: int,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm_topk_prob: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(output, tuple):
            return self._adjust_tuple(layer_index, output, norm_topk_prob)
        return self._adjust_logits(layer_index, output)

    def _adjust_tuple(
        self,
        layer_index: int,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        norm_topk_prob: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = output[0]
        adjusted_logits, shaped_logits, flatten = self._shaped_logits(logits)
        router_scores = output[1].clone()
        router_indices = output[2].clone()
        if flatten:
            router_scores = router_scores.reshape(self.batch_size, self.seq_len, self.top_k)
            router_indices = router_indices.reshape(self.batch_size, self.seq_len, self.top_k)

        for token_index in range(self.seq_len):
            for batch_index in range(self.batch_size):
                if not self.attention_mask[batch_index, token_index].item():
                    continue
                token_logits = shaped_logits[batch_index, token_index].clone()
                selected, fresh_topk = self._select(
                    batch_index,
                    layer_index,
                    token_index,
                    token_logits,
                )
                if selected != fresh_topk:
                    self._force_logits(
                        shaped_logits[batch_index, token_index],
                        token_logits,
                        selected,
                    )
                    selected_tensor = torch.tensor(
                        selected,
                        dtype=router_indices.dtype,
                        device=logits.device,
                    )
                    token_probs = torch.softmax(token_logits.float(), dim=-1)
                    selected_scores = token_probs[selected_tensor.long()].to(router_scores.dtype)
                    if norm_topk_prob:
                        selected_scores /= selected_scores.sum(dim=-1, keepdim=True)
                    router_indices[batch_index, token_index] = selected_tensor
                    router_scores[batch_index, token_index] = selected_scores

        if flatten:
            router_scores = router_scores.reshape(output[1].shape)
            router_indices = router_indices.reshape(output[2].shape)
        return adjusted_logits, router_scores, router_indices

    def _adjust_logits(self, layer_index: int, output: torch.Tensor) -> torch.Tensor:
        adjusted_logits, shaped_logits, _flatten = self._shaped_logits(output)
        for token_index in range(self.seq_len):
            for batch_index in range(self.batch_size):
                if not self.attention_mask[batch_index, token_index].item():
                    continue
                token_logits = shaped_logits[batch_index, token_index].clone()
                selected, fresh_topk = self._select(
                    batch_index,
                    layer_index,
                    token_index,
                    token_logits,
                )
                if selected != fresh_topk:
                    self._force_logits(
                        shaped_logits[batch_index, token_index],
                        token_logits,
                        selected,
                    )
        return adjusted_logits

    def _shaped_logits(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        original_shape = output.shape
        adjusted_logits = output.clone()
        if adjusted_logits.ndim == 2 and adjusted_logits.shape[0] == self.batch_size * self.seq_len:
            shaped_logits = adjusted_logits.reshape(
                self.batch_size,
                self.seq_len,
                adjusted_logits.shape[-1],
            )
            return adjusted_logits, shaped_logits, True
        if adjusted_logits.ndim == 3 and adjusted_logits.shape[:2] == (
            self.batch_size,
            self.seq_len,
        ):
            return adjusted_logits, adjusted_logits, False
        raise RuntimeError(f"unsupported gate output shape for call-and-return: {original_shape}")

    def _select(
        self,
        batch_index: int,
        layer_index: int,
        token_index: int,
        token_logits: torch.Tensor,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        fresh_topk = _topk(token_logits, self.top_k)
        current_option = self.options[batch_index][layer_index]
        terminate = self._should_terminate(
            batch_index,
            layer_index,
            token_logits,
            current_option,
            fresh_topk,
        )
        if terminate:
            self.controller_calls += 1
            if current_option is not None:
                self.option_lengths.append(self.ages[batch_index][layer_index])
                edit = _edit_distance(current_option, fresh_topk)
                self.switch_edit_distance += edit
                if edit:
                    self.option_switches += 1
            selected = fresh_topk
            self.options[batch_index][layer_index] = selected
            self.ages[batch_index][layer_index] = 0
        else:
            if current_option is None:
                raise RuntimeError("continuation without an active option")
            selected = current_option

        self.events += 1
        self.router_logit_regret_sum += _option_logit_regret(token_logits, selected, fresh_topk)
        self.router_kl_sum += _forced_option_kl(token_logits, selected, fresh_topk)
        _update_resident(self.caches[batch_index][layer_index], list(selected), self.capacity)
        self.ages[batch_index][layer_index] += 1
        return selected, fresh_topk

    def _should_terminate(
        self,
        batch_index: int,
        layer_index: int,
        token_logits: torch.Tensor,
        current_option: tuple[int, ...] | None,
        fresh_topk: tuple[int, ...],
    ) -> bool:
        if current_option is None:
            return True
        if self.mode == "polling":
            return True
        if self.mode == "fixed_duration":
            if self.option_length is None or self.option_length <= 0:
                raise ValueError("fixed_duration mode requires a positive option_length")
            return self.ages[batch_index][layer_index] >= self.option_length
        if self.mode in {"call_return", "threshold"}:
            if self.termination_threshold is None:
                raise ValueError("call_return mode requires termination_threshold")
            return _option_logit_regret(token_logits, current_option, fresh_topk) > (
                self.termination_threshold
            )
        if self.mode == "oracle_one_step":
            resident = self.caches[batch_index][layer_index]
            continue_objective = _option_logit_regret(token_logits, current_option, fresh_topk)
            continue_objective += self.load_cost_weight * self._load_cost(
                current_option,
                resident,
            )
            switch_objective = self.load_cost_weight * self._load_cost(fresh_topk, resident)
            switch_objective += self.deliberation_cost_unit
            switch_objective += self.switch_cost_unit * _edit_distance(current_option, fresh_topk)
            return switch_objective < continue_objective
        raise ValueError(f"unsupported call-and-return mode: {self.mode}")

    def _load_cost(self, selected: tuple[int, ...], resident: list[int]) -> float:
        before = set(resident)
        misses = [expert for expert in selected if expert not in before]
        if not misses:
            return 0.0
        if self.expert_costs is None:
            return float(len(misses))
        return float(self.expert_costs[misses].sum())

    def _force_logits(
        self,
        target: torch.Tensor,
        original: torch.Tensor,
        selected: tuple[int, ...],
    ) -> None:
        high_score = original.max().to(target.dtype)
        for offset, expert in enumerate(selected):
            target[expert] = high_score + float(len(selected) - offset)

    def finalize(self) -> None:
        for batch_options, batch_ages in zip(self.options, self.ages, strict=True):
            for option, age in zip(batch_options, batch_ages, strict=True):
                if option is not None:
                    self.option_lengths.append(age)

    @property
    def mean_option_length(self) -> float:
        if self.option_lengths:
            return sum(self.option_lengths) / len(self.option_lengths)
        return 0.0

    @property
    def router_logit_regret(self) -> float:
        return self.router_logit_regret_sum / self.events if self.events else 0.0

    @property
    def router_kl_to_vanilla(self) -> float:
        return self.router_kl_sum / self.events if self.events else 0.0


class SparseMoeCallReturnController:
    """Call-and-return controller that runs the MoE gate only at option boundaries."""

    def __init__(
        self,
        attention_mask: torch.Tensor,
        num_layers: int,
        capacity: int,
        top_k: int,
        mode: str,
        option_length: int | None = None,
        continuation_weighting: str = "stored_weights",
    ) -> None:
        if mode not in {"polling", "fixed_duration"}:
            raise ValueError(
                "no-gate call-and-return supports polling and fixed_duration only; "
                f"{mode!r} requires current gate logits"
            )
        if mode == "fixed_duration" and (option_length is None or option_length <= 0):
            raise ValueError("fixed_duration mode requires a positive option_length")
        if continuation_weighting not in {"restricted_softmax", "stored_weights"}:
            raise ValueError(
                "continuation_weighting must be restricted_softmax or stored_weights"
            )

        self.attention_mask = attention_mask.detach().bool().cpu()
        self.batch_size, self.seq_len = self.attention_mask.shape
        self.capacity = capacity
        self.top_k = top_k
        self.mode = mode
        self.option_length = option_length
        self.continuation_weighting = continuation_weighting
        self.caches: list[list[list[int]]] = []
        self.options: list[list[tuple[int, ...] | None]] = []
        self.option_weights: list[list[torch.Tensor | None]] = []
        self.option_logits: list[list[torch.Tensor | None]] = []
        self.ages: list[list[int]] = []
        self.option_lengths: list[int] = []
        self.router_logits_by_layer: list[torch.Tensor | None] = []
        self.events = 0
        self.controller_calls = 0
        self.gate_evaluated_tokens = 0
        self.option_switches = 0
        self.switch_edit_distance = 0
        self.reset(num_layers)

    def reset(self, num_layers: int | None = None) -> None:
        layers = num_layers if num_layers is not None else len(self.caches[0])
        self.caches = [[[] for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.options = [[None for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.option_weights = [
            [None for _layer in range(layers)] for _batch in range(self.batch_size)
        ]
        self.option_logits = [
            [None for _layer in range(layers)] for _batch in range(self.batch_size)
        ]
        self.ages = [[0 for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.option_lengths = []
        self.router_logits_by_layer = [None for _layer in range(layers)]
        self.events = 0
        self.controller_calls = 0
        self.gate_evaluated_tokens = 0
        self.option_switches = 0
        self.switch_edit_distance = 0

    def forward_sparse_moe(
        self,
        block: torch.nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if (batch_size, sequence_length) != (self.batch_size, self.seq_len):
            raise RuntimeError(
                "no-gate call-and-return controller was built for "
                f"{(self.batch_size, self.seq_len)} but saw {(batch_size, sequence_length)}"
            )

        num_experts = getattr(block, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(getattr(block, "gate", None), "num_experts", None)
        if num_experts is None and isinstance(block.experts, torch.nn.ModuleList):
            num_experts = len(block.experts)
        if num_experts is None:
            raise RuntimeError("could not infer number of experts from sparse block")
        num_experts = int(num_experts)
        block_top_k = getattr(block, "top_k", None)
        if block_top_k is None:
            block_top_k = getattr(getattr(block, "gate", None), "top_k", self.top_k)
        block_top_k = int(block_top_k)
        if block_top_k != self.top_k:
            raise RuntimeError(
                f"controller top_k={self.top_k} does not match sparse block top_k={block_top_k}"
            )

        flat_states = hidden_states.reshape(-1, hidden_dim)
        total_tokens = flat_states.shape[0]
        selected_experts = torch.empty(
            (total_tokens, self.top_k),
            dtype=torch.long,
            device=hidden_states.device,
        )
        routing_weights = torch.empty(
            (total_tokens, self.top_k),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        router_logits = torch.empty(
            (total_tokens, num_experts),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for token_index in range(sequence_length):
            boundary_rows: list[int] = []
            boundary_metadata: list[tuple[int, bool]] = []
            for batch_index in range(batch_size):
                row_index = batch_index * sequence_length + token_index
                active = bool(self.attention_mask[batch_index, token_index].item())
                if active and not self._should_terminate(batch_index, layer_index):
                    self._fill_continuation(
                        block,
                        batch_index,
                        layer_index,
                        row_index,
                        flat_states[row_index],
                        selected_experts,
                        routing_weights,
                        router_logits,
                    )
                    continue

                boundary_rows.append(row_index)
                boundary_metadata.append((batch_index, active))

            if boundary_rows:
                boundary_index = torch.tensor(
                    boundary_rows,
                    dtype=torch.long,
                    device=flat_states.device,
                )
                gate_output = block.gate(flat_states.index_select(0, boundary_index))
                if isinstance(gate_output, tuple):
                    boundary_logits = gate_output[0]
                    boundary_weights = gate_output[1].to(hidden_states.dtype)
                    boundary_selected = gate_output[2]
                else:
                    boundary_logits = gate_output
                    boundary_probs = torch.softmax(boundary_logits, dim=1, dtype=torch.float)
                    boundary_weights, boundary_selected = torch.topk(
                        boundary_probs,
                        self.top_k,
                        dim=-1,
                    )
                    if bool(getattr(block, "norm_topk_prob", False)):
                        boundary_weights /= boundary_weights.sum(dim=-1, keepdim=True)
                    boundary_weights = boundary_weights.to(hidden_states.dtype)

                for local_index, row_index in enumerate(boundary_rows):
                    batch_index, active = boundary_metadata[local_index]
                    selected_experts[row_index] = boundary_selected[local_index]
                    routing_weights[row_index] = boundary_weights[local_index]
                    router_logits[row_index] = boundary_logits[local_index].to(
                        router_logits.dtype
                    )
                    if active:
                        selected = tuple(
                            int(expert)
                            for expert in boundary_selected[local_index].detach().cpu().tolist()
                        )
                        self._record_boundary(
                            batch_index,
                            layer_index,
                            selected,
                            boundary_weights[local_index].detach(),
                            boundary_logits[local_index].detach(),
                        )

        if isinstance(block.experts, torch.nn.ModuleList):
            final_hidden_states = torch.zeros(
                (total_tokens, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            expert_mask = torch.nn.functional.one_hot(
                selected_experts,
                num_classes=num_experts,
            ).permute(2, 1, 0)

            for expert_idx in range(num_experts):
                expert_layer = block.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])
                if top_x.numel() == 0:
                    continue
                current_state = flat_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state) * routing_weights[top_x, idx, None]
                )
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    current_hidden_states.to(hidden_states.dtype),
                )
        else:
            final_hidden_states = block.experts(
                flat_states,
                selected_experts,
                routing_weights,
            )

        self.router_logits_by_layer[layer_index] = router_logits.detach()
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

    def _should_terminate(self, batch_index: int, layer_index: int) -> bool:
        current_option = self.options[batch_index][layer_index]
        if current_option is None:
            return True
        if self.mode == "polling":
            return True
        if self.option_length is None:
            raise RuntimeError("fixed_duration mode lost its option_length")
        return self.ages[batch_index][layer_index] >= self.option_length

    def _fill_continuation(
        self,
        block: torch.nn.Module,
        batch_index: int,
        layer_index: int,
        row_index: int,
        hidden_state: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> None:
        option = self.options[batch_index][layer_index]
        option_weights = self.option_weights[batch_index][layer_index]
        option_logits = self.option_logits[batch_index][layer_index]
        if option is None or option_weights is None or option_logits is None:
            raise RuntimeError("continuation without an active option")
        weights = self._continuation_weights(block, hidden_state, option, option_weights)

        selected_experts[row_index] = torch.tensor(
            option,
            dtype=selected_experts.dtype,
            device=selected_experts.device,
        )
        routing_weights[row_index] = weights.to(
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        )
        router_logits[row_index] = self._continuation_router_logits(
            option,
            weights,
            option_logits,
            router_logits.shape[-1],
            router_logits.device,
            router_logits.dtype,
        )
        self._record_event(batch_index, layer_index, option)

    def _continuation_weights(
        self,
        block: torch.nn.Module,
        hidden_state: torch.Tensor,
        option: tuple[int, ...],
        stored_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.continuation_weighting == "stored_weights":
            return stored_weights.to(device=hidden_state.device, dtype=hidden_state.dtype)
        scores = _restricted_gate_scores(block, hidden_state, option)
        return torch.softmax(scores.float(), dim=-1).to(hidden_state.dtype)

    def _continuation_router_logits(
        self,
        option: tuple[int, ...],
        weights: torch.Tensor,
        stored_logits: torch.Tensor,
        num_experts: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.continuation_weighting == "stored_weights":
            return stored_logits.to(device=device, dtype=dtype)
        logits = torch.full((num_experts,), -1.0e4, dtype=dtype, device=device)
        option_tensor = torch.tensor(option, dtype=torch.long, device=device)
        logits[option_tensor] = weights.to(device=device, dtype=dtype)
        return logits

    def _record_boundary(
        self,
        batch_index: int,
        layer_index: int,
        selected: tuple[int, ...],
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> None:
        current_option = self.options[batch_index][layer_index]
        if current_option is not None:
            self.option_lengths.append(self.ages[batch_index][layer_index])
            edit = _edit_distance(current_option, selected)
            self.switch_edit_distance += edit
            if edit:
                self.option_switches += 1

        self.controller_calls += 1
        self.gate_evaluated_tokens += 1
        self.options[batch_index][layer_index] = selected
        self.option_weights[batch_index][layer_index] = routing_weights.detach()
        self.option_logits[batch_index][layer_index] = router_logits.detach()
        self.ages[batch_index][layer_index] = 0
        self._record_event(batch_index, layer_index, selected)

    def _record_event(
        self,
        batch_index: int,
        layer_index: int,
        selected: tuple[int, ...],
    ) -> None:
        self.events += 1
        _update_resident(self.caches[batch_index][layer_index], list(selected), self.capacity)
        self.ages[batch_index][layer_index] += 1

    def finalize(self) -> None:
        for batch_options, batch_ages in zip(self.options, self.ages, strict=True):
            for option, age in zip(batch_options, batch_ages, strict=True):
                if option is not None:
                    self.option_lengths.append(age)

    @property
    def mean_option_length(self) -> float:
        if self.option_lengths:
            return sum(self.option_lengths) / len(self.option_lengths)
        return 0.0

    def router_stack(self, batch_size: int, seq_len: int) -> torch.Tensor:
        layers: list[torch.Tensor] = []
        for layer_index, layer_logits in enumerate(self.router_logits_by_layer):
            if layer_logits is None:
                raise RuntimeError(f"missing no-gate router logits for layer {layer_index}")
            layers.append(layer_logits.detach().float().reshape(batch_size, seq_len, -1).cpu())
        return torch.stack(layers, dim=1)


class CallReturnSparseMoeWrapper(torch.nn.Module):
    def __init__(
        self,
        block: torch.nn.Module,
        layer_index: int,
        controller: SparseMoeCallReturnController,
    ) -> None:
        super().__init__()
        self.block = block
        self.layer_index = layer_index
        self.controller = controller
        self.returns_router_logits = isinstance(getattr(block, "gate", None), torch.nn.Linear)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output, router_logits = self.controller.forward_sparse_moe(
            self.block,
            self.layer_index,
            hidden_states,
        )
        if self.returns_router_logits:
            return output, router_logits
        return output


def _layer_index_from_name(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\..*\.gate$", f".{name}")
    return None if match is None else int(match.group(1))


def _layer_child_index_from_name(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", f".{name}.")
    return None if match is None else int(match.group(1))


def _gate_modules(model: torch.nn.Module) -> tuple[tuple[int, torch.nn.Module], ...]:
    modules: list[tuple[int, torch.nn.Module]] = []
    for name, module in model.named_modules():
        layer_index = _layer_index_from_name(name)
        if layer_index is not None:
            modules.append((layer_index, module))
    if not modules:
        raise RuntimeError("no OLMoE gate modules matched pattern model.layers.*.mlp.gate")
    return tuple(sorted(modules, key=lambda item: item[0]))


def _sparse_moe_modules(
    model: torch.nn.Module,
) -> tuple[tuple[int, torch.nn.Module, str, torch.nn.Module], ...]:
    modules: list[tuple[int, torch.nn.Module, str, torch.nn.Module]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            layer_index = _layer_child_index_from_name(full_name)
            if layer_index is None:
                continue
            if not all(hasattr(child, attribute) for attribute in ("gate", "experts")):
                continue
            modules.append((layer_index, parent, child_name, child))
    if not modules:
        raise RuntimeError("no OLMoE sparse MoE modules with gate and experts were found")
    return tuple(sorted(modules, key=lambda item: item[0]))


def _simulate_topk_with_costs(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int,
    capacity: int,
    expert_costs: torch.Tensor | None,
) -> tuple[int, float, int]:
    if expert_costs is None:
        hard_misses, weighted_cost, requests, _online_kl = _simulate_topk(
            scores,
            attention_mask,
            top_k=top_k,
            capacity=capacity,
        )
        return hard_misses, weighted_cost, requests

    batch_size, num_layers, seq_len, _num_experts = scores.shape
    hard_misses = 0
    weighted_cost = 0.0
    requests = 0
    costs = expert_costs.detach().float().cpu()
    caches: list[list[list[int]]] = [
        [[] for _layer in range(num_layers)] for _batch in range(batch_size)
    ]

    for token_index in range(seq_len):
        for batch_index in range(batch_size):
            if not attention_mask[batch_index, token_index].item():
                continue
            for layer_index in range(num_layers):
                resident = caches[batch_index][layer_index]
                selected = (
                    torch.topk(
                        scores[batch_index, layer_index, token_index],
                        k=top_k,
                        dim=-1,
                    )
                    .indices.detach()
                    .cpu()
                    .tolist()
                )
                before = set(resident)
                misses = [expert for expert in selected if expert not in before]
                hard_misses += len(misses)
                weighted_cost += float(costs[misses].sum()) if misses else 0.0
                requests += len(selected)
                _update_resident(resident, selected, capacity)

    return hard_misses, weighted_cost, requests


def _evaluate(
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    vanilla_router_logits: torch.Tensor | None,
    capacity: int,
    lambda_cache: float,
    metric_expert_costs: torch.Tensor | None = None,
    router_logits_provider: Callable[[], torch.Tensor] | None = None,
) -> tuple[RerankEvalRow, torch.Tensor]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            **encoded,
            labels=labels,
            output_router_logits=router_logits_provider is None,
            use_cache=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    forward_elapsed_ms = (time.perf_counter() - start) * 1000.0

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"].detach().cpu().bool()
    router_logits = (
        _router_stack(outputs.router_logits, input_ids.shape[0], input_ids.shape[1])
        if router_logits_provider is None
        else router_logits_provider()
    )
    top_k = int(getattr(model.config, "num_experts_per_tok", 2))
    hard_misses, weighted_cost, requests = _simulate_topk_with_costs(
        router_logits,
        attention_mask,
        top_k=top_k,
        capacity=capacity,
        expert_costs=metric_expert_costs,
    )
    router_kl = (
        0.0
        if vanilla_router_logits is None
        else _router_kl(vanilla_router_logits, router_logits, attention_mask)
    )
    loss = outputs.loss.detach().float()
    aux_loss = getattr(outputs, "aux_loss", None)
    if aux_loss is not None:
        router_aux_loss_coef = float(
            getattr(
                model,
                "router_aux_loss_coef",
                getattr(model.config, "router_aux_loss_coef", 0.0),
            )
        )
        loss = loss - router_aux_loss_coef * aux_loss.detach().float().to(loss.device)
    nll = float(loss.cpu())
    return (
        RerankEvalRow(
            method=method,
            cache_capacity=capacity,
            lambda_cache=lambda_cache,
            nll=nll,
            perplexity=math.exp(nll),
            hard_misses=hard_misses,
            hard_miss_rate=hard_misses / requests if requests else 0.0,
            weighted_miss_cost=weighted_cost,
            requests=requests,
            router_kl_to_vanilla=router_kl,
            total_cost=weighted_cost,
            forward_elapsed_ms=forward_elapsed_ms,
        ),
        router_logits,
    )


def _evaluate_with_cache_rerank(
    method_prefix: str,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    vanilla_router_logits: torch.Tensor,
    capacity: int,
    lambda_cache: float,
    expert_costs: torch.Tensor | None = None,
    weighted_penalty: bool = False,
    metric_expert_costs: torch.Tensor | None = None,
) -> RerankEvalRow:
    gates = _gate_modules(model)
    controller = CacheRerankController(
        attention_mask=encoded["attention_mask"],
        num_layers=len(gates),
        capacity=capacity,
        top_k=int(getattr(model.config, "num_experts_per_tok", 2)),
        lambda_cache=lambda_cache,
        expert_costs=expert_costs,
        weighted_penalty=weighted_penalty,
    )
    handles = [
        module.register_forward_hook(
            lambda module, _inputs, output, layer_index=layer_index: controller.adjust(
                layer_index,
                output,
                norm_topk_prob=bool(getattr(module, "norm_topk_prob", False)),
            )
        )
        for layer_index, module in gates
    ]
    try:
        controller.reset()
        row, _router_logits = _evaluate(
            f"{method_prefix}_lambda_{lambda_cache:g}",
            model,
            encoded,
            labels,
            vanilla_router_logits=vanilla_router_logits,
            capacity=capacity,
            lambda_cache=lambda_cache,
            metric_expert_costs=metric_expert_costs,
        )
    finally:
        for handle in handles:
            handle.remove()
    return row


def _evaluate_with_mocce(
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    vanilla_router_logits: torch.Tensor,
    capacity: int,
    value: float,
    *,
    mode: str,
    max_rank: int,
    min_rank: int,
    probability_threshold: float,
    prior_lambda: float = 0.0,
) -> RerankEvalRow:
    gates = _gate_modules(model)
    controller = MoccEController(
        attention_mask=encoded["attention_mask"],
        num_layers=len(gates),
        capacity=capacity,
        top_k=int(getattr(model.config, "num_experts_per_tok", 2)),
        mode=mode,
        max_rank=max_rank,
        min_rank=min_rank,
        probability_threshold=probability_threshold,
        prior_lambda=prior_lambda,
    )
    handles = [
        module.register_forward_hook(
            lambda module, _inputs, output, layer_index=layer_index: controller.adjust(
                layer_index,
                output,
                norm_topk_prob=bool(getattr(module, "norm_topk_prob", False)),
            )
        )
        for layer_index, module in gates
    ]
    try:
        controller.reset()
        row, _router_logits = _evaluate(
            method,
            model,
            encoded,
            labels,
            vanilla_router_logits=vanilla_router_logits,
            capacity=capacity,
            lambda_cache=value,
        )
    finally:
        for handle in handles:
            handle.remove()
    return row


def _evaluate_with_call_return(
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    vanilla_router_logits: torch.Tensor,
    capacity: int,
    *,
    mode: str,
    option_length: int | None,
    termination_threshold: float | None,
    expert_costs: torch.Tensor | None,
    deliberation_cost_unit: float,
    switch_cost_unit: float,
    load_cost_weight: float,
) -> RerankEvalRow:
    gates = _gate_modules(model)
    controller = CallReturnController(
        attention_mask=encoded["attention_mask"],
        num_layers=len(gates),
        capacity=capacity,
        top_k=int(getattr(model.config, "num_experts_per_tok", 2)),
        mode=mode,
        option_length=option_length,
        termination_threshold=termination_threshold,
        expert_costs=expert_costs,
        deliberation_cost_unit=deliberation_cost_unit,
        switch_cost_unit=switch_cost_unit,
        load_cost_weight=load_cost_weight,
    )
    handles = [
        module.register_forward_hook(
            lambda module, _inputs, output, layer_index=layer_index: controller.adjust(
                layer_index,
                output,
                norm_topk_prob=bool(getattr(module, "norm_topk_prob", False)),
            )
        )
        for layer_index, module in gates
    ]
    try:
        controller.reset()
        row, _router_logits = _evaluate(
            method,
            model,
            encoded,
            labels,
            vanilla_router_logits=vanilla_router_logits,
            capacity=capacity,
            lambda_cache=termination_threshold if termination_threshold is not None else 0.0,
            metric_expert_costs=expert_costs,
        )
        controller.finalize()
    finally:
        for handle in handles:
            handle.remove()

    deliberation_cost = deliberation_cost_unit * controller.controller_calls
    switch_cost = switch_cost_unit * controller.switch_edit_distance
    total_cost = load_cost_weight * row.weighted_miss_cost + deliberation_cost + switch_cost
    return replace(
        row,
        controller_calls=controller.controller_calls,
        controller_call_rate=controller.controller_calls / controller.events
        if controller.events
        else 0.0,
        option_switches=controller.option_switches,
        switch_edit_distance=controller.switch_edit_distance,
        gate_evaluated_tokens=controller.events,
        gate_evaluation_rate=1.0 if controller.events else 0.0,
        mean_option_length=controller.mean_option_length,
        router_logit_regret=controller.router_logit_regret,
        deliberation_cost=deliberation_cost,
        switch_cost=switch_cost,
        total_cost=total_cost,
        controller_mode=mode,
        call_return_runtime="gate_hook",
        option_length=option_length,
        termination_threshold=termination_threshold,
    )


def _evaluate_with_sparse_moe_call_return(
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    vanilla_router_logits: torch.Tensor,
    capacity: int,
    *,
    mode: str,
    option_length: int | None,
    continuation_weighting: str,
    expert_costs: torch.Tensor | None,
    deliberation_cost_unit: float,
    switch_cost_unit: float,
    load_cost_weight: float,
) -> RerankEvalRow:
    sparse_blocks = _sparse_moe_modules(model)
    controller = SparseMoeCallReturnController(
        attention_mask=encoded["attention_mask"],
        num_layers=len(sparse_blocks),
        capacity=capacity,
        top_k=int(getattr(model.config, "num_experts_per_tok", 2)),
        mode=mode,
        option_length=option_length,
        continuation_weighting=continuation_weighting,
    )
    originals: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    for layer_index, parent, child_name, child in sparse_blocks:
        setattr(parent, child_name, CallReturnSparseMoeWrapper(child, layer_index, controller))
        originals.append((parent, child_name, child))

    try:
        controller.reset()
        row, _router_logits = _evaluate(
            method,
            model,
            encoded,
            labels,
            vanilla_router_logits=vanilla_router_logits,
            capacity=capacity,
            lambda_cache=0.0,
            metric_expert_costs=expert_costs,
            router_logits_provider=lambda: controller.router_stack(
                encoded["input_ids"].shape[0],
                encoded["input_ids"].shape[1],
            ),
        )
        controller.finalize()
    finally:
        for parent, child_name, child in originals:
            setattr(parent, child_name, child)

    deliberation_cost = deliberation_cost_unit * controller.controller_calls
    switch_cost = switch_cost_unit * controller.switch_edit_distance
    total_cost = load_cost_weight * row.weighted_miss_cost + deliberation_cost + switch_cost
    return replace(
        row,
        controller_calls=controller.controller_calls,
        controller_call_rate=controller.controller_calls / controller.events
        if controller.events
        else 0.0,
        option_switches=controller.option_switches,
        switch_edit_distance=controller.switch_edit_distance,
        gate_evaluated_tokens=controller.gate_evaluated_tokens,
        gate_evaluation_rate=controller.gate_evaluated_tokens / controller.events
        if controller.events
        else 0.0,
        mean_option_length=controller.mean_option_length,
        deliberation_cost=deliberation_cost,
        switch_cost=switch_cost,
        total_cost=total_cost,
        controller_mode=mode,
        call_return_runtime="sparse_moe_no_gate",
        continuation_weighting=continuation_weighting,
        option_length=option_length,
        termination_threshold=None,
    )


def _evaluate_with_learned_beta_call_return(
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    vanilla_router_logits: torch.Tensor,
    capacity: int,
    *,
    termination_head: LearnedTerminationHead,
    termination_threshold: float,
    continuation_weighting: str,
    expert_costs: torch.Tensor | None,
    deliberation_cost_unit: float,
    switch_cost_unit: float,
    load_cost_weight: float,
) -> RerankEvalRow:
    sparse_blocks = _sparse_moe_modules(model)
    controller = LearnedBetaSparseMoeController(
        attention_mask=encoded["attention_mask"],
        num_layers=len(sparse_blocks),
        capacity=capacity,
        top_k=int(getattr(model.config, "num_experts_per_tok", 2)),
        termination_head=termination_head,
        mode="hard",
        termination_threshold=termination_threshold,
        continuation_weighting=continuation_weighting,
        expert_costs=expert_costs,
    )
    originals: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    for layer_index, parent, child_name, child in sparse_blocks:
        setattr(parent, child_name, LearnedBetaSparseMoeWrapper(child, layer_index, controller))
        originals.append((parent, child_name, child))

    try:
        controller.reset()
        row, _router_logits = _evaluate(
            method,
            model,
            encoded,
            labels,
            vanilla_router_logits=vanilla_router_logits,
            capacity=capacity,
            lambda_cache=termination_threshold,
            metric_expert_costs=expert_costs,
            router_logits_provider=lambda: controller.router_stack(
                encoded["input_ids"].shape[0],
                encoded["input_ids"].shape[1],
            ),
        )
        controller.finalize()
    finally:
        for parent, child_name, child in originals:
            setattr(parent, child_name, child)

    deliberation_cost = deliberation_cost_unit * controller.controller_calls
    switch_cost = switch_cost_unit * controller.switch_edit_distance
    total_cost = load_cost_weight * row.weighted_miss_cost + deliberation_cost + switch_cost
    return replace(
        row,
        controller_calls=controller.controller_calls,
        controller_call_rate=controller.controller_calls / controller.events
        if controller.events
        else 0.0,
        option_switches=controller.option_switches,
        switch_edit_distance=controller.switch_edit_distance,
        gate_evaluated_tokens=controller.gate_evaluated_tokens,
        gate_evaluation_rate=controller.gate_evaluated_tokens / controller.events
        if controller.events
        else 0.0,
        mean_option_length=controller.mean_option_length,
        deliberation_cost=deliberation_cost,
        switch_cost=switch_cost,
        total_cost=total_cost,
        controller_mode="learned_beta",
        call_return_runtime="sparse_moe_no_gate",
        continuation_weighting=continuation_weighting,
        option_length=None,
        termination_threshold=termination_threshold,
        beta_calls=controller.beta_calls,
    )


def evaluate(
    config: dict[str, Any],
    output_dir: Path,
    router_state_path: Path | None = None,
    learned_beta_state_path: Path | None = None,
    adapter_state_path: Path | None = None,
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
    capacities = tuple(
        int(value)
        for value in eval_config.get(
            "cache_capacities",
            [eval_config.get("cache_capacity", 8)],
        )
    )
    lambdas = tuple(float(value) for value in eval_config.get("cache_lambdas", [0.1, 0.3, 1.0]))
    mocce_min_rank = int(eval_config.get("mocce_min_rank", int(eval_config.get("min_rank", 0))))
    mocce_max_ranks = tuple(int(value) for value in eval_config.get("mocce_max_ranks", []))
    mocce_thresholds = tuple(
        float(value) for value in eval_config.get("mocce_probability_thresholds", [])
    )
    mocce_prior_lambdas = tuple(
        float(value) for value in eval_config.get("mocce_prior_lambdas", [])
    )
    cost_profile_name = eval_config.get("expert_cost_profile")
    weighted_cache_lambdas = tuple(
        float(value) for value in eval_config.get("weighted_cache_lambdas", [])
    )
    controller_config = eval_config.get("controller", {})
    controller_modes = tuple(str(value) for value in controller_config.get("modes", []))
    controller_mode = controller_config.get("mode")
    if controller_mode is not None:
        controller_modes = (str(controller_mode),)
    termination_thresholds = tuple(
        float(value) for value in controller_config.get("termination_thresholds", [])
    )
    option_lengths = tuple(int(value) for value in controller_config.get("option_lengths", []))
    controller_costs = controller_config.get("costs", {})
    if not isinstance(controller_costs, dict):
        raise TypeError("eval.controller.costs must be a mapping when provided")
    deliberation_cost_unit = float(
        controller_config.get("deliberation_cost", controller_costs.get("deliberation", 0.0))
    )
    switch_cost_unit = float(
        controller_config.get("switch_cost", controller_costs.get("switch", 0.0))
    )
    load_cost_weight = float(
        controller_config.get("load_cost_weight", controller_costs.get("load", 1.0))
    )
    continuation_weighting = str(
        controller_config.get("continuation_weighting", "restricted_softmax")
    )
    if continuation_weighting not in {"restricted_softmax", "stored_weights"}:
        raise ValueError(
            "eval.controller.continuation_weighting must be restricted_softmax or stored_weights"
        )
    call_return_runtime = str(controller_config.get("runtime", "gate_hook"))
    if call_return_runtime in {"no_gate", "sparse_moe"}:
        call_return_runtime = "sparse_moe_no_gate"
    if call_return_runtime not in {"gate_hook", "sparse_moe_no_gate"}:
        raise ValueError(f"unsupported call-and-return runtime: {call_return_runtime}")
    config_learned_beta_path = controller_config.get("beta_checkpoint") or controller_config.get(
        "learned_beta_checkpoint"
    )
    if learned_beta_state_path is None and config_learned_beta_path is not None:
        learned_beta_state_path = Path(str(config_learned_beta_path))
    controller_cost_profile = controller_config.get("expert_cost_profile")
    if controller_cost_profile is not None:
        cost_profile_name = controller_cost_profile
    preload_router_state = bool(
        eval_config.get("preload_router_state", eval_config.get("router_state_preload", False))
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = _load_prompts({"trace": eval_config})[: int(eval_config.get("max_sequences", 4))]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(eval_config.get("max_length", 128)),
    ).to(device)
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    loaded_names: tuple[str, ...] = ()
    router_state_preloaded = False
    if router_state_path is not None and preload_router_state:
        loaded_names = _load_router_state(model, router_state_path)
        router_state_preloaded = True

    if adapter_state_path is None:
        config_adapter = eval_config.get("adapter_state")
        if config_adapter:
            adapter_state_path = Path(os.path.expandvars(str(config_adapter)))
    adapter_loaded_names: tuple[str, ...] = ()
    if adapter_state_path is not None:
        lora_eval_raw = eval_config.get("lora", {})
        if not isinstance(lora_eval_raw, dict):
            raise TypeError("eval.lora must be a mapping when provided")
        # rank/alpha/targets must match the training run so the LoRA scaling reproduces it;
        # load_lora_state validates tensor shapes and will raise on a mismatch.
        lora_eval_config = LoRAExpertsConfig(**{**lora_eval_raw, "enabled": True})
        attach_lora_experts(model, lora_eval_config)
        adapter_loaded_names = load_lora_state(
            model,
            torch.load(adapter_state_path, map_location="cpu"),
        )

    expert_costs = (
        _cost_profile(int(model.config.num_experts), str(cost_profile_name))
        if cost_profile_name
        else None
    )
    learned_beta_head: LearnedTerminationHead | None = None
    learned_beta_checkpoint_metadata: dict[str, Any] | None = None
    if "learned_beta" in controller_modes:
        if call_return_runtime != "sparse_moe_no_gate":
            raise ValueError("learned_beta evaluation requires controller.runtime: no_gate")
        if learned_beta_state_path is None:
            raise ValueError(
                "learned_beta evaluation requires --learned-beta-state or "
                "eval.controller.beta_checkpoint"
            )
        learned_beta_head = load_learned_beta_checkpoint(learned_beta_state_path, device)
        learned_beta_checkpoint_metadata = learned_beta_metadata(learned_beta_state_path)
        learned_beta_head.eval()

    vanilla_rows: list[RerankEvalRow] = []
    vanilla_router_logits: torch.Tensor | None = None
    for capacity in capacities:
        vanilla_row, current_vanilla_router_logits = _evaluate(
            "vanilla",
            model,
            encoded,
            labels,
            vanilla_router_logits=None,
            capacity=capacity,
            lambda_cache=0.0,
            metric_expert_costs=expert_costs,
        )
        vanilla_rows.append(vanilla_row)
        if vanilla_router_logits is None:
            vanilla_router_logits = current_vanilla_router_logits

    if vanilla_router_logits is None:
        raise RuntimeError("no capacities configured")

    rows = list(vanilla_rows)
    for capacity in capacities:
        for lambda_cache in lambdas:
            rows.append(
                _evaluate_with_cache_rerank(
                    "posthoc_cache_rerank",
                    model,
                    encoded,
                    labels,
                    vanilla_router_logits,
                    capacity=capacity,
                    lambda_cache=lambda_cache,
                    metric_expert_costs=expert_costs,
                )
            )
        for lambda_cache in weighted_cache_lambdas:
            rows.append(
                _evaluate_with_cache_rerank(
                    "weighted_posthoc_cache_rerank",
                    model,
                    encoded,
                    labels,
                    vanilla_router_logits,
                    capacity=capacity,
                    lambda_cache=lambda_cache,
                    expert_costs=expert_costs,
                    weighted_penalty=True,
                    metric_expert_costs=expert_costs,
                )
            )

    for capacity in capacities:
        for max_rank in mocce_max_ranks:
            rows.append(
                _evaluate_with_mocce(
                    f"mocce_max_rank_{max_rank}_min_{mocce_min_rank}",
                    model,
                    encoded,
                    labels,
                    vanilla_router_logits,
                    capacity=capacity,
                    value=float(max_rank),
                    mode="max_rank",
                    max_rank=max_rank,
                    min_rank=mocce_min_rank,
                    probability_threshold=1.0,
                )
            )
        for threshold in mocce_thresholds:
            rows.append(
                _evaluate_with_mocce(
                    f"mocce_cumsum_threshold_{threshold:g}_min_{mocce_min_rank}",
                    model,
                    encoded,
                    labels,
                    vanilla_router_logits,
                    capacity=capacity,
                    value=threshold,
                    mode="cumsum_threshold",
                    max_rank=int(getattr(model.config, "num_experts", 0) or 0),
                    min_rank=mocce_min_rank,
                    probability_threshold=threshold,
                )
            )
        for prior_lambda in mocce_prior_lambdas:
            rows.append(
                _evaluate_with_mocce(
                    f"mocce_cache_prior_{prior_lambda:g}_min_{mocce_min_rank}",
                    model,
                    encoded,
                    labels,
                    vanilla_router_logits,
                    capacity=capacity,
                    value=prior_lambda,
                    mode="cache_prior",
                    max_rank=int(getattr(model.config, "num_experts", 0) or 0),
                    min_rank=mocce_min_rank,
                    probability_threshold=1.0,
                    prior_lambda=prior_lambda,
                )
            )

    for capacity in capacities:
        if "polling" in controller_modes:
            evaluator = (
                _evaluate_with_sparse_moe_call_return
                if call_return_runtime == "sparse_moe_no_gate"
                else _evaluate_with_call_return
            )
            rows.append(
                evaluator(
                    "polling_controller"
                    if call_return_runtime == "gate_hook"
                    else "polling_controller_no_gate",
                    model,
                    encoded,
                    labels,
                    vanilla_router_logits,
                    capacity=capacity,
                    mode="polling",
                    option_length=None,
                    expert_costs=expert_costs,
                    deliberation_cost_unit=deliberation_cost_unit,
                    switch_cost_unit=switch_cost_unit,
                    load_cost_weight=load_cost_weight,
                    **(
                        {"termination_threshold": None}
                        if call_return_runtime == "gate_hook"
                        else {"continuation_weighting": continuation_weighting}
                    ),
                )
            )
        if "fixed_duration" in controller_modes:
            evaluator = (
                _evaluate_with_sparse_moe_call_return
                if call_return_runtime == "sparse_moe_no_gate"
                else _evaluate_with_call_return
            )
            for option_length in option_lengths:
                rows.append(
                    evaluator(
                        f"fixed_duration_{option_length}"
                        if call_return_runtime == "gate_hook"
                        else f"fixed_duration_no_gate_{option_length}",
                        model,
                        encoded,
                        labels,
                        vanilla_router_logits,
                        capacity=capacity,
                        mode="fixed_duration",
                        option_length=option_length,
                        expert_costs=expert_costs,
                        deliberation_cost_unit=deliberation_cost_unit,
                        switch_cost_unit=switch_cost_unit,
                        load_cost_weight=load_cost_weight,
                        **(
                            {"termination_threshold": None}
                            if call_return_runtime == "gate_hook"
                            else {"continuation_weighting": continuation_weighting}
                        ),
                    )
                )
        if "call_return" in controller_modes or "threshold" in controller_modes:
            if call_return_runtime == "sparse_moe_no_gate":
                raise ValueError(
                    "threshold call-and-return requires current gate logits; use "
                    "runtime: gate_hook or implement a pre-gate termination head"
                )
            for threshold in termination_thresholds:
                rows.append(
                    _evaluate_with_call_return(
                        f"call_return_threshold_{threshold:g}",
                        model,
                        encoded,
                        labels,
                        vanilla_router_logits,
                        capacity=capacity,
                        mode="call_return",
                        option_length=None,
                        termination_threshold=threshold,
                        expert_costs=expert_costs,
                        deliberation_cost_unit=deliberation_cost_unit,
                        switch_cost_unit=switch_cost_unit,
                        load_cost_weight=load_cost_weight,
                    )
                )
        if "learned_beta" in controller_modes:
            if learned_beta_head is None:
                raise RuntimeError("learned_beta mode was not initialized")
            for threshold in termination_thresholds:
                rows.append(
                    _evaluate_with_learned_beta_call_return(
                        f"learned_beta_no_gate_{threshold:g}",
                        model,
                        encoded,
                        labels,
                        vanilla_router_logits,
                        capacity=capacity,
                        termination_head=learned_beta_head,
                        termination_threshold=threshold,
                        continuation_weighting=continuation_weighting,
                        expert_costs=expert_costs,
                        deliberation_cost_unit=deliberation_cost_unit,
                        switch_cost_unit=switch_cost_unit,
                        load_cost_weight=load_cost_weight,
                    )
                )
        if "oracle_one_step" in controller_modes:
            if call_return_runtime == "sparse_moe_no_gate":
                raise ValueError(
                    "oracle_one_step requires current gate logits; use runtime: gate_hook"
                )
            rows.append(
                _evaluate_with_call_return(
                    "oracle_one_step",
                    model,
                    encoded,
                    labels,
                    vanilla_router_logits,
                    capacity=capacity,
                    mode="oracle_one_step",
                    option_length=None,
                    termination_threshold=None,
                    expert_costs=expert_costs,
                    deliberation_cost_unit=deliberation_cost_unit,
                    switch_cost_unit=switch_cost_unit,
                    load_cost_weight=load_cost_weight,
                )
            )

    if router_state_path is not None and not router_state_preloaded:
        loaded_names = _load_router_state(model, router_state_path)
        for capacity in capacities:
            for lambda_cache in lambdas:
                rows.append(
                    _evaluate_with_cache_rerank(
                        "trained_cache_rerank",
                        model,
                        encoded,
                        labels,
                        vanilla_router_logits,
                        capacity=capacity,
                        lambda_cache=lambda_cache,
                        metric_expert_costs=expert_costs,
                    )
                )
            for lambda_cache in weighted_cache_lambdas:
                rows.append(
                    _evaluate_with_cache_rerank(
                        "weighted_trained_cache_rerank",
                        model,
                        encoded,
                        labels,
                        vanilla_router_logits,
                        capacity=capacity,
                        lambda_cache=lambda_cache,
                        expert_costs=expert_costs,
                        weighted_penalty=True,
                        metric_expert_costs=expert_costs,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cache_rerank_eval.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")

    metadata = {
        "seed": seed,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": _resolved_model_revision(model),
        "dtype": str(dtype).removeprefix("torch."),
        "cache_capacities": capacities,
        "cache_lambdas": lambdas,
        "expert_cost_profile": cost_profile_name,
        "weighted_cache_lambdas": weighted_cache_lambdas,
        "expert_costs": [float(value) for value in expert_costs.tolist()]
        if expert_costs is not None
        else None,
        "mocce_min_rank": mocce_min_rank,
        "mocce_max_ranks": mocce_max_ranks,
        "mocce_probability_thresholds": mocce_thresholds,
        "mocce_prior_lambdas": mocce_prior_lambdas,
        "controller_modes": controller_modes,
        "controller_termination_thresholds": termination_thresholds,
        "controller_option_lengths": option_lengths,
        "controller_deliberation_cost": deliberation_cost_unit,
        "controller_switch_cost": switch_cost_unit,
        "controller_load_cost_weight": load_cost_weight,
        "controller_runtime": call_return_runtime,
        "controller_continuation_weighting": continuation_weighting,
        "learned_beta_state_path": str(learned_beta_state_path)
        if learned_beta_state_path is not None
        else None,
        "learned_beta_checkpoint_metadata": learned_beta_checkpoint_metadata,
        "router_state_path": str(router_state_path) if router_state_path is not None else None,
        "router_state_preloaded": router_state_preloaded,
        "adapter_state_path": str(adapter_state_path) if adapter_state_path is not None else None,
        "adapter_parameter_count": len(adapter_loaded_names),
        "loaded_router_parameter_count": len(loaded_names),
        "loaded_router_parameter_names": loaded_names,
        "num_prompts": len(prompts),
        "sequence_length": int(encoded["input_ids"].shape[1]),
        "gate_module_count": len(_gate_modules(model)),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "note": (
            "Serving-time cache-conditioned reranking and deterministic call-and-return "
            "routing through OLMoE gate hooks or sparse-block no-gate continuation."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate in-forward cache-conditioned reranking.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--router-state", type=Path)
    parser.add_argument("--learned-beta-state", type=Path)
    parser.add_argument("--adapter-state", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluate(
        config,
        args.output_dir,
        args.router_state,
        args.learned_beta_state,
        args.adapter_state,
    )


if __name__ == "__main__":
    main()
