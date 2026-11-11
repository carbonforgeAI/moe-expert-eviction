from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from moe_eviction.experiments.call_return_frontier import _edit_distance


@dataclass(frozen=True, slots=True)
class BetaHeadConfig:
    hidden_dim: int = 128
    expert_embed_dim: int = 32
    shared_across_layers: bool = True
    include_cache_features: bool = False
    feature_mode: str = "summary"
    max_age: int = 32


@dataclass(frozen=True, slots=True)
class LearnedBetaCheckpoint:
    num_experts: int
    top_k: int
    num_layers: int
    hidden_size: int
    config: BetaHeadConfig
    continuation_weighting: str


def beta_feature_dim(top_k: int, include_cache_features: bool = False) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    dim = 4  # hidden summary: mean, std, RMS, max abs
    dim += 1  # normalized option age
    dim += 1  # normalized layer index
    dim += top_k  # previous option expert ids
    dim += top_k  # previous option weights
    dim += 1  # previous boundary confidence
    if include_cache_features:
        dim += 3  # cache fill, resident option fraction, expected keep load
    return dim


class LearnedTerminationHead(torch.nn.Module):
    """Small MLP that predicts option termination before evaluating the full gate."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        num_layers: int,
        config: BetaHeadConfig | None = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.config = config or BetaHeadConfig()
        if self.config.feature_mode not in {"summary", "henderson"}:
            raise ValueError("beta head feature_mode must be summary or henderson")
        if self.config.expert_embed_dim <= 0:
            raise ValueError("expert_embed_dim must be positive")
        if self.config.feature_mode == "summary":
            self.input_dim = beta_feature_dim(
                self.top_k,
                include_cache_features=self.config.include_cache_features,
            )
        else:
            self.expert_embedding = torch.nn.Embedding(num_experts, self.config.expert_embed_dim)
            self.set_phi = torch.nn.Sequential(
                torch.nn.Linear(self.config.expert_embed_dim, self.config.hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            )
            self.input_dim = self.hidden_size + self.config.hidden_dim
        if self.config.shared_across_layers:
            self.net = _make_beta_mlp(self.input_dim, self.config.hidden_dim)
        else:
            self.nets = torch.nn.ModuleList(
                _make_beta_mlp(self.input_dim, self.config.hidden_dim)
                for _layer in range(self.num_layers)
            )

    def forward(
        self,
        features: torch.Tensor,
        layer_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_features(features)
        if self.config.shared_across_layers:
            return torch.sigmoid(self.net(features.float())).squeeze(-1)

        if layer_indices is None:
            raise RuntimeError("per-layer termination head requires layer_indices")
        if layer_indices.ndim != 1 or layer_indices.shape[0] != features.shape[0]:
            raise RuntimeError(
                "layer_indices must have shape [rows], "
                f"got {tuple(layer_indices.shape)} for {features.shape[0]} rows"
            )

        outputs = torch.empty(
            (features.shape[0],),
            dtype=torch.float32,
            device=features.device,
        )
        layer_indices = layer_indices.to(device=features.device, dtype=torch.long)
        for layer_index, net in enumerate(self.nets):
            rows = torch.where(layer_indices == layer_index)[0]
            if rows.numel() == 0:
                continue
            outputs[rows] = torch.sigmoid(net(features.index_select(0, rows).float())).squeeze(-1)

        invalid = (layer_indices < 0) | (layer_indices >= self.num_layers)
        if bool(invalid.any().item()):
            bad_layer = int(layer_indices[torch.where(invalid)[0][0]].detach().cpu())
            raise RuntimeError(f"layer index {bad_layer} is outside [0, {self.num_layers})")
        return outputs

    def set_embedding(self, option_mask: torch.Tensor) -> torch.Tensor:
        if self.config.feature_mode != "henderson":
            raise RuntimeError("set embeddings are only available in henderson feature mode")
        embedded = self.expert_embedding(option_mask)
        return self.set_phi(embedded).sum(dim=-2)

    def _validate_features(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"termination features must have shape [rows, {self.input_dim}], "
                f"got {tuple(features.shape)}"
            )

    def checkpoint_metadata(self, continuation_weighting: str) -> LearnedBetaCheckpoint:
        return LearnedBetaCheckpoint(
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            config=self.config,
            continuation_weighting=continuation_weighting,
        )


def _make_beta_mlp(input_dim: int, hidden_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


# Policy over options (mu): per-expert demand-anticipation features.
POLICY_PER_EXPERT_FEATURES = 5  # gate_logit, router_prob, pop_ema, recency, resident indicator
POLICY_HIDDEN_SUMMARY = 4  # mean, std, RMS, max-abs of the gate hidden state (broadcast)


def policy_feature_dim() -> int:
    return POLICY_PER_EXPERT_FEATURES + POLICY_HIDDEN_SUMMARY


@dataclass(frozen=True, slots=True)
class PolicyHeadConfig:
    hidden_dim: int = 64
    shared_across_layers: bool = False
    temperature: float = 1.0


class PolicyOverOptionsHead(torch.nn.Module):
    """Per-expert demand-anticipation scorer for the policy over options (mu).

    Scores each expert from causal features; when termination fires, the next resident pool is the
    top-C experts by score. Mirrors ``LearnedTerminationHead``'s per-layer / shared MLP structure.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        num_layers: int,
        feature_dim: int,
        config: PolicyHeadConfig | None = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.num_experts = int(num_experts)
        self.num_layers = int(num_layers)
        self.feature_dim = int(feature_dim)
        self.config = config or PolicyHeadConfig()
        if self.config.shared_across_layers:
            self.net = _make_beta_mlp(self.feature_dim, self.config.hidden_dim)
        else:
            self.nets = torch.nn.ModuleList(
                _make_beta_mlp(self.feature_dim, self.config.hidden_dim)
                for _layer in range(self.num_layers)
            )

    def forward(self, features: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Score every expert: ``features`` ``[num_experts, feature_dim]`` -> ``[num_experts]``."""
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"policy features must have shape [experts, {self.feature_dim}], "
                f"got {tuple(features.shape)}"
            )
        net = self.net if self.config.shared_across_layers else self.nets[int(layer_index)]
        return net(features.float()).squeeze(-1)


def save_learned_beta_checkpoint(
    path: Path,
    head: LearnedTerminationHead,
    *,
    continuation_weighting: str,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    metadata = asdict(head.checkpoint_metadata(continuation_weighting))
    if extra_metadata:
        metadata.update(extra_metadata)
    torch.save(
        {
            "termination_head": head.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_learned_beta_checkpoint(path: Path, device: torch.device) -> LearnedTerminationHead:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"learned beta checkpoint must be a dict, got {type(state)!r}")
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise KeyError("learned beta checkpoint is missing metadata")
    config_raw = metadata.get("config", {})
    if not isinstance(config_raw, dict):
        raise TypeError("learned beta checkpoint metadata config must be a dict")
    head = LearnedTerminationHead(
        num_experts=int(metadata["num_experts"]),
        top_k=int(metadata["top_k"]),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        config=BetaHeadConfig(**config_raw),
    ).to(device)
    head.load_state_dict(state["termination_head"])
    head.eval()
    return head


def learned_beta_metadata(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("metadata"), dict):
        raise KeyError("learned beta checkpoint is missing metadata")
    return dict(state["metadata"])


def write_learned_beta_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def bernoulli_log_prob(
    probability: torch.Tensor,
    action: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probability = probability.float().clamp(eps, 1.0 - eps)
    action = action.to(device=probability.device, dtype=probability.dtype)
    return action * torch.log(probability) + (1.0 - action) * torch.log1p(-probability)


def bernoulli_entropy(probability: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    probability = probability.float().clamp(eps, 1.0 - eps)
    return -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log1p(-probability)
    )


def plackett_luce_log_prob(logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    if selected.ndim != 1:
        raise RuntimeError(f"selected must be 1D, got {tuple(selected.shape)}")
    if selected.unique().numel() != selected.numel():
        raise RuntimeError("Plackett-Luce selected experts must be unique")
    if selected.numel() > logits.numel():
        raise RuntimeError("cannot select more experts than logits contain")

    available = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    log_prob = logits.new_tensor(0.0, dtype=torch.float32)
    for expert in selected.to(device=logits.device, dtype=torch.long):
        masked_logits = logits.float().masked_fill(~available, -torch.inf)
        log_prob = log_prob + F.log_softmax(masked_logits, dim=-1)[expert]
        available = available.clone()
        available[expert] = False
    return log_prob


def plackett_luce_sample(
    logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 1:
        raise RuntimeError(f"logits must be 1D, got {tuple(logits.shape)}")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > logits.numel():
        raise ValueError("k cannot exceed number of logits")

    available = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    selected: list[torch.Tensor] = []
    log_prob = logits.new_tensor(0.0, dtype=torch.float32)
    entropy = logits.new_tensor(0.0, dtype=torch.float32)
    for _ in range(k):
        masked_logits = logits.float().masked_fill(~available, -torch.inf)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probabilities = torch.softmax(masked_logits, dim=-1)
        finite = torch.isfinite(log_probs)
        entropy = entropy - (probabilities[finite] * log_probs[finite]).sum()
        expert = torch.multinomial(probabilities, num_samples=1).squeeze(0)
        selected.append(expert)
        log_prob = log_prob + log_probs[expert]
        available = available.clone()
        available[expert] = False
    return torch.stack(selected).to(dtype=torch.long), log_prob, entropy


class LearnedBetaSparseMoeController:
    """Sparse-MoE controller for learned option termination.

    In hard mode, beta is computed before the gate. Continuation reuses the
    persisted expert subset and skips the full gate. In soft mode, active rows
    with an option execute both the keep branch and a fresh full-gate branch,
    then blend them with beta.
    """

    def __init__(
        self,
        *,
        attention_mask: torch.Tensor,
        num_layers: int,
        capacity: int,
        top_k: int,
        termination_head: LearnedTerminationHead,
        mode: str,
        termination_threshold: float = 0.5,
        continuation_weighting: str = "restricted_softmax",
        expert_costs: torch.Tensor | None = None,
        state_update_threshold: float = 0.5,
        reference_router_logits: torch.Tensor | None = None,
        lambda_cov: float = 1.0,
        lambda_call: float = 0.05,
        lambda_cache: float = 0.0,
        lambda_edit: float = 0.0,
        counterfactual_eta: float = 0.1,
        mask_temperature: float = 1.0,
    ) -> None:
        if mode not in {"hard", "soft", "policy_gradient"}:
            raise ValueError("learned beta mode must be 'hard', 'soft', or 'policy_gradient'")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if continuation_weighting not in {"restricted_softmax", "stored_weights"}:
            raise ValueError(
                "continuation_weighting must be restricted_softmax or stored_weights"
            )
        if not 0.0 <= termination_threshold <= 1.0:
            raise ValueError("termination_threshold must be in [0, 1]")

        self.attention_mask = attention_mask.detach().bool().cpu()
        self.batch_size, self.seq_len = self.attention_mask.shape
        self.capacity = int(capacity)
        self.top_k = int(top_k)
        self.termination_head = termination_head
        self.mode = mode
        self.termination_threshold = float(termination_threshold)
        self.continuation_weighting = continuation_weighting
        self.expert_costs = (
            expert_costs.detach().float().cpu() if expert_costs is not None else None
        )
        self.state_update_threshold = float(state_update_threshold)
        if reference_router_logits is not None:
            expected_shape = (self.batch_size, num_layers, self.seq_len)
            if tuple(reference_router_logits.shape[:3]) != expected_shape:
                raise ValueError(
                    "reference_router_logits must have shape "
                    f"{expected_shape} + (num_experts,), got {tuple(reference_router_logits.shape)}"
                )
            self.reference_router_logits = reference_router_logits.detach().float().cpu()
        else:
            self.reference_router_logits = None
        self.lambda_cov = float(lambda_cov)
        self.lambda_call = float(lambda_call)
        self.lambda_cache = float(lambda_cache)
        self.lambda_edit = float(lambda_edit)
        self.counterfactual_eta = float(counterfactual_eta)
        self.mask_temperature = float(mask_temperature)

        self.caches: list[list[list[int]]] = []
        self.options: list[list[tuple[int, ...] | None]] = []
        self.option_weights: list[list[torch.Tensor | None]] = []
        self.option_logits: list[list[torch.Tensor | None]] = []
        self.boundary_confidences: list[list[float]] = []
        self.ages: list[list[int]] = []
        self.option_lengths: list[int] = []
        self.router_logits_by_layer: list[torch.Tensor | None] = []
        self.events = 0
        self.controller_calls = 0
        self.gate_evaluated_tokens = 0
        self.beta_calls = 0
        self.option_switches = 0
        self.switch_edit_distance = 0
        self.soft_keep_executions = 0
        self.soft_refresh_executions = 0
        self.policy_termination_decisions = 0
        self.policy_switch_decisions = 0
        self.cache_load_cost = 0.0
        self._soft_beta_terms: list[torch.Tensor] = []
        self._soft_load_terms: list[torch.Tensor] = []
        self._soft_switch_terms: list[torch.Tensor] = []
        self._counterfactual_beta_losses: list[torch.Tensor] = []
        self._counterfactual_mask_losses: list[torch.Tensor] = []
        self._counterfactual_targets: list[torch.Tensor] = []
        self._continue_coverages: list[torch.Tensor] = []
        self._switch_coverages: list[torch.Tensor] = []
        self._policy_beta_log_probs: list[torch.Tensor] = []
        self._policy_betas: list[torch.Tensor] = []
        self._policy_mask_log_probs: list[torch.Tensor] = []
        self._policy_entropies: list[torch.Tensor] = []
        self._router_kl_terms: list[torch.Tensor] = []
        self.reset(num_layers)

    def reset(self, num_layers: int | None = None) -> None:
        layers = num_layers if num_layers is not None else len(self.router_logits_by_layer)
        self.caches = [[[] for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.options = [[None for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.option_weights = [
            [None for _layer in range(layers)] for _batch in range(self.batch_size)
        ]
        self.option_logits = [
            [None for _layer in range(layers)] for _batch in range(self.batch_size)
        ]
        self.boundary_confidences = [
            [0.0 for _layer in range(layers)] for _batch in range(self.batch_size)
        ]
        self.ages = [[0 for _layer in range(layers)] for _batch in range(self.batch_size)]
        self.option_lengths = []
        self.router_logits_by_layer = [None for _layer in range(layers)]
        self.events = 0
        self.controller_calls = 0
        self.gate_evaluated_tokens = 0
        self.beta_calls = 0
        self.option_switches = 0
        self.switch_edit_distance = 0
        self.soft_keep_executions = 0
        self.soft_refresh_executions = 0
        self.policy_termination_decisions = 0
        self.policy_switch_decisions = 0
        self.cache_load_cost = 0.0
        self._soft_beta_terms = []
        self._soft_load_terms = []
        self._soft_switch_terms = []
        self._counterfactual_beta_losses = []
        self._counterfactual_mask_losses = []
        self._counterfactual_targets = []
        self._continue_coverages = []
        self._switch_coverages = []
        self._policy_beta_log_probs = []
        self._policy_betas = []
        self._policy_mask_log_probs = []
        self._policy_entropies = []
        self._router_kl_terms = []

    def forward_sparse_moe(
        self,
        block: torch.nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "soft":
            return self._forward_soft(block, layer_index, hidden_states)
        if self.mode == "policy_gradient":
            return self._forward_policy_gradient(block, layer_index, hidden_states)
        return self._forward_hard(block, layer_index, hidden_states)

    def _forward_hard(
        self,
        block: torch.nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim, num_experts = self._validate_block(
            block,
            hidden_states,
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
                if active and not self._hard_should_terminate(
                    block,
                    batch_index,
                    layer_index,
                    flat_states[row_index],
                ):
                    self._fill_hard_continuation(
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
                boundary_logits, boundary_weights, boundary_selected = _full_gate_topk(
                    block,
                    flat_states.index_select(0, boundary_index),
                    self.top_k,
                    hidden_states.dtype,
                )
                for local_index, row_index in enumerate(boundary_rows):
                    batch_index, active = boundary_metadata[local_index]
                    selected_experts[row_index] = boundary_selected[local_index]
                    routing_weights[row_index] = boundary_weights[local_index]
                    router_logits[row_index] = boundary_logits[local_index].to(
                        router_logits.dtype
                    )
                    if active:
                        selected = _tensor_to_option(boundary_selected[local_index])
                        self._record_boundary(
                            batch_index,
                            layer_index,
                            selected,
                            boundary_weights[local_index].detach(),
                            boundary_logits[local_index].detach(),
                            count_controller=True,
                        )

        final_hidden_states = _execute_sparse_experts(
            block,
            flat_states,
            selected_experts,
            routing_weights,
            num_experts,
            hidden_states.dtype,
        )
        self.router_logits_by_layer[layer_index] = router_logits.detach()
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

    def _forward_policy_gradient(
        self,
        block: torch.nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim, num_experts = self._validate_block(
            block,
            hidden_states,
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
        router_logit_rows: list[torch.Tensor | None] = [None for _row in range(total_tokens)]

        for token_index in range(sequence_length):
            boundary_rows: list[int] = []
            boundary_metadata: list[tuple[int, bool]] = []
            for batch_index in range(batch_size):
                row_index = batch_index * sequence_length + token_index
                active = bool(self.attention_mask[batch_index, token_index].item())
                if active:
                    current_option = self.options[batch_index][layer_index]
                    if current_option is not None:
                        beta = self._termination_probability(
                            batch_index,
                            layer_index,
                            flat_states[row_index],
                            block,
                        )
                        should_terminate = self._sample_policy_termination(beta)
                        if not should_terminate:
                            row_logits = self._fill_hard_continuation(
                                block,
                                batch_index,
                                layer_index,
                                row_index,
                                flat_states[row_index],
                                selected_experts,
                                routing_weights,
                                router_logits,
                            )
                            router_logit_rows[row_index] = row_logits
                            continue

                boundary_rows.append(row_index)
                boundary_metadata.append((batch_index, active))

            if boundary_rows:
                boundary_index = torch.tensor(
                    boundary_rows,
                    dtype=torch.long,
                    device=flat_states.device,
                )
                boundary_logits, deterministic_weights, deterministic_selected = _full_gate_topk(
                    block,
                    flat_states.index_select(0, boundary_index),
                    self.top_k,
                    hidden_states.dtype,
                )
                for local_index, row_index in enumerate(boundary_rows):
                    batch_index, active = boundary_metadata[local_index]
                    if active:
                        sampled, log_prob, entropy = plackett_luce_sample(
                            boundary_logits[local_index],
                            self.top_k,
                        )
                        weights = _routing_weights_for_selected(
                            block,
                            boundary_logits[local_index],
                            sampled,
                            hidden_states.dtype,
                        )
                        self._record_policy_mask(log_prob, entropy)
                        selected = sampled
                    else:
                        selected = deterministic_selected[local_index]
                        weights = deterministic_weights[local_index]

                    selected_experts[row_index] = selected
                    routing_weights[row_index] = weights
                    row_logits = boundary_logits[local_index].to(router_logits.dtype)
                    router_logits[row_index] = row_logits
                    router_logit_rows[row_index] = row_logits
                    if active:
                        self._append_router_kl_term(
                            boundary_logits[local_index],
                            batch_index=batch_index,
                            layer_index=layer_index,
                            token_index=token_index,
                        )
                        self._record_boundary(
                            batch_index,
                            layer_index,
                            _tensor_to_option(selected),
                            weights.detach(),
                            boundary_logits[local_index].detach(),
                            count_controller=True,
                        )

        final_hidden_states = _execute_sparse_experts(
            block,
            flat_states,
            selected_experts,
            routing_weights,
            num_experts,
            hidden_states.dtype,
        )
        if any(row is None for row in router_logit_rows):
            raise RuntimeError("policy-gradient routing left unset router-logit rows")
        differentiable_router_logits = torch.stack(
            [row for row in router_logit_rows if row is not None],
        )
        self.router_logits_by_layer[layer_index] = differentiable_router_logits.detach()
        return (
            final_hidden_states.reshape(batch_size, sequence_length, hidden_dim),
            differentiable_router_logits,
        )

    def _forward_soft(
        self,
        block: torch.nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim, num_experts = self._validate_block(
            block,
            hidden_states,
        )
        flat_states = hidden_states.reshape(-1, hidden_dim)
        total_tokens = flat_states.shape[0]
        final_hidden_states = torch.zeros_like(flat_states)
        router_logits = torch.empty(
            (total_tokens, num_experts),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for token_index in range(sequence_length):
            token_rows = [
                batch_index * sequence_length + token_index for batch_index in range(batch_size)
            ]
            token_index_tensor = torch.tensor(
                token_rows,
                dtype=torch.long,
                device=flat_states.device,
            )
            fresh_logits, fresh_weights, fresh_selected = _full_gate_topk(
                block,
                flat_states.index_select(0, token_index_tensor),
                self.top_k,
                hidden_states.dtype,
            )
            fresh_outputs = _execute_sparse_experts(
                block,
                flat_states.index_select(0, token_index_tensor),
                fresh_selected,
                fresh_weights,
                num_experts,
                hidden_states.dtype,
            )
            self.soft_refresh_executions += int(fresh_selected.shape[0])

            keep_rows: list[int] = []
            keep_local_indices: list[int] = []
            keep_selected_values: list[tuple[int, ...]] = []
            keep_weight_values: list[torch.Tensor] = []
            beta_by_local_index: dict[int, torch.Tensor] = {}

            for local_index, batch_index in enumerate(range(batch_size)):
                row_index = token_rows[local_index]
                active = bool(self.attention_mask[batch_index, token_index].item())
                router_logits[row_index] = fresh_logits[local_index].to(router_logits.dtype)
                if not active:
                    final_hidden_states[row_index] = fresh_outputs[local_index]
                    continue
                self._append_router_kl_term(
                    fresh_logits[local_index],
                    batch_index=batch_index,
                    layer_index=layer_index,
                    token_index=token_index,
                )

                current_option = self.options[batch_index][layer_index]
                if current_option is None:
                    forced_beta = fresh_outputs.new_tensor(1.0)
                    self._append_counterfactual_terms(
                        beta=None,
                        keep_option=None,
                        fresh_logits=fresh_logits[local_index],
                        fresh_option=_tensor_to_option(fresh_selected[local_index]),
                        batch_index=batch_index,
                        layer_index=layer_index,
                        token_index=token_index,
                    )
                    self._append_soft_costs(
                        forced_beta,
                        keep_option=None,
                        fresh_option=_tensor_to_option(fresh_selected[local_index]),
                        resident=self.caches[batch_index][layer_index],
                    )
                    final_hidden_states[row_index] = fresh_outputs[local_index]
                    self._record_boundary(
                        batch_index,
                        layer_index,
                        _tensor_to_option(fresh_selected[local_index]),
                        fresh_weights[local_index].detach(),
                        fresh_logits[local_index].detach(),
                        count_controller=True,
                    )
                    continue

                beta = self._termination_probability(
                    batch_index,
                    layer_index,
                    flat_states[row_index],
                    block,
                )
                beta_by_local_index[local_index] = beta
                self._append_counterfactual_terms(
                    beta=beta,
                    keep_option=current_option,
                    fresh_logits=fresh_logits[local_index],
                    fresh_option=_tensor_to_option(fresh_selected[local_index]),
                    batch_index=batch_index,
                    layer_index=layer_index,
                    token_index=token_index,
                )
                keep_weights = self._continuation_weights(
                    block,
                    flat_states[row_index],
                    current_option,
                    self.option_weights[batch_index][layer_index],
                )
                keep_rows.append(row_index)
                keep_local_indices.append(local_index)
                keep_selected_values.append(current_option)
                keep_weight_values.append(keep_weights)

            if keep_rows:
                keep_index_tensor = torch.tensor(
                    keep_rows,
                    dtype=torch.long,
                    device=flat_states.device,
                )
                keep_selected = torch.tensor(
                    keep_selected_values,
                    dtype=torch.long,
                    device=flat_states.device,
                )
                keep_weights = torch.stack(keep_weight_values).to(
                    device=flat_states.device,
                    dtype=hidden_states.dtype,
                )
                keep_outputs = _execute_sparse_experts(
                    block,
                    flat_states.index_select(0, keep_index_tensor),
                    keep_selected,
                    keep_weights,
                    num_experts,
                    hidden_states.dtype,
                )
                self.soft_keep_executions += int(keep_selected.shape[0])

                for keep_output, local_index in zip(
                    keep_outputs,
                    keep_local_indices,
                    strict=True,
                ):
                    batch_index = local_index
                    row_index = token_rows[local_index]
                    beta = beta_by_local_index[local_index].to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    fresh_option = _tensor_to_option(fresh_selected[local_index])
                    current_option = self.options[batch_index][layer_index]
                    if current_option is None:
                        raise RuntimeError("soft keep branch lost its active option")
                    final_hidden_states[row_index] = (
                        (1.0 - beta) * keep_output + beta * fresh_outputs[local_index]
                    )
                    self._append_soft_costs(
                        beta,
                        keep_option=current_option,
                        fresh_option=fresh_option,
                        resident=self.caches[batch_index][layer_index],
                    )
                    update_to_fresh = float(beta.detach().cpu()) > self.state_update_threshold
                    if update_to_fresh:
                        self._record_boundary(
                            batch_index,
                            layer_index,
                            fresh_option,
                            fresh_weights[local_index].detach(),
                            fresh_logits[local_index].detach(),
                            count_controller=True,
                        )
                    else:
                        self._record_event(batch_index, layer_index, current_option)

        self.router_logits_by_layer[layer_index] = router_logits.detach()
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

    def _validate_block(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if (batch_size, sequence_length) != (self.batch_size, self.seq_len):
            raise RuntimeError(
                "learned-beta controller was built for "
                f"{(self.batch_size, self.seq_len)} but saw {(batch_size, sequence_length)}"
            )
        num_experts = _infer_num_experts(block)
        if num_experts != self.termination_head.num_experts:
            raise RuntimeError(
                f"termination head expects {self.termination_head.num_experts} experts, "
                f"but sparse block has {num_experts}"
            )
        block_top_k = getattr(block, "top_k", None)
        if block_top_k is None:
            block_top_k = getattr(getattr(block, "gate", None), "top_k", self.top_k)
        block_top_k = int(block_top_k)
        if block_top_k != self.top_k:
            raise RuntimeError(
                f"controller top_k={self.top_k} does not match sparse block top_k={block_top_k}"
            )
        return batch_size, sequence_length, hidden_dim, num_experts

    def _hard_should_terminate(
        self,
        block: torch.nn.Module,
        batch_index: int,
        layer_index: int,
        hidden_state: torch.Tensor,
    ) -> bool:
        current_option = self.options[batch_index][layer_index]
        if current_option is None:
            return True
        beta = self._termination_probability(batch_index, layer_index, hidden_state, block)
        return bool(float(beta.detach().cpu()) > self.termination_threshold)

    def _sample_policy_termination(self, beta: torch.Tensor) -> bool:
        action = torch.bernoulli(beta.detach().float().clamp(0.0, 1.0)).to(beta.dtype)
        self._record_policy_termination(beta, action)
        return bool(float(action.detach().cpu()) >= 0.5)

    def _record_policy_termination(self, beta: torch.Tensor, action: torch.Tensor) -> None:
        action = action.to(device=beta.device, dtype=beta.dtype)
        self.policy_termination_decisions += 1
        if bool(float(action.detach().cpu()) >= 0.5):
            self.policy_switch_decisions += 1
        self._policy_betas.append(beta.float())
        self._policy_beta_log_probs.append(bernoulli_log_prob(beta, action))
        self._policy_entropies.append(bernoulli_entropy(beta))

    def _record_policy_mask(self, log_prob: torch.Tensor, entropy: torch.Tensor) -> None:
        self._policy_mask_log_probs.append(log_prob.float())
        self._policy_entropies.append(entropy.float())

    def _termination_probability(
        self,
        batch_index: int,
        layer_index: int,
        hidden_state: torch.Tensor,
        block: torch.nn.Module,
    ) -> torch.Tensor:
        option = self.options[batch_index][layer_index]
        option_weights = self.option_weights[batch_index][layer_index]
        if option is None or option_weights is None:
            raise RuntimeError("termination probability requested without an active option")
        features = self._features(
            batch_index,
            layer_index,
            hidden_state,
            option,
            option_weights,
        )
        self.beta_calls += 1
        batched_features = features.unsqueeze(0)
        if self.termination_head.config.shared_across_layers:
            beta = self.termination_head(batched_features)
        else:
            layer_indices = torch.tensor(
                [layer_index],
                dtype=torch.long,
                device=hidden_state.device,
            )
            beta = self.termination_head(batched_features, layer_indices=layer_indices)
        return beta.squeeze(0).to(hidden_state.device)

    def _features(
        self,
        batch_index: int,
        layer_index: int,
        hidden_state: torch.Tensor,
        option: tuple[int, ...],
        option_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.termination_head.config.feature_mode == "henderson":
            option_tensor = torch.tensor(
                list(option),
                dtype=torch.long,
                device=hidden_state.device,
            ).unsqueeze(0)
            set_features = self.termination_head.set_embedding(option_tensor).squeeze(0)
            return torch.cat((hidden_state.float(), set_features.float()), dim=-1)

        dtype = torch.float32
        hidden = hidden_state.detach().float()
        hidden_summary = torch.stack(
            (
                hidden.mean(),
                hidden.std(unbiased=False),
                torch.sqrt(torch.mean(hidden * hidden).clamp_min(0.0)),
                hidden.abs().max(),
            )
        )
        age = self.ages[batch_index][layer_index]
        age_denominator = torch.tensor(
            float(max(1, self.termination_head.config.max_age)),
            dtype=dtype,
            device=hidden_state.device,
        )
        age_feature = torch.log1p(torch.tensor(float(age), dtype=dtype, device=hidden_state.device))
        age_feature = age_feature / torch.log1p(age_denominator)
        if self.termination_head.num_layers <= 1:
            layer_feature = hidden.new_tensor(0.0)
        else:
            layer_feature = hidden.new_tensor(layer_index / (self.termination_head.num_layers - 1))
        expert_denominator = max(1, self.termination_head.num_experts - 1)
        option_ids = torch.tensor(
            list(option),
            dtype=dtype,
            device=hidden_state.device,
        ) / float(expert_denominator)
        weights = option_weights.detach().to(device=hidden_state.device, dtype=dtype)
        confidence = hidden.new_tensor(self.boundary_confidences[batch_index][layer_index])
        parts = [
            hidden_summary.to(device=hidden_state.device),
            age_feature.reshape(1),
            layer_feature.reshape(1),
            option_ids,
            weights,
            confidence.reshape(1),
        ]
        if self.termination_head.config.include_cache_features:
            resident = self.caches[batch_index][layer_index]
            resident_set = set(resident)
            resident_fraction = sum(1 for expert in option if expert in resident_set) / len(option)
            keep_load = self._load_cost_float(option, resident)
            normalizer = self._cost_normalizer()
            cache_features = torch.tensor(
                [
                    len(resident) / self.capacity,
                    resident_fraction,
                    keep_load / normalizer,
                ],
                dtype=dtype,
                device=hidden_state.device,
            )
            parts.append(cache_features)
        return torch.cat(parts)

    def _fill_hard_continuation(
        self,
        block: torch.nn.Module,
        batch_index: int,
        layer_index: int,
        row_index: int,
        hidden_state: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        option = self.options[batch_index][layer_index]
        option_logits = self.option_logits[batch_index][layer_index]
        if option is None or option_logits is None:
            raise RuntimeError("continuation without an active option")
        weights = self._continuation_weights(
            block,
            hidden_state,
            option,
            self.option_weights[batch_index][layer_index],
        )
        selected_experts[row_index] = torch.tensor(
            option,
            dtype=selected_experts.dtype,
            device=selected_experts.device,
        )
        routing_weights[row_index] = weights.to(
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        )
        row_logits = self._continuation_router_logits(
            option,
            weights,
            option_logits,
            router_logits.shape[-1],
            router_logits.device,
            router_logits.dtype,
        )
        router_logits[row_index] = row_logits
        self._record_event(batch_index, layer_index, option)
        return row_logits

    def _continuation_weights(
        self,
        block: torch.nn.Module,
        hidden_state: torch.Tensor,
        option: tuple[int, ...],
        stored_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if stored_weights is None:
            raise RuntimeError("continuation without stored option weights")
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
        return logits.scatter(0, option_tensor, weights.to(device=device, dtype=dtype))

    def _record_boundary(
        self,
        batch_index: int,
        layer_index: int,
        selected: tuple[int, ...],
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        count_controller: bool,
    ) -> None:
        current_option = self.options[batch_index][layer_index]
        if current_option is not None:
            self.option_lengths.append(self.ages[batch_index][layer_index])
            edit = _edit_distance(current_option, selected)
            self.switch_edit_distance += edit
            if edit:
                self.option_switches += 1
        if count_controller:
            self.controller_calls += 1
            self.gate_evaluated_tokens += 1
        self.options[batch_index][layer_index] = selected
        self.option_weights[batch_index][layer_index] = routing_weights.detach().float().cpu()
        self.option_logits[batch_index][layer_index] = router_logits.detach().float().cpu()
        self.boundary_confidences[batch_index][layer_index] = float(
            routing_weights.detach().float().max().cpu()
        )
        self.ages[batch_index][layer_index] = 0
        self._record_event(batch_index, layer_index, selected)

    def _record_event(
        self,
        batch_index: int,
        layer_index: int,
        selected: tuple[int, ...],
    ) -> None:
        self.events += 1
        self.cache_load_cost += self._load_cost_float(
            selected,
            self.caches[batch_index][layer_index],
        )
        _update_resident(self.caches[batch_index][layer_index], selected, self.capacity)
        self.ages[batch_index][layer_index] += 1

    def _append_soft_costs(
        self,
        beta: torch.Tensor,
        *,
        keep_option: tuple[int, ...] | None,
        fresh_option: tuple[int, ...],
        resident: list[int],
    ) -> None:
        beta = beta.float()
        if keep_option is None:
            keep_load = beta.new_tensor(0.0)
            switch_cost = beta.new_tensor(0.0)
        else:
            keep_load = self._load_cost_tensor(keep_option, resident, beta.device)
            switch_cost = beta * float(_edit_distance(keep_option, fresh_option))
        fresh_load = self._load_cost_tensor(fresh_option, resident, beta.device)
        expected_load = (1.0 - beta) * keep_load + beta * fresh_load
        self._soft_beta_terms.append(beta)
        self._soft_load_terms.append(expected_load)
        self._soft_switch_terms.append(switch_cost)

    def _append_counterfactual_terms(
        self,
        *,
        beta: torch.Tensor | None,
        keep_option: tuple[int, ...] | None,
        fresh_logits: torch.Tensor,
        fresh_option: tuple[int, ...],
        batch_index: int,
        layer_index: int,
        token_index: int,
    ) -> None:
        if self.reference_router_logits is None:
            return
        reference_logits = self.reference_router_logits[
            batch_index,
            layer_index,
            token_index,
        ].to(device=fresh_logits.device)
        demand = torch.topk(reference_logits, k=self.top_k, dim=-1).indices
        if keep_option is None:
            continue_coverage = fresh_logits.new_tensor(1.0)
            edit_cost = fresh_logits.new_tensor(0.0)
            target = fresh_logits.new_tensor(1.0)
        else:
            continue_coverage = _hard_missing_coverage(
                keep_option,
                demand,
                device=fresh_logits.device,
            )
            edit_cost = fresh_logits.new_tensor(
                float(_edit_distance(keep_option, fresh_option)) / float(max(1, self.top_k))
            )
        switch_coverage = _st_missing_coverage(
            fresh_logits,
            demand,
            top_k=self.top_k,
            temperature=self.mask_temperature,
        )
        if keep_option is not None:
            q_continue = (
                self.lambda_cov * continue_coverage
                + self.lambda_cache * fresh_logits.new_tensor(0.0)
            )
            q_switch = (
                self.lambda_cov * switch_coverage.detach()
                + fresh_logits.new_tensor(self.lambda_call)
                + self.lambda_edit * edit_cost
            )
            eta = max(1.0e-6, self.counterfactual_eta)
            target = torch.sigmoid((q_continue - q_switch) / eta).detach()
            if beta is None:
                raise RuntimeError("counterfactual beta loss requires beta for active option")
            beta_float = beta.float().clamp(1.0e-6, 1.0 - 1.0e-6)
            self._counterfactual_beta_losses.append(
                F.binary_cross_entropy(beta_float, target.to(beta_float.device))
            )
        self._counterfactual_mask_losses.append(target.detach() * switch_coverage)
        self._counterfactual_targets.append(target.detach())
        self._continue_coverages.append(continue_coverage.detach())
        self._switch_coverages.append(switch_coverage.detach())

    def _append_router_kl_term(
        self,
        current_logits: torch.Tensor,
        *,
        batch_index: int,
        layer_index: int,
        token_index: int,
    ) -> None:
        if self.reference_router_logits is None:
            return
        reference_logits = self.reference_router_logits[
            batch_index,
            layer_index,
            token_index,
        ].to(device=current_logits.device)
        reference_probs = torch.softmax(reference_logits.float(), dim=-1)
        current_log_probs = torch.log_softmax(current_logits.float(), dim=-1)
        self._router_kl_terms.append(
            F.kl_div(
                current_log_probs,
                reference_probs,
                reduction="sum",
            )
        )

    def _load_cost_tensor(
        self,
        selected: tuple[int, ...],
        resident: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        before = set(resident)
        misses = [expert for expert in selected if expert not in before]
        if not misses:
            return torch.zeros((), device=device)
        if self.expert_costs is None:
            return torch.tensor(float(len(misses)), device=device)
        return self.expert_costs[misses].to(device=device).sum()

    def _load_cost_float(self, selected: tuple[int, ...], resident: list[int]) -> float:
        before = set(resident)
        misses = [expert for expert in selected if expert not in before]
        if not misses:
            return 0.0
        if self.expert_costs is None:
            return float(len(misses))
        return float(self.expert_costs[misses].sum())

    def _cost_normalizer(self) -> float:
        if self.expert_costs is None:
            return float(max(1, self.top_k))
        mean_cost = float(self.expert_costs.mean())
        return max(1.0e-6, mean_cost * self.top_k)

    def soft_loss_terms(self, device: torch.device) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device)
        if not self._soft_beta_terms:
            return {
                "deliberation": zero,
                "load": zero,
                "switch": zero,
                "commitment": zero,
                "beta_counterfactual": zero,
                "mask_coverage": zero,
                "counterfactual_target": zero,
                "continue_coverage": zero,
                "switch_coverage": zero,
            }
        beta_stack = torch.stack([term.to(device) for term in self._soft_beta_terms])
        terms = {
            "deliberation": beta_stack.mean(),
            "load": torch.stack([term.to(device) for term in self._soft_load_terms]).mean(),
            "switch": torch.stack([term.to(device) for term in self._soft_switch_terms]).mean(),
            # Commitment penalty: pushes soft beta toward {0,1} to close the soft-train /
            # hard-eval gap. Zero when beta is already saturated.
            "commitment": (beta_stack * (1.0 - beta_stack)).mean(),
        }
        terms["beta_counterfactual"] = (
            torch.stack([term.to(device) for term in self._counterfactual_beta_losses]).mean()
            if self._counterfactual_beta_losses
            else zero
        )
        terms["mask_coverage"] = (
            torch.stack([term.to(device) for term in self._counterfactual_mask_losses]).mean()
            if self._counterfactual_mask_losses
            else zero
        )
        terms["counterfactual_target"] = (
            torch.stack([term.to(device) for term in self._counterfactual_targets]).mean()
            if self._counterfactual_targets
            else zero
        )
        terms["continue_coverage"] = (
            torch.stack([term.to(device) for term in self._continue_coverages]).mean()
            if self._continue_coverages
            else zero
        )
        terms["switch_coverage"] = (
            torch.stack([term.to(device) for term in self._switch_coverages]).mean()
            if self._switch_coverages
            else zero
        )
        return terms

    def policy_gradient_terms(self, device: torch.device) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device)
        beta_values = (
            torch.stack([term.to(device).float() for term in self._policy_betas])
            if self._policy_betas
            else None
        )
        expected_hazard = beta_values.mean() if beta_values is not None else zero
        beta_log_prob = (
            torch.stack([term.to(device) for term in self._policy_beta_log_probs]).sum()
            if self._policy_beta_log_probs
            else zero
        )
        mask_log_prob = (
            torch.stack([term.to(device) for term in self._policy_mask_log_probs]).sum()
            if self._policy_mask_log_probs
            else zero
        )
        entropy = (
            torch.stack([term.to(device) for term in self._policy_entropies]).sum()
            if self._policy_entropies
            else zero
        )
        log_prob = beta_log_prob + mask_log_prob
        log_count = len(self._policy_beta_log_probs) + len(self._policy_mask_log_probs)
        normalizer = float(max(1, log_count))
        beta_count = len(self._policy_beta_log_probs)
        mask_count = len(self._policy_mask_log_probs)
        return {
            "beta_log_prob": beta_log_prob,
            "mask_log_prob": mask_log_prob,
            "log_prob": log_prob,
            "beta_normalized_log_prob": beta_log_prob / normalizer,
            "mask_normalized_log_prob": mask_log_prob / normalizer,
            "beta_log_prob_per_decision": beta_log_prob / float(max(1, beta_count)),
            "mask_log_prob_per_decision": mask_log_prob / float(max(1, mask_count)),
            "normalized_log_prob": log_prob / normalizer,
            "entropy": entropy,
            "normalized_entropy": entropy / normalizer,
            "expected_hazard": expected_hazard,
            "log_prob_count": torch.tensor(float(log_count), device=device),
            "beta_log_prob_count": torch.tensor(
                float(len(self._policy_beta_log_probs)),
                device=device,
            ),
            "mask_log_prob_count": torch.tensor(
                float(len(self._policy_mask_log_probs)),
                device=device,
            ),
        }

    def beta_statistics(self, device: torch.device) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device)
        count = torch.tensor(float(len(self._policy_betas)), device=device)
        if not self._policy_betas:
            return {
                "beta_sum": zero,
                "beta_mean": zero,
                "beta_std": zero,
                "beta_min": zero,
                "beta_max": zero,
                "beta_p10": zero,
                "beta_p25": zero,
                "beta_p50": zero,
                "beta_p75": zero,
                "beta_p90": zero,
                "beta_count": count,
            }
        values = torch.stack([term.detach().to(device).float() for term in self._policy_betas])
        quantiles = torch.quantile(
            values,
            torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=device),
        )
        return {
            "beta_sum": values.sum(),
            "beta_mean": values.mean(),
            "beta_std": values.std(unbiased=False),
            "beta_min": values.min(),
            "beta_max": values.max(),
            "beta_p10": quantiles[0],
            "beta_p25": quantiles[1],
            "beta_p50": quantiles[2],
            "beta_p75": quantiles[3],
            "beta_p90": quantiles[4],
            "beta_count": count,
        }

    def router_kl_loss(self, device: torch.device) -> torch.Tensor:
        if not self._router_kl_terms:
            return torch.zeros((), device=device)
        return torch.stack([term.to(device) for term in self._router_kl_terms]).mean()

    @property
    def expected_deliberation_cost(self) -> float:
        if self.mode in {"hard", "policy_gradient"}:
            return float(self.controller_calls)
        if not self._soft_beta_terms:
            return 0.0
        return float(
            torch.stack([term.detach().cpu() for term in self._soft_beta_terms]).mean().item()
        )

    @property
    def mean_option_length(self) -> float:
        if self.option_lengths:
            return sum(self.option_lengths) / len(self.option_lengths)
        return 0.0

    @property
    def call_rate(self) -> float:
        return self.policy_switch_decisions / max(1, self.policy_termination_decisions)

    @property
    def edit_rate(self) -> float:
        return self.switch_edit_distance / max(1, self.top_k * self.policy_switch_decisions)

    @property
    def cache_cost_rate(self) -> float:
        return self.cache_load_cost / max(1, self.events)

    def finalize(self) -> None:
        for batch_options, batch_ages in zip(self.options, self.ages, strict=True):
            for option, age in zip(batch_options, batch_ages, strict=True):
                if option is not None:
                    self.option_lengths.append(age)

    def router_stack(self, batch_size: int, seq_len: int) -> torch.Tensor:
        layers: list[torch.Tensor] = []
        for layer_index, layer_logits in enumerate(self.router_logits_by_layer):
            if layer_logits is None:
                raise RuntimeError(f"missing learned-beta router logits for layer {layer_index}")
            layers.append(layer_logits.detach().float().reshape(batch_size, seq_len, -1).cpu())
        return torch.stack(layers, dim=1)


class BestOfPoolBetaController(LearnedBetaSparseMoeController):
    """Learned joint termination (beta) + policy-over-options (mu) on a best-of-pool cache.

    An *option* is a per-layer resident pool ``R`` of size ``capacity``, held frozen during the
    option. keep = best-of-pool over the frozen ``R_old`` (top-k of the frozen router logits masked
    to ``R_old``); fresh = best-of-pool over ``R_new = topC(mu)``; the soft blend is
    ``(1-beta)*keep + beta*fresh``. The transfer price rides ``switch = beta * |R_new \\ R_old|``
    (soft set-difference). Router + experts stay frozen; only the beta and mu heads learn. Only
    ``restricted_layers`` are pooled; other layers pass through vanilla.
    """

    POOL_FILL = -1.0e4
    POP_DECAY = 0.9

    def __init__(
        self,
        *args: Any,
        policy_head: PolicyOverOptionsHead,
        restricted_layers: frozenset[int] | None = None,
        coverage_target: str = "current",
        **kwargs: Any,
    ) -> None:
        if coverage_target not in {"current", "union_dwell"}:
            raise ValueError("coverage_target must be 'current' or 'union_dwell'")
        self.policy_head = policy_head
        self._restricted_layers_arg = restricted_layers
        # 'current': supervise mu's top-C to cover the current token's demand (myopic, dwell-1).
        # 'union_dwell': cover the running union of demand over the current option (comparison 4) --
        # trains mu to provision the pool for the whole option horizon, not just one token.
        self.coverage_target = coverage_target
        super().__init__(*args, **kwargs)
        if self.termination_head.config.feature_mode != "henderson":
            raise ValueError("best_of_pool requires the beta head feature_mode 'henderson'")
        num_layers = len(self.router_logits_by_layer)
        self.restricted_layers = (
            restricted_layers if restricted_layers is not None else frozenset(range(num_layers))
        )

    def reset(self, num_layers: int | None = None) -> None:
        super().reset(num_layers)
        layers = len(self.router_logits_by_layer)
        num_experts = self.termination_head.num_experts
        self.pop_ema = [
            [torch.zeros(num_experts) for _layer in range(layers)]
            for _batch in range(self.batch_size)
        ]
        self.last_seen = [
            [torch.full((num_experts,), 1.0e9) for _layer in range(layers)]
            for _batch in range(self.batch_size)
        ]
        # Running union of realized top-k demand over the current option (union_dwell coverage).
        self._option_demand: list[list[torch.Tensor | None]] = [
            [None for _layer in range(layers)] for _batch in range(self.batch_size)
        ]
        self._mu_coverage_terms: list[torch.Tensor] = []

    def soft_loss_terms(self, device: torch.device) -> dict[str, torch.Tensor]:
        terms = super().soft_loss_terms(device)
        terms["mu_coverage"] = (
            torch.stack([term.to(device) for term in self._mu_coverage_terms]).mean()
            if self._mu_coverage_terms
            else torch.zeros((), device=device)
        )
        return terms

    # -- mu features + pool selection ------------------------------------------------------------
    def _policy_features(
        self,
        batch_index: int,
        layer_index: int,
        hidden_state: torch.Tensor,
        full_logits: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        return assemble_policy_features(
            hidden_state,
            full_logits,
            self.pop_ema[batch_index][layer_index],
            self.last_seen[batch_index][layer_index],
            self.caches[batch_index][layer_index],
            num_experts,
            self.seq_len,
        )

    def _select_new_pool(
        self,
        batch_index: int,
        layer_index: int,
        hidden_state: torch.Tensor,
        full_logits: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._policy_features(
            batch_index, layer_index, hidden_state, full_logits, num_experts
        )
        scores = self.policy_head(features, layer_index)
        st_mask, indices = st_topc_pool_mask(
            scores, self.capacity, temperature=self.mask_temperature
        )
        return st_mask, indices, scores

    def _update_demand(
        self, batch_index: int, layer_index: int, full_logits: torch.Tensor
    ) -> None:
        update_pool_demand(
            self.pop_ema[batch_index][layer_index],
            self.last_seen[batch_index][layer_index],
            full_logits,
            self.top_k,
            self.POP_DECAY,
        )

    # -- best-of-pool routing --------------------------------------------------------------------
    def _bop_route(
        self,
        block: torch.nn.Module,
        state_row: torch.Tensor,
        full_logits: torch.Tensor,
        pool_bool: torch.Tensor,
        num_experts: int,
        dtype: torch.dtype,
        st_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        masked = full_logits.float().masked_fill(~pool_bool, self.POOL_FILL)
        probs = torch.softmax(masked, dim=-1)
        weights, selected = torch.topk(probs, self.top_k, dim=-1)
        if st_mask is not None:  # inject mu gradient through the fresh pool membership
            weights = weights * st_mask.index_select(0, selected)
        output = _execute_sparse_experts(
            block,
            state_row.unsqueeze(0),
            selected.unsqueeze(0),
            weights.to(dtype).unsqueeze(0),
            num_experts,
            dtype,
        )
        return output[0], _tensor_to_option(selected)

    def _pool_bool(self, pool: tuple[int, ...], num_experts: int, device: torch.device):
        mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
        mask[list(pool)] = True
        return mask

    def _append_bop_costs(
        self, beta: torch.Tensor, st_mask: torch.Tensor, resident_mask: torch.Tensor
    ) -> None:
        beta = beta.float()
        switch = beta * soft_set_difference(st_mask, resident_mask)
        self._soft_beta_terms.append(beta)
        self._soft_switch_terms.append(switch)
        self._soft_load_terms.append(beta.new_tensor(0.0))

    def _bop_route_batched(
        self,
        block: torch.nn.Module,
        flat: torch.Tensor,
        row_indices: list[int],
        pool_bools: list[torch.Tensor],
        fresh_logits_all: torch.Tensor,
        num_experts: int,
        dtype: torch.dtype,
        st_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Batched best-of-pool routing: one ``_execute_sparse_experts`` over many rows.

        Equivalent, row-for-row, to calling ``_bop_route`` per row (softmax/top-k/dispatch are all
        per-row independent), but collapses the per-token, per-batch expert dispatch into a single
        kernel sweep. ``pool_bools[i]`` is the membership mask for ``row_indices[i]``; ``st_masks``,
        when given, injects the mu gradient through the fresh-pool membership.
        """
        idx = torch.tensor(row_indices, dtype=torch.long, device=flat.device)
        logits = fresh_logits_all.index_select(0, idx).float()
        pool_stack = torch.stack(pool_bools)
        masked = logits.masked_fill(~pool_stack, self.POOL_FILL)
        probs = torch.softmax(masked, dim=-1)
        weights, selected = torch.topk(probs, self.top_k, dim=-1)
        if st_masks is not None:
            st_stack = torch.stack(st_masks)
            weights = weights * st_stack.gather(1, selected)
        states = flat.index_select(0, idx)
        return _execute_sparse_experts(
            block, states, selected, weights.to(dtype), num_experts, dtype
        )

    def _coverage_demand(
        self,
        batch_index: int,
        layer_index: int,
        current_demand: torch.Tensor,
        num_experts: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Coverage target for mu at this token: current-token demand, or the option-so-far union.

        Returns ``(demand_index_tensor, new_union_mask)``. For ``current`` the union is ``None``
        (no state). For ``union_dwell`` the demand is the running union of realized top-k demand
        over the current option (from the last boundary through this token) -- a causal, monotone
        target that trains mu to provision for the whole option horizon, not one token.
        """
        if self.coverage_target == "current":
            return current_demand, None
        acc = self._option_demand[batch_index][layer_index]
        union = torch.zeros(num_experts, dtype=torch.bool, device=device)
        if acc is not None:
            union |= acc.to(device)
        union[current_demand.to(device)] = True
        return union.nonzero(as_tuple=False).reshape(-1), union

    def _apply_coverage_state(
        self,
        batch_index: int,
        layer_index: int,
        info: dict[str, Any],
        num_experts: int,
        device: torch.device,
        *,
        reset: bool,
    ) -> None:
        """Advance the option-demand union used by 'union_dwell' coverage (no-op for 'current').

        ``reset`` (a fired/bootstrap boundary) starts a fresh option whose union is this token's
        demand; otherwise the current option's union is extended to include this token.
        """
        if self.coverage_target != "union_dwell":
            return
        if reset:
            mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
            mask[info["cur_demand"].to(device)] = True
            self._option_demand[batch_index][layer_index] = mask
        else:
            self._option_demand[batch_index][layer_index] = info["cov_union"]

    def _forward_soft(
        self,
        block: torch.nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim, num_experts = self._validate_block(
            block, hidden_states
        )
        flat = hidden_states.reshape(-1, hidden_dim)
        dtype = hidden_states.dtype
        device = hidden_states.device
        # Gate over ALL tokens at once (state-free) -- 1 call instead of one-per-token.
        fresh_logits_all, fresh_weights_all, fresh_selected_all = _full_gate_topk(
            block, flat, self.top_k, dtype
        )
        router_logits = fresh_logits_all.to(dtype)

        if layer_index not in self.restricted_layers:
            # Unrestricted layer: every token is vanilla routing (no pool, no cost, no state).
            # The whole layer collapses to a single batched expert dispatch.
            final = _execute_sparse_experts(
                block, flat, fresh_selected_all, fresh_weights_all, num_experts, dtype
            )
            self.router_logits_by_layer[layer_index] = router_logits.detach()
            return final.reshape(batch_size, sequence_length, hidden_dim), router_logits

        final = torch.zeros_like(flat)
        # Hoist the attention-mask device->host sync out of the per-token/per-row loop.
        active_flags = self.attention_mask.to(torch.bool).tolist()

        for token_index in range(sequence_length):
            rows = [b * sequence_length + token_index for b in range(batch_size)]
            vanilla_rows: list[int] = []
            keep_rows: list[int] = []
            keep_pools: list[torch.Tensor] = []
            keep_local: list[int] = []
            fresh_rows: list[int] = []
            fresh_pools: list[torch.Tensor] = []
            fresh_masks: list[torch.Tensor] = []
            fresh_local: list[int] = []
            staged: dict[int, dict[str, Any]] = {}

            # --- gather phase: stateful mu/beta reads (per row, order-independent slots) ---
            for local in range(batch_size):
                row = rows[local]
                batch_index = local
                if not active_flags[batch_index][token_index]:
                    vanilla_rows.append(row)
                    continue

                self._append_router_kl_term(
                    fresh_logits_all[row],
                    batch_index=batch_index,
                    layer_index=layer_index,
                    token_index=token_index,
                )
                st_mask, new_idx, mu_scores = self._select_new_pool(
                    batch_index, layer_index, flat[row], fresh_logits_all[row], num_experts
                )
                # Coverage-BCE: supervise mu to include the realized top-k demand in its top-C pool
                # -- a strong, direct NLL-aligned signal (the straight-through NLL path is weak
                # because selected experts sit in the pool interior). The target is the current
                # token's demand ('current') or the running option-union ('union_dwell').
                cov_demand, cov_union = self._coverage_demand(
                    batch_index, layer_index, fresh_selected_all[row], num_experts, device
                )
                self._mu_coverage_terms.append(
                    _st_missing_coverage(
                        mu_scores, cov_demand,
                        top_k=self.capacity, temperature=self.mask_temperature,
                    )
                )
                new_pool = tuple(sorted(int(e) for e in new_idx.tolist()))
                new_bool = self._pool_bool(new_pool, num_experts, device)
                resident = self.caches[batch_index][layer_index]
                resident_mask = torch.zeros(num_experts, device=device)
                for expert in resident:
                    resident_mask[expert] = 1.0
                old_pool = self.options[batch_index][layer_index]
                fresh_rows.append(row)
                fresh_pools.append(new_bool)
                fresh_masks.append(st_mask)
                fresh_local.append(local)
                info: dict[str, Any] = {
                    "new_pool": new_pool,
                    "st_mask": st_mask,
                    "resident_mask": resident_mask,
                    "fresh_logits": fresh_logits_all[row],
                    "old_pool": old_pool,
                    "cov_union": cov_union,
                    "cur_demand": fresh_selected_all[row],
                }
                if old_pool is None:
                    info["kind"] = "bootstrap"
                else:
                    info["kind"] = "normal"
                    info["beta"] = self._termination_probability(
                        batch_index, layer_index, flat[row], block
                    ).to(device=device)
                    keep_rows.append(row)
                    keep_pools.append(self._pool_bool(old_pool, num_experts, device))
                    keep_local.append(local)
                staged[local] = info

            # --- batched expert dispatch (keep + fresh + vanilla) ---
            fresh_out: dict[int, torch.Tensor] = {}
            if fresh_rows:
                outs = self._bop_route_batched(
                    block, flat, fresh_rows, fresh_pools, fresh_logits_all,
                    num_experts, dtype, st_masks=fresh_masks,
                )
                for i, local in enumerate(fresh_local):
                    fresh_out[local] = outs[i]
            keep_out: dict[int, torch.Tensor] = {}
            if keep_rows:
                outs = self._bop_route_batched(
                    block, flat, keep_rows, keep_pools, fresh_logits_all,
                    num_experts, dtype,
                )
                for i, local in enumerate(keep_local):
                    keep_out[local] = outs[i]
            if vanilla_rows:
                vidx = torch.tensor(vanilla_rows, dtype=torch.long, device=device)
                vout = _execute_sparse_experts(
                    block, flat.index_select(0, vidx),
                    fresh_selected_all.index_select(0, vidx),
                    fresh_weights_all.index_select(0, vidx),
                    num_experts, dtype,
                )
                for i, row in enumerate(vanilla_rows):
                    final[row] = vout[i]

            # --- update phase: blend + bookkeeping (per row, order-independent slots) ---
            for local, info in staged.items():
                row = rows[local]
                batch_index = local
                if info["kind"] == "bootstrap":  # adopt R_new, forced termination
                    final[row] = fresh_out[local]
                    self._append_bop_costs(
                        fresh_logits_all.new_tensor(1.0), info["st_mask"], info["resident_mask"]
                    )
                    self._record_boundary(
                        batch_index, layer_index, info["new_pool"],
                        torch.ones(len(info["new_pool"])), info["fresh_logits"].detach(),
                        count_controller=True,
                    )
                    self._update_demand(batch_index, layer_index, info["fresh_logits"])
                    self._apply_coverage_state(
                        batch_index, layer_index, info, num_experts, device, reset=True
                    )
                    continue
                beta = info["beta"]
                beta_d = beta.to(dtype)
                final[row] = (1.0 - beta_d) * keep_out[local] + beta_d * fresh_out[local]
                self._append_bop_costs(beta, info["st_mask"], info["resident_mask"])
                fired = float(beta.detach().cpu()) > self.state_update_threshold
                if fired:
                    self._record_boundary(
                        batch_index, layer_index, info["new_pool"],
                        torch.ones(len(info["new_pool"])), info["fresh_logits"].detach(),
                        count_controller=True,
                    )
                else:
                    self._record_event(batch_index, layer_index, info["old_pool"])
                self._update_demand(batch_index, layer_index, info["fresh_logits"])
                # A fired boundary starts a new option (reset the union to this token's demand);
                # a continuation extends the current option's demand union.
                self._apply_coverage_state(
                    batch_index, layer_index, info, num_experts, device, reset=fired
                )

        self.router_logits_by_layer[layer_index] = router_logits.detach()
        return final.reshape(batch_size, sequence_length, hidden_dim), router_logits

    def _forward_soft_reference(
        self,
        block: torch.nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-reorg per-token/per-row soft forward, kept only as the equivalence-test oracle.

        ``_forward_soft`` is the optimized (batched-dispatch, gate-precomputed) path; this method is
        the row-at-a-time original. ``tests/test_bestpool_forward_equivalence.py`` asserts the two
        agree within tolerance on outputs, soft-loss terms, and controller bookkeeping. Remove once
        the on-cluster scale-up run has confirmed the reorg.
        """
        batch_size, sequence_length, hidden_dim, num_experts = self._validate_block(
            block, hidden_states
        )
        flat = hidden_states.reshape(-1, hidden_dim)
        total_tokens = flat.shape[0]
        final = torch.zeros_like(flat)
        router_logits = torch.empty(
            (total_tokens, num_experts), dtype=hidden_states.dtype, device=hidden_states.device
        )
        restricted = layer_index in self.restricted_layers

        for token_index in range(sequence_length):
            rows = [b * sequence_length + token_index for b in range(batch_size)]
            idx = torch.tensor(rows, dtype=torch.long, device=flat.device)
            fresh_logits, fresh_weights, fresh_selected = _full_gate_topk(
                block, flat.index_select(0, idx), self.top_k, hidden_states.dtype
            )
            for local, batch_index in enumerate(range(batch_size)):
                row = rows[local]
                router_logits[row] = fresh_logits[local].to(router_logits.dtype)
                active = bool(self.attention_mask[batch_index, token_index].item())
                if not restricted or not active:
                    # vanilla routing (unrestricted layer, or padded position): no pool, no cost.
                    out = _execute_sparse_experts(
                        block,
                        flat[row].unsqueeze(0),
                        fresh_selected[local].unsqueeze(0),
                        fresh_weights[local].unsqueeze(0),
                        num_experts,
                        hidden_states.dtype,
                    )
                    final[row] = out[0]
                    continue

                self._append_router_kl_term(
                    fresh_logits[local],
                    batch_index=batch_index,
                    layer_index=layer_index,
                    token_index=token_index,
                )
                st_mask, new_idx, mu_scores = self._select_new_pool(
                    batch_index, layer_index, flat[row], fresh_logits[local], num_experts
                )
                self._mu_coverage_terms.append(
                    _st_missing_coverage(
                        mu_scores, fresh_selected[local],
                        top_k=self.capacity, temperature=self.mask_temperature,
                    )
                )
                new_pool = tuple(sorted(int(e) for e in new_idx.tolist()))
                new_bool = self._pool_bool(new_pool, num_experts, flat.device)
                resident = self.caches[batch_index][layer_index]
                resident_mask = torch.zeros(num_experts, device=flat.device)
                for expert in resident:
                    resident_mask[expert] = 1.0
                old_pool = self.options[batch_index][layer_index]

                if old_pool is None:  # bootstrap: adopt R_new, forced termination
                    fresh_out, _sel = self._bop_route(
                        block, flat[row], fresh_logits[local], new_bool, num_experts,
                        hidden_states.dtype, st_mask=st_mask,
                    )
                    final[row] = fresh_out
                    self._append_bop_costs(
                        fresh_logits.new_tensor(1.0), st_mask, resident_mask
                    )
                    self._record_boundary(
                        batch_index, layer_index, new_pool,
                        torch.ones(len(new_pool)), fresh_logits[local].detach(),
                        count_controller=True,
                    )
                    self._update_demand(batch_index, layer_index, fresh_logits[local])
                    continue

                beta = self._termination_probability(
                    batch_index, layer_index, flat[row], block
                ).to(device=hidden_states.device)
                old_bool = self._pool_bool(old_pool, num_experts, flat.device)
                keep_out, _ = self._bop_route(
                    block, flat[row], fresh_logits[local], old_bool, num_experts,
                    hidden_states.dtype,
                )
                fresh_out, _ = self._bop_route(
                    block, flat[row], fresh_logits[local], new_bool, num_experts,
                    hidden_states.dtype, st_mask=st_mask,
                )
                beta_d = beta.to(hidden_states.dtype)
                final[row] = (1.0 - beta_d) * keep_out + beta_d * fresh_out
                self._append_bop_costs(beta, st_mask, resident_mask)
                if float(beta.detach().cpu()) > self.state_update_threshold:
                    self._record_boundary(
                        batch_index, layer_index, new_pool,
                        torch.ones(len(new_pool)), fresh_logits[local].detach(),
                        count_controller=True,
                    )
                else:
                    self._record_event(batch_index, layer_index, old_pool)
                self._update_demand(batch_index, layer_index, fresh_logits[local])

        self.router_logits_by_layer[layer_index] = router_logits.detach()
        return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


class LearnedBetaSparseMoeWrapper(torch.nn.Module):
    def __init__(
        self,
        block: torch.nn.Module,
        layer_index: int,
        controller: LearnedBetaSparseMoeController,
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


def _infer_num_experts(block: torch.nn.Module) -> int:
    num_experts = getattr(block, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(getattr(block, "gate", None), "num_experts", None)
    if num_experts is None and isinstance(getattr(block, "experts", None), torch.nn.ModuleList):
        num_experts = len(block.experts)
    if num_experts is None:
        raise RuntimeError("could not infer number of experts from sparse block")
    return int(num_experts)


def _full_gate_topk(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_output = block.gate(hidden_states)
    if isinstance(gate_output, tuple):
        logits = gate_output[0]
        weights = gate_output[1].to(output_dtype)
        selected = gate_output[2]
    else:
        logits = gate_output
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float)
        weights, selected = torch.topk(probabilities, top_k, dim=-1)
        if bool(getattr(block, "norm_topk_prob", False)):
            weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.to(output_dtype)
    return logits, weights, selected


def _routing_weights_for_selected(
    block: torch.nn.Module,
    logits: torch.Tensor,
    selected: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    weights = probabilities.index_select(0, selected.to(device=logits.device, dtype=torch.long))
    if bool(getattr(block, "norm_topk_prob", False)):
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return weights.to(output_dtype)


def _execute_sparse_experts(
    block: torch.nn.Module,
    flat_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    hidden_dim = flat_states.shape[-1]
    if selected_experts.numel() == 0:
        return torch.zeros(
            (flat_states.shape[0], hidden_dim),
            dtype=output_dtype,
            device=flat_states.device,
        )
    if isinstance(block.experts, torch.nn.ModuleList):
        final_hidden_states = torch.zeros(
            (flat_states.shape[0], hidden_dim),
            dtype=output_dtype,
            device=flat_states.device,
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
                current_hidden_states.to(output_dtype),
            )
        return final_hidden_states
    return block.experts(flat_states, selected_experts, routing_weights).to(output_dtype)


def _restricted_gate_scores(
    block: torch.nn.Module,
    hidden_state: torch.Tensor,
    option: tuple[int, ...],
) -> torch.Tensor:
    gate = block.gate
    option_tensor = torch.tensor(option, dtype=torch.long, device=hidden_state.device)
    gate_weight = getattr(gate, "weight", None)
    if isinstance(gate_weight, torch.Tensor) and gate_weight.ndim == 2:
        weight = gate_weight.index_select(0, option_tensor).to(hidden_state.dtype)
        bias = None
        gate_bias = getattr(gate, "bias", None)
        if isinstance(gate_bias, torch.Tensor):
            bias = gate_bias.index_select(0, option_tensor).to(hidden_state.dtype)
        return F.linear(hidden_state.unsqueeze(0), weight, bias).squeeze(0)
    if hidden_state.shape[-1] >= _infer_num_experts(block):
        return hidden_state.index_select(0, option_tensor)
    raise RuntimeError(
        "restricted_softmax continuation requires a linear gate or hidden states "
        "whose last dimension indexes experts"
    )


def _tensor_to_option(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(expert) for expert in values.detach().cpu().tolist())


def _hard_missing_coverage(
    option: tuple[int, ...],
    demand: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    option_set = set(option)
    covered = sum(1 for expert in demand.detach().cpu().tolist() if int(expert) in option_set)
    return torch.tensor(1.0 - covered / max(1, int(demand.numel())), device=device)


def _st_missing_coverage(
    logits: torch.Tensor,
    demand: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    hard = torch.zeros_like(logits, dtype=torch.float32)
    selected = torch.topk(logits.detach(), k=top_k, dim=-1).indices
    hard.scatter_(0, selected, 1.0)
    soft = float(top_k) * torch.softmax(logits.float() / max(1.0e-6, temperature), dim=-1)
    st_mask = hard + soft - soft.detach()
    demand_values = st_mask.index_select(0, demand.to(device=logits.device))
    return (1.0 - torch.clamp(demand_values, 0.0, 1.0)).mean()


def st_topc_pool_mask(
    scores: torch.Tensor,
    capacity: int,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight-through top-C pool membership for the policy over options (mu).

    Forward: a hard one-hot mask over the top-``capacity`` experts by ``scores`` (the discrete
    pool). Backward: gradient flows through a soft membership ``sigmoid((scores - tau_C) / T)``,
    where ``tau_C`` is the (detached) capacity-th largest score, so experts near the selection
    boundary receive the strongest gradient. Returns ``(st_mask [E], hard_indices [C])``.
    """
    num_experts = int(scores.shape[-1])
    capacity = min(int(capacity), num_experts)
    top = torch.topk(scores.detach(), k=capacity, dim=-1)
    hard = torch.zeros_like(scores, dtype=torch.float32)
    hard.scatter_(0, top.indices, 1.0)
    tau_c = top.values[..., -1]
    soft = torch.sigmoid((scores.float() - tau_c) / max(1.0e-6, temperature))
    st_mask = hard + soft - soft.detach()
    return st_mask, top.indices


def soft_set_difference(st_mask: torch.Tensor, resident_mask: torch.Tensor) -> torch.Tensor:
    """Expected ``|R_new \\ R_old|`` = ``sum_e st_mask_e * (1 - resident_e)``.

    The forward value equals the exact hard set-difference (the real transfer count); gradient
    flows to ``mu`` through the soft part of ``st_mask``.
    """
    resident = resident_mask.to(dtype=st_mask.dtype, device=st_mask.device)
    return (st_mask * (1.0 - resident)).sum()


def assemble_policy_features(
    hidden_state: torch.Tensor,
    full_logits: torch.Tensor,
    pop_ema: torch.Tensor,
    last_seen: torch.Tensor,
    resident: list[int],
    num_experts: int,
    seq_len: int,
) -> torch.Tensor:
    """Per-expert demand-anticipation features ``[num_experts, policy_feature_dim()]`` for mu.

    Shared by training (``BestOfPoolBetaController``) and hard eval so the two never diverge.
    """
    device = full_logits.device
    gate_logit = full_logits.detach().float()
    router_prob = torch.softmax(gate_logit, dim=-1)
    pop = pop_ema.to(device)
    seen = last_seen.to(device)
    recency = torch.log1p(seen) / torch.log1p(seen.new_tensor(float(max(1, seq_len))))
    resident_ind = torch.zeros(num_experts, device=device)
    for expert in resident:
        resident_ind[expert] = 1.0
    per_expert = torch.stack((gate_logit, router_prob, pop, recency, resident_ind), dim=-1)
    hidden = hidden_state.detach().float()
    summary = torch.stack(
        (
            hidden.mean(),
            hidden.std(unbiased=False),
            torch.sqrt(torch.mean(hidden * hidden).clamp_min(0.0)),
            hidden.abs().max(),
        )
    ).to(device)
    summary = summary.unsqueeze(0).expand(num_experts, POLICY_HIDDEN_SUMMARY)
    return torch.cat((per_expert, summary), dim=-1)


def update_pool_demand(
    pop_ema: torch.Tensor,
    last_seen: torch.Tensor,
    full_logits: torch.Tensor,
    top_k: int,
    decay: float,
) -> None:
    """Causal per-expert popularity EMA + recency update from a token's realized top-k demand."""
    demand = torch.topk(full_logits.detach().float(), top_k, dim=-1).indices.cpu()
    pop_ema.mul_(decay)
    pop_ema[demand] += 1.0
    last_seen += 1.0
    last_seen[demand] = 0.0


def load_policy_head(path: Path, device: torch.device) -> PolicyOverOptionsHead:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or "policy_head" not in state:
        raise TypeError("policy head checkpoint must be a dict with a 'policy_head' entry")
    metadata = state.get("metadata", {})
    head = PolicyOverOptionsHead(
        num_experts=int(metadata["num_experts"]),
        num_layers=int(metadata["num_layers"]),
        feature_dim=int(metadata["feature_dim"]),
        config=PolicyHeadConfig(**metadata.get("config", {})),
    ).to(device)
    head.load_state_dict(state["policy_head"])
    head.eval()
    return head


def _update_resident(resident: list[int], selected: tuple[int, ...], capacity: int) -> None:
    for expert in selected:
        if expert in resident:
            resident.remove(expert)
        resident.append(expert)
        while len(resident) > capacity:
            resident.pop(0)
