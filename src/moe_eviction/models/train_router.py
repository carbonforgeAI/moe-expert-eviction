from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_eviction.cache import (
    melinoe_soft_cache_states,
    topk_request_vectors,
)

from .collect_trace import _dtype, _load_prompts, _resolved_model_revision, _router_layers


def _is_router_parameter(name: str, parameter: torch.nn.Parameter, num_experts: int) -> bool:
    lowered = name.lower()
    if ".experts." in lowered or lowered.endswith(".experts.gate_up_proj"):
        return False
    if "router" in lowered:
        return True
    if not lowered.endswith(".mlp.gate.weight"):
        return False
    if parameter.ndim < 1:
        return False
    return min(parameter.shape) <= num_experts


def _freeze_non_router_parameters(
    model: torch.nn.Module,
    num_experts: int,
) -> tuple[int, tuple[str, ...]]:
    trainable = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = _is_router_parameter(name, parameter, num_experts)
        if parameter.requires_grad:
            trainable += parameter.numel()
            trainable_names.append(name)
    if trainable == 0:
        raise RuntimeError("no router/gate parameters matched; inspect model parameter names")
    if trainable > 20_000_000:
        joined_names = ", ".join(trainable_names[:20])
        raise RuntimeError(
            f"matched {trainable} trainable parameters, which is too large for router-only "
            f"smoke training; first matches: {joined_names}"
        )
    return trainable, tuple(trainable_names)


def _router_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _router_gate_module_name(parameter_name: str) -> str:
    if not parameter_name.endswith(".weight"):
        raise ValueError(f"expected router gate weight parameter, got {parameter_name}")
    return parameter_name.removesuffix(".weight")


def _register_reference_router_hooks(
    model: torch.nn.Module,
    trainable_parameter_names: tuple[str, ...],
) -> tuple[list[torch.utils.hooks.RemovableHandle], list[torch.Tensor]]:
    """Record frozen-gate logits on the current hidden states.

    ReMoE's Trust-KL anchors the trainable router to a frozen snapshot gate
    evaluated on the current hidden state, not to a separate frozen model's
    stale hidden states.
    """

    model_parameters = dict(model.named_parameters())
    reference_logits: list[torch.Tensor] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for parameter_name in trainable_parameter_names:
        module_name = _router_gate_module_name(parameter_name)
        if parameter_name not in model_parameters:
            raise KeyError(f"missing trainable router parameter: {parameter_name}")
        module = model.get_submodule(module_name)
        frozen_weight = model_parameters[parameter_name].detach().clone().float()

        def hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            weight: torch.Tensor = frozen_weight,
            gate_name: str = module_name,
        ) -> None:
            if not inputs:
                raise RuntimeError(f"router gate {gate_name} received no positional input")
            hidden_states = inputs[0]
            with torch.no_grad():
                reference_logits.append(
                    F.linear(
                        hidden_states.detach().float(),
                        weight.to(device=hidden_states.device),
                    ).detach()
                )

        handles.append(module.register_forward_pre_hook(hook))

    return handles, reference_logits


def _router_kl(
    current: Any,
    reference: Any,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    batch_size = attention_mask.shape[0] if attention_mask is not None else 0
    seq_len = attention_mask.shape[1] if attention_mask is not None else 0
    mask = attention_mask.bool() if attention_mask is not None else None

    for current_layer_raw, reference_layer_raw in zip(
        _router_layers(current),
        _router_layers(reference),
        strict=True,
    ):
        if mask is None:
            current_layer = current_layer_raw.float()
            reference_layer = reference_layer_raw.float()
        else:
            current_layer = _reshape_router_logits(current_layer_raw, batch_size, seq_len)
            reference_layer = _reshape_router_logits(reference_layer_raw, batch_size, seq_len)
        current_log_probs = F.log_softmax(current_layer, dim=-1)
        reference_probs = F.softmax(reference_layer, dim=-1)
        per_position = F.kl_div(
            current_log_probs,
            reference_probs,
            reduction="none",
        ).sum(dim=-1)
        if mask is not None:
            selected = per_position.masked_select(mask)
            if selected.numel() == 0:
                continue
            losses.append(selected.mean())
        else:
            losses.append(per_position.mean())
    if not losses:
        if attention_mask is not None:
            return _zero_from_mask(attention_mask)
        first_layer = next(iter(_router_layers(current)))
        return first_layer.float().sum() * 0.0
    return torch.stack(losses).mean()


def _reshape_router_logits(logits: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    if logits.ndim == 3:
        return logits.float()
    if logits.ndim == 2 and logits.shape[0] == batch_size * seq_len:
        return logits.float().reshape(batch_size, seq_len, logits.shape[-1])
    raise RuntimeError(f"unsupported router logit shape: {tuple(logits.shape)}")


def _cache_cost_loss(
    router_logits: Any,
    attention_mask: torch.Tensor,
    top_k: int,
    capacity: int,
    default_expert_cost: float,
) -> torch.Tensor:
    """Expected cold-expert mass under hard pre-step LRU residency."""

    batch_size, seq_len = attention_mask.shape
    losses: list[torch.Tensor] = []
    expert_costs: torch.Tensor | None = None

    for layer_logits in _router_layers(router_logits):
        logits = _reshape_router_logits(layer_logits, batch_size, seq_len)
        if expert_costs is None:
            expert_costs = torch.full(
                (logits.shape[-1],),
                float(default_expert_cost),
                device=logits.device,
                dtype=torch.float32,
            )
        probabilities = torch.softmax(logits, dim=-1)
        selected = torch.topk(logits.detach(), k=top_k, dim=-1).indices
        caches: list[list[int]] = [[] for _ in range(batch_size)]

        for token_index in range(seq_len):
            for batch_index in range(batch_size):
                if attention_mask[batch_index, token_index].item() == 0:
                    continue
                resident = caches[batch_index]
                cold = torch.ones_like(expert_costs)
                if resident:
                    cold[torch.tensor(resident, device=logits.device)] = 0.0
                losses.append((probabilities[batch_index, token_index] * cold * expert_costs).sum())

                for expert in selected[batch_index, token_index].tolist():
                    if expert in resident:
                        resident.remove(expert)
                    resident.append(expert)
                    while len(resident) > capacity:
                        resident.pop(0)

    if not losses:
        return attention_mask.float().sum() * 0.0
    return torch.stack(losses).mean()


def _melinoe_cache_simulation_loss(
    router_logits: Any,
    attention_mask: torch.Tensor,
    top_k: int,
    capacity: float,
    gamma: float,
    initialization: str,
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    mask = attention_mask.bool()
    losses: list[torch.Tensor] = []

    for layer_logits in _router_layers(router_logits):
        logits = _reshape_router_logits(layer_logits, batch_size, seq_len)
        probabilities = torch.softmax(logits.float(), dim=-1)
        requests = topk_request_vectors(probabilities, top_k=top_k, straight_through=True)
        requests = requests * mask[:, :, None].to(dtype=requests.dtype)
        pre_states, _post_states = melinoe_soft_cache_states(
            requests,
            capacity=capacity,
            gamma=gamma,
            initialization=initialization,
        )
        miss_proxy = (requests * (1.0 - pre_states)).sum(dim=-1)
        if mask.any():
            losses.append(miss_proxy.masked_select(mask).mean())

    if not losses:
        return _zero_from_mask(attention_mask)
    return torch.stack(losses).mean()


def _melinoe_rank_matching_loss(
    current: Any,
    reference: Any,
    attention_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    mask = attention_mask.bool()
    losses: list[torch.Tensor] = []

    for current_layer_raw, reference_layer_raw in zip(
        _router_layers(current),
        _router_layers(reference),
        strict=True,
    ):
        current_layer = _reshape_router_logits(current_layer_raw, batch_size, seq_len)
        reference_layer = _reshape_router_logits(reference_layer_raw, batch_size, seq_len)
        current_probs = torch.softmax(current_layer.float(), dim=-1)
        reference_probs = torch.softmax(reference_layer.float(), dim=-1)
        base_preference = reference_probs.unsqueeze(-1) > reference_probs.unsqueeze(-2)
        finetuned_margin = margin - (
            current_probs.unsqueeze(-1) - current_probs.unsqueeze(-2)
        )
        mistakes = torch.where(
            base_preference,
            F.relu(finetuned_margin),
            torch.zeros_like(finetuned_margin),
        )
        per_position = mistakes.sum(dim=(-1, -2))
        if mask.any():
            losses.append(per_position.masked_select(mask).mean())

    if not losses:
        return _zero_from_mask(attention_mask)
    return torch.stack(losses).mean()


def _active_pair_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask[:, 1:] != 0) & (attention_mask[:, :-1] != 0)


def _zero_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.float().sum() * 0.0


def _temporal_reuse_loss(
    router_logits: Any,
    attention_mask: torch.Tensor,
    top_k: int,
    lag: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    losses: list[torch.Tensor] = []
    if seq_len <= lag:
        return _zero_from_mask(attention_mask)

    active = (attention_mask[:, lag:] != 0) & (attention_mask[:, :-lag] != 0)
    for layer_logits in _router_layers(router_logits):
        logits = _reshape_router_logits(layer_logits, batch_size, seq_len)
        probabilities = torch.softmax(logits[:, lag:], dim=-1)
        previous_topk = torch.topk(logits[:, :-lag].detach(), k=top_k, dim=-1).indices
        reuse_mass = probabilities.gather(dim=-1, index=previous_topk).sum(dim=-1)
        if active.any():
            rho = reuse_mass.masked_select(active).mean()
            losses.append(-torch.log(rho.clamp_min(float(epsilon))))

    if not losses:
        return _zero_from_mask(attention_mask)
    return torch.stack(losses).mean()


def _symmetric_kl_from_logits(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_probs = torch.softmax(first.float(), dim=-1)
    second_probs = torch.softmax(second.float(), dim=-1)
    first_log_probs = torch.log_softmax(first.float(), dim=-1)
    second_log_probs = torch.log_softmax(second.float(), dim=-1)
    forward = F.kl_div(second_log_probs, first_probs, reduction="none").sum(dim=-1)
    backward = F.kl_div(first_log_probs, second_probs, reduction="none").sum(dim=-1)
    return 0.5 * (forward + backward)


def _temporal_symmetric_kl_loss(router_logits: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    if seq_len <= 1:
        return _zero_from_mask(attention_mask)
    active = _active_pair_mask(attention_mask)
    losses: list[torch.Tensor] = []

    for layer_logits in _router_layers(router_logits):
        logits = _reshape_router_logits(layer_logits, batch_size, seq_len)
        previous = logits[:, :-1].float()
        current = logits[:, 1:].float()
        symmetric = _symmetric_kl_from_logits(previous, current)
        if active.any():
            losses.append(symmetric.masked_select(active).mean())

    if not losses:
        return _zero_from_mask(attention_mask)
    return torch.stack(losses).mean()


def _lagged_symmetric_kl_loss(
    router_logits: Any,
    attention_mask: torch.Tensor,
    lag_steps: tuple[int, ...],
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    if seq_len <= 1 or not lag_steps:
        return _zero_from_mask(attention_mask)
    if any(lag <= 0 for lag in lag_steps):
        raise ValueError(f"lag steps must be positive, got {lag_steps}")

    active_adjacent = _active_pair_mask(attention_mask)
    normalizer = int(active_adjacent.sum().item()) * len(lag_steps)
    if normalizer == 0:
        return _zero_from_mask(attention_mask)

    losses: list[torch.Tensor] = []
    for layer_logits in _router_layers(router_logits):
        logits = _reshape_router_logits(layer_logits, batch_size, seq_len)
        layer_sum = logits.float().sum() * 0.0
        for lag in lag_steps:
            if seq_len <= lag:
                continue
            active = (attention_mask[:, lag:] != 0) & (attention_mask[:, :-lag] != 0)
            if not active.any():
                continue
            symmetric = _symmetric_kl_from_logits(logits[:, :-lag], logits[:, lag:])
            layer_sum = layer_sum + symmetric.masked_select(active).sum()
        losses.append(layer_sum / normalizer)

    if not losses:
        return _zero_from_mask(attention_mask)
    return torch.stack(losses).mean()


def _working_set_entropy_loss(
    router_logits: Any,
    attention_mask: torch.Tensor,
    window_size: int = 16,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    batch_size, seq_len = attention_mask.shape
    mask = attention_mask.bool()
    losses: list[torch.Tensor] = []

    for layer_logits in _router_layers(router_logits):
        logits = _reshape_router_logits(layer_logits, batch_size, seq_len)
        probabilities = torch.softmax(logits.float(), dim=-1)
        layer_losses: list[torch.Tensor] = []
        full_windows = seq_len // window_size
        for batch_index in range(batch_size):
            for window_index in range(full_windows):
                start = window_index * window_size
                end = start + window_size
                window_mask = mask[batch_index, start:end]
                if not window_mask.any():
                    continue
                window_probs = probabilities[batch_index, start:end]
                masked_probs = window_probs * window_mask[:, None]
                mean_probs = masked_probs.sum(dim=0) / window_mask.float().sum()
                entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-8))).sum()
                layer_losses.append(entropy)
        if layer_losses:
            losses.append(torch.stack(layer_losses).mean())

    if not losses:
        return _zero_from_mask(attention_mask)
    return torch.stack(losses).mean()


def _linear_warmup_multiplier(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def _parse_lag_steps(train_config: dict[str, Any]) -> tuple[int, ...]:
    raw = train_config.get("lag_steps")
    if raw is None:
        if "inertia_lag" in train_config:
            return (int(train_config["inertia_lag"]),)
        return (1, 2, 4, 8, 16)
    if isinstance(raw, int):
        return (raw,)
    return tuple(int(value) for value in raw)


def _effective_batch_size(train_config: dict[str, Any], num_prompts: int) -> int:
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    raw_batch_size = train_config.get("batch_size")
    if raw_batch_size is None:
        return num_prompts
    batch_size = int(raw_batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return min(batch_size, num_prompts)


def _select_training_batch(
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    total_rows = labels.shape[0]
    if batch_size >= total_rows:
        return encoded, labels
    indices = torch.tensor(
        random.sample(range(total_rows), k=batch_size),
        dtype=torch.long,
        device=labels.device,
    )
    batch = {
        name: tensor.index_select(0, indices)
        for name, tensor in encoded.items()
        if isinstance(tensor, torch.Tensor)
    }
    return batch, labels.index_select(0, indices)


def train_router(config: dict[str, Any], output_dir: Path) -> None:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_config = config["model"]
    train_config = config.get("train", {})
    model_id = model_config["id"]
    revision = model_config.get("revision")
    device = torch.device(model_config.get("device", "cuda"))
    dtype = _dtype(model_config.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.train()
    num_experts = int(
        getattr(model.config, "num_experts", 0)
        or getattr(model.config, "num_local_experts", 0)
    )
    if num_experts <= 0:
        raise RuntimeError("model config does not expose num_experts/num_local_experts")
    trainable_params, trainable_names = _freeze_non_router_parameters(model, num_experts)

    kl_weight = float(train_config.get("router_kl_weight", 0.01))
    cache_cost_weight = float(train_config.get("cache_cost_weight", 0.001))
    reuse_weight = float(train_config.get("reuse_weight", 0.0))
    smoothness_weight = float(train_config.get("smoothness_weight", 0.0))
    lag_weight = float(train_config.get("lag_weight", train_config.get("inertia_weight", 0.0)))
    working_set_weight = float(train_config.get("working_set_weight", 0.0))
    melinoe_cache_weight = float(train_config.get("melinoe_cache_weight", 0.0))
    melinoe_rank_weight = float(train_config.get("melinoe_rank_weight", 0.0))
    melinoe_gamma = float(train_config.get("melinoe_gamma", 0.9))
    melinoe_cache_initialization = str(
        train_config.get("melinoe_cache_initialization", "zero_fill")
    )
    melinoe_rank_margin = float(train_config.get("melinoe_rank_margin", 0.1))
    reuse_warmup_steps = int(train_config.get("reuse_warmup_steps", 0))
    loc_warmup_steps = int(train_config.get("loc_warmup_steps", 0))
    lr_warmup_steps = int(train_config.get("learning_rate_warmup_steps", 0))
    gradient_accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    max_grad_norm = float(train_config.get("max_grad_norm", 0.0))
    subtract_router_aux_loss = bool(train_config.get("subtract_router_aux_loss", True))
    lag_steps = _parse_lag_steps(train_config)
    working_set_window = int(train_config.get("working_set_window", 16))
    cache_capacity = int(train_config.get("cache_capacity", 8))
    default_expert_cost = float(train_config.get("default_expert_cost", 1.0))
    steps = int(train_config.get("steps", 1))
    top_k = int(getattr(model.config, "num_experts_per_tok", train_config.get("top_k", 2)))
    base_learning_rate = float(train_config.get("learning_rate", 1e-5))
    needs_router_logits = any(
        weight != 0.0
        for weight in (
            kl_weight,
            cache_cost_weight,
            reuse_weight,
            smoothness_weight,
            lag_weight,
            working_set_weight,
            melinoe_cache_weight,
            melinoe_rank_weight,
        )
    )

    prompts = _load_prompts({"trace": train_config})[: int(train_config.get("max_sequences", 4))]
    batch_size = _effective_batch_size(train_config, len(prompts))
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(train_config.get("max_length", 128)),
    ).to(device)
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    reference_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    reference_router_logits: list[torch.Tensor] = []
    if kl_weight != 0.0 or melinoe_rank_weight != 0.0:
        reference_hook_handles, reference_router_logits = _register_reference_router_hooks(
            model,
            trainable_names,
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=base_learning_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            learning_rate_multiplier = _linear_warmup_multiplier(step, lr_warmup_steps)
            current_learning_rate = base_learning_rate * learning_rate_multiplier
            for group in optimizer.param_groups:
                group["lr"] = current_learning_rate

            step_sums: dict[str, float] = {}
            for _micro_step in range(gradient_accumulation_steps):
                batch_encoded, batch_labels = _select_training_batch(
                    encoded,
                    labels,
                    batch_size=batch_size,
                )
                reference_router_logits.clear()
                outputs = model(
                    **batch_encoded,
                    labels=batch_labels,
                    output_router_logits=needs_router_logits,
                    use_cache=False,
                )

                lm_loss = outputs.loss
                aux_loss = getattr(outputs, "aux_loss", None)
                if subtract_router_aux_loss and aux_loss is not None:
                    router_aux_loss_coef = float(
                        getattr(
                            model,
                            "router_aux_loss_coef",
                            getattr(model.config, "router_aux_loss_coef", 0.0),
                        )
                    )
                    lm_loss = lm_loss - router_aux_loss_coef * aux_loss.to(lm_loss.device)

                zero_loss = lm_loss * 0.0
                kl = (
                    _router_kl(
                        outputs.router_logits,
                        tuple(reference_router_logits),
                        batch_encoded["attention_mask"],
                    )
                    if kl_weight != 0.0
                    else zero_loss
                )
                cache_cost = (
                    _cache_cost_loss(
                        outputs.router_logits,
                        batch_encoded["attention_mask"],
                        top_k=top_k,
                        capacity=cache_capacity,
                        default_expert_cost=default_expert_cost,
                    )
                    if cache_cost_weight != 0.0
                    else zero_loss
                )
                reuse_loss = (
                    _temporal_reuse_loss(
                        outputs.router_logits,
                        batch_encoded["attention_mask"],
                        top_k=top_k,
                        lag=1,
                    )
                    if reuse_weight != 0.0
                    else zero_loss
                )
                smoothness_loss = (
                    _temporal_symmetric_kl_loss(
                        outputs.router_logits,
                        batch_encoded["attention_mask"],
                    )
                    if smoothness_weight != 0.0
                    else zero_loss
                )
                lag_loss = (
                    _lagged_symmetric_kl_loss(
                        outputs.router_logits,
                        batch_encoded["attention_mask"],
                        lag_steps=lag_steps,
                    )
                    if lag_weight != 0.0
                    else zero_loss
                )
                working_set_loss = (
                    _working_set_entropy_loss(
                        outputs.router_logits,
                        batch_encoded["attention_mask"],
                        window_size=working_set_window,
                    )
                    if working_set_weight != 0.0
                    else zero_loss
                )
                melinoe_cache_loss = (
                    _melinoe_cache_simulation_loss(
                        outputs.router_logits,
                        batch_encoded["attention_mask"],
                        top_k=top_k,
                        capacity=float(cache_capacity),
                        gamma=melinoe_gamma,
                        initialization=melinoe_cache_initialization,
                    )
                    if melinoe_cache_weight != 0.0
                    else zero_loss
                )
                melinoe_rank_loss = (
                    _melinoe_rank_matching_loss(
                        outputs.router_logits,
                        tuple(reference_router_logits),
                        batch_encoded["attention_mask"],
                        margin=melinoe_rank_margin,
                    )
                    if melinoe_rank_weight != 0.0
                    else zero_loss
                )
                reuse_multiplier = _linear_warmup_multiplier(step, reuse_warmup_steps)
                loc_multiplier = _linear_warmup_multiplier(step, loc_warmup_steps)
                effective_reuse_weight = reuse_weight * reuse_multiplier
                effective_smoothness_weight = smoothness_weight * loc_multiplier
                effective_lag_weight = lag_weight * loc_multiplier
                effective_working_set_weight = working_set_weight * loc_multiplier
                loss = (
                    lm_loss
                    + kl_weight * kl
                    + cache_cost_weight * cache_cost
                    + effective_reuse_weight * reuse_loss
                    + effective_smoothness_weight * smoothness_loss
                    + effective_lag_weight * lag_loss
                    + effective_working_set_weight * working_set_loss
                    + melinoe_cache_weight * melinoe_cache_loss
                    + melinoe_rank_weight * melinoe_rank_loss
                )
                (loss / gradient_accumulation_steps).backward()

                micro_row = {
                    "loss": float(loss.detach().cpu()),
                    "lm_loss": float(lm_loss.detach().cpu()),
                    "router_kl": float(kl.detach().cpu()),
                    "cache_cost": float(cache_cost.detach().cpu()),
                    "reuse_loss": float(reuse_loss.detach().cpu()),
                    "smoothness_loss": float(smoothness_loss.detach().cpu()),
                    "lag_loss": float(lag_loss.detach().cpu()),
                    "working_set_loss": float(working_set_loss.detach().cpu()),
                    "melinoe_cache_loss": float(melinoe_cache_loss.detach().cpu()),
                    "melinoe_rank_loss": float(melinoe_rank_loss.detach().cpu()),
                    "effective_reuse_weight": effective_reuse_weight,
                    "effective_smoothness_weight": effective_smoothness_weight,
                    "effective_lag_weight": effective_lag_weight,
                    "effective_working_set_weight": effective_working_set_weight,
                    "learning_rate": current_learning_rate,
                }
                for name, value in micro_row.items():
                    step_sums[name] = step_sums.get(name, 0.0) + value

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()

            step_row = {
                name: value / float(gradient_accumulation_steps)
                for name, value in step_sums.items()
            }
            row = {
                "step": step,
                **step_row,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()

    for handle in reference_hook_handles:
        handle.remove()

    torch.save(_router_state_dict(model), output_dir / "router_state.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    metadata = {
        "seed": seed,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": _resolved_model_revision(model),
        "dtype": str(dtype).removeprefix("torch."),
        "trainable_router_parameters": trainable_params,
        "trainable_router_parameter_names": trainable_names,
        "steps": steps,
        "router_kl_weight": kl_weight,
        "cache_cost_weight": cache_cost_weight,
        "reuse_weight": reuse_weight,
        "smoothness_weight": smoothness_weight,
        "lag_weight": lag_weight,
        "working_set_weight": working_set_weight,
        "melinoe_cache_weight": melinoe_cache_weight,
        "melinoe_rank_weight": melinoe_rank_weight,
        "melinoe_gamma": melinoe_gamma,
        "melinoe_cache_initialization": melinoe_cache_initialization,
        "melinoe_rank_margin": melinoe_rank_margin,
        "reuse_warmup_steps": reuse_warmup_steps,
        "loc_warmup_steps": loc_warmup_steps,
        "learning_rate_warmup_steps": lr_warmup_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
        "max_grad_norm": max_grad_norm,
        "subtract_router_aux_loss": subtract_router_aux_loss,
        "lag_steps": lag_steps,
        "working_set_window": working_set_window,
        "cache_capacity": cache_capacity,
        "default_expert_cost": default_expert_cost,
        "reference_router_mode": (
            "frozen_gate_on_current_hidden"
            if kl_weight != 0.0 or melinoe_rank_weight != 0.0
            else "none"
        ),
        "num_prompts": len(prompts),
        "sequence_length": int(encoded["input_ids"].shape[1]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "note": (
            "Router-only fine-tune with optional ReMoE-style temporal locality losses "
            "and MELINOE-style cache/rank losses."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small router-only fine-tuning smoke job.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config.get("output_dir", "outputs/router_train_smoke"))
    train_router(config, output_dir)


if __name__ == "__main__":
    main()
