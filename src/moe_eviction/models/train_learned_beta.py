from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_eviction.experiments.heterogeneous_cost_frontier import _cost_profile

from .collect_trace import _dtype, _load_prompts, _resolved_model_revision
from .evaluate_cache_rerank import _sparse_moe_modules
from .evaluate_router_state import (
    _load_router_state,
    _router_stack,
)
from .learned_beta import (
    BestOfPoolBetaController,
    BetaHeadConfig,
    LearnedBetaSparseMoeController,
    LearnedBetaSparseMoeWrapper,
    LearnedTerminationHead,
    PolicyHeadConfig,
    PolicyOverOptionsHead,
    learned_beta_metadata,
    load_learned_beta_checkpoint,
    policy_feature_dim,
    save_learned_beta_checkpoint,
    write_learned_beta_metadata,
)
from .lora_experts import (
    LoRAExperts,
    LoRAExpertsConfig,
    attach_lora_experts,
    lora_state_dict,
)
from .train_cache_rerank_router import _select_training_batch
from .train_router import (
    _freeze_non_router_parameters,
    _router_state_dict,
)


def _controller_config(config: dict[str, Any], train_config: dict[str, Any]) -> dict[str, Any]:
    candidate = train_config.get("controller", config.get("controller", {}))
    if not isinstance(candidate, dict):
        raise TypeError("learned beta controller config must be a mapping")
    return candidate


def _cost_weights(controller_config: dict[str, Any]) -> tuple[float, float, float, float]:
    costs = controller_config.get("costs", {})
    if not isinstance(costs, dict):
        raise TypeError("controller.costs must be a mapping when provided")
    deliberation = float(
        controller_config.get("deliberation_cost", costs.get("deliberation", 1.0))
    )
    load = float(controller_config.get("load_cost_weight", costs.get("load", 1.0)))
    switch = float(controller_config.get("switch_cost", costs.get("switch", 0.1)))
    kl = float(controller_config.get("router_kl_weight", costs.get("kl", 0.01)))
    return deliberation, load, switch, kl


def _counterfactual_weights(controller_config: dict[str, Any]) -> dict[str, float]:
    costs = controller_config.get("costs", {})
    if not isinstance(costs, dict):
        raise TypeError("controller.costs must be a mapping when provided")
    return {
        "alpha_beta": float(controller_config.get("alpha_beta", costs.get("alpha_beta", 1.0))),
        "alpha_mask": float(controller_config.get("alpha_mask", costs.get("alpha_mask", 1.0))),
        "lambda_cov": float(controller_config.get("lambda_cov", costs.get("coverage", 1.0))),
        "lambda_call": float(controller_config.get("lambda_call", costs.get("call", 0.05))),
        "lambda_cache": float(controller_config.get("lambda_cache", costs.get("cache", 0.0))),
        "lambda_edit": float(controller_config.get("lambda_edit", costs.get("edit", 0.0))),
        "lambda_router": float(
            controller_config.get("lambda_router", costs.get("router", costs.get("kl", 0.1)))
        ),
        "eta": float(controller_config.get("counterfactual_eta", costs.get("eta", 0.1))),
        "mask_temperature": float(
            controller_config.get("mask_temperature", costs.get("mask_temperature", 1.0))
        ),
    }


def _policy_gradient_weights(
    train_config: dict[str, Any],
    controller_config: dict[str, Any],
) -> dict[str, float]:
    costs = controller_config.get("costs", {})
    if not isinstance(costs, dict):
        raise TypeError("controller.costs must be a mapping when provided")
    return {
        "lambda_call": float(
            train_config.get(
                "lambda_call",
                controller_config.get("lambda_call", costs.get("call", 0.05)),
            )
        ),
        "lambda_edit": float(
            train_config.get(
                "lambda_edit",
                controller_config.get("lambda_edit", costs.get("edit", 0.0)),
            )
        ),
        "lambda_cache": float(
            train_config.get(
                "lambda_cache",
                controller_config.get("lambda_cache", costs.get("cache", 0.0)),
            )
        ),
        "lambda_beta": float(
            train_config.get(
                "lambda_beta",
                controller_config.get("lambda_beta", costs.get("beta", 0.1)),
            )
        ),
        "lambda_router": float(
            train_config.get(
                "lambda_router",
                controller_config.get("lambda_router", costs.get("router", 0.1)),
            )
        ),
        "entropy_weight": float(
            train_config.get(
                "entropy_weight",
                controller_config.get("entropy_weight", costs.get("entropy", 0.001)),
            )
        ),
    }


def _group_relative_advantages(
    rewards: torch.Tensor,
    baseline: str,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if rewards.ndim != 1:
        raise RuntimeError(f"rewards must be 1D, got {tuple(rewards.shape)}")
    if baseline not in {"grpo_zscore", "rloo_mean"}:
        raise ValueError("baseline must be grpo_zscore or rloo_mean")
    if rewards.numel() == 0:
        raise ValueError("at least one reward is required")
    if rewards.numel() == 1:
        return torch.zeros_like(rewards).detach()

    rewards = rewards.float()
    if baseline == "grpo_zscore":
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        return ((rewards - mean) / (std + eps)).detach()

    total = rewards.sum()
    others = (total - rewards) / float(rewards.numel() - 1)
    return (rewards - others).detach()


def _policy_gradient_loss(
    normalized_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    if normalized_log_probs.shape != advantages.shape:
        raise RuntimeError(
            "normalized_log_probs and advantages must have matching shapes, "
            f"got {tuple(normalized_log_probs.shape)} and {tuple(advantages.shape)}"
        )
    return -(advantages.detach() * normalized_log_probs.float()).mean()


def _hazard_loss(
    expected_hazard: torch.Tensor,
    *,
    lambda_beta: float,
    beta_regularizer: str,
) -> torch.Tensor:
    if beta_regularizer not in {"none", "expected_hazard"}:
        raise ValueError("beta_regularizer must be none or expected_hazard")
    if beta_regularizer == "none":
        return expected_hazard.new_zeros(())
    return expected_hazard.float() * expected_hazard.new_tensor(float(lambda_beta))


def _differentiable_cache_loss(
    lm_loss: torch.Tensor,
    soft_terms: dict[str, torch.Tensor],
    router_kl: torch.Tensor,
    weights: dict[str, float],
) -> torch.Tensor:
    """NLL + differentiable cache pressure (deliberation/switch/load) + KL anchor + commitment.

    Persistence is driven by the differentiable costs, which couple to NLL through the soft
    ``(1-beta)*keep + beta*fresh`` blend, rather than by the coverage-BCE surrogate.
    """
    mu_coverage = soft_terms.get("mu_coverage")
    coverage_term = (
        weights.get("mu_coverage", 0.0) * mu_coverage
        if mu_coverage is not None
        else lm_loss.new_tensor(0.0)
    )
    return (
        lm_loss
        + weights["deliberation"] * soft_terms["deliberation"]
        + weights["switch"] * soft_terms["switch"]
        + weights["load"] * soft_terms["load"]
        + weights["commit"] * soft_terms["commitment"]
        + weights["router"] * router_kl
        + coverage_term
    )


def _reward_decomposition(
    *,
    lm_loss: torch.Tensor,
    call_rate: float,
    edit_rate: float,
    cache_cost_rate: float,
    lambda_call: float,
    lambda_edit: float,
    lambda_cache: float,
) -> dict[str, torch.Tensor]:
    lm_term = -lm_loss.detach().float()
    call_penalty = -lm_loss.new_tensor(float(lambda_call)) * float(call_rate)
    edit_penalty = -lm_loss.new_tensor(float(lambda_edit)) * float(edit_rate)
    cache_penalty = -lm_loss.new_tensor(float(lambda_cache)) * float(cache_cost_rate)
    return {
        "reward_lm_term": lm_term,
        "reward_call_penalty": call_penalty,
        "reward_edit_penalty": edit_penalty,
        "reward_cache_penalty": cache_penalty,
        "reward": lm_term + call_penalty + edit_penalty + cache_penalty,
    }


def _grad_l2_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool = True,
) -> torch.Tensor:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable or not loss.requires_grad:
        return loss.detach().new_zeros(())
    grads = torch.autograd.grad(
        loss,
        trainable,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = loss.detach().new_zeros(())
    for grad in grads:
        if grad is None:
            continue
        total = total + grad.detach().float().pow(2).sum()
    return torch.sqrt(total)


def _hidden_size(model: torch.nn.Module) -> int:
    value = int(
        getattr(model.config, "hidden_size", 0)
        or getattr(model.config, "d_model", 0)
        or getattr(model.config, "n_embd", 0)
    )
    if value <= 0:
        raise RuntimeError("model config does not expose hidden_size/d_model/n_embd")
    return value


def _reference_router_stack(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> torch.Tensor:
    total_rows = int(encoded["input_ids"].shape[0])
    if batch_size <= 0:
        raise ValueError("reference batch_size must be positive")
    stacks: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, total_rows, batch_size):
            stop = min(total_rows, start + batch_size)
            indices = torch.arange(
                start,
                stop,
                dtype=torch.long,
                device=encoded["input_ids"].device,
            )
            batch = {
                name: tensor.index_select(0, indices)
                for name, tensor in encoded.items()
                if isinstance(tensor, torch.Tensor)
            }
            outputs = model(**batch, output_router_logits=True, use_cache=False)
            stacks.append(
                _router_stack(
                    outputs.router_logits,
                    batch["input_ids"].shape[0],
                    batch["input_ids"].shape[1],
                )
            )
    if was_training:
        model.train()
    return torch.cat(stacks, dim=0)


def _make_controller(
    *,
    attention_mask: torch.Tensor,
    num_layers: int,
    capacity: int,
    top_k: int,
    beta_head: LearnedTerminationHead,
    continuation_weighting: str,
    expert_costs: torch.Tensor | None,
    state_update_threshold: float,
    termination_threshold: float,
    reference_router_logits: torch.Tensor | None,
    objective_weights: dict[str, float],
    mode: str = "soft",
    mechanism: str = "exact_reuse",
    policy_head: PolicyOverOptionsHead | None = None,
    restricted_layers: frozenset[int] | None = None,
    coverage_target: str = "current",
) -> LearnedBetaSparseMoeController:
    kwargs: dict[str, Any] = dict(
        attention_mask=attention_mask,
        num_layers=num_layers,
        capacity=capacity,
        top_k=top_k,
        termination_head=beta_head,
        mode=mode,
        termination_threshold=termination_threshold,
        continuation_weighting=continuation_weighting,
        expert_costs=expert_costs,
        state_update_threshold=state_update_threshold,
        reference_router_logits=reference_router_logits,
        lambda_cov=objective_weights["lambda_cov"],
        lambda_call=objective_weights["lambda_call"],
        lambda_cache=objective_weights["lambda_cache"],
        lambda_edit=objective_weights["lambda_edit"],
        counterfactual_eta=objective_weights["eta"],
        mask_temperature=objective_weights["mask_temperature"],
    )
    if mechanism == "best_of_pool":
        if policy_head is None:
            raise ValueError("best_of_pool mechanism requires a policy_head")
        return BestOfPoolBetaController(
            policy_head=policy_head, restricted_layers=restricted_layers,
            coverage_target=coverage_target, **kwargs
        )
    return LearnedBetaSparseMoeController(**kwargs)


def train(config: dict[str, Any], output_dir: Path) -> None:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_config = config["model"]
    train_config = config.get("train", {})
    controller_config = _controller_config(config, train_config)
    objective = str(train_config.get("objective", "group_policy_gradient"))
    if objective not in {"group_policy_gradient", "counterfactual_st", "differentiable_cache"}:
        raise ValueError(
            "train.objective must be group_policy_gradient, counterfactual_st, "
            "or differentiable_cache"
        )
    group_size = int(train_config.get("group_size", 4))
    if group_size <= 0:
        raise ValueError("train.group_size must be positive")
    baseline = str(train_config.get("baseline", "grpo_zscore"))
    if baseline not in {"grpo_zscore", "rloo_mean"}:
        raise ValueError("train.baseline must be grpo_zscore or rloo_mean")
    beta_regularizer = str(
        train_config.get(
            "beta_regularizer",
            "expected_hazard" if objective == "group_policy_gradient" else "none",
        )
    )
    if beta_regularizer not in {"none", "expected_hazard"}:
        raise ValueError("train.beta_regularizer must be none or expected_hazard")
    model_id = model_config["id"]
    revision = model_config.get("revision")
    device = torch.device(model_config.get("device", "cuda"))
    dtype = _dtype(model_config.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = _load_prompts({"trace": train_config})[: int(train_config.get("max_sequences", 8))]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(train_config.get("max_length", 96)),
    ).to(device)
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.train()
    initial_router_state = train_config.get("initial_router_state")
    warm_start_router_state = train_config.get("warm_start_router_state")
    router_state_to_load = (
        warm_start_router_state if warm_start_router_state is not None else initial_router_state
    )
    loaded_initial_names: tuple[str, ...] = ()
    if router_state_to_load is not None:
        initial_router_state_path = Path(os.path.expandvars(str(router_state_to_load)))
        loaded_initial_names = _load_router_state(model, initial_router_state_path)

    sparse_blocks = _sparse_moe_modules(model)
    num_experts = int(
        getattr(model.config, "num_experts", 0)
        or getattr(model.config, "num_local_experts", 0)
    )
    if num_experts <= 0:
        raise RuntimeError("model config does not expose num_experts/num_local_experts")
    top_k = int(getattr(model.config, "num_experts_per_tok", train_config.get("top_k", 2)))
    hidden_size = _hidden_size(model)
    train_router = bool(controller_config.get("train_router", True))
    trainable_router_parameters = 0
    trainable_router_names: tuple[str, ...] = ()
    if train_router:
        trainable_router_parameters, trainable_router_names = _freeze_non_router_parameters(
            model,
            num_experts,
        )

    beta_head_config_raw = controller_config.get("beta_head", {})
    if not isinstance(beta_head_config_raw, dict):
        raise TypeError("controller.beta_head must be a mapping when provided")
    beta_head = LearnedTerminationHead(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        num_layers=len(sparse_blocks),
        config=BetaHeadConfig(**beta_head_config_raw),
    ).to(device)

    # Best-of-pool joint termination (beta) + policy-over-options (mu): mu is a per-expert
    # demand-anticipation head; restrict_layers scopes the pooling to the reuse-tolerant layers.
    mechanism = str(controller_config.get("mechanism", "exact_reuse"))
    policy_head: PolicyOverOptionsHead | None = None
    restricted_layers: frozenset[int] | None = None
    # 'current' (myopic dwell-1) or 'union_dwell' (cover the option-horizon demand union).
    coverage_target = str(controller_config.get("coverage_target", "current"))
    if mechanism == "best_of_pool":
        policy_head_config_raw = controller_config.get("policy_head", {})
        if not isinstance(policy_head_config_raw, dict):
            raise TypeError("controller.policy_head must be a mapping when provided")
        policy_head = PolicyOverOptionsHead(
            num_experts=num_experts,
            num_layers=len(sparse_blocks),
            feature_dim=policy_feature_dim(),
            config=PolicyHeadConfig(**policy_head_config_raw),
        ).to(device)
        restrict_layers_cfg = controller_config.get("restrict_layers")
        if restrict_layers_cfg is not None:
            restricted_layers = frozenset(int(layer) for layer in restrict_layers_cfg)
    warm_start_beta_checkpoint = train_config.get("warm_start_beta_checkpoint")
    if warm_start_beta_checkpoint is not None:
        warm_head = load_learned_beta_checkpoint(
            Path(os.path.expandvars(str(warm_start_beta_checkpoint))),
            device,
        )
        beta_head.load_state_dict(warm_head.state_dict())
        beta_head.train()

    batch_size = int(train_config.get("batch_size", encoded["input_ids"].shape[0]))
    gradient_accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    if batch_size <= 0:
        raise ValueError("train.batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive")
    reference_batch_size = int(train_config.get("reference_batch_size", batch_size))
    reference_router_logits = _reference_router_stack(
        model,
        encoded,
        batch_size=reference_batch_size,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Co-adaptation lever: attach LoRA adapters to the (frozen) experts AFTER the frozen-base
    # reference routing is captured, so the KL anchor reflects the original routing.
    lora_config_raw = train_config.get("lora", controller_config.get("lora", {}))
    if not isinstance(lora_config_raw, dict):
        raise TypeError("train.lora must be a mapping when provided")
    lora_config = LoRAExpertsConfig(**lora_config_raw)
    lora_modules: list[LoRAExperts] = []
    if lora_config.enabled:
        lora_modules = attach_lora_experts(model, lora_config)
    lora_parameter_count = sum(
        parameter.numel()
        for module in lora_modules
        for parameter in module.adapter_parameters()
    )

    continuation_weighting = str(
        controller_config.get("continuation_weighting", "restricted_softmax")
    )
    if continuation_weighting not in {"restricted_softmax", "stored_weights"}:
        raise ValueError("continuation_weighting must be restricted_softmax or stored_weights")
    capacity = int(train_config.get("cache_capacity", controller_config.get("cache_capacity", 8)))
    cost_profile_name = train_config.get(
        "expert_cost_profile",
        controller_config.get("expert_cost_profile"),
    )
    expert_costs = (
        _cost_profile(num_experts, str(cost_profile_name)) if cost_profile_name else None
    )
    state_update_threshold = float(controller_config.get("state_update_threshold", 0.5))
    _legacy_deliberation_weight, _legacy_load_weight, _legacy_switch_weight, _legacy_kl_weight = (
        _cost_weights(controller_config)
    )
    objective_weights = _counterfactual_weights(controller_config)
    policy_weights = _policy_gradient_weights(train_config, controller_config)
    # Differentiable cache-pressure weights: drive option persistence directly through the
    # differentiable switch/load/call costs (which couple to NLL via the soft blend) plus a
    # router-KL quality anchor and a commitment penalty, instead of the coverage-BCE surrogate.
    differentiable_weights = {
        "deliberation": float(train_config.get("lambda_deliberation", 0.0)),
        "switch": float(train_config.get("lambda_switch", 0.0)),
        "load": float(train_config.get("lambda_load", 0.0)),
        "commit": float(train_config.get("lambda_commit", 0.0)),
        "router": objective_weights["lambda_router"],
        # Best-of-pool only: supervise mu to cover the realized top-k demand in its pool.
        "mu_coverage": float(train_config.get("lambda_mu_coverage", 0.0)),
    }

    termination_threshold = float(controller_config.get("termination_threshold", 0.5))
    controller = _make_controller(
        attention_mask=encoded["attention_mask"],
        num_layers=len(sparse_blocks),
        capacity=capacity,
        top_k=top_k,
        beta_head=beta_head,
        termination_threshold=termination_threshold,
        continuation_weighting=continuation_weighting,
        expert_costs=expert_costs,
        state_update_threshold=state_update_threshold,
        reference_router_logits=reference_router_logits,
        objective_weights=objective_weights,
        mechanism=mechanism,
        policy_head=policy_head,
        restricted_layers=restricted_layers,
        coverage_target=coverage_target,
    )
    originals: list[tuple[torch.nn.Module, str, torch.nn.Module, LearnedBetaSparseMoeWrapper]] = []
    for layer_index, parent, child_name, child in sparse_blocks:
        wrapper = LearnedBetaSparseMoeWrapper(child, layer_index, controller)
        setattr(parent, child_name, wrapper)
        originals.append((parent, child_name, child, wrapper))

    trainable_heads = list(beta_head.parameters())
    if policy_head is not None:
        trainable_heads += list(policy_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in trainable_heads + list(model.parameters())
            if parameter.requires_grad
        ],
        lr=float(train_config.get("learning_rate", 1e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    steps = int(train_config.get("steps", 3))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                step_sums: dict[str, float] = {}
                for _micro_step in range(gradient_accumulation_steps):
                    batch, batch_labels, indices = _select_training_batch(
                        encoded,
                        labels,
                        batch_size=batch_size,
                    )
                    reference_batch = reference_router_logits.index_select(
                        0,
                        indices.detach().cpu(),
                    )
                    if objective in {"counterfactual_st", "differentiable_cache"}:
                        controller = _make_controller(
                            attention_mask=batch["attention_mask"],
                            num_layers=len(sparse_blocks),
                            capacity=capacity,
                            top_k=top_k,
                            beta_head=beta_head,
                            termination_threshold=termination_threshold,
                            continuation_weighting=continuation_weighting,
                            expert_costs=expert_costs,
                            state_update_threshold=state_update_threshold,
                            reference_router_logits=reference_batch,
                            objective_weights=objective_weights,
                            mode="soft",
                            mechanism=mechanism,
                            policy_head=policy_head,
                            restricted_layers=restricted_layers,
                            coverage_target=coverage_target,
                        )
                        for _parent, _child_name, _child, wrapper in originals:
                            wrapper.controller = controller
                        outputs = model(
                            **batch,
                            labels=batch_labels,
                            output_router_logits=False,
                            use_cache=False,
                        )
                        controller.finalize()
                        soft_terms = controller.soft_loss_terms(outputs.loss.device)
                        router_kl = controller.router_kl_loss(outputs.loss.device)
                        router_kl_value = float(router_kl.detach().cpu())
                        if objective == "differentiable_cache":
                            loss = _differentiable_cache_loss(
                                outputs.loss,
                                soft_terms,
                                router_kl,
                                differentiable_weights,
                            )
                        else:
                            loss = (
                                outputs.loss
                                + objective_weights["alpha_beta"]
                                * soft_terms["beta_counterfactual"]
                                + objective_weights["alpha_mask"] * soft_terms["mask_coverage"]
                                + objective_weights["lambda_router"] * router_kl
                            )
                        micro_row = {
                            "loss": float(loss.detach().cpu()),
                            "lm_loss": float(outputs.loss.detach().cpu()),
                            "expected_deliberation": float(
                                soft_terms["deliberation"].detach().cpu()
                            ),
                            "expected_load_cost": float(soft_terms["load"].detach().cpu()),
                            "expected_switch_cost": float(soft_terms["switch"].detach().cpu()),
                            "expected_commitment": float(soft_terms["commitment"].detach().cpu()),
                            "beta_counterfactual_loss": float(
                                soft_terms["beta_counterfactual"].detach().cpu()
                            ),
                            "mask_coverage_loss": float(
                                soft_terms["mask_coverage"].detach().cpu()
                            ),
                            "counterfactual_target": float(
                                soft_terms["counterfactual_target"].detach().cpu()
                            ),
                            "continue_coverage": float(
                                soft_terms["continue_coverage"].detach().cpu()
                            ),
                            "switch_coverage": float(
                                soft_terms["switch_coverage"].detach().cpu()
                            ),
                            "router_kl_to_frozen": router_kl_value,
                            "router_kl_loss": float(router_kl.detach().cpu()),
                            "beta_calls": float(controller.beta_calls),
                            "controller_calls": float(controller.controller_calls),
                            "soft_keep_executions": float(controller.soft_keep_executions),
                            "soft_refresh_executions": float(controller.soft_refresh_executions),
                            "mean_option_length": controller.mean_option_length,
                            "gate_evaluated_tokens": float(controller.gate_evaluated_tokens),
                        }
                    else:
                        rewards: list[torch.Tensor] = []
                        normalized_log_probs: list[torch.Tensor] = []
                        beta_normalized_log_probs: list[torch.Tensor] = []
                        mask_normalized_log_probs: list[torch.Tensor] = []
                        entropies: list[torch.Tensor] = []
                        hazard_losses: list[torch.Tensor] = []
                        expected_hazards: list[torch.Tensor] = []
                        router_kls: list[torch.Tensor] = []
                        rollout_sums: dict[str, float] = {}
                        for _rollout in range(group_size):
                            controller = _make_controller(
                                attention_mask=batch["attention_mask"],
                                num_layers=len(sparse_blocks),
                                capacity=capacity,
                                top_k=top_k,
                                beta_head=beta_head,
                                termination_threshold=termination_threshold,
                                continuation_weighting=continuation_weighting,
                                expert_costs=expert_costs,
                                state_update_threshold=state_update_threshold,
                                reference_router_logits=reference_batch,
                                objective_weights=objective_weights,
                                mode="policy_gradient",
                            )
                            for _parent, _child_name, _child, wrapper in originals:
                                wrapper.controller = controller
                            outputs = model(
                                **batch,
                                labels=batch_labels,
                                output_router_logits=False,
                                use_cache=False,
                            )
                            controller.finalize()
                            pg_terms = controller.policy_gradient_terms(outputs.loss.device)
                            beta_stats = controller.beta_statistics(outputs.loss.device)
                            rollout_hazard_loss = _hazard_loss(
                                pg_terms["expected_hazard"],
                                lambda_beta=policy_weights["lambda_beta"],
                                beta_regularizer=beta_regularizer,
                            )
                            router_kl = controller.router_kl_loss(outputs.loss.device)
                            router_kl_value = float(router_kl.detach().cpu())
                            reward_parts = _reward_decomposition(
                                lm_loss=outputs.loss,
                                call_rate=controller.call_rate,
                                edit_rate=controller.edit_rate,
                                cache_cost_rate=controller.cache_cost_rate,
                                lambda_call=policy_weights["lambda_call"],
                                lambda_edit=policy_weights["lambda_edit"],
                                lambda_cache=policy_weights["lambda_cache"],
                            )
                            reward = reward_parts["reward"]
                            rewards.append(reward)
                            normalized_log_probs.append(pg_terms["normalized_log_prob"])
                            beta_normalized_log_probs.append(
                                pg_terms["beta_normalized_log_prob"]
                            )
                            mask_normalized_log_probs.append(
                                pg_terms["mask_normalized_log_prob"]
                            )
                            entropies.append(pg_terms["normalized_entropy"])
                            hazard_losses.append(rollout_hazard_loss)
                            expected_hazards.append(pg_terms["expected_hazard"])
                            router_kls.append(router_kl)
                            rollout_row = {
                                "lm_loss": float(outputs.loss.detach().cpu()),
                                "reward": float(reward.detach().cpu()),
                                "hazard_loss": float(
                                    rollout_hazard_loss.detach().cpu()
                                ),
                                "expected_hazard": float(
                                    pg_terms["expected_hazard"].detach().cpu()
                                ),
                                "mean_beta": float(
                                    pg_terms["expected_hazard"].detach().cpu()
                                ),
                                "objective_beta_penalty": float(
                                    rollout_hazard_loss.detach().cpu()
                                ),
                                "call_rate": controller.call_rate,
                                "edit_rate": controller.edit_rate,
                                "cache_cost_rate": controller.cache_cost_rate,
                                "policy_log_prob": float(pg_terms["log_prob"].detach().cpu()),
                                "policy_log_prob_normalized": float(
                                    pg_terms["normalized_log_prob"].detach().cpu()
                                ),
                                "beta_log_prob_normalized": float(
                                    pg_terms["beta_normalized_log_prob"].detach().cpu()
                                ),
                                "mask_log_prob_normalized": float(
                                    pg_terms["mask_normalized_log_prob"].detach().cpu()
                                ),
                                "beta_log_prob_per_decision": float(
                                    pg_terms["beta_log_prob_per_decision"].detach().cpu()
                                ),
                                "mask_log_prob_per_decision": float(
                                    pg_terms["mask_log_prob_per_decision"].detach().cpu()
                                ),
                                "policy_entropy": float(pg_terms["entropy"].detach().cpu()),
                                "normalized_entropy": float(
                                    pg_terms["normalized_entropy"].detach().cpu()
                                ),
                                "policy_log_prob_count": float(
                                    pg_terms["log_prob_count"].detach().cpu()
                                ),
                                "beta_log_prob": float(
                                    pg_terms["beta_log_prob"].detach().cpu()
                                ),
                                "mask_log_prob": float(
                                    pg_terms["mask_log_prob"].detach().cpu()
                                ),
                                "beta_log_prob_count": float(
                                    pg_terms["beta_log_prob_count"].detach().cpu()
                                ),
                                "mask_log_prob_count": float(
                                    pg_terms["mask_log_prob_count"].detach().cpu()
                                ),
                                "router_kl_to_frozen": router_kl_value,
                                "router_kl_loss": float(router_kl.detach().cpu()),
                                "beta_calls": float(controller.beta_calls),
                                "controller_calls": float(controller.controller_calls),
                                "mean_option_length": controller.mean_option_length,
                                "gate_evaluated_tokens": float(controller.gate_evaluated_tokens),
                            }
                            for name, value in reward_parts.items():
                                if name == "reward":
                                    continue
                                rollout_row[name] = float(value.detach().cpu())
                            for name, value in beta_stats.items():
                                rollout_row[name] = float(value.detach().cpu())
                            for name, value in rollout_row.items():
                                rollout_sums[name] = rollout_sums.get(name, 0.0) + value

                        reward_tensor = torch.stack(rewards)
                        advantages = _group_relative_advantages(reward_tensor, baseline)
                        log_prob_tensor = torch.stack(normalized_log_probs)
                        beta_log_prob_tensor = torch.stack(beta_normalized_log_probs)
                        mask_log_prob_tensor = torch.stack(mask_normalized_log_probs)
                        policy_loss = _policy_gradient_loss(log_prob_tensor, advantages)
                        beta_policy_loss = _policy_gradient_loss(
                            beta_log_prob_tensor,
                            advantages,
                        )
                        mask_policy_loss = _policy_gradient_loss(
                            mask_log_prob_tensor,
                            advantages,
                        )
                        entropy = torch.stack(entropies).mean()
                        hazard_loss = torch.stack(hazard_losses).mean()
                        expected_hazard = torch.stack(expected_hazards).mean()
                        router_kl = torch.stack(router_kls).mean()
                        beta_head_parameters = list(beta_head.parameters())
                        beta_policy_grad_norm = _grad_l2_norm(
                            beta_policy_loss,
                            beta_head_parameters,
                        )
                        hazard_grad_norm = _grad_l2_norm(
                            hazard_loss,
                            beta_head_parameters,
                        )
                        loss = (
                            policy_loss
                            + hazard_loss
                            + policy_weights["lambda_router"] * router_kl
                            - policy_weights["entropy_weight"] * entropy
                        )
                        micro_row = {
                            "loss": float(loss.detach().cpu()),
                            "policy_loss": float(policy_loss.detach().cpu()),
                            "beta_policy_loss": float(beta_policy_loss.detach().cpu()),
                            "mask_policy_loss": float(mask_policy_loss.detach().cpu()),
                            "beta_policy_loss_abs": float(
                                beta_policy_loss.detach().abs().cpu()
                            ),
                            "mask_policy_loss_abs": float(
                                mask_policy_loss.detach().abs().cpu()
                            ),
                            "hazard_to_beta_policy_abs": float(
                                (
                                    hazard_loss.detach().abs()
                                    / beta_policy_loss.detach().abs().clamp_min(1.0e-12)
                                ).cpu()
                            ),
                            "beta_policy_grad_norm": float(
                                beta_policy_grad_norm.detach().cpu()
                            ),
                            "hazard_grad_norm": float(hazard_grad_norm.detach().cpu()),
                            "hazard_to_beta_policy_grad_norm": float(
                                (
                                    hazard_grad_norm.detach()
                                    / beta_policy_grad_norm.detach().clamp_min(1.0e-12)
                                ).cpu()
                            ),
                            "hazard_loss": float(hazard_loss.detach().cpu()),
                            "expected_hazard": float(expected_hazard.detach().cpu()),
                            "mean_beta": float(expected_hazard.detach().cpu()),
                            "advantage_mean": float(advantages.mean().detach().cpu()),
                            "advantage_std": float(
                                advantages.std(unbiased=False).detach().cpu()
                            ),
                            **{
                                name: value / float(group_size)
                                for name, value in rollout_sums.items()
                            },
                        }
                    (loss / gradient_accumulation_steps).backward()
                    for name, value in micro_row.items():
                        step_sums[name] = step_sums.get(name, 0.0) + value
                optimizer.step()

                row = {
                    "step": step,
                    **{
                        name: value / float(gradient_accumulation_steps)
                        for name, value in step_sums.items()
                    },
                }
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
    finally:
        for parent, child_name, child, _wrapper in originals:
            setattr(parent, child_name, child)

    checkpoint_path = output_dir / "learned_beta_state.pt"
    extra_metadata = {
        "seed": seed,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": _resolved_model_revision(model),
        "dtype": str(dtype).removeprefix("torch."),
        "steps": steps,
        "learning_rate": float(train_config.get("learning_rate", 1e-4)),
        "weight_decay": float(train_config.get("weight_decay", 0.0)),
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "reference_batch_size": reference_batch_size,
        "cache_capacity": capacity,
        "objective": objective,
        "group_size": group_size,
        "baseline": baseline,
        "entropy_weight": policy_weights["entropy_weight"],
        "policy_lambda_call": policy_weights["lambda_call"],
        "policy_lambda_beta": policy_weights["lambda_beta"],
        "policy_lambda_edit": policy_weights["lambda_edit"],
        "policy_lambda_cache": policy_weights["lambda_cache"],
        "policy_lambda_router": policy_weights["lambda_router"],
        "beta_regularizer": beta_regularizer,
        "warm_start_beta_checkpoint": str(warm_start_beta_checkpoint)
        if warm_start_beta_checkpoint
        else None,
        "warm_start_router_state": str(warm_start_router_state)
        if warm_start_router_state
        else None,
        "initial_router_state": str(initial_router_state) if initial_router_state else None,
        "loaded_router_state": str(router_state_to_load) if router_state_to_load else None,
        "loaded_initial_router_parameter_count": len(loaded_initial_names),
        "loaded_initial_router_parameter_names": loaded_initial_names,
        "state_update_threshold": state_update_threshold,
        "alpha_beta": objective_weights["alpha_beta"],
        "alpha_mask": objective_weights["alpha_mask"],
        "lambda_cov": objective_weights["lambda_cov"],
        "lambda_call": objective_weights["lambda_call"],
        "lambda_cache": objective_weights["lambda_cache"],
        "lambda_edit": objective_weights["lambda_edit"],
        "lambda_router": objective_weights["lambda_router"],
        "counterfactual_eta": objective_weights["eta"],
        "mask_temperature": objective_weights["mask_temperature"],
        "legacy_deliberation_weight": _legacy_deliberation_weight,
        "legacy_load_weight": _legacy_load_weight,
        "legacy_switch_weight": _legacy_switch_weight,
        "legacy_router_kl_weight": _legacy_kl_weight,
        "train_router": train_router,
        "trainable_router_parameters": trainable_router_parameters,
        "trainable_router_parameter_names": trainable_router_names,
        "expert_cost_profile": cost_profile_name,
        "expert_costs": [float(value) for value in expert_costs.tolist()]
        if expert_costs is not None
        else None,
        "num_prompts": len(prompts),
        "sequence_length": int(encoded["input_ids"].shape[1]),
        "trainable_parameters": sum(parameter.numel() for parameter in beta_head.parameters()),
        "lora_enabled": lora_config.enabled,
        "lora_rank": lora_config.rank if lora_config.enabled else None,
        "lora_alpha": lora_config.alpha if lora_config.enabled else None,
        "lora_targets": list(lora_config.targets) if lora_config.enabled else None,
        "lora_parameter_count": lora_parameter_count,
        "differentiable_lambda_deliberation": differentiable_weights["deliberation"],
        "differentiable_lambda_switch": differentiable_weights["switch"],
        "differentiable_lambda_load": differentiable_weights["load"],
        "differentiable_lambda_commit": differentiable_weights["commit"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "note": (
            "Learned call-and-return training supports counterfactual_st warm starts "
            "and critic-free grouped policy-gradient fine-tuning. Hard no-gate "
            "evaluation remains deterministic."
        ),
    }
    adapter_state = lora_state_dict(model) if lora_modules else {}
    if train_router:
        router_state = _router_state_dict(model)
        # Keep adapter tensors out of router_state.pt (they ship in adapter_state.pt) so the
        # router checkpoint stays loadable by models without LoRA attached.
        router_state = {
            name: value for name, value in router_state.items() if name not in adapter_state
        }
        torch.save(router_state, output_dir / "router_state.pt")
    if adapter_state:
        torch.save(adapter_state, output_dir / "adapter_state.pt")
    save_learned_beta_checkpoint(
        checkpoint_path,
        beta_head,
        continuation_weighting=continuation_weighting,
        extra_metadata=extra_metadata,
    )
    if policy_head is not None:
        torch.save(
            {
                "policy_head": policy_head.state_dict(),
                "metadata": {
                    "num_experts": policy_head.num_experts,
                    "num_layers": policy_head.num_layers,
                    "feature_dim": policy_head.feature_dim,
                    "config": {
                        "hidden_dim": policy_head.config.hidden_dim,
                        "shared_across_layers": policy_head.config.shared_across_layers,
                        "temperature": policy_head.config.temperature,
                    },
                    "capacity": capacity,
                    "restrict_layers": sorted(restricted_layers) if restricted_layers else None,
                    "coverage_target": coverage_target,
                },
            },
            output_dir / "policy_head_state.pt",
        )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    write_learned_beta_metadata(
        output_dir / "metadata.json",
        learned_beta_metadata(checkpoint_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned option-termination head.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config.get("output_dir", "outputs/learned_beta_train"))
    train(config, output_dir)


if __name__ == "__main__":
    main()
