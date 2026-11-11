from __future__ import annotations

import argparse
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
from .evaluate_cache_rerank import CacheRerankController, _gate_modules
from .evaluate_router_state import _load_router_state
from .train_router import (
    _effective_batch_size,
    _freeze_non_router_parameters,
    _linear_warmup_multiplier,
    _register_reference_router_hooks,
    _router_kl,
    _router_state_dict,
)


def _select_training_batch(
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    total_rows = labels.shape[0]
    if batch_size >= total_rows:
        indices = torch.arange(total_rows, dtype=torch.long, device=labels.device)
        return encoded, labels, indices
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
    return batch, labels.index_select(0, indices), indices


def _select_reference_layers(
    reference_layers: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    total_batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, ...]:
    selected_layers: list[torch.Tensor] = []
    for layer in reference_layers:
        layer_indices = indices.detach().to(device=layer.device)
        if layer.ndim == 3:
            selected_layers.append(layer.index_select(0, layer_indices))
            continue
        if layer.ndim == 2 and layer.shape[0] == total_batch_size * sequence_length:
            reshaped = layer.reshape(total_batch_size, sequence_length, layer.shape[-1])
            selected = reshaped.index_select(0, layer_indices)
            selected_layers.append(selected.reshape(-1, layer.shape[-1]))
            continue
        raise RuntimeError(f"unsupported reference router logit shape: {tuple(layer.shape)}")
    return tuple(selected_layers)


def _register_cache_rerank_hooks(
    model: torch.nn.Module,
    attention_mask: torch.Tensor,
    capacity: int,
    lambda_cache: float,
    expert_costs: torch.Tensor | None = None,
    weighted_penalty: bool = False,
) -> tuple[CacheRerankController, list[torch.utils.hooks.RemovableHandle]]:
    gates = _gate_modules(model)
    controller = CacheRerankController(
        attention_mask=attention_mask,
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
    return controller, handles


def train(config: dict[str, Any], output_dir: Path) -> None:
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

    prompts = _load_prompts({"trace": train_config})[: int(train_config.get("max_sequences", 16))]
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
    model.train()

    num_experts = int(
        getattr(model.config, "num_experts", 0)
        or getattr(model.config, "num_local_experts", 0)
    )
    if num_experts <= 0:
        raise RuntimeError("model config does not expose num_experts/num_local_experts")
    trainable_params, trainable_names = _freeze_non_router_parameters(model, num_experts)
    initial_router_state = train_config.get("initial_router_state")
    loaded_initial_names: tuple[str, ...] = ()
    if initial_router_state is not None:
        initial_router_state_path = Path(os.path.expandvars(str(initial_router_state)))
        loaded_initial_names = _load_router_state(model, initial_router_state_path)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_config.get("learning_rate", 5e-6)),
    )
    steps = int(train_config.get("steps", 3))
    batch_size = _effective_batch_size(train_config, len(prompts))
    gradient_accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    lr_warmup_steps = int(train_config.get("learning_rate_warmup_steps", 0))
    max_grad_norm = float(train_config.get("max_grad_norm", 0.0))
    kl_weight = float(train_config.get("router_kl_weight", 0.01))
    capacity = int(train_config.get("cache_capacity", 8))
    lambda_cache = float(train_config.get("lambda_cache", 0.03))
    cost_profile_name = train_config.get("expert_cost_profile")
    weighted_penalty = bool(train_config.get("weighted_penalty", False))
    expert_costs = (
        _cost_profile(num_experts, str(cost_profile_name)) if cost_profile_name else None
    )
    reference_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    reference_router_logits: list[torch.Tensor] = []
    if kl_weight != 0.0:
        reference_hook_handles, reference_router_logits = _register_reference_router_hooks(
            model,
            trainable_names,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                learning_rate_multiplier = _linear_warmup_multiplier(step, lr_warmup_steps)
                current_learning_rate = float(train_config.get("learning_rate", 5e-6)) * (
                    learning_rate_multiplier
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_learning_rate

                step_sums: dict[str, float] = {}
                for _micro_step in range(gradient_accumulation_steps):
                    batch_encoded, batch_labels, _batch_indices = _select_training_batch(
                        encoded,
                        labels,
                        batch_size=batch_size,
                    )
                    reference_router_logits.clear()
                    controller, handles = _register_cache_rerank_hooks(
                        model,
                        batch_encoded["attention_mask"],
                        capacity=capacity,
                        lambda_cache=lambda_cache,
                        expert_costs=expert_costs,
                        weighted_penalty=weighted_penalty,
                    )
                    try:
                        controller.reset()
                        outputs = model(
                            **batch_encoded,
                            labels=batch_labels,
                            output_router_logits=True,
                            use_cache=False,
                        )
                        zero_loss = outputs.loss * 0.0
                        kl = (
                            _router_kl(
                                outputs.router_logits,
                                tuple(reference_router_logits),
                                batch_encoded["attention_mask"],
                            )
                            if kl_weight != 0.0
                            else zero_loss
                        )
                        loss = outputs.loss + kl_weight * kl
                        (loss / gradient_accumulation_steps).backward()
                    finally:
                        for handle in handles:
                            handle.remove()
                    micro_row = {
                        "loss": float(loss.detach().cpu()),
                        "lm_loss": float(outputs.loss.detach().cpu()),
                        "router_kl": float(kl.detach().cpu()),
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
        "initial_router_state": str(initial_router_state) if initial_router_state else None,
        "loaded_initial_router_parameter_count": len(loaded_initial_names),
        "loaded_initial_router_parameter_names": loaded_initial_names,
        "steps": steps,
        "learning_rate": float(train_config.get("learning_rate", 5e-6)),
        "learning_rate_warmup_steps": lr_warmup_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "router_kl_weight": kl_weight,
        "cache_capacity": capacity,
        "lambda_cache": lambda_cache,
        "expert_cost_profile": cost_profile_name,
        "weighted_penalty": weighted_penalty,
        "expert_costs": [float(value) for value in expert_costs.tolist()]
        if expert_costs is not None
        else None,
        "reference_router_mode": "frozen_gate_on_current_hidden" if kl_weight != 0.0 else "none",
        "num_prompts": len(prompts),
        "sequence_length": int(encoded["input_ids"].shape[1]),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "note": "Router-only training with serving-time cache-rerank hook active.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train router gates under cache-reranked routing.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train(config, args.output_dir)


if __name__ == "__main__":
    main()
