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

from moe_eviction.experiments.trace_frontier import _simulate_topk

from .collect_trace import (
    DEFAULT_PROMPTS,
    _dtype,
    _load_prompts,
    _resolved_model_revision,
    _router_layers,
)


@dataclass(frozen=True, slots=True)
class EvalRow:
    method: str
    cache_capacity: int
    nll: float
    perplexity: float
    hard_misses: int
    hard_miss_rate: float
    weighted_miss_cost: float
    requests: int
    router_kl_to_vanilla: float
    expert_overlap_ratio: float
    normalized_router_entropy: float
    load_cv: float


def _load_router_state(model: torch.nn.Module, router_state_path: Path) -> tuple[str, ...]:
    state = torch.load(router_state_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"router state must be a dict, got {type(state)!r}")

    model_params = dict(model.named_parameters())
    missing = sorted(name for name in state if name not in model_params)
    if missing:
        raise KeyError(f"router state contains unknown parameter names: {missing[:5]}")

    loaded_names: list[str] = []
    with torch.no_grad():
        for name, value in state.items():
            parameter = model_params[name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(
                    f"shape mismatch for {name}: checkpoint {tuple(value.shape)} "
                    f"model {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            loaded_names.append(name)
    return tuple(sorted(loaded_names))


def _router_stack(router_logits: Any, batch_size: int, seq_len: int) -> torch.Tensor:
    layers: list[torch.Tensor] = []
    for layer in _router_layers(router_logits):
        if layer.ndim == 3:
            layers.append(layer.detach().float().cpu())
        elif layer.ndim == 2 and layer.shape[0] == batch_size * seq_len:
            layers.append(
                layer.detach().float().reshape(batch_size, seq_len, layer.shape[-1]).cpu()
            )
        else:
            raise RuntimeError(f"unsupported router logit shape: {tuple(layer.shape)}")
    return torch.stack(layers, dim=1)


def _router_kl(
    vanilla: torch.Tensor,
    candidate: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    mask = attention_mask[:, None, :, None].bool()
    original_probs = torch.softmax(vanilla, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate, dim=-1)
    per_position = torch.nn.functional.kl_div(
        candidate_log_probs,
        original_probs,
        reduction="none",
    ).sum(dim=-1, keepdim=True)
    masked = per_position.masked_select(mask)
    return float(masked.mean()) if masked.numel() else 0.0


def _routing_locality_metrics(
    router_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int,
) -> tuple[float, float, float]:
    selected = torch.topk(router_logits, k=top_k, dim=-1).indices
    active = attention_mask.bool()
    batch_size, num_layers, seq_len, num_experts = router_logits.shape

    overlap_sum = 0.0
    overlap_count = 0
    if seq_len > 1:
        for batch_index in range(batch_size):
            for layer_index in range(num_layers):
                for token_index in range(1, seq_len):
                    if not active[batch_index, token_index].item():
                        continue
                    if not active[batch_index, token_index - 1].item():
                        continue
                    current = set(
                        selected[batch_index, layer_index, token_index].detach().cpu().tolist()
                    )
                    previous = set(
                        selected[
                            batch_index,
                            layer_index,
                            token_index - 1,
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    overlap_sum += len(current & previous) / top_k
                    overlap_count += 1

    probabilities = torch.softmax(router_logits.float(), dim=-1)
    masked_probabilities = probabilities.masked_select(active[:, None, :, None]).reshape(
        -1,
        num_experts,
    )
    if masked_probabilities.numel() == 0:
        normalized_entropy = 0.0
    else:
        entropy = -(
            masked_probabilities * torch.log(masked_probabilities.clamp_min(1e-8))
        ).sum(dim=-1)
        normalized_entropy = float((entropy / math.log(num_experts)).mean())

    counts = torch.zeros((num_experts,), dtype=torch.float32)
    for batch_index in range(batch_size):
        for layer_index in range(num_layers):
            for token_index in range(seq_len):
                if not active[batch_index, token_index].item():
                    continue
                counts.scatter_add_(
                    0,
                    selected[batch_index, layer_index, token_index].cpu(),
                    torch.ones((top_k,), dtype=torch.float32),
                )
    mean_count = counts.mean()
    load_cv = (
        float(counts.std(unbiased=False) / mean_count.clamp_min(1e-8))
        if counts.numel()
        else 0.0
    )

    eor = overlap_sum / overlap_count if overlap_count else 0.0
    return eor, normalized_entropy, load_cv


def _causal_lm_nll(model: torch.nn.Module, outputs: Any) -> float:
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
    return float(loss.cpu())


def _evaluate_loaded_model(
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    vanilla_router_logits: torch.Tensor | None,
    capacity: int,
) -> tuple[EvalRow, torch.Tensor]:
    with torch.inference_mode():
        outputs = model(**encoded, labels=labels, output_router_logits=True, use_cache=False)

    nll = _causal_lm_nll(model, outputs)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"].detach().cpu()
    router_logits = _router_stack(outputs.router_logits, input_ids.shape[0], input_ids.shape[1])
    top_k = int(getattr(model.config, "num_experts_per_tok", 2))
    hard_misses, weighted_cost, requests, _online_kl = _simulate_topk(
        router_logits,
        attention_mask.bool(),
        top_k=top_k,
        capacity=capacity,
    )
    eor, normalized_entropy, load_cv = _routing_locality_metrics(
        router_logits,
        attention_mask,
        top_k,
    )
    router_kl = (
        0.0
        if vanilla_router_logits is None
        else _router_kl(vanilla_router_logits, router_logits, attention_mask)
    )
    row = EvalRow(
        method=method,
        cache_capacity=capacity,
        nll=nll,
        perplexity=math.exp(nll),
        hard_misses=hard_misses,
        hard_miss_rate=hard_misses / requests if requests else 0.0,
        weighted_miss_cost=weighted_cost,
        requests=requests,
        router_kl_to_vanilla=router_kl,
        expert_overlap_ratio=eor,
        normalized_router_entropy=normalized_entropy,
        load_cv=load_cv,
    )
    return row, router_logits


def evaluate(config: dict[str, Any], router_state_path: Path, output_dir: Path) -> None:
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = _load_prompts({"trace": eval_config})[: int(eval_config.get("max_sequences", 4))]
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
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

    rows: list[EvalRow] = []
    vanilla_router_logits: torch.Tensor | None = None
    for capacity in capacities:
        vanilla_row, current_vanilla_router_logits = _evaluate_loaded_model(
            "vanilla",
            model,
            encoded,
            labels,
            vanilla_router_logits=None,
            capacity=capacity,
        )
        rows.append(vanilla_row)
        if vanilla_router_logits is None:
            vanilla_router_logits = current_vanilla_router_logits
    if vanilla_router_logits is None:
        raise RuntimeError("no capacities configured")
    loaded_names = _load_router_state(model, router_state_path)
    for capacity in capacities:
        trained_row, _trained_router_logits = _evaluate_loaded_model(
            "trained_router",
            model,
            encoded,
            labels,
            vanilla_router_logits=vanilla_router_logits,
            capacity=capacity,
        )
        rows.append(trained_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")

    metadata = {
        "seed": seed,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": _resolved_model_revision(model),
        "dtype": str(dtype).removeprefix("torch."),
        "router_state_path": str(router_state_path),
        "loaded_router_parameter_count": len(loaded_names),
        "loaded_router_parameter_names": loaded_names,
        "cache_capacities": capacities,
        "num_prompts": len(prompts),
        "sequence_length": int(encoded["input_ids"].shape[1]),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vanilla vs saved router-state quality.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--router-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluate(config, args.router_state, args.output_dir)


if __name__ == "__main__":
    main()
