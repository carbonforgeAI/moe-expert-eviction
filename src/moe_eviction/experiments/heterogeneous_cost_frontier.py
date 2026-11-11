from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class HeterogeneousCostRow:
    method: str
    capacity: int
    lambda_cache: float
    router_kl: float
    hard_misses: int
    hard_miss_rate: float
    weighted_miss_cost: float
    weighted_miss_cost_per_request: float
    requests: int


def _load_trace(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    data = np.load(path)
    logits = torch.from_numpy(data["router_logits"]).float()
    attention_mask = torch.from_numpy(data["attention_mask"]).bool()
    selected = data["selected_experts"]
    return logits, attention_mask, int(selected.shape[-1])


def _cost_profile(num_experts: int, mode: str) -> torch.Tensor:
    if mode == "uniform":
        return torch.ones(num_experts, dtype=torch.float32)
    if mode == "tiered":
        costs = torch.ones(num_experts, dtype=torch.float32)
        costs[: max(1, num_experts // 4)] = 4.0
        costs[max(1, num_experts // 4) : max(2, num_experts // 2)] = 2.0
        return costs
    if mode == "linear":
        return torch.linspace(1.0, 4.0, num_experts, dtype=torch.float32)
    raise ValueError(f"unsupported cost profile: {mode}")


def _simulate(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
    expert_costs: torch.Tensor,
    *,
    top_k: int,
    capacity: int,
    lambda_cache: float,
    weighted_penalty: bool,
    original_scores: torch.Tensor,
) -> HeterogeneousCostRow:
    batch_size, num_layers, seq_len, _num_experts = scores.shape
    hard_misses = 0
    weighted_cost = 0.0
    requests = 0
    kl_sum = 0.0
    kl_count = 0
    caches: list[list[list[int]]] = [
        [[] for _layer in range(num_layers)] for _batch in range(batch_size)
    ]

    for token_index in range(seq_len):
        for batch_index in range(batch_size):
            if not attention_mask[batch_index, token_index].item():
                continue
            for layer_index in range(num_layers):
                resident = caches[batch_index][layer_index]
                adjusted = scores[batch_index, layer_index, token_index].clone()
                if lambda_cache:
                    cold = torch.ones_like(adjusted)
                    if resident:
                        cold[torch.tensor(resident)] = 0.0
                    penalty = cold * expert_costs if weighted_penalty else cold
                    adjusted = adjusted - lambda_cache * penalty

                original_probs = torch.softmax(
                    original_scores[batch_index, layer_index, token_index],
                    dim=-1,
                )
                adjusted_log_probs = torch.log_softmax(adjusted, dim=-1)
                kl_sum += float(
                    F.kl_div(adjusted_log_probs, original_probs, reduction="sum").detach()
                )
                kl_count += 1

                selected = torch.topk(adjusted, k=top_k, dim=-1).indices.tolist()
                before = set(resident)
                misses = [expert for expert in selected if expert not in before]
                hard_misses += len(misses)
                weighted_cost += float(expert_costs[misses].sum()) if misses else 0.0
                requests += len(selected)

                for expert in selected:
                    if expert in resident:
                        resident.remove(expert)
                    resident.append(expert)
                    while len(resident) > capacity:
                        resident.pop(0)

    method = "weighted_cache_bias" if weighted_penalty else "unit_cache_bias"
    if lambda_cache == 0.0:
        method = "vanilla"
    return HeterogeneousCostRow(
        method=method,
        capacity=capacity,
        lambda_cache=lambda_cache,
        router_kl=kl_sum / kl_count if kl_count else 0.0,
        hard_misses=hard_misses,
        hard_miss_rate=hard_misses / requests if requests else 0.0,
        weighted_miss_cost=weighted_cost,
        weighted_miss_cost_per_request=weighted_cost / requests if requests else 0.0,
        requests=requests,
    )


def run(
    trace_path: Path,
    output_dir: Path,
    capacities: tuple[int, ...],
    lambdas: tuple[float, ...],
    cost_profile: str,
) -> None:
    logits, attention_mask, top_k = _load_trace(trace_path)
    expert_costs = _cost_profile(logits.shape[-1], cost_profile)
    rows: list[HeterogeneousCostRow] = []
    for capacity in capacities:
        rows.append(
            _simulate(
                logits,
                attention_mask,
                expert_costs,
                top_k=top_k,
                capacity=capacity,
                lambda_cache=0.0,
                weighted_penalty=False,
                original_scores=logits,
            )
        )
        for lambda_cache in lambdas:
            for weighted_penalty in (False, True):
                rows.append(
                    _simulate(
                        logits,
                        attention_mask,
                        expert_costs,
                        top_k=top_k,
                        capacity=capacity,
                        lambda_cache=lambda_cache,
                        weighted_penalty=weighted_penalty,
                        original_scores=logits,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "heterogeneous_cost_frontier.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")

    metadata = {
        "trace_path": str(trace_path),
        "capacities": capacities,
        "lambdas": lambdas,
        "cost_profile": cost_profile,
        "expert_cost_min": float(expert_costs.min()),
        "expert_cost_max": float(expert_costs.max()),
        "expert_cost_mean": float(expert_costs.mean()),
        "expert_costs": [float(value) for value in expert_costs.tolist()],
        "top_k": top_k,
        "note": (
            "Trace-level synthetic heterogeneous-cost proxy; router KL is a quality "
            "proxy and costs are not device-calibrated."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _parse_csv_numbers(
    raw: str,
    cast: type[int] | type[float],
) -> tuple[int, ...] | tuple[float, ...]:
    return tuple(cast(item.strip()) for item in raw.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-level heterogeneous-cost frontier.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--capacities", default="8,16,32")
    parser.add_argument("--lambdas", default="0.01,0.03,0.1,0.3")
    parser.add_argument("--cost-profile", default="tiered")
    args = parser.parse_args()

    run(
        trace_path=args.trace,
        output_dir=args.output_dir,
        capacities=_parse_csv_numbers(args.capacities, int),
        lambdas=_parse_csv_numbers(args.lambdas, float),
        cost_profile=args.cost_profile,
    )


if __name__ == "__main__":
    main()
