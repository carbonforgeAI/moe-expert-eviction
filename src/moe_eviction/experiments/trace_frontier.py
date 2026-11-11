from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class FrontierRow:
    method: str
    lambda_cache: float
    train_steps: int
    router_kl: float
    hard_misses: int
    hard_miss_rate: float
    weighted_miss_cost: float
    requests: int


def _load_trace(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    data = np.load(path)
    logits = torch.from_numpy(data["router_logits"]).float()
    attention_mask = torch.from_numpy(data["attention_mask"]).bool()
    selected = data["selected_experts"]
    return logits, attention_mask, int(selected.shape[-1])


def _simulate_topk(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int,
    capacity: int,
    lambda_cache: float = 0.0,
    original_scores: torch.Tensor | None = None,
) -> tuple[int, float, int, float]:
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
                    adjusted = adjusted - lambda_cache * cold
                if original_scores is not None:
                    original_probs = torch.softmax(
                        original_scores[batch_index, layer_index, token_index],
                        dim=-1,
                    )
                    adjusted_log_probs = torch.log_softmax(adjusted, dim=-1)
                    kl_sum += float(
                        F.kl_div(
                            adjusted_log_probs,
                            original_probs,
                            reduction="sum",
                        ).detach()
                    )
                    kl_count += 1

                selected = torch.topk(adjusted, k=top_k, dim=-1).indices.tolist()
                before = set(resident)
                misses = [expert for expert in selected if expert not in before]
                hard_misses += len(misses)
                weighted_cost += float(len(misses))
                requests += len(selected)

                for expert in selected:
                    if expert in resident:
                        resident.remove(expert)
                    resident.append(expert)
                    while len(resident) > capacity:
                        resident.pop(0)

    return hard_misses, weighted_cost, requests, kl_sum / kl_count if kl_count else 0.0


def _router_kl(original_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    original_probs = torch.softmax(original_logits, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)
    kl = F.kl_div(candidate_log_probs, original_probs, reduction="batchmean")
    return float(kl.detach().cpu())


def _expected_cold_cost(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int,
    capacity: int,
) -> torch.Tensor:
    batch_size, num_layers, seq_len, _num_experts = logits.shape
    probabilities = torch.softmax(logits, dim=-1)
    losses: list[torch.Tensor] = []
    caches: list[list[list[int]]] = [
        [[] for _layer in range(num_layers)] for _batch in range(batch_size)
    ]

    for token_index in range(seq_len):
        for batch_index in range(batch_size):
            if not attention_mask[batch_index, token_index].item():
                continue
            for layer_index in range(num_layers):
                resident = caches[batch_index][layer_index]
                cold = torch.ones(logits.shape[-1], dtype=logits.dtype)
                if resident:
                    cold[torch.tensor(resident)] = 0.0
                losses.append((probabilities[batch_index, layer_index, token_index] * cold).sum())

                selected = torch.topk(
                    logits[batch_index, layer_index, token_index].detach(),
                    k=top_k,
                    dim=-1,
                ).indices.tolist()
                for expert in selected:
                    if expert in resident:
                        resident.remove(expert)
                    resident.append(expert)
                    while len(resident) > capacity:
                        resident.pop(0)

    return torch.stack(losses).mean()


def _evaluate(
    method: str,
    original_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int,
    capacity: int,
    lambda_cache: float,
    train_steps: int,
) -> FrontierRow:
    hard_misses, weighted_cost, requests, _online_kl = _simulate_topk(
        candidate_logits,
        attention_mask,
        top_k=top_k,
        capacity=capacity,
    )
    return FrontierRow(
        method=method,
        lambda_cache=lambda_cache,
        train_steps=train_steps,
        router_kl=_router_kl(original_logits, candidate_logits),
        hard_misses=hard_misses,
        hard_miss_rate=hard_misses / requests if requests else 0.0,
        weighted_miss_cost=weighted_cost,
        requests=requests,
    )


def run_frontier(
    trace_path: Path,
    output_dir: Path,
    capacity: int,
    lambdas: tuple[float, ...],
    train_steps: int,
    learning_rate: float,
    kl_weight: float,
) -> None:
    original_logits, attention_mask, top_k = _load_trace(trace_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[FrontierRow] = [
        _evaluate(
            "vanilla",
            original_logits,
            original_logits,
            attention_mask,
            top_k,
            capacity,
            lambda_cache=0.0,
            train_steps=0,
        )
    ]

    for lambda_cache in lambdas:
        posthoc_logits = original_logits.clone()
        hard_misses, weighted_cost, requests, online_kl = _simulate_topk(
            posthoc_logits,
            attention_mask,
            top_k=top_k,
            capacity=capacity,
            lambda_cache=lambda_cache,
            original_scores=original_logits,
        )
        rows.append(
            FrontierRow(
                method="posthoc_cache_bias",
                lambda_cache=lambda_cache,
                train_steps=0,
                router_kl=online_kl,
                hard_misses=hard_misses,
                hard_miss_rate=hard_misses / requests if requests else 0.0,
                weighted_miss_cost=weighted_cost,
                requests=requests,
            )
        )

    for lambda_cache in lambdas:
        bias = torch.zeros(
            original_logits.shape[1],
            original_logits.shape[-1],
            dtype=original_logits.dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.AdamW([bias], lr=learning_rate)
        for _step in range(train_steps):
            optimizer.zero_grad(set_to_none=True)
            candidate_logits = original_logits + bias[None, :, None, :]
            cache_cost = _expected_cold_cost(candidate_logits, attention_mask, top_k, capacity)
            kl = F.kl_div(
                torch.log_softmax(candidate_logits, dim=-1),
                torch.softmax(original_logits, dim=-1),
                reduction="batchmean",
            )
            loss = lambda_cache * cache_cost + kl_weight * kl
            loss.backward()
            optimizer.step()

        candidate_logits = original_logits + bias.detach()[None, :, None, :]
        rows.append(
            _evaluate(
                "trained_layer_expert_bias",
                original_logits,
                candidate_logits,
                attention_mask,
                top_k,
                capacity,
                lambda_cache=lambda_cache,
                train_steps=train_steps,
            )
        )

    metrics_path = output_dir / "frontier.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")

    metadata = {
        "trace_path": str(trace_path),
        "capacity": capacity,
        "top_k": top_k,
        "lambdas": lambdas,
        "train_steps": train_steps,
        "learning_rate": learning_rate,
        "kl_weight": kl_weight,
        "note": "Trace-level proxy experiment; router KL is a quality proxy, not LM NLL.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trace-level cache/quality proxy frontier.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    args = parser.parse_args()

    run_frontier(
        trace_path=args.trace,
        output_dir=args.output_dir,
        capacity=args.capacity,
        lambdas=tuple(args.lambdas),
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
    )


if __name__ == "__main__":
    main()
