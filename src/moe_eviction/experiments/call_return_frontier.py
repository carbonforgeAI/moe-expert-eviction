from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from moe_eviction.experiments.heterogeneous_cost_frontier import _cost_profile


@dataclass(frozen=True, slots=True)
class CallReturnRow:
    method: str
    capacity: int
    top_k: int
    policy: str
    option_length: int | None
    termination_threshold: float | None
    events: int
    requests: int
    hard_misses: int
    hard_miss_rate: float
    weighted_miss_cost: float
    controller_calls: int
    controller_call_rate: float
    option_switches: int
    switch_edit_distance: int
    mean_option_length: float
    option_lengths: tuple[int, ...]
    router_kl_to_vanilla: float
    router_logit_regret: float
    deliberation_cost: float
    switch_cost: float
    total_cost: float


def _load_trace(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    data = np.load(path)
    logits = torch.from_numpy(data["router_logits"]).float()
    attention_mask = torch.from_numpy(data["attention_mask"]).bool()
    selected = data["selected_experts"]
    return logits, attention_mask, int(selected.shape[-1])


def _expert_costs(num_experts: int, profile: str) -> torch.Tensor:
    if profile == "uniform":
        return torch.ones(num_experts, dtype=torch.float32)
    if profile == "measured":
        raise ValueError(
            "measured expert costs require an explicit measured profile; "
            "use uniform, tiered, or linear for this trace proxy"
        )
    return _cost_profile(num_experts, profile)


def _topk(logits: torch.Tensor, top_k: int) -> tuple[int, ...]:
    return tuple(int(value) for value in torch.topk(logits, k=top_k, dim=-1).indices.tolist())


def _edit_distance(previous: tuple[int, ...], current: tuple[int, ...]) -> int:
    return len(set(previous).symmetric_difference(current))


def _option_logit_regret(
    logits: torch.Tensor,
    selected: tuple[int, ...],
    fresh_topk: tuple[int, ...] | None = None,
) -> float:
    fresh = fresh_topk if fresh_topk is not None else _topk(logits, len(selected))
    fresh_score = float(logits[list(fresh)].sum().detach().cpu())
    selected_score = float(logits[list(selected)].sum().detach().cpu())
    return max(0.0, fresh_score - selected_score)


def _forced_option_kl(
    logits: torch.Tensor,
    selected: tuple[int, ...],
    fresh_topk: tuple[int, ...] | None = None,
) -> float:
    fresh = fresh_topk if fresh_topk is not None else _topk(logits, len(selected))
    if selected == fresh:
        return 0.0

    adjusted = logits.clone()
    high_score = logits.max()
    for offset, expert in enumerate(selected):
        adjusted[expert] = high_score + float(len(selected) - offset)

    original_probs = torch.softmax(logits, dim=-1)
    adjusted_log_probs = torch.log_softmax(adjusted, dim=-1)
    return float(F.kl_div(adjusted_log_probs, original_probs, reduction="sum").detach().cpu())


def _update_resident(resident: list[int], selected: tuple[int, ...], capacity: int) -> None:
    for expert in selected:
        if expert in resident:
            resident.remove(expert)
        resident.append(expert)
        while len(resident) > capacity:
            resident.pop(0)


def _load_cost(
    selected: tuple[int, ...],
    resident: list[int],
    expert_costs: torch.Tensor,
) -> float:
    before = set(resident)
    misses = [expert for expert in selected if expert not in before]
    return float(expert_costs[misses].sum()) if misses else 0.0


def _should_terminate(
    policy: str,
    *,
    age: int,
    option_length: int | None,
    termination_threshold: float | None,
    logits: torch.Tensor,
    current_option: tuple[int, ...] | None,
    fresh_topk: tuple[int, ...],
    resident: list[int],
    expert_costs: torch.Tensor,
    deliberation_cost_unit: float,
    switch_cost_unit: float,
    load_cost_weight: float,
) -> bool:
    if current_option is None:
        return True
    if policy == "polling":
        return True
    if policy == "fixed_duration":
        if option_length is None or option_length <= 0:
            raise ValueError("fixed_duration policy requires a positive option_length")
        return age >= option_length
    if policy == "threshold":
        if termination_threshold is None:
            raise ValueError("threshold policy requires termination_threshold")
        regret = _option_logit_regret(logits, current_option, fresh_topk)
        return regret > termination_threshold
    if policy == "oracle_one_step":
        continue_objective = _option_logit_regret(logits, current_option, fresh_topk)
        continue_objective += load_cost_weight * _load_cost(
            current_option,
            resident,
            expert_costs,
        )
        switch_objective = load_cost_weight * _load_cost(fresh_topk, resident, expert_costs)
        switch_objective += deliberation_cost_unit
        switch_objective += switch_cost_unit * _edit_distance(current_option, fresh_topk)
        return switch_objective < continue_objective
    raise ValueError(f"unsupported call-and-return policy: {policy}")


def simulate_call_return(
    router_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    top_k: int,
    cache_capacity: int,
    policy: str,
    method: str | None = None,
    option_length: int | None = None,
    termination_threshold: float | None = None,
    expert_costs: torch.Tensor | None = None,
    deliberation_cost_unit: float = 0.0,
    switch_cost_unit: float = 0.0,
    load_cost_weight: float = 1.0,
) -> CallReturnRow:
    if cache_capacity <= 0:
        raise ValueError("cache_capacity must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if router_logits.ndim != 4:
        raise ValueError("router_logits must have shape [batch, layer, token, expert]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, token]")
    batch_size, num_layers, seq_len, num_experts = router_logits.shape
    if tuple(attention_mask.shape) != (batch_size, seq_len):
        raise ValueError("attention_mask shape must match router_logits batch and token axes")
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed number of experts")

    costs = (
        expert_costs.detach().float().cpu()
        if expert_costs is not None
        else torch.ones(num_experts, dtype=torch.float32)
    )
    if costs.shape != (num_experts,):
        raise ValueError("expert_costs must have shape [num_experts]")
    if bool((costs < 0).any()):
        raise ValueError("expert_costs must be non-negative")

    logits = router_logits.detach().float().cpu()
    mask = attention_mask.detach().bool().cpu()
    caches: list[list[list[int]]] = [
        [[] for _layer in range(num_layers)] for _batch in range(batch_size)
    ]
    options: list[list[tuple[int, ...] | None]] = [
        [None for _layer in range(num_layers)] for _batch in range(batch_size)
    ]
    ages: list[list[int]] = [[0 for _layer in range(num_layers)] for _batch in range(batch_size)]

    events = 0
    requests = 0
    hard_misses = 0
    weighted_miss_cost = 0.0
    controller_calls = 0
    option_switches = 0
    switch_edit_distance = 0
    option_lengths: list[int] = []
    kl_sum = 0.0
    regret_sum = 0.0

    for token_index in range(seq_len):
        for batch_index in range(batch_size):
            if not mask[batch_index, token_index].item():
                continue
            for layer_index in range(num_layers):
                token_logits = logits[batch_index, layer_index, token_index]
                fresh_topk = _topk(token_logits, top_k)
                resident = caches[batch_index][layer_index]
                current_option = options[batch_index][layer_index]
                terminate = _should_terminate(
                    policy,
                    age=ages[batch_index][layer_index],
                    option_length=option_length,
                    termination_threshold=termination_threshold,
                    logits=token_logits,
                    current_option=current_option,
                    fresh_topk=fresh_topk,
                    resident=resident,
                    expert_costs=costs,
                    deliberation_cost_unit=deliberation_cost_unit,
                    switch_cost_unit=switch_cost_unit,
                    load_cost_weight=load_cost_weight,
                )

                if terminate:
                    controller_calls += 1
                    if current_option is not None:
                        option_lengths.append(ages[batch_index][layer_index])
                        edit = _edit_distance(current_option, fresh_topk)
                        switch_edit_distance += edit
                        if edit:
                            option_switches += 1
                    selected = fresh_topk
                    options[batch_index][layer_index] = selected
                    ages[batch_index][layer_index] = 0
                else:
                    if current_option is None:
                        raise RuntimeError("continuation without an active option")
                    selected = current_option

                before = set(resident)
                misses = [expert for expert in selected if expert not in before]
                hard_misses += len(misses)
                weighted_miss_cost += float(costs[misses].sum()) if misses else 0.0
                requests += len(selected)
                events += 1

                regret_sum += _option_logit_regret(token_logits, selected, fresh_topk)
                kl_sum += _forced_option_kl(token_logits, selected, fresh_topk)

                _update_resident(resident, selected, cache_capacity)
                ages[batch_index][layer_index] += 1

    for batch_index in range(batch_size):
        for layer_index in range(num_layers):
            if options[batch_index][layer_index] is not None:
                option_lengths.append(ages[batch_index][layer_index])

    deliberation_cost = deliberation_cost_unit * controller_calls
    switch_cost = switch_cost_unit * switch_edit_distance
    total_cost = load_cost_weight * weighted_miss_cost + deliberation_cost + switch_cost
    return CallReturnRow(
        method=method or policy,
        capacity=cache_capacity,
        top_k=top_k,
        policy=policy,
        option_length=option_length,
        termination_threshold=termination_threshold,
        events=events,
        requests=requests,
        hard_misses=hard_misses,
        hard_miss_rate=hard_misses / requests if requests else 0.0,
        weighted_miss_cost=weighted_miss_cost,
        controller_calls=controller_calls,
        controller_call_rate=controller_calls / events if events else 0.0,
        option_switches=option_switches,
        switch_edit_distance=switch_edit_distance,
        mean_option_length=sum(option_lengths) / len(option_lengths) if option_lengths else 0.0,
        option_lengths=tuple(option_lengths),
        router_kl_to_vanilla=kl_sum / events if events else 0.0,
        router_logit_regret=regret_sum / events if events else 0.0,
        deliberation_cost=deliberation_cost,
        switch_cost=switch_cost,
        total_cost=total_cost,
    )


def run_call_return_frontier(
    trace_path: Path,
    output_dir: Path,
    capacities: tuple[int, ...],
    fixed_durations: tuple[int, ...],
    thresholds: tuple[float, ...],
    cost_profile: str,
    deliberation_cost_unit: float,
    switch_cost_unit: float,
    load_cost_weight: float,
    include_oracle: bool,
) -> None:
    logits, attention_mask, top_k = _load_trace(trace_path)
    expert_costs = _expert_costs(logits.shape[-1], cost_profile)
    rows: list[CallReturnRow] = []

    for capacity in capacities:
        rows.append(
            simulate_call_return(
                logits,
                attention_mask,
                top_k=top_k,
                cache_capacity=capacity,
                policy="polling",
                method="polling",
                expert_costs=expert_costs,
                deliberation_cost_unit=deliberation_cost_unit,
                switch_cost_unit=switch_cost_unit,
                load_cost_weight=load_cost_weight,
            )
        )
        for duration in fixed_durations:
            rows.append(
                simulate_call_return(
                    logits,
                    attention_mask,
                    top_k=top_k,
                    cache_capacity=capacity,
                    policy="fixed_duration",
                    method=f"fixed_duration_{duration}",
                    option_length=duration,
                    expert_costs=expert_costs,
                    deliberation_cost_unit=deliberation_cost_unit,
                    switch_cost_unit=switch_cost_unit,
                    load_cost_weight=load_cost_weight,
                )
            )
        for threshold in thresholds:
            rows.append(
                simulate_call_return(
                    logits,
                    attention_mask,
                    top_k=top_k,
                    cache_capacity=capacity,
                    policy="threshold",
                    method=f"call_return_threshold_{threshold:g}",
                    termination_threshold=threshold,
                    expert_costs=expert_costs,
                    deliberation_cost_unit=deliberation_cost_unit,
                    switch_cost_unit=switch_cost_unit,
                    load_cost_weight=load_cost_weight,
                )
            )
        if include_oracle:
            rows.append(
                simulate_call_return(
                    logits,
                    attention_mask,
                    top_k=top_k,
                    cache_capacity=capacity,
                    policy="oracle_one_step",
                    method="oracle_one_step",
                    expert_costs=expert_costs,
                    deliberation_cost_unit=deliberation_cost_unit,
                    switch_cost_unit=switch_cost_unit,
                    load_cost_weight=load_cost_weight,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "call_return_frontier.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")

    metadata = {
        "trace_path": str(trace_path),
        "capacities": capacities,
        "fixed_durations": fixed_durations,
        "thresholds": thresholds,
        "cost_profile": cost_profile,
        "expert_cost_min": float(expert_costs.min()),
        "expert_cost_max": float(expert_costs.max()),
        "expert_cost_mean": float(expert_costs.mean()),
        "top_k": top_k,
        "deliberation_cost_unit": deliberation_cost_unit,
        "switch_cost_unit": switch_cost_unit,
        "load_cost_weight": load_cost_weight,
        "include_oracle": include_oracle,
        "note": (
            "Trace-level call-and-return proxy. Router KL/logit regret are quality "
            "proxies, not language-model NLL. Costs are measured only if the input "
            "profile came from a measured calibration."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-level call-and-return frontier.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--capacities", default="8")
    parser.add_argument("--fixed-durations", default="2,4,8,16")
    parser.add_argument("--thresholds", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--cost-profile", default="uniform")
    parser.add_argument("--deliberation-cost", type=float, default=0.0)
    parser.add_argument("--switch-cost", type=float, default=0.0)
    parser.add_argument("--load-cost-weight", type=float, default=1.0)
    parser.add_argument("--include-oracle", action="store_true")
    args = parser.parse_args()

    run_call_return_frontier(
        trace_path=args.trace,
        output_dir=args.output_dir,
        capacities=_parse_csv_ints(args.capacities),
        fixed_durations=_parse_csv_ints(args.fixed_durations),
        thresholds=_parse_csv_floats(args.thresholds),
        cost_profile=args.cost_profile,
        deliberation_cost_unit=args.deliberation_cost,
        switch_cost_unit=args.switch_cost,
        load_cost_weight=args.load_cost_weight,
        include_oracle=args.include_oracle,
    )


if __name__ == "__main__":
    main()
