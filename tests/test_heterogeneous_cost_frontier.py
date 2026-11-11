from __future__ import annotations

import torch

from moe_eviction.experiments.heterogeneous_cost_frontier import _cost_profile, _simulate


def test_cost_profile_is_heterogeneous() -> None:
    costs = _cost_profile(8, "tiered")

    assert float(costs.min()) == 1.0
    assert float(costs.max()) == 4.0


def test_uniform_cost_profile_is_explicit() -> None:
    costs = _cost_profile(4, "uniform")

    assert costs.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_weighted_penalty_can_reduce_weighted_cost_more_than_unit_penalty() -> None:
    logits = torch.tensor([[[[5.0, 4.9, 4.8, 1.0], [5.0, 4.9, 4.8, 1.0]]]])
    attention_mask = torch.ones((1, 2), dtype=torch.bool)
    expert_costs = torch.tensor([4.0, 1.0, 1.0, 1.0])

    unit = _simulate(
        logits,
        attention_mask,
        expert_costs,
        top_k=1,
        capacity=1,
        lambda_cache=0.2,
        weighted_penalty=False,
        original_scores=logits,
    )
    weighted = _simulate(
        logits,
        attention_mask,
        expert_costs,
        top_k=1,
        capacity=1,
        lambda_cache=0.2,
        weighted_penalty=True,
        original_scores=logits,
    )

    assert weighted.weighted_miss_cost < unit.weighted_miss_cost
