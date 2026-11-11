from __future__ import annotations

import torch

from moe_eviction.experiments.call_return_frontier import simulate_call_return


def test_call_return_continuation_does_not_call_option_policy() -> None:
    logits = torch.tensor([[[[5.0, 1.0, 0.0], [1.0, 5.0, 0.0], [0.0, 1.0, 5.0]]]])
    attention_mask = torch.ones((1, 3), dtype=torch.bool)

    row = simulate_call_return(
        logits,
        attention_mask,
        top_k=1,
        cache_capacity=1,
        policy="fixed_duration",
        option_length=3,
    )

    assert row.controller_calls == 1
    assert row.option_lengths == (3,)
    assert row.hard_misses == 1
    assert row.mean_option_length == 3.0
    assert row.router_logit_regret > 0.0


def test_polling_calls_policy_once_per_active_token_layer_decision() -> None:
    logits = torch.zeros((1, 2, 3, 4))
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    row = simulate_call_return(
        logits,
        attention_mask,
        top_k=2,
        cache_capacity=2,
        policy="polling",
    )

    assert row.events == 4
    assert row.controller_calls == 4
    assert row.controller_call_rate == 1.0
    assert row.mean_option_length == 1.0


def test_fixed_duration_option_lengths_match_controller_calls() -> None:
    logits = torch.zeros((1, 1, 5, 3))
    attention_mask = torch.ones((1, 5), dtype=torch.bool)

    row = simulate_call_return(
        logits,
        attention_mask,
        top_k=1,
        cache_capacity=1,
        policy="fixed_duration",
        option_length=2,
    )

    assert row.controller_calls == 3
    assert row.option_lengths == (2, 2, 1)


def test_cache_miss_accounting_matches_when_selected_sets_match() -> None:
    logits = torch.tensor([[[[5.0, 4.0], [5.0, 4.0], [5.0, 4.0]]]])
    attention_mask = torch.ones((1, 3), dtype=torch.bool)

    polling = simulate_call_return(
        logits,
        attention_mask,
        top_k=1,
        cache_capacity=1,
        policy="polling",
    )
    fixed = simulate_call_return(
        logits,
        attention_mask,
        top_k=1,
        cache_capacity=1,
        policy="fixed_duration",
        option_length=8,
    )

    assert fixed.hard_misses == polling.hard_misses
    assert fixed.weighted_miss_cost == polling.weighted_miss_cost
    assert fixed.requests == polling.requests


def test_deliberation_cost_is_paid_only_at_boundaries() -> None:
    logits = torch.zeros((1, 1, 3, 2))
    attention_mask = torch.ones((1, 3), dtype=torch.bool)

    row = simulate_call_return(
        logits,
        attention_mask,
        top_k=1,
        cache_capacity=1,
        policy="fixed_duration",
        option_length=3,
        deliberation_cost_unit=2.5,
    )

    assert row.controller_calls == 1
    assert row.deliberation_cost == 2.5
    assert row.total_cost == row.weighted_miss_cost + 2.5


def test_weighted_load_cost_matches_hand_computed_example() -> None:
    logits = torch.tensor([[[[5.0, 4.0], [4.0, 5.0]]]])
    attention_mask = torch.ones((1, 2), dtype=torch.bool)

    row = simulate_call_return(
        logits,
        attention_mask,
        top_k=1,
        cache_capacity=1,
        policy="polling",
        expert_costs=torch.tensor([3.0, 5.0]),
    )

    assert row.hard_misses == 2
    assert row.weighted_miss_cost == 8.0


def test_high_threshold_persists_without_temporal_smoothness_loss() -> None:
    logits = torch.tensor([[[[5.0, 1.0], [1.0, 5.0], [1.0, 5.0]]]])
    attention_mask = torch.ones((1, 3), dtype=torch.bool)

    row = simulate_call_return(
        logits,
        attention_mask,
        top_k=1,
        cache_capacity=1,
        policy="threshold",
        termination_threshold=100.0,
    )

    assert row.controller_calls == 1
    assert row.mean_option_length == 3.0
