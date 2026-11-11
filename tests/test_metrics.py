import pytest

from moe_eviction.metrics import simulate_lru


def test_trace_metrics_and_weighted_cost() -> None:
    metrics = simulate_lru(
        requests=[(1, 2), (2, 3), (1, 3)],
        capacity=2,
        expert_costs={1: 2.0, 2: 1.0, 3: 4.0},
    )

    assert metrics.requests == 6
    assert metrics.hits == 2
    assert metrics.misses == 4
    assert metrics.hit_rate == 2 / 6
    assert metrics.weighted_miss_cost == 9.0


def test_missing_expert_cost_defaults_to_unit_cost() -> None:
    metrics = simulate_lru(
        requests=[(1, 2), (3,)],
        capacity=1,
        expert_costs={1: 2.0},
    )

    assert metrics.misses == 3
    assert metrics.weighted_miss_cost == 4.0


def test_negative_expert_cost_is_rejected_even_if_not_missed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        simulate_lru(
            requests=[(1,)],
            capacity=1,
            expert_costs={2: -1.0},
        )
