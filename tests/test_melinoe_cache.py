import pytest
import torch

from moe_eviction.cache import (
    MelinoeSoftCache,
    melinoe_cache_simulation_loss,
    melinoe_rank_matching_loss,
    melinoe_soft_cache_states,
    topk_request_vectors,
)


def test_melinoe_zero_fill_allows_state_mass_to_grow_before_capacity() -> None:
    cache = MelinoeSoftCache(num_experts=4, capacity=3.0, gamma=0.5)

    first = cache.update(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    second = cache.update(torch.tensor([0.0, 1.0, 0.0, 0.0]))

    assert first.sum() == pytest.approx(1.0)
    assert second.sum() == pytest.approx(1.5)
    assert torch.allclose(second, torch.tensor([0.5, 1.0, 0.0, 0.0]))


def test_melinoe_zero_fill_normalizes_after_capacity_is_reached() -> None:
    cache = MelinoeSoftCache(num_experts=4, capacity=2.0, gamma=1.0)

    cache.update(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    second = cache.update(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    third = cache.update(torch.tensor([0.0, 0.0, 1.0, 0.0]))

    assert second.sum() == pytest.approx(2.0)
    assert third.sum() == pytest.approx(2.0)
    assert torch.allclose(third, torch.tensor([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0]))


def test_melinoe_uniform_initialization_uses_paper_normalizer_update() -> None:
    cache = MelinoeSoftCache(
        num_experts=4,
        capacity=2.0,
        gamma=0.5,
        initialization="uniform",
    )

    updated = cache.update(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    expected = torch.tensor([1.25, 0.25, 0.25, 0.25]) / 1.0
    assert updated.sum() == pytest.approx(2.0)
    assert torch.allclose(updated, expected)


def test_melinoe_cache_simulation_loss_uses_pre_step_cache_state() -> None:
    requests = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    loss = melinoe_cache_simulation_loss(
        requests,
        capacity=1.0,
        gamma=1.0,
        initialization="zero_fill",
    )

    assert loss == pytest.approx(0.5)


def test_melinoe_soft_cache_states_support_prefix_dimensions() -> None:
    requests = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
        ]
    )

    pre_states, post_states = melinoe_soft_cache_states(requests, capacity=1.0, gamma=1.0)

    assert pre_states.shape == requests.shape
    assert post_states.shape == requests.shape
    assert torch.allclose(pre_states[0, 0, 1], torch.tensor([1.0, 0.0]))
    assert torch.allclose(pre_states[1, 0, 1], torch.tensor([0.0, 1.0]))


def test_topk_request_vectors_builds_binary_requests() -> None:
    probs = torch.tensor([[[0.2, 0.7, 0.1], [0.5, 0.4, 0.1]]])

    requests = topk_request_vectors(probs, top_k=2)

    assert torch.equal(requests, torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]))


def test_topk_request_vectors_can_use_straight_through_gradients() -> None:
    probs = torch.tensor([[0.2, 0.7, 0.1]], requires_grad=True)

    requests = topk_request_vectors(probs, top_k=1, straight_through=True)
    loss = (requests * torch.tensor([[0.0, 0.0, 1.0]])).sum()
    loss.backward()

    assert torch.equal(requests.detach(), torch.tensor([[0.0, 1.0, 0.0]]))
    assert probs.grad is not None
    assert torch.equal(probs.grad, torch.tensor([[0.0, 0.0, 1.0]]))


def test_melinoe_rank_matching_loss_is_zero_when_margin_is_satisfied() -> None:
    base = torch.tensor([[[0.7, 0.2, 0.1]]])
    finetuned = torch.tensor([[[0.8, 0.1, 0.0]]])

    loss = melinoe_rank_matching_loss(base, finetuned, margin=0.1)

    assert loss == pytest.approx(0.0)


def test_melinoe_rank_matching_loss_penalizes_base_order_inversions() -> None:
    base = torch.tensor([[[0.7, 0.2]]])
    finetuned = torch.tensor([[[0.2, 0.7]]])

    loss = melinoe_rank_matching_loss(base, finetuned, margin=0.1)

    assert loss == pytest.approx(0.6)
