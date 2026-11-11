from __future__ import annotations

import torch

from moe_eviction.models.evaluate_cache_rerank import (
    CacheRerankController,
    CallReturnController,
    MoccEController,
    SparseMoeCallReturnController,
    _simulate_topk_with_costs,
    _sparse_moe_modules,
)


class CountingGate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.evaluated_rows = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.evaluated_rows += int(hidden_states.shape[0])
        return hidden_states


class ScaleExpert(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


class TinySparseMoe(torch.nn.Module):
    def __init__(self, *, top_k: int = 1) -> None:
        super().__init__()
        self.num_experts = 3
        self.top_k = top_k
        self.norm_topk_prob = False
        self.gate = CountingGate()
        self.experts = torch.nn.ModuleList(
            [ScaleExpert(1.0), ScaleExpert(10.0), ScaleExpert(100.0)]
        )


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router_block = TinySparseMoe()


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer()])


def test_mocce_max_rank_promotes_cached_expert_after_protected_prefix() -> None:
    controller = MoccEController(
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        num_layers=1,
        capacity=4,
        top_k=2,
        mode="max_rank",
        max_rank=5,
        min_rank=1,
        probability_threshold=1.0,
    )
    logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    selected = controller._select(logits, resident=[4])

    assert selected == [0, 4]


def test_mocce_max_rank_does_not_promote_outside_allowed_rank() -> None:
    controller = MoccEController(
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        num_layers=1,
        capacity=4,
        top_k=2,
        mode="max_rank",
        max_rank=4,
        min_rank=1,
        probability_threshold=1.0,
    )
    logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    selected = controller._select(logits, resident=[4])

    assert selected == [0, 1]


def test_mocce_cumsum_threshold_controls_allowed_rank() -> None:
    controller = MoccEController(
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        num_layers=1,
        capacity=4,
        top_k=2,
        mode="cumsum_threshold",
        max_rank=6,
        min_rank=1,
        probability_threshold=0.99,
    )
    logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    selected = controller._select(logits, resident=[3])

    assert selected == [0, 3]


def test_mocce_cache_prior_boosts_cached_expert_using_range_average() -> None:
    controller = MoccEController(
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        num_layers=1,
        capacity=4,
        top_k=2,
        mode="cache_prior",
        max_rank=6,
        min_rank=1,
        probability_threshold=1.0,
        prior_lambda=0.5,
    )
    logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    controller._current_range_average = 10.0

    selected = controller._select(logits, resident=[4])

    assert selected == [0, 4]


def test_weighted_cache_rerank_penalizes_expensive_cold_expert_more() -> None:
    controller = CacheRerankController(
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        num_layers=1,
        capacity=1,
        top_k=1,
        lambda_cache=0.2,
        expert_costs=torch.tensor([4.0, 1.0]),
        weighted_penalty=True,
    )
    logits = torch.tensor([[[5.0, 4.8]]])

    adjusted = controller.adjust(layer_index=0, output=logits, norm_topk_prob=False)

    assert torch.topk(adjusted, k=1, dim=-1).indices.item() == 1


def test_simulate_topk_with_costs_reports_weighted_miss_cost() -> None:
    scores = torch.tensor([[[[5.0, 4.0], [5.0, 4.0]]]])
    attention_mask = torch.ones((1, 2), dtype=torch.bool)

    hard_misses, weighted_cost, requests = _simulate_topk_with_costs(
        scores,
        attention_mask,
        top_k=1,
        capacity=1,
        expert_costs=torch.tensor([3.0, 1.0]),
    )

    assert hard_misses == 1
    assert weighted_cost == 3.0
    assert requests == 2


def test_call_return_gate_hook_reuses_option_on_continuation() -> None:
    controller = CallReturnController(
        attention_mask=torch.ones((1, 3), dtype=torch.bool),
        num_layers=1,
        capacity=1,
        top_k=1,
        mode="fixed_duration",
        option_length=3,
    )
    logits = torch.tensor([[[5.0, 1.0], [1.0, 5.0], [1.0, 5.0]]])
    scores = torch.ones((1, 3, 1))
    indices = torch.tensor([[[0], [1], [1]]])

    _adjusted_logits, _scores, adjusted_indices = controller.adjust(
        layer_index=0,
        output=(logits, scores, indices),
        norm_topk_prob=False,
    )
    controller.finalize()

    assert adjusted_indices.squeeze(-1).tolist() == [[0, 0, 0]]
    assert controller.controller_calls == 1
    assert controller.mean_option_length == 3.0


def test_sparse_moe_call_return_skips_gate_on_fixed_duration_continuation() -> None:
    block = TinySparseMoe()
    controller = SparseMoeCallReturnController(
        attention_mask=torch.ones((1, 3), dtype=torch.bool),
        num_layers=1,
        capacity=1,
        top_k=1,
        mode="fixed_duration",
        option_length=3,
    )
    hidden_states = torch.tensor(
        [[[9.0, 1.0, 0.0], [0.0, 9.0, 1.0], [0.0, 1.0, 9.0]]]
    )

    _output, router_logits = controller.forward_sparse_moe(
        block,
        layer_index=0,
        hidden_states=hidden_states,
    )
    controller.finalize()

    assert block.gate.evaluated_rows == 1
    assert controller.gate_evaluated_tokens == 1
    assert controller.controller_calls == 1
    assert controller.mean_option_length == 3.0
    selected = torch.topk(router_logits.reshape(1, 3, 3), k=1, dim=-1).indices.squeeze(-1)
    assert selected.tolist() == [[0, 0, 0]]


def test_sparse_moe_call_return_restricted_softmax_scores_option_subset() -> None:
    block = TinySparseMoe(top_k=2)
    controller = SparseMoeCallReturnController(
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        num_layers=1,
        capacity=2,
        top_k=2,
        mode="fixed_duration",
        option_length=2,
        continuation_weighting="restricted_softmax",
    )
    hidden_states = torch.tensor([[[5.0, 4.0, 0.0], [1.0, 10.0, 100.0]]])

    output, _router_logits = controller.forward_sparse_moe(
        block,
        layer_index=0,
        hidden_states=hidden_states,
    )

    restricted_weights = torch.softmax(torch.tensor([1.0, 10.0]), dim=-1)
    expected = (
        hidden_states[0, 1] * 1.0 * restricted_weights[0]
        + hidden_states[0, 1] * 10.0 * restricted_weights[1]
    )
    assert block.gate.evaluated_rows == 1
    assert torch.allclose(output[0, 1], expected, atol=1e-5)


def test_sparse_moe_call_return_stored_weights_preserves_boundary_weights() -> None:
    block = TinySparseMoe(top_k=2)
    controller = SparseMoeCallReturnController(
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        num_layers=1,
        capacity=2,
        top_k=2,
        mode="fixed_duration",
        option_length=2,
        continuation_weighting="stored_weights",
    )
    hidden_states = torch.tensor([[[5.0, 4.0, 0.0], [1.0, 10.0, 100.0]]])

    output, _router_logits = controller.forward_sparse_moe(
        block,
        layer_index=0,
        hidden_states=hidden_states,
    )

    boundary_probs = torch.softmax(hidden_states[0, 0], dim=-1)
    stored_weights, _stored_indices = torch.topk(boundary_probs, k=2)
    expected = (
        hidden_states[0, 1] * 1.0 * stored_weights[0]
        + hidden_states[0, 1] * 10.0 * stored_weights[1]
    )
    assert block.gate.evaluated_rows == 1
    assert torch.allclose(output[0, 1], expected, atol=1e-5)


def test_sparse_moe_module_discovery_is_not_tied_to_mlp_name() -> None:
    model = TinyModel()

    modules = _sparse_moe_modules(model)

    assert len(modules) == 1
    layer_index, _parent, child_name, child = modules[0]
    assert layer_index == 0
    assert child_name == "router_block"
    assert child is model.layers[0].router_block


def test_sparse_moe_polling_gates_every_active_token() -> None:
    block = TinySparseMoe()
    controller = SparseMoeCallReturnController(
        attention_mask=torch.ones((1, 3), dtype=torch.bool),
        num_layers=1,
        capacity=1,
        top_k=1,
        mode="polling",
    )
    hidden_states = torch.tensor(
        [[[9.0, 1.0, 0.0], [0.0, 9.0, 1.0], [0.0, 1.0, 9.0]]]
    )

    _output, router_logits = controller.forward_sparse_moe(
        block,
        layer_index=0,
        hidden_states=hidden_states,
    )

    assert block.gate.evaluated_rows == 3
    assert controller.gate_evaluated_tokens == 3
    assert controller.controller_calls == 3
    selected = torch.topk(router_logits.reshape(1, 3, 3), k=1, dim=-1).indices.squeeze(-1)
    assert selected.tolist() == [[0, 1, 2]]
