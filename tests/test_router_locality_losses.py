from __future__ import annotations

import torch

from moe_eviction.models.train_router import (
    _effective_batch_size,
    _lagged_symmetric_kl_loss,
    _linear_warmup_multiplier,
    _melinoe_cache_simulation_loss,
    _melinoe_rank_matching_loss,
    _register_reference_router_hooks,
    _select_training_batch,
    _temporal_reuse_loss,
    _temporal_symmetric_kl_loss,
    _working_set_entropy_loss,
)


def test_temporal_reuse_loss_is_smaller_when_topk_repeats() -> None:
    attention_mask = torch.ones((1, 3), dtype=torch.long)
    repeated = torch.tensor([[[5.0, 1.0], [5.0, 1.0], [5.0, 1.0]]])
    alternating = torch.tensor([[[5.0, 1.0], [1.0, 5.0], [5.0, 1.0]]])

    repeated_loss = _temporal_reuse_loss((repeated,), attention_mask, top_k=1, lag=1)
    alternating_loss = _temporal_reuse_loss((alternating,), attention_mask, top_k=1, lag=1)

    assert repeated_loss < alternating_loss


def test_temporal_reuse_loss_uses_log_sequence_reuse_mass() -> None:
    attention_mask = torch.ones((1, 3), dtype=torch.long)
    logits = torch.tensor([[[5.0, 1.0], [5.0, 1.0], [5.0, 1.0]]])

    loss = _temporal_reuse_loss((logits,), attention_mask, top_k=1, lag=1)
    probs = torch.softmax(logits, dim=-1)
    expected = -torch.log(probs[0, 1:, 0].mean())

    assert torch.allclose(loss, expected)


def test_temporal_symmetric_kl_is_zero_for_identical_adjacent_logits() -> None:
    attention_mask = torch.ones((1, 2), dtype=torch.long)
    logits = torch.tensor([[[2.0, 1.0], [2.0, 1.0]]])

    loss = _temporal_symmetric_kl_loss((logits,), attention_mask)

    assert loss.item() == 0.0


def test_lagged_symmetric_kl_penalizes_longer_horizon_drift() -> None:
    attention_mask = torch.ones((1, 4), dtype=torch.long)
    stable = torch.tensor([[[3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0]]])
    drift = torch.tensor([[[3.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 3.0]]])

    stable_loss = _lagged_symmetric_kl_loss((stable,), attention_mask, lag_steps=(1, 2))
    drift_loss = _lagged_symmetric_kl_loss((drift,), attention_mask, lag_steps=(1, 2))

    assert torch.allclose(stable_loss, torch.tensor(0.0), atol=1e-7)
    assert drift_loss > stable_loss


def test_working_set_entropy_is_smaller_for_concentrated_router() -> None:
    attention_mask = torch.ones((1, 2), dtype=torch.long)
    concentrated = torch.tensor([[[5.0, 0.0], [5.0, 0.0]]])
    spread = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])

    concentrated_loss = _working_set_entropy_loss((concentrated,), attention_mask, window_size=2)
    spread_loss = _working_set_entropy_loss((spread,), attention_mask, window_size=2)

    assert concentrated_loss < spread_loss


def test_melinoe_cache_loss_uses_topk_requests_and_attention_mask() -> None:
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    logits = torch.tensor([[[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]])

    loss = _melinoe_cache_simulation_loss(
        (logits,),
        attention_mask,
        top_k=1,
        capacity=1.0,
        gamma=1.0,
        initialization="zero_fill",
    )

    assert torch.allclose(loss, torch.tensor(0.5))


def test_melinoe_cache_loss_backpropagates_to_router_logits() -> None:
    attention_mask = torch.ones((1, 3), dtype=torch.long)
    logits = torch.tensor(
        [[[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]],
        requires_grad=True,
    )

    loss = _melinoe_cache_simulation_loss(
        (logits,),
        attention_mask,
        top_k=1,
        capacity=1.0,
        gamma=0.9,
        initialization="zero_fill",
    )
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0.0


def test_melinoe_rank_matching_loss_is_zero_when_reference_order_is_preserved() -> None:
    attention_mask = torch.ones((1, 1), dtype=torch.long)
    current = torch.tensor([[[8.0, -8.0]]])
    reference = torch.tensor([[[3.0, 1.0]]])

    loss = _melinoe_rank_matching_loss((current,), (reference,), attention_mask, margin=0.1)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_melinoe_rank_matching_loss_penalizes_reference_order_inversions() -> None:
    attention_mask = torch.ones((1, 1), dtype=torch.long)
    current = torch.tensor([[[0.0, 5.0]]])
    reference = torch.tensor([[[5.0, 0.0]]])

    loss = _melinoe_rank_matching_loss((current,), (reference,), attention_mask, margin=0.1)

    current_probs = torch.softmax(current, dim=-1)
    expected = torch.relu(
        torch.tensor(0.1) - (current_probs[0, 0, 0] - current_probs[0, 0, 1])
    )
    assert torch.allclose(loss, expected)


def test_linear_warmup_multiplier_reaches_one() -> None:
    assert _linear_warmup_multiplier(step=0, warmup_steps=4) == 0.25
    assert _linear_warmup_multiplier(step=3, warmup_steps=4) == 1.0
    assert _linear_warmup_multiplier(step=10, warmup_steps=4) == 1.0
    assert _linear_warmup_multiplier(step=0, warmup_steps=0) == 1.0


def test_effective_batch_size_defaults_to_full_prompt_set() -> None:
    assert _effective_batch_size({}, num_prompts=8) == 8
    assert _effective_batch_size({"batch_size": 4}, num_prompts=8) == 4
    assert _effective_batch_size({"batch_size": 16}, num_prompts=8) == 8


def test_select_training_batch_keeps_inputs_and_labels_aligned() -> None:
    torch.manual_seed(0)
    encoded = {
        "input_ids": torch.arange(20).reshape(5, 4),
        "attention_mask": torch.ones((5, 4), dtype=torch.long),
    }
    labels = encoded["input_ids"] + 100

    batch, batch_labels = _select_training_batch(encoded, labels, batch_size=3)

    assert batch["input_ids"].shape == (3, 4)
    assert batch["attention_mask"].shape == (3, 4)
    assert torch.equal(batch_labels, batch["input_ids"] + 100)


def test_reference_router_hook_uses_frozen_gate_on_current_hidden() -> None:
    class TinyRouter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Linear(2, 2, bias=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.gate(hidden_states)

    model = TinyRouter()
    with torch.no_grad():
        model.gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    handles, reference_logits = _register_reference_router_hooks(model, ("gate.weight",))
    try:
        with torch.no_grad():
            model.gate.weight.copy_(torch.tensor([[10.0, 0.0], [0.0, 10.0]]))
        hidden_states = torch.tensor([[2.0, 3.0]])

        current_logits = model(hidden_states)

        assert torch.equal(current_logits, torch.tensor([[20.0, 30.0]]))
        assert len(reference_logits) == 1
        assert torch.equal(reference_logits[0], torch.tensor([[2.0, 3.0]]))
    finally:
        for handle in handles:
            handle.remove()
