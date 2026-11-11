import torch

from moe_eviction.models.evaluate_working_set import (
    WorkingSetController,
    dynamic_admit,
    hot_sets,
    per_layer_popularity,
    restrict_to_set,
    select_cheap_layers,
)


def test_select_cheap_layers_picks_lowest_delta() -> None:
    rows = [
        {"layer": 0, "delta_nll": 0.50},
        {"layer": 1, "delta_nll": 0.02},
        {"layer": 2, "delta_nll": 0.30},
        {"layer": 3, "delta_nll": 0.01},
        {"layer": 4, "delta_nll": 0.09},
    ]
    # cheapest three by delta_nll are layers 3, 1, 4; returned sorted ascending.
    assert select_cheap_layers(rows, 3) == [1, 3, 4]


def test_select_cheap_layers_breaks_ties_by_layer_index() -> None:
    rows = [
        {"layer": 5, "delta_nll": 0.1},
        {"layer": 2, "delta_nll": 0.1},
        {"layer": 9, "delta_nll": 0.9},
    ]
    assert select_cheap_layers(rows, 1) == [2]  # tie on delta -> lower layer index wins


def test_per_layer_popularity_counts_topk() -> None:
    # 1 batch, 1 layer, 3 tokens, 4 experts; top_k=2. Last token is padding.
    logits = torch.tensor(
        [[[[5.0, 4.0, 1.0, 0.0], [0.0, 1.0, 5.0, 4.0], [9.0, 9.0, 9.0, 9.0]]]]
    )
    mask = torch.tensor([[1, 1, 0]])
    counts = per_layer_popularity(logits, mask, top_k=2)
    assert counts.shape == (1, 4)
    # token0 -> {0,1}, token1 -> {2,3}; token2 ignored (pad).
    assert counts[0].tolist() == [1, 1, 1, 1]


def test_per_layer_popularity_concentrates() -> None:
    logits = torch.tensor([[[[5.0, 4.0, 1.0, 0.0], [5.0, 4.0, 1.0, 0.0]]]])
    mask = torch.tensor([[1, 1]])
    counts = per_layer_popularity(logits, mask, top_k=2)
    assert counts[0].tolist() == [2, 2, 0, 0]  # experts 0,1 always selected


def test_hot_sets_topc() -> None:
    popularity = torch.tensor([[2, 9, 0, 7, 1]])  # one layer, 5 experts
    hot = hot_sets(popularity, capacity=2)
    assert hot[0].tolist() == [1, 3]  # the two most popular, sorted


def test_restrict_to_set_masks_non_resident() -> None:
    logits = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0])
    out = restrict_to_set(logits, torch.tensor([1, 3]))
    assert float(out[1]) == 1.0 and float(out[3]) == 5.0
    assert float(out[0]) < -1000 and float(out[2]) < -1000 and float(out[4]) < -1000
    # top-2 of the restricted logits are exactly the resident set.
    assert sorted(torch.topk(out, 2).indices.tolist()) == [1, 3]


def _controller(hot, restricted, top_k=2, seq=2, batch=1):
    return WorkingSetController(
        hot_sets=hot,
        restricted_layers=frozenset(restricted),
        top_k=top_k,
        batch_size=batch,
        seq_len=seq,
    )


def test_controller_restricts_only_targeted_layers() -> None:
    # gate output is the OlmoeTopKRouter tuple (logits, scores, indices) of shape [batch*seq, E].
    num_experts = 5
    logits = torch.randn(2, num_experts)  # batch*seq = 2 rows
    probs = torch.softmax(logits, dim=-1)
    scores, indices = torch.topk(probs, 2, dim=-1)
    hot = {0: torch.tensor([1, 3]), 1: torch.tensor([1, 3])}
    controller = _controller(hot, restricted={0})

    # Layer 0 is restricted -> indices become the resident set {1,3}.
    out0_logits, out0_scores, out0_indices = controller.adjust(
        0, (logits, scores.to(logits.dtype), indices), norm_topk_prob=False
    )
    for row in range(2):
        assert sorted(out0_indices[row].tolist()) == [1, 3]
    # Layer 1 is NOT restricted -> passthrough unchanged.
    out1 = controller.adjust(1, (logits, scores.to(logits.dtype), indices), norm_topk_prob=False)
    assert torch.equal(out1[0], logits)
    assert torch.equal(out1[2], indices)


def test_controller_tensor_passthrough_for_unrestricted_layer() -> None:
    logits = torch.randn(2, 4)
    controller = _controller({0: torch.tensor([0, 1])}, restricted={0})
    # Bare-tensor gate output on an unrestricted layer returns unchanged.
    assert torch.equal(controller.adjust(5, logits, norm_topk_prob=False), logits)


# logits -> softmax probs: expert3 p~0.636, expert4 p~0.234, expert0 p~0.086.
_DYN_LOGITS = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0])


def test_dynamic_admit_threshold_zero_recovers_true_topk() -> None:
    # Admit every miss -> behaves like vanilla (true top-k = {3,4}); both are transfers.
    _masked, selected, transfers = dynamic_admit(
        _DYN_LOGITS, resident=[0, 1], top_k=2, threshold=0.0
    )
    assert sorted(selected) == [3, 4]
    assert transfers == 2


def test_dynamic_admit_threshold_one_restricts_to_resident() -> None:
    # Admit nothing -> pick best of the resident pool {0,1}; zero transfers.
    _masked, selected, transfers = dynamic_admit(
        _DYN_LOGITS, resident=[0, 1], top_k=2, threshold=1.0
    )
    assert sorted(selected) == [0, 1]
    assert transfers == 0


def test_dynamic_admit_partial_admits_high_confidence_only() -> None:
    # threshold 0.3 admits expert3 (p~0.64) but not expert4 (p~0.23); slot falls back to best
    # resident (expert0, logit 3).
    _masked, selected, transfers = dynamic_admit(
        _DYN_LOGITS, resident=[0, 1], top_k=2, threshold=0.3
    )
    assert sorted(selected) == [0, 3]
    assert transfers == 1  # only expert3 was admitted/non-resident
