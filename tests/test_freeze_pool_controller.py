"""Pure (CPU, no GPU/model) tests for the M0 FreezePoolWindowController.

These lock the properties the dwell probe relies on:
- ``window_size == 1`` re-selects every token, so (since top_k <= capacity) the pool always
  contains the token's true top-k -> selections == vanilla;
- a boundary occurs exactly every ``window_size`` attended tokens; between boundaries the pool is
  frozen and incurs zero transfers; at a boundary transfers == |new_pool \\ old_pool|;
- ``window_size >= seq_len`` selects once -> a single one-time load of ``capacity`` per (batch,
  restricted layer);
- unrestricted layers pass through untouched; batches are independent.
"""

from __future__ import annotations

import torch

from moe_eviction.models.evaluate_working_set import FreezePoolWindowController


def _controller(*, batch, seq, window, capacity, top_k, restricted, mask=None):
    if mask is None:
        mask = torch.ones(batch, seq, dtype=torch.bool)
    return FreezePoolWindowController(
        attention_mask=mask,
        num_layers=1,
        capacity=capacity,
        top_k=top_k,
        window_size=window,
        restricted_layers=frozenset(restricted),
    )


# Deterministic 1-batch/4-token/4-expert trace: tokens 0,1 want expert 0; tokens 2,3 want expert 1.
_DET = torch.tensor([[[5.0, 0.0, 0.0, 0.0],
                      [5.0, 0.0, 0.0, 0.0],
                      [0.0, 5.0, 0.0, 0.0],
                      [0.0, 5.0, 0.0, 0.0]]])


def test_window_one_reselects_every_token_transfers() -> None:
    c = _controller(batch=1, seq=4, window=1, capacity=1, top_k=1, restricted={0})
    c.adjust(0, _DET.clone(), norm_topk_prob=False)
    # pools per token = [0],[0],[1],[1]; transfers = 1 + 0 + 1 + 0.
    assert c.transfers == 2


def test_window_two_batches_the_reselection() -> None:
    c = _controller(batch=1, seq=4, window=2, capacity=1, top_k=1, restricted={0})
    c.adjust(0, _DET.clone(), norm_topk_prob=False)
    # boundaries at ranks 0 and 2: pool0={0} (load) then pool1={1}; transfers = 1 + 1.
    assert c.transfers == 2


def test_window_ge_seq_is_single_one_time_load() -> None:
    c = _controller(batch=1, seq=4, window=4, capacity=1, top_k=1, restricted={0})
    out = c.adjust(0, _DET.clone(), norm_topk_prob=False)
    # one boundary: pool = the single most-popular expert over all 4 tokens (a 0-vs-1 tie).
    assert c.transfers == 1  # one-time load of `capacity` experts, once
    # every token is forced to the same frozen single-expert pool (whichever won the tie).
    selected = torch.topk(out, 1, dim=-1).indices.reshape(-1).tolist()
    assert len(set(selected)) == 1 and selected[0] in (0, 1)


def test_window_ge_seq_one_time_load_scales_with_capacity() -> None:
    # capacity=3 -> the single load transfers 3 experts; still exactly one boundary.
    c = _controller(batch=1, seq=4, window=4, capacity=3, top_k=1, restricted={0})
    c.adjust(0, _DET.clone(), norm_topk_prob=False)
    assert c.transfers == 3


def test_window_one_reproduces_vanilla_selection() -> None:
    # capacity >= top_k so the freshly re-selected pool always contains the token's true top-k.
    torch.manual_seed(0)
    grid = torch.randn(1, 5, 5)  # batch=1, seq=5, experts=5
    c = _controller(batch=1, seq=5, window=1, capacity=3, top_k=2, restricted={0})
    out = c.adjust(0, grid.clone(), norm_topk_prob=False)
    for token in range(5):
        vanilla = sorted(torch.topk(grid[0, token], 2).indices.tolist())
        frozen = sorted(torch.topk(out[0, token], 2).indices.tolist())
        assert frozen == vanilla


def test_unrestricted_layer_is_untouched() -> None:
    c = _controller(batch=1, seq=4, window=2, capacity=1, top_k=1, restricted={0})
    grid = _DET.clone()
    out = c.adjust(5, grid, norm_topk_prob=False)  # layer 5 is not restricted
    assert c.transfers == 0
    assert torch.equal(out, grid)


def test_batches_are_independent() -> None:
    # Two identical sequences -> exactly double the single-batch transfers.
    c = _controller(batch=2, seq=4, window=4, capacity=1, top_k=1, restricted={0})
    grid = _DET.repeat(2, 1, 1)  # [2, 4, 4]
    c.adjust(0, grid, norm_topk_prob=False)
    assert c.transfers == 2  # 1 one-time load per batch


def _causal(*, window, capacity, top_k, restricted, seq):
    return FreezePoolWindowController(
        attention_mask=torch.ones(1, seq, dtype=torch.bool),
        num_layers=1, capacity=capacity, top_k=top_k, window_size=window,
        restricted_layers=frozenset(restricted), pool_selection="causal_recency",
    )


def test_causal_window_one_covers_current_demand() -> None:
    # causal + window 1: pool = top-C over the *current* token -> contains its true top-k -> the
    # frozen routing reproduces the token's selection (delta NLL ~ 0), same as oracle window 1.
    torch.manual_seed(0)
    grid = torch.randn(1, 5, 5)
    c = _causal(window=1, capacity=3, top_k=2, restricted={0}, seq=5)
    out = c.adjust(0, grid.clone(), norm_topk_prob=False)
    for token in range(5):
        assert sorted(torch.topk(out[0, token], 2).indices.tolist()) == sorted(
            torch.topk(grid[0, token], 2).indices.tolist()
        )


def test_causal_trailing_differs_from_oracle_lookahead() -> None:
    # token 0 wants expert 0; tokens 1-3 want expert 3. At window 4 the single pool is chosen at
    # token 0: causal looks back (only token 0 -> expert 0), oracle looks ahead over the whole
    # sequence (e3 majority -> expert 3). token 3 routes to e0 (causal, stale) vs e3 (oracle).
    trace = torch.tensor([[[5.0, 0.0, 0.0, 0.0]] + [[0.0, 0.0, 0.0, 5.0]] * 3])
    def run(mode):
        c = FreezePoolWindowController(
            attention_mask=torch.ones(1, 4, dtype=torch.bool), num_layers=1, capacity=1,
            top_k=1, window_size=4, restricted_layers=frozenset({0}), pool_selection=mode,
        )
        return c.adjust(0, trace.clone(), norm_topk_prob=False)
    assert int(torch.topk(run("oracle")[0, 3], 1).indices) == 3
    assert int(torch.topk(run("causal_recency")[0, 3], 1).indices) == 0


def _mode(*, mode, window, capacity, top_k, restricted, seq):
    return FreezePoolWindowController(
        attention_mask=torch.ones(1, seq, dtype=torch.bool),
        num_layers=1, capacity=capacity, top_k=top_k, window_size=window,
        restricted_layers=frozenset(restricted), pool_selection=mode,
    )


def test_logit_selection_ranks_by_logit_not_count() -> None:
    # window of 3 tokens: e0 is the top-1 twice (count picks it), but e1 has the highest mean
    # logit without ever being the argmax (logit selection picks it) -- so the two modes diverge.
    c = _mode(mode="trailing_logit_topc", window=3, capacity=1, top_k=1, restricted={0}, seq=3)
    window = torch.tensor([[5.0, 4.0, 0.0], [5.0, 4.0, 0.0], [0.0, 4.0, 5.0]])
    assert c._select_pool(window) == [0]  # count: e0 is top-1 in 2 of 3 rows
    assert c._select_pool_logit(window) == [1]  # mean logit: e1 = 4.0 > e0 = 3.33 > e2 = 1.67


def test_current_logit_window_one_covers_current_demand() -> None:
    # current_logit_topc re-selecting every token picks the current token's top-C logits; since
    # capacity >= top_k that pool contains the token's true top-k, so routing == vanilla (dNLL ~ 0).
    torch.manual_seed(0)
    grid = torch.randn(1, 5, 5)
    c = _mode(mode="current_logit_topc", window=1, capacity=3, top_k=2, restricted={0}, seq=5)
    out = c.adjust(0, grid.clone(), norm_topk_prob=False)
    for token in range(5):
        assert sorted(torch.topk(out[0, token], 2).indices.tolist()) == sorted(
            torch.topk(grid[0, token], 2).indices.tolist()
        )


def test_trailing_logit_holds_pool_between_boundaries() -> None:
    # window 2: the pool is chosen at rank 0 from the trailing window {token 0} and frozen for
    # token 1 -- so token 1 routes within token 0's top-C, not its own. Transfers count once here.
    trace = torch.tensor([[[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]]])
    c = _mode(mode="trailing_logit_topc", window=2, capacity=1, top_k=1, restricted={0}, seq=2)
    out = c.adjust(0, trace.clone(), norm_topk_prob=False)
    # boundary at rank 0 loads {0}; token 1 (wants e3) is forced into the frozen {0}.
    assert int(torch.topk(out[0, 1], 1).indices) == 0
    assert c.transfers == 1


def test_unknown_pool_selection_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="pool_selection"):
        _mode(mode="not_a_mode", window=1, capacity=1, top_k=1, restricted={0}, seq=2)


def test_padding_positions_do_not_start_windows() -> None:
    # Token 3 is padding; only ranks over the 3 attended tokens {0,1,2} form windows.
    mask = torch.tensor([[True, True, True, False]])
    c = _controller(batch=1, seq=4, window=2, capacity=1, top_k=1, restricted={0}, mask=mask)
    c.adjust(0, _DET.clone(), norm_topk_prob=False)
    # attended = [0,1,2]; boundaries at rank 0 (tokens 0,1 -> {0}) and rank 2 (token 2 -> {1}).
    assert c.transfers == 2
