"""Pure-CPU tests for the wall-clock offload harness (no GPU; tiny random OLMoE, no downloads)."""

import json

import pytest
import torch
import torch.nn.functional as F

from moe_eviction.experiments.offload_wallclock import (
    LayerPolicy,
    OffloadMoeBlock,
    PinnedExpertStore,
    recency_pool,
    restricted_selection,
    run,
    slabs_from_block,
)

E, C, K, HID, INT = 16, 8, 4, 32, 24
CPU = torch.device("cpu")


def _slabs(seed=0):
    torch.manual_seed(seed)
    return {
        "gate": torch.randn(E, INT, HID),
        "up": torch.randn(E, INT, HID),
        "down": torch.randn(E, HID, INT),
    }


def _ref_expert(slabs, eid, x):
    h = F.silu(F.linear(x, slabs["gate"][eid])) * F.linear(x, slabs["up"][eid])
    return F.linear(h, slabs["down"][eid])


def _store(slabs=None, capacity=C):
    return PinnedExpertStore(slabs or _slabs(), capacity, CPU, pin=False)


# ---------------------------------------------------------------- pure helpers
def test_recency_pool_matches_naive_counts():
    torch.manual_seed(1)
    hist = torch.randint(0, E, (12, K))
    pool = recency_pool(hist, E, C)
    counts = torch.bincount(hist.reshape(-1), minlength=E)
    excluded = [i for i in range(E) if i not in pool]
    assert len(pool) == C == len(set(pool))
    assert min(counts[i] for i in pool) >= max(counts[i] for i in excluded)


def test_restricted_selection_unrestricted_equals_vanilla():
    torch.manual_seed(2)
    logits = torch.randn(E)
    ids, w = restricted_selection(logits, torch.arange(E), K, norm_topk_prob=False)
    ref_ids = torch.topk(logits, K).indices
    ref_w = torch.softmax(logits, dim=-1, dtype=torch.float)[ref_ids]
    assert torch.equal(ids, ref_ids)
    assert torch.allclose(w.float(), ref_w, atol=1e-6)


def test_restricted_selection_weights_from_full_softmax_no_renorm():
    torch.manual_seed(3)
    logits = torch.randn(E)
    resident = torch.topk(logits, E, largest=False).indices[:C]  # the C worst experts
    ids, w = restricted_selection(logits, resident, K, norm_topk_prob=False)
    assert all(i in set(resident.tolist()) for i in ids.tolist())
    full = torch.softmax(logits, dim=-1, dtype=torch.float)
    assert torch.allclose(w.float(), full[ids], atol=1e-6)


# ------------------------------------------------------- v5 fused slab packing
def test_slabs_from_fused_v5_block_gate_first():
    torch.manual_seed(4)
    block = torch.nn.Module()
    block.experts = torch.nn.Module()
    block.experts.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * INT, HID))
    block.experts.down_proj = torch.nn.Parameter(torch.randn(E, HID, INT))
    slabs = slabs_from_block(block)
    x = torch.randn(3, HID)
    for eid in (0, E - 1):
        # v5 reference: linear(x, gate_up_proj[e]).chunk(2, -1) -> silu(gate) * up -> down
        gate, up = F.linear(x, block.experts.gate_up_proj[eid]).chunk(2, dim=-1)
        want = F.linear(F.silu(gate) * up, block.experts.down_proj[eid])
        got = _ref_expert(slabs, eid, x)
        assert torch.allclose(got, want, atol=1e-5), eid


def test_store_run_expert_matches_reference():
    slabs = _slabs()
    store = _store(slabs)
    store.refresh(list(range(4, 4 + C)))
    x = torch.randn(1, HID)
    for eid in (4, 7, 4 + C - 1):
        assert torch.allclose(store.run_expert(eid, x), _ref_expert(slabs, eid, x), atol=1e-6)


# ------------------------------------------------------------------- the store
def test_store_refresh_copies_only_difference():
    store = _store()
    store.refresh(list(range(C)))
    assert store.initial_loads == C and store.transfers == 0
    n = store.refresh(list(range(4, 4 + C)))
    assert n == 4 and store.transfers == 4
    assert sorted(store.slot_of) == list(range(4, 4 + C))
    assert sorted(store.slot_of.values()) == list(range(C))  # each slot used exactly once


def test_store_lru_never_evicts_currently_selected():
    """Review regression: LRU-oldest resident expert selected together with a miss."""
    store = _store(capacity=4)
    store.ensure([0, 1, 2, 3])
    store.ensure([1, 2, 3])       # 0 becomes LRU-oldest
    n = store.ensure([0, 7])      # 0 is selected AND oldest; 7 is a miss
    assert n == 1
    assert 0 in store.slot_of and 7 in store.slot_of      # 0 survived
    assert 1 not in store.slot_of                          # oldest NON-selected evicted
    x = torch.randn(1, HID)
    store.run_expert(0, x)                                 # and its weights are intact


def test_store_cold_fill_priced_as_initial_for_lru_too():
    store = _store(capacity=4)
    store.ensure([0, 1])       # pool not yet full -> initial
    store.ensure([2, 3])       # fills the pool -> still initial
    assert store.initial_loads == 4 and store.transfers == 0
    store.ensure([9])          # pool has been full once -> steady-state transfer
    assert store.transfers == 1


def test_store_reset_sequence_reprices_reload_as_initial():
    store = _store()
    store.refresh(list(range(C)))
    store.reset_sequence()
    store.refresh(list(range(C)))
    assert store.transfers == 0 and store.initial_loads == 2 * C


# ----------------------------------------------------------------- the policies
def test_dwell1_pool_always_contains_true_topk():
    torch.manual_seed(5)
    policy = LayerPolicy("dwell", _store(), C, K, dwell=1)
    policy.reset_sequence()
    for _ in range(20):
        logits = torch.randn(E)
        ids, _ = policy.route(logits, norm_topk_prob=False)
        ref = torch.topk(logits, K).indices
        assert torch.equal(torch.sort(ids).values, torch.sort(ref).values)


def test_dwell_k_warm_starts_then_freezes():
    """Referee fix: refresh every token until the window fills, THEN lock to the cadence."""
    torch.manual_seed(6)
    policy = LayerPolicy("dwell", _store(), C, K, dwell=8)
    policy.reset_sequence()
    pools = []
    for _ in range(24):
        policy.route(torch.randn(E), norm_topk_prob=False)
        pools.append(tuple(policy.pool_ids.tolist()))
    # warm phase (window filling): pool tracks demand, so it changes
    assert len(set(pools[:8])) > 1, "warm phase must refresh every token"
    # after the window fills, frozen inside each dwell window
    assert len(set(pools[8:16])) == 1 and len(set(pools[16:24])) == 1


def test_static_warm_pool_uses_whole_warmup_window():
    """Review regression: the frozen pool must reflect the full warmup, not the last token."""
    warmup = 6
    policy = LayerPolicy("static_warm", _store(), C, K, warmup_tokens=warmup)
    policy.reset_sequence()
    # experts 0..3 demanded on every warmup token except the last, which demands 8..11
    hot = torch.full((E,), -10.0)
    hot[[0, 1, 2, 3]] = 10.0
    last = torch.full((E,), -10.0)
    last[[8, 9, 10, 11]] = 10.0
    for _ in range(warmup - 1):
        policy.route(hot, norm_topk_prob=False)
    policy.route(last, norm_topk_prob=False)
    frozen = set(policy.pool_ids.tolist())
    assert {0, 1, 2, 3} <= frozen, "warmup-long demand must dominate the frozen pool"
    tr = policy.store.transfers + policy.store.initial_loads
    for _ in range(10):
        policy.route(torch.randn(E), norm_topk_prob=False)
    assert set(policy.pool_ids.tolist()) == frozen
    assert policy.store.transfers + policy.store.initial_loads == tr  # zero copies after freeze


def test_resident_pool_mode_matches_vanilla_and_never_recopies():
    torch.manual_seed(7)
    policy = LayerPolicy("resident_pool", _store(capacity=E), E, K)
    policy.reset_sequence()
    for _ in range(10):
        logits = torch.randn(E)
        ids, w = policy.route(logits, norm_topk_prob=False)
        assert torch.equal(ids, torch.topk(logits, K).indices)
    assert policy.store.initial_loads == E and policy.store.transfers == 0


# --------------------------------------------------------------- patched block
def test_offload_block_v5_contract_and_math():
    torch.manual_seed(8)
    slabs = _slabs()
    gate = torch.nn.Module()
    gate.weight = torch.nn.Parameter(torch.randn(E, HID))
    policy = LayerPolicy("dwell", _store(slabs), C, K, dwell=1)
    policy.reset_sequence()
    block = OffloadMoeBlock(gate, policy, norm_topk_prob=False)
    x = torch.randn(1, 1, HID)
    out = block(x)
    assert isinstance(out, torch.Tensor) and out.shape == x.shape  # bare tensor, v5 contract
    flat = x.reshape(1, HID)
    logits = F.linear(flat, gate.weight)[0]
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    ref = sum(
        probs[i].to(x.dtype) * _ref_expert(slabs, i, flat)
        for i in torch.topk(logits, K).indices.tolist()
    )
    assert torch.allclose(out.reshape(1, HID), ref, atol=1e-5)


# ------------------------------------------- end-to-end on a real (tiny) OLMoE
def test_end_to_end_tiny_olmoe_cpu(tmp_path):
    """The gold check: the full run() path on a random-weight transformers OLMoE, CPU.

    Exercises the exact code that runs on the cluster — v5 block replacement, KV-cache
    decode loop, every policy, and the in-run NLL cross-checks (resident_pool / lru /
    dwell_1 == resident_gpu), which raise AssertionError on any semantic divergence.
    """
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(9)
    config = OlmoeConfig(
        vocab_size=50304,  # covers the gpt2 tokenizer ids used below
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_local_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        max_position_embeddings=64,
    )
    model = OlmoeForCausalLM(config)
    model_dir = tmp_path / "tiny_olmoe"
    model.save_pretrained(model_dir)
    # minimal tokenizer: reuse a real slow tokenizer is overkill — save a char-level stub
    # via tokenizer-free path: the harness only needs tokenizer(...)["input_ids"], so give
    # it a GPT-2-style tokenizer if available, else skip.
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("no tokenizer available offline")
    tok.save_pretrained(model_dir)

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "the quick brown fox jumps over the lazy dog and keeps on running\n"
        "pack my box with five dozen liquor jugs before the storm arrives\n"
    )
    cfg = {
        "seed": 0,
        "model": {"id": str(model_dir), "dtype": "float32", "device": "cpu"},
        "eval": {
            "prompt_file": str(prompt_file),
            "max_sequences": 2,
            "max_length": 12,
            "capacity": 4,
            "restrict_layers": [1, 2],
            "policies": ["resident_gpu", "resident_pool", "lru", "dwell", "static_warm"],
            "dwells": [1, 3],
            "static_warmup_tokens": 4,
            "timing_warmup_steps": 0,
            "residency_trace_dwell": 3,
        },
    }
    out_dir = tmp_path / "out"
    run(cfg, out_dir, router_state_path=None)  # in-run asserts are the test

    rows = [
        json.loads(l) for l in (out_dir / "offload_wallclock.jsonl").read_text().splitlines()
    ]
    by = {(r["policy"], r["dwell"]): r for r in rows}
    base = by[("resident_gpu", None)]["nll"]
    assert abs(by[("resident_pool", None)]["nll"] - base) < 5e-3
    assert abs(by[("lru", None)]["nll"] - base) < 5e-3
    assert abs(by[("dwell", 1)]["nll"] - base) < 5e-3
    # dwell 3 restricts to 4 of 8 experts from a stale window: quality may drop, never NaN
    assert by[("dwell", 3)]["nll"] == by[("dwell", 3)]["nll"]
    assert by[("dwell", 3)]["transfers_incl_initial"] >= by[("dwell", 3)]["transfers"]
    assert (out_dir / "residency_trace.npz").exists()
    assert (out_dir / "metadata.json").exists()


def test_scored_eviction_picks_lowest_score_nonselected():
    """Hawkeye hook: victim = argmin keep-score among non-selected residents."""
    store = _store(capacity=4)
    store.ensure([0, 1, 2, 3])
    scores = torch.tensor([9.0, 0.1, 5.0, 9.0] + [0.0] * (E - 4))
    n = store.ensure([0, 9], victim_scores=scores)   # 1 has lowest score -> victim
    assert n == 1
    assert 1 not in store.slot_of and 9 in store.slot_of and 0 in store.slot_of


def test_scored_eviction_never_evicts_selected():
    store = _store(capacity=4)
    store.ensure([0, 1, 2, 3])
    scores = torch.zeros(E); scores[1] = -100.0      # 1 has the worst score BUT is selected
    store.ensure([1, 9], victim_scores=scores)
    assert 1 in store.slot_of and 9 in store.slot_of


def test_lru_evict_variants_stay_exact():
    torch.manual_seed(11)
    for ev in ("lfu", "logit"):
        policy = LayerPolicy("lru", _store(), C, K)
        policy.evict = ev
        policy.reset_sequence()
        for _ in range(30):
            logits = torch.randn(E)
            ids, _ = policy.route(logits, norm_topk_prob=False)
            assert torch.equal(ids, torch.topk(logits, K).indices), ev
