"""Wall-clock expert-offloading harness (E1): does temporally extended routing pay off in time?

Everything else in this repo counts transfers or times replayed copies open-loop
(``cache_serving_stall_benchmark``).  This module runs the model CLOSED-LOOP, token by token
with a KV cache, with the restricted layers' experts genuinely absent from the GPU: expert
weights live in pinned host memory and reach a fixed-size per-layer resident pool only when
the serving policy asks for them.  Copies are synchronous, on the critical path, and timed
with CUDA events.

Written against the transformers >= 5.x OLMoE API (fused ``OlmoeExperts`` with
``gate_up_proj [E, 2I, H]`` / ``down_proj [E, H, I]``, ``OlmoeTopKRouter`` gate whose
``.weight`` is the router matrix, MoE block returning a bare tensor) — the API pinned by this
repo's uv.lock (transformers 5.12.1).

Policies (applied to every restricted layer; other layers stay fully resident):

``resident_gpu``
    The intact HuggingFace block: all experts on GPU, free routing.  Quality ceiling; also a
    latency reference, but it runs HF's expert kernel, so latency gaps against it conflate
    policy cost with implementation cost — use ``resident_pool`` for matched-latency deltas.
``resident_pool``
    The harness's own pool path with capacity = num_experts: pool filled once per sequence,
    free routing, zero steady-state copies.  The matched-implementation latency ceiling.
    NLL must equal ``resident_gpu``.
``lru``
    Free routing; per token, every selected-but-absent expert is copied in, evicting the
    least-recently-used slot among experts NOT selected by the current token (per-token
    admission, the no-dwell offloading baseline in the style of vLLM PR #37190 /
    Mixtral-offloading).  NLL must equal ``resident_gpu``.
``dwell_<k>``
    The pool is frozen for ``k`` tokens.  At each refresh the pool becomes the top-C experts
    by causal recency — appearances in the router's UNRESTRICTED top-k over the trailing
    ``k`` tokens — and only ``|R_new \\ R_old|`` experts are copied.  The closed-loop twin of
    ``FreezePoolWindowController`` with ``pool_selection: causal_recency``.
``static_warm``
    Refresh every token for the first ``warmup_tokens`` of each sequence (recency window =
    the whole warmup), then the pool freezes for the rest of the sequence.

Routing under restriction copies ``evaluate_working_set._recompute_tuple`` semantics exactly:
selection = top-k over masked logits (``RESTRICTED_FILL``), weights = the router's real full
softmax gathered at the selected experts (OLMoE has ``norm_topk_prob=False``: no renorm).

Measurement conventions (all per-token numbers are steady-state):
  * Each sequence's first token and the first ``timing_warmup_steps`` global steps are
    excluded from step timing; per-sequence initial pool fills are booked as
    ``initial_loads`` / ``initial_copy_ms``, never as transfers, for EVERY policy (a pool is
    "initial" until it has been full once in the sequence).
  * ``copy_ms_per_token`` is steady-state copy time only and is CONTAINED IN
    ``ms_per_token_mean`` (copies happen inside the step); ``compute_ms_per_token`` is the
    difference.  ``tok_per_s`` = 1000 / ms_per_token_mean (CUDA events, scoring excluded);
    ``wall_tok_per_s`` additionally includes NLL scoring overhead — use it only for context.
  * Compare NLL as deltas against this harness's own ``resident_gpu`` row; bf16 kernel-path
    differences put absolute NLL within ~1e-3 of the recorded batched-eval base, not equal.

Built-in cross-checks (asserted, not eyeballed): ``resident_pool``, ``lru`` and ``dwell_1``
NLL must all equal ``resident_gpu`` NLL to 5e-3.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from moe_eviction.models.collect_trace import _dtype, _load_prompts
from moe_eviction.models.evaluate_router_state import _load_router_state
from moe_eviction.models.evaluate_working_set import RESTRICTED_FILL
from moe_eviction.models.learned_beta import (
    assemble_policy_features,
    load_learned_beta_checkpoint,
    load_policy_head,
    update_pool_demand,
)

SLAB_NAMES = ("gate", "up", "down")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU)
# ---------------------------------------------------------------------------
def recency_pool(topk_history: torch.Tensor, num_experts: int, capacity: int) -> list[int]:
    """Top-``capacity`` experts by appearance count in ``topk_history`` ([w, top_k] int64).

    Ties (including zero counts) resolve by ascending expert id, matching
    ``FreezePoolWindowController._select_pool``'s ``torch.topk`` determinism.
    """
    counts = torch.bincount(topk_history.reshape(-1), minlength=num_experts)
    return sorted(torch.topk(counts, capacity).indices.tolist())


def restricted_selection(
    logits: torch.Tensor, resident_ids: torch.Tensor, top_k: int, norm_topk_prob: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """(expert_ids [top_k], weights [top_k]) under pool restriction, semantics of
    ``_recompute_tuple``: mask-then-topk for selection, full-softmax gather for weights."""
    masked = torch.full_like(logits, RESTRICTED_FILL)
    masked[resident_ids] = logits[resident_ids]
    indices = torch.topk(masked, top_k, dim=-1).indices
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    scores = probs.gather(-1, indices)
    if norm_topk_prob:
        scores = scores / scores.sum(dim=-1, keepdim=True)
    return indices, scores.to(logits.dtype)


def slabs_from_block(block: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract per-expert weight slabs from an OLMoE MoE block, on CPU, model dtype.

    Returns ``{"gate": [E, I, H], "up": [E, I, H], "down": [E, H, I]}``.  Supports the
    transformers >= 5.x fused layout (``experts.gate_up_proj`` [E, 2I, H] where the FIRST I
    output rows are the gate — v5 computes ``linear(x, gate_up_proj[e]).chunk(2, dim=-1)``)
    and the 4.x ``ModuleList`` layout as a fallback for tests.
    """
    experts = block.experts
    if hasattr(experts, "gate_up_proj"):  # fused, transformers >= 5.x
        fused = experts.gate_up_proj.detach().to("cpu")
        inter = fused.shape[1] // 2
        return {
            "gate": fused[:, :inter, :].contiguous(),
            "up": fused[:, inter:, :].contiguous(),
            "down": experts.down_proj.detach().to("cpu").contiguous(),
        }
    return {
        "gate": torch.stack([m.gate_proj.weight.detach().to("cpu") for m in experts]).contiguous(),
        "up": torch.stack([m.up_proj.weight.detach().to("cpu") for m in experts]).contiguous(),
        "down": torch.stack([m.down_proj.weight.detach().to("cpu") for m in experts]).contiguous(),
    }


# ---------------------------------------------------------------------------
# Pinned host slabs + GPU resident pool
# ---------------------------------------------------------------------------
class PinnedExpertStore:
    """One restricted layer's experts: pinned host slabs, fixed-size GPU pool, timed copies.

    ``slabs`` may be shared between stores (e.g. a capacity-C store and a capacity-E
    resident_pool store for the same layer).  Copy time is accumulated via CUDA event pairs
    on the current stream, separately for initial fills and steady-state transfers.  A pool
    is "initial" until it has been full once in the current sequence, so cold-fill pricing
    is identical across policies.
    """

    def __init__(
        self,
        slabs: dict[str, torch.Tensor],
        capacity: int,
        device: torch.device,
        pin: bool = True,
        tau_multiplier: int = 1,
    ) -> None:
        assert set(slabs) == set(SLAB_NAMES)
        # Emulate a slower memory tier by physically repeating each copy: real DMA traffic
        # at tau' = tau_multiplier x tau. Transfers/bytes stay semantic (counted once).
        self.tau_multiplier = max(1, int(tau_multiplier))
        self.num_experts = slabs["gate"].shape[0]
        self.capacity = capacity
        self.device = device
        self._use_cuda_events = device.type == "cuda"
        self.slabs = {
            name: (t.pin_memory() if pin and not t.is_pinned() else t) for name, t in slabs.items()
        }
        self.pool = {
            name: torch.empty(
                (capacity, *self.slabs[name].shape[1:]),
                dtype=self.slabs[name].dtype,
                device=device,
            )
            for name in SLAB_NAMES
        }
        self.bytes_per_expert = sum(
            self.slabs[name][0].numel() * self.slabs[name][0].element_size()
            for name in SLAB_NAMES
        )
        # expert id -> slot, insertion-ordered so popitem(last=False) evicts LRU
        self.slot_of: OrderedDict[int, int] = OrderedDict()
        self._free_slots = list(range(capacity))
        self.transfers = 0
        self.initial_loads = 0
        self.bytes_copied = 0
        self._events_initial: list[tuple[Any, Any]] = []
        self._events_steady: list[tuple[Any, Any]] = []
        self._full_once = False

    # -- copy mechanics ------------------------------------------------------
    def _timed_copy(self, expert_ids: list[int], slots: list[int]) -> None:
        if not expert_ids:
            return
        initial = not self._full_once
        if self._use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        for _rep in range(self.tau_multiplier):
            for expert_id, slot in zip(expert_ids, slots):
                for name, slab in self.slabs.items():
                    self.pool[name][slot].copy_(slab[expert_id], non_blocking=True)
        if self._use_cuda_events:
            end.record()
            (self._events_initial if initial else self._events_steady).append((start, end))
        if initial:
            self.initial_loads += len(expert_ids)
        else:
            self.transfers += len(expert_ids)
        self.bytes_copied += len(expert_ids) * self.bytes_per_expert

    # -- policy-facing API ---------------------------------------------------
    def refresh(self, new_ids: list[int]) -> int:
        """Make the pool exactly ``new_ids`` (len == capacity); copy only the difference."""
        assert len(new_ids) == self.capacity, (len(new_ids), self.capacity)
        new_set = set(new_ids)
        keep = {eid: slot for eid, slot in self.slot_of.items() if eid in new_set}
        freed = [slot for eid, slot in self.slot_of.items() if eid not in new_set]
        freed += self._free_slots
        incoming = [eid for eid in new_ids if eid not in keep]
        slots = freed[: len(incoming)]
        self._timed_copy(incoming, slots)
        self.slot_of = OrderedDict(keep)
        for eid, slot in zip(incoming, slots):
            self.slot_of[eid] = slot
        self._free_slots = freed[len(incoming) :]
        if len(self.slot_of) == self.capacity:
            self._full_once = True
        return len(incoming)

    def ensure(self, expert_ids: list[int], victim_scores=None) -> int:
        """Exact serving: make each id resident.

        Present requested experts are touched FIRST, so eviction can never pick an expert the
        current token selected. Victim choice is LRU by default; when ``victim_scores`` (a
        tensor or dict of keep-scores over experts, higher = keep) is given, the resident
        with the LOWEST score among non-selected residents is evicted instead — the hook for
        Belady-imitating eviction (frequency, router-logit, or learned scores).
        """
        for eid in expert_ids:
            if eid in self.slot_of:
                self.slot_of.move_to_end(eid)
        missing = [eid for eid in expert_ids if eid not in self.slot_of]
        slots: list[int] = []
        for _ in missing:
            if self._free_slots:
                slots.append(self._free_slots.pop())
            elif victim_scores is None:
                _evicted, slot = self.slot_of.popitem(last=False)
                slots.append(slot)
            else:
                selected = set(expert_ids)
                cands = [r for r in self.slot_of if r not in selected]
                victim = min(cands, key=lambda r: float(victim_scores[r]))
                slots.append(self.slot_of.pop(victim))
        self._timed_copy(missing, slots)
        for eid, slot in zip(missing, slots):
            self.slot_of[eid] = slot
        if len(self.slot_of) == self.capacity:
            self._full_once = True
        return len(missing)

    def reset_sequence(self) -> None:
        """New sequence: forget residency; the refill is priced as initial_loads again."""
        self.slot_of = OrderedDict()
        self._free_slots = list(range(self.capacity))
        self._full_once = False

    def reset_counters(self) -> None:
        self.transfers = 0
        self.initial_loads = 0
        self.bytes_copied = 0
        self._events_initial = []
        self._events_steady = []

    def drain_copy_ms(self) -> tuple[float, float]:
        """(initial_ms, steady_ms) accumulated since the last drain."""
        if not self._use_cuda_events:
            return 0.0, 0.0
        torch.cuda.synchronize()
        initial = sum(s.elapsed_time(e) for s, e in self._events_initial)
        steady = sum(s.elapsed_time(e) for s, e in self._events_steady)
        self._events_initial = []
        self._events_steady = []
        return initial, steady

    def run_expert(self, expert_id: int, x: torch.Tensor) -> torch.Tensor:
        slot = self.slot_of[expert_id]
        gate = F.linear(x, self.pool["gate"][slot])
        up = F.linear(x, self.pool["up"][slot])
        return F.linear(F.silu(gate) * up, self.pool["down"][slot])


# ---------------------------------------------------------------------------
# Serving policies (per restricted layer)
# ---------------------------------------------------------------------------
class LayerPolicy:
    """Decode-time policy state for one restricted layer."""

    def __init__(
        self,
        mode: str,
        store: PinnedExpertStore,
        capacity: int,
        top_k: int,
        dwell: int = 1,
        warmup_tokens: int = 64,
        trace: bool = False,
        window_mult: int = 1,
    ) -> None:
        assert mode in ("lru", "dwell", "static_warm", "resident_pool")
        self.mode = mode
        self.evict = "lru"          # lru | lfu | logit  (Family-A victim scoring)
        self._freq = None           # per-expert EMA, lazily created
        self.store = store
        self.capacity = capacity
        self.top_k = top_k
        self.dwell = dwell
        self.warmup_tokens = warmup_tokens
        # recency window: trailing dwell x window_mult for dwell mode (window_mult > 1
        # decouples demand-profiling length from refresh cadence), whole warmup for static
        self.window_mult = max(1, int(window_mult))
        self.window = warmup_tokens if mode == "static_warm" else max(1, dwell) * self.window_mult
        self.t = 0
        self._history: list[torch.Tensor] = []  # unrestricted top-k ids per token
        self.pool_ids: torch.Tensor | None = None
        self.trace_enabled = trace
        self.trace_resident: list[list[int]] = []
        self.trace_demand: list[list[int]] = []
        self.trace_refresh: list[bool] = []
        self.trace_probs: list[np.ndarray] = []  # full router softmax per token (fp16)

    def reset_sequence(self) -> None:
        self.t = 0
        self._history = []
        self.pool_ids = None
        self._freq = None
        self.store.reset_sequence()

    def _record(self, logits: torch.Tensor) -> torch.Tensor:
        top = torch.topk(logits, self.top_k, dim=-1).indices
        self._history.append(top)
        if len(self._history) > self.window:
            self._history.pop(0)
        return top

    def _refresh_from_recency(self) -> bool:
        hist = torch.stack(self._history)
        new_ids = recency_pool(hist.cpu(), self.store.num_experts, self.capacity)
        copied = self.store.refresh(new_ids)
        self.pool_ids = torch.tensor(new_ids, dtype=torch.long, device=hist.device)
        return copied > 0 or self.t == 0

    def _trace(
        self, demand: torch.Tensor, refreshed: bool, probs: torch.Tensor | None = None
    ) -> None:
        if not self.trace_enabled:
            return
        self.trace_resident.append(sorted(self.store.slot_of))
        self.trace_demand.append(demand.tolist())
        self.trace_refresh.append(refreshed)
        if probs is not None:
            self.trace_probs.append(
                probs.detach().float().cpu().numpy().astype(np.float16)
            )

    def route(
        self, logits: torch.Tensor, norm_topk_prob: bool, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """logits: [num_experts] for the single decode token.  Returns (ids, weights)."""
        if self.mode == "lru":
            probs = torch.softmax(logits, dtype=torch.float, dim=-1)
            ids = torch.topk(logits, self.top_k, dim=-1).indices
            weights = probs.gather(-1, ids)
            if norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            scores = None
            if self.evict == "lfu":
                if self._freq is None:
                    self._freq = torch.zeros(self.store.num_experts)
                self._freq *= 0.95
                self._freq[ids.cpu()] += 1.0
                scores = self._freq
            elif self.evict == "logit":
                scores = logits.detach().float().cpu()
            self.store.ensure(ids.tolist(), victim_scores=scores)
            self._trace(ids, refreshed=False, probs=probs)
            self.t += 1
            return ids, weights.to(logits.dtype)

        if self.mode == "resident_pool":
            if self.pool_ids is None:
                all_ids = list(range(self.store.num_experts))
                self.store.refresh(all_ids)
                self.pool_ids = torch.tensor(all_ids, dtype=torch.long, device=logits.device)
            self.t += 1
            return restricted_selection(logits, self.pool_ids, self.top_k, norm_topk_prob)

        demand = self._record(logits)
        refreshed = False
        if self.mode == "dwell":
            # warm phase: until the demand window first fills, refresh every token so the
            # pool is never dominated by zero-count tie-break filler experts (referee fix)
            warm = len(self._history) < self.window
            if self.pool_ids is None or warm or self.t % self.dwell == 0:
                refreshed = self._refresh_from_recency()
        else:  # static_warm
            if self.pool_ids is None or self.t < self.warmup_tokens:
                refreshed = self._refresh_from_recency()
        trace_probs = (
            torch.softmax(logits, dtype=torch.float, dim=-1) if self.trace_enabled else None
        )
        self._trace(demand, refreshed, probs=trace_probs)
        self.t += 1
        return restricted_selection(logits, self.pool_ids, self.top_k, norm_topk_prob)


class LearnedLayerPolicy:
    """Learned joint termination (beta) + policy-over-options (mu), closed-loop.

    A faithful per-token port of ``LearnedBetaPoolController`` (evaluate_working_set):
    fire = pool is None or beta(hidden, pool) > threshold; on fire the pool becomes
    top-C of the trained causal mu head's per-expert scores; every attended token updates
    the causal popularity EMA / recency features via ``update_pool_demand``.  Head
    inference runs on the decode critical path and is therefore included in step timing —
    the learned policy pays its own controller cost, exactly as a deployed one would.
    """

    def __init__(
        self,
        store: PinnedExpertStore,
        capacity: int,
        top_k: int,
        layer_index: int,
        beta_head: Any,
        policy_head: Any,
        threshold: float,
        seq_len: int,
        pop_decay: float = 0.9,
        trace: bool = False,
    ) -> None:
        self.mode = "learned"
        self.store = store
        self.capacity = capacity
        self.top_k = top_k
        self.layer_index = layer_index
        self.beta_head = beta_head
        self.policy_head = policy_head
        self.threshold = float(threshold)
        self.seq_len = seq_len
        self.pop_decay = pop_decay
        self.num_experts = store.num_experts
        self.pool_ids: torch.Tensor | None = None
        self.pop_ema = torch.zeros(self.num_experts)
        self.last_seen = torch.full((self.num_experts,), 1.0e9)
        self.age = 0
        self.option_lengths: list[int] = []
        self.trace_enabled = trace
        self.trace_resident: list[list[int]] = []
        self.trace_demand: list[list[int]] = []
        self.trace_refresh: list[bool] = []
        self.trace_probs: list[np.ndarray] = []  # full router softmax per token (fp16)

    def reset_sequence(self) -> None:
        if self.pool_ids is not None:
            self.option_lengths.append(self.age)
        self.pool_ids = None
        self.pop_ema = torch.zeros(self.num_experts)
        self.last_seen = torch.full((self.num_experts,), 1.0e9)
        self.age = 0
        self.store.reset_sequence()

    @torch.no_grad()
    def _beta(self, hidden: torch.Tensor) -> float:
        option = self.pool_ids.unsqueeze(0)
        set_features = self.beta_head.set_embedding(option).squeeze(0)
        features = torch.cat((hidden.float(), set_features.float())).unsqueeze(0)
        if self.beta_head.config.shared_across_layers:
            beta = self.beta_head(features)
        else:
            layer_indices = torch.tensor([self.layer_index], dtype=torch.long, device=hidden.device)
            beta = self.beta_head(features, layer_indices=layer_indices)
        return float(beta.squeeze(0))

    @torch.no_grad()
    def _select_pool(self, hidden: torch.Tensor, logits: torch.Tensor) -> list[int]:
        resident = list(self.store.slot_of)
        features = assemble_policy_features(
            hidden, logits, self.pop_ema, self.last_seen, resident,
            self.num_experts, self.seq_len,
        )
        scores = self.policy_head(features, self.layer_index)
        indices = torch.topk(scores, self.capacity, dim=-1).indices
        return sorted(int(e) for e in indices.tolist())

    def route(
        self, logits: torch.Tensor, norm_topk_prob: bool, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden is not None, "learned policy needs the layer hidden state"
        fire = self.pool_ids is None or self._beta(hidden) > self.threshold
        if fire:
            new_ids = self._select_pool(hidden, logits)
            self.store.refresh(new_ids)
            if self.pool_ids is not None:
                self.option_lengths.append(self.age)
            self.pool_ids = torch.tensor(new_ids, dtype=torch.long, device=logits.device)
            self.age = 0
        self.age += 1
        update_pool_demand(self.pop_ema, self.last_seen, logits, self.top_k, self.pop_decay)
        if self.trace_enabled:
            self.trace_resident.append(sorted(self.store.slot_of))
            self.trace_demand.append(torch.topk(logits, self.top_k).indices.tolist())
            self.trace_refresh.append(bool(fire))
        return restricted_selection(logits, self.pool_ids, self.top_k, norm_topk_prob)

    @property
    def mean_dwell(self) -> float:
        lens = self.option_lengths + ([self.age] if self.pool_ids is not None else [])
        return sum(lens) / len(lens) if lens else 0.0


# ---------------------------------------------------------------------------
# Patched MoE block (transformers >= 5.x contract: mlp returns a bare tensor)
# ---------------------------------------------------------------------------
class OffloadMoeBlock(torch.nn.Module):
    """Drop-in replacement for the v5 ``OlmoeSparseMoeBlock`` on a restricted layer.

    Keeps the original ``OlmoeTopKRouter`` module (so router weights, including a preloaded
    ReMoE state, are used verbatim) but computes the logits directly from ``gate.weight`` —
    the router's own forward returns a (logits, scores, indices) tuple we don't want.
    Serves expert FFNs from the pool.  Batch-1, single-token calls only.
    """

    def __init__(self, gate: torch.nn.Module, policy: LayerPolicy, norm_topk_prob: bool):
        super().__init__()
        self.gate = gate
        self.policy = policy
        self.norm_topk_prob = norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = hidden_states.shape
        if batch != 1 or seq != 1:
            raise RuntimeError(
                f"OffloadMoeBlock is decode-only (batch=1, seq=1); got {batch}x{seq}"
            )
        x = hidden_states.reshape(1, hidden)
        router_logits = F.linear(x, self.gate.weight)  # [1, E]
        ids, weights = self.policy.route(router_logits[0], self.norm_topk_prob, hidden=x[0])
        out = torch.zeros_like(x)
        for eid, w in zip(ids.tolist(), weights):
            out = out + w * self.policy.store.run_expert(eid, x)
        return out.reshape(batch, seq, hidden)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WallclockRow:
    policy: str
    dwell: int | None
    window_mult: int | None
    beta_threshold: float | None
    mean_dwell: float | None
    tau_multiplier: int
    capacity: int
    restricted_layers: int
    sequences: int
    tokens_scored: int
    tokens_timed: int
    nll: float
    nll_head: float | None
    nll_tail: float | None
    per_sequence_nll: list[float]
    tok_per_s: float
    wall_tok_per_s: float
    ms_per_token_mean: float
    ms_per_token_p50: float
    copy_ms_per_token: float
    compute_ms_per_token: float
    initial_copy_ms: float
    transfers: int
    initial_loads: int
    transfers_incl_initial: int
    bytes_copied: int
    gpu_mem_gb: float


def _norm_topk_prob(block: torch.nn.Module) -> bool:
    gate = block.gate
    if hasattr(gate, "norm_topk_prob"):
        return bool(gate.norm_topk_prob)
    return bool(getattr(block, "norm_topk_prob", False))


def _install_policies(
    model: torch.nn.Module,
    layer_indices: list[int],
    originals: dict[int, torch.nn.Module],
    stores: dict[int, PinnedExpertStore],
    mode: str,
    capacity: int,
    top_k: int,
    dwell: int,
    warmup_tokens: int,
    trace: bool,
    window_mult: int = 1,
) -> list[LayerPolicy]:
    policies = []
    for i in layer_indices:
        block = originals[i]
        if mode == "resident_gpu":
            model.model.layers[i].mlp = block
            continue
        policy = LayerPolicy(
            mode, stores[i], stores[i].capacity, top_k,
            dwell=dwell, warmup_tokens=warmup_tokens, trace=trace, window_mult=window_mult,
        )
        model.model.layers[i].mlp = OffloadMoeBlock(block.gate, policy, _norm_topk_prob(block))
        policies.append(policy)
    return policies


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def _run_policy(
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    policies: list[LayerPolicy],
    stores: dict[int, PinnedExpertStore],
    device: torch.device,
    warmup_steps: int,
    phase_split_tokens: int = 64,
) -> dict[str, Any]:
    from transformers import DynamicCache

    for store in stores.values():
        store.reset_counters()

    nll_sum = torch.zeros((), dtype=torch.float32, device=device)
    nll_count = 0
    head_sum = torch.zeros((), dtype=torch.float32, device=device)
    head_count = 0
    per_seq: list[float] = []
    step_ms: list[float] = []
    initial_copy_ms = 0.0
    steady_copy_ms = 0.0
    step = 0
    wall_s = 0.0

    for seq in sequences:
        for policy in policies:
            policy.reset_sequence()
        seq_sum = torch.zeros((), dtype=torch.float32, device=device)
        seq_count = 0
        cache = DynamicCache()
        ids = seq.to(device).unsqueeze(0)  # [1, L]
        length = ids.shape[1]
        for t in range(length):
            # steady-state timing: skip global warmup AND each sequence's first token
            # (which pays the initial pool fill)
            timed = step >= warmup_steps and t >= 1
            if timed:
                _sync(device)
                t0 = time.perf_counter()
                if device.type == "cuda":
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
            out = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            if timed and device.type == "cuda":
                ev_e.record()
            if t < length - 1:
                logits = out.logits[0, -1].float()
                tok_nll = F.cross_entropy(logits.unsqueeze(0), ids[0, t + 1].unsqueeze(0))
                nll_sum += tok_nll
                nll_count += 1
                seq_sum += tok_nll
                seq_count += 1
                if t < phase_split_tokens:
                    head_sum += tok_nll
                    head_count += 1
            if timed:
                _sync(device)
                wall_s += time.perf_counter() - t0
                if device.type == "cuda":
                    step_ms.append(ev_s.elapsed_time(ev_e))
            step += 1
        per_seq.append((seq_sum / max(seq_count, 1)).item())
        for store in stores.values():
            ini, ste = store.drain_copy_ms()
            initial_copy_ms += ini
            steady_copy_ms += ste

    _sync(device)
    tokens_timed = len(step_ms)
    ms_mean = statistics.fmean(step_ms) if step_ms else float("nan")
    copy_per_tok = steady_copy_ms / max(tokens_timed, 1)
    total = nll_sum.item()
    head = head_sum.item()
    return {
        "nll": total / max(nll_count, 1),
        "nll_head": head / max(head_count, 1) if head_count else None,
        "nll_tail": (total - head) / max(nll_count - head_count, 1)
        if nll_count > head_count else None,
        "per_sequence_nll": [round(x, 5) for x in per_seq],
        "tokens_scored": nll_count,
        "tokens_timed": tokens_timed,
        "tok_per_s": 1000.0 / ms_mean if step_ms else float("nan"),
        "wall_tok_per_s": tokens_timed / wall_s if wall_s > 0 else float("nan"),
        "ms_per_token_mean": ms_mean,
        "ms_per_token_p50": statistics.median(step_ms) if step_ms else float("nan"),
        "copy_ms_per_token": copy_per_tok,
        "compute_ms_per_token": ms_mean - copy_per_tok if step_ms else float("nan"),
        "initial_copy_ms": initial_copy_ms,
        "transfers": sum(s.transfers for s in stores.values()),
        "initial_loads": sum(s.initial_loads for s in stores.values()),
        "transfers_incl_initial": sum(
            s.transfers + s.initial_loads for s in stores.values()
        ),
        "bytes_copied": sum(s.bytes_copied for s in stores.values()),
    }


def _pad_ragged(rows: list[list[int]], width: int) -> np.ndarray:
    """Rows may be shorter than ``width`` during a cold fill (lru): pad with -1."""
    out = np.full((len(rows), width), -1, dtype=np.int16)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def _dump_residency(policies: list[LayerPolicy], layer_indices: list[int],
                    num_experts: int, seq_lengths: list[int], out: Path) -> None:
    arrays: dict[str, np.ndarray] = {"num_experts": np.array(num_experts)}
    width = max((len(r) for p_ in policies for r in p_.trace_resident), default=0)
    for layer_idx, policy in zip(layer_indices, policies):
        offset = 0
        for s, length in enumerate(seq_lengths):
            end = offset + length
            arrays[f"layer{layer_idx}_seq{s}_resident"] = _pad_ragged(
                policy.trace_resident[offset:end], width
            )
            arrays[f"layer{layer_idx}_seq{s}_demand"] = np.array(
                policy.trace_demand[offset:end], dtype=np.int16
            )
            arrays[f"layer{layer_idx}_seq{s}_refresh"] = np.array(
                policy.trace_refresh[offset:end], dtype=bool
            )
            if policy.trace_probs:
                arrays[f"layer{layer_idx}_seq{s}_probs"] = np.stack(
                    policy.trace_probs[offset:end]
                )
            offset = end
    np.savez_compressed(out, **arrays)
    print(f"[offload] residency trace -> {out}")


def run(
    config: dict[str, Any],
    output_dir: Path,
    router_state_path: Path | None,
    learned_beta_state: Path | None = None,
    policy_head_state: Path | None = None,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(int(config.get("seed", 0)))
    model_config = config["model"]
    eval_config = config["eval"]
    device = torch.device(model_config.get("device", "cuda"))
    dtype = _dtype(model_config.get("dtype", "bfloat16"))
    capacity = int(eval_config.get("capacity", 32))
    restrict_layers = [int(x) for x in eval_config["restrict_layers"]]
    dwells = [int(x) for x in eval_config.get("dwells", [1, 2, 4, 8, 16, 32])]
    warmup_tokens = int(eval_config.get("static_warmup_tokens", 64))
    warmup_steps = int(eval_config.get("timing_warmup_steps", 32))
    trace_dwell = eval_config.get("residency_trace_dwell")  # dwell value to trace, or None
    window_mults = [int(x) for x in eval_config.get("dwell_window_mults", [1])]
    tau_multiplier = max(1, int(eval_config.get("tau_multiplier", 1)))
    learned_thresholds = [float(t) for t in eval_config.get("learned_beta_thresholds", [0.1, 0.5, 0.9])]
    policies_cfg = list(
        eval_config.get(
            "policies", ["resident_gpu", "resident_pool", "lru", "dwell", "static_warm"]
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config.get("revision")
    )
    prompts = _load_prompts({"trace": eval_config})[: int(eval_config.get("max_sequences", 16))]
    prefix = eval_config.get("prompt_prefix")
    if prefix:
        prompts = [str(prefix) + p for p in prompts]
    max_length = int(eval_config.get("max_length", 384))
    sequences = [
        tokenizer(p, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        for p in prompts
    ]
    seq_lengths = [int(s.shape[0]) for s in sequences]

    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config.get("revision"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if router_state_path is not None:
        loaded = _load_router_state(model, router_state_path)
        print(f"[offload] loaded router state: {len(loaded)} tensors")

    top_k = int(model.config.num_experts_per_tok)
    num_experts = int(
        getattr(model.config, "num_local_experts", getattr(model.config, "num_experts", 0))
    )
    blocks = [layer.mlp for layer in model.model.layers]
    originals = {i: blocks[i] for i in restrict_layers}

    # resident_gpu must run before slabs/pools exist (clean memory reading) and before the
    # GPU expert weights are freed (it needs them).
    policies_cfg.sort(key=lambda m: 0 if m == "resident_gpu" else 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[WallclockRow] = []
    reference_nll: float | None = None
    slabs: dict[int, dict[str, torch.Tensor]] = {}
    stores_c: dict[int, PinnedExpertStore] = {}
    stores_full: dict[int, PinnedExpertStore] = {}

    def prepare_offload() -> None:
        nonlocal slabs, stores_c, stores_full
        if stores_c:
            return
        print(f"[offload] packing pinned slabs for layers {restrict_layers} ...")
        pin = device.type == "cuda"
        for i in restrict_layers:
            raw = slabs_from_block(originals[i])
            # pin exactly once per layer; every store for this layer shares the pinned slabs
            slabs[i] = {n: (t.pin_memory() if pin else t) for n, t in raw.items()}
        stores_c = {
            i: PinnedExpertStore(slabs[i], capacity, device, pin=False,
                                 tau_multiplier=tau_multiplier)
            for i in restrict_layers
        }
        if "resident_pool" in policies_cfg:
            stores_full = {
                i: PinnedExpertStore(slabs[i], num_experts, device, pin=False,
                                     tau_multiplier=tau_multiplier)
                for i in restrict_layers
            }
        # Drop the restricted layers' GPU expert weights: pools are now the only device copy.
        for i in restrict_layers:
            originals[i].experts = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
            print(
                f"[offload] freed GPU experts; allocated now "
                f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB"
            )

    def emit(name: str, dwell: int | None, stats: dict[str, Any], **extra: Any) -> None:
        nonlocal reference_nll
        row = WallclockRow(
            policy=name,
            dwell=dwell,
            window_mult=extra.get("window_mult"),
            beta_threshold=extra.get("beta_threshold"),
            mean_dwell=extra.get("mean_dwell"),
            tau_multiplier=tau_multiplier,
            capacity=num_experts if name == "resident_pool" else capacity,
            restricted_layers=len(restrict_layers),
            sequences=len(sequences),
            gpu_mem_gb=round(torch.cuda.memory_allocated() / 2**30, 3)
            if device.type == "cuda"
            else 0.0,
            **stats,
        )
        rows.append(row)
        with (output_dir / "offload_wallclock.jsonl").open("a") as f:
            f.write(json.dumps(asdict(row)) + "\n")
        print(
            f"[offload] {name:>13s} dwell={dwell!s:>4s}  nll={row.nll:.4f}  "
            f"tok/s={row.tok_per_s:7.2f}  ms/tok={row.ms_per_token_mean:6.2f}  "
            f"copy_ms/tok={row.copy_ms_per_token:5.3f}  transfers={row.transfers}"
        )
        if name == "resident_gpu":
            reference_nll = row.nll
        exact = (name in ("resident_pool", "lru")) or (
            name == "dwell" and dwell == 1 and extra.get("window_mult", 1) in (None, 1)
        )
        if exact and reference_nll is not None:
            assert abs(row.nll - reference_nll) < 5e-3, (
                f"{name} NLL {row.nll} != resident_gpu NLL {reference_nll}: "
                "routing or expert math diverged"
            )

    beta_head = policy_head = None
    if learned_beta_state is not None and policy_head_state is not None:
        beta_head = load_learned_beta_checkpoint(learned_beta_state, device)
        policy_head = load_policy_head(policy_head_state, device)
        print(f"[offload] loaded learned heads: {learned_beta_state}")

    for mode in policies_cfg:
        if mode != "resident_gpu":
            prepare_offload()
        if mode == "dwell":
            for wm in window_mults:
                for k in dwells:
                    trace = trace_dwell is not None and int(trace_dwell) == k and wm == window_mults[-1]
                    policies = _install_policies(
                        model, restrict_layers, originals, stores_c,
                        "dwell", capacity, top_k, k, warmup_tokens, trace, window_mult=wm,
                    )
                    stats = _run_policy(model, sequences, policies, stores_c, device, warmup_steps)
                    emit("dwell", k, stats, window_mult=wm)
                    if trace:
                        _dump_residency(
                            policies, restrict_layers, num_experts, seq_lengths,
                            output_dir / "residency_trace.npz",
                        )
        elif mode == "lru_evict":
            for ev in [str(x) for x in eval_config.get("lru_evict_variants", ["lfu", "logit"])]:
                policies = _install_policies(
                    model, restrict_layers, originals, stores_c,
                    "lru", capacity, top_k, 1, warmup_tokens, trace=False,
                )
                for pol in policies:
                    pol.evict = ev
                stats = _run_policy(model, sequences, policies, stores_c, device, warmup_steps)
                emit(f"lru_{ev}", None, stats)
                if reference_nll is not None:
                    assert abs(stats["nll"] - reference_nll) < 5e-3, (
                        f"lru_{ev} must be exact: {stats['nll']} vs {reference_nll}"
                    )
        elif mode == "learned":
            if beta_head is None:
                print("[offload] SKIP learned arm: no --learned-beta-state/--policy-head-state")
                continue
            for thr in learned_thresholds:
                policies = []
                for i in restrict_layers:
                    pol = LearnedLayerPolicy(
                        stores_c[i], capacity, top_k, i, beta_head, policy_head,
                        thr, max_length,
                    )
                    model.model.layers[i].mlp = OffloadMoeBlock(
                        originals[i].gate, pol, _norm_topk_prob(originals[i])
                    )
                    policies.append(pol)
                stats = _run_policy(model, sequences, policies, stores_c, device, warmup_steps)
                dwell_mean = sum(p_.mean_dwell for p_ in policies) / len(policies)
                emit("learned", None, stats, beta_threshold=thr, mean_dwell=round(dwell_mean, 2))
        else:
            stores = (
                {} if mode == "resident_gpu"
                else stores_full if mode == "resident_pool"
                else stores_c
            )
            trace = mode == "lru" and bool(eval_config.get("dump_demand_trace", False))
            policies = _install_policies(
                model, restrict_layers, originals, stores or stores_c,
                mode, capacity, top_k, 1, warmup_tokens, trace=trace,
            )
            stats = _run_policy(model, sequences, policies, stores, device, warmup_steps)
            emit(mode, None, stats)
            if trace:
                _dump_residency(
                    policies, restrict_layers, num_experts, seq_lengths,
                    output_dir / "demand_trace.npz",
                )

    meta = {
        "model_id": model_config["id"],
        "revision": model_config.get("revision"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "capacity": capacity,
        "restrict_layers": restrict_layers,
        "top_k": top_k,
        "num_experts": num_experts,
        "bytes_per_expert": next(iter(stores_c.values())).bytes_per_expert if stores_c else 0,
        "max_length": max_length,
        "sequences": len(sequences),
        "router_state": str(router_state_path) if router_state_path else None,
        "torch": torch.__version__,
        "notes": "copy_ms_per_token is steady-state and contained in ms_per_token_mean; "
        "compare NLL as deltas vs this run's resident_gpu row",
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[offload] wrote {len(rows)} rows to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--router-state", type=Path, default=None)
    parser.add_argument("--learned-beta-state", type=Path, default=None)
    parser.add_argument("--policy-head-state", type=Path, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    run(config, args.output_dir, args.router_state, args.learned_beta_state, args.policy_head_state)


if __name__ == "__main__":
    main()
