# Recorded demand traces

These two files are the recorded per-token expert-demand streams that the offline
results replay (§3.2, the eviction-ladder appendix, and the formal appendix all rest on
them). Each `.npz` holds, per `(layer, seq)` stream: `layer{L}_seq{S}_demand` (`[T, 8]`
int16, the router's true top-8 expert ids per token), `_resident`, `_refresh`, and
`_probs` (`[T, 64]` fp16, the router's full softmax at each token; rows sum to 1).
Eight restricted layers {4,5,6,7,8,11,12,13}, 64 experts, pool capacity C=32, ragged
sequences of up to 384 tokens, frozen OLMoE-1B-7B with the reuse-biased router.

- `demand_trace_eval16.npz` — the evaluation trace: 16 OpenHermes sequences
  (stream skip 2080). Its demand arrays are bit-identical to the trace behind every
  reported replay number (LRU 31,776 / LFU-EMA 27,285 / Belady 15,087 steady transfers;
  held-out seqs 12–15: 10,635 / 9,011 / 4,985).
- `demand_trace_train48.npz` — the disjoint training trace for the learned evictors:
  48 sequences (stream skip 4000), ~105k eviction events.

Regenerate with `configs/offload_wallclock_traceprobs16.yaml` /
`...traceprobs48.yaml` (LRU arm, `dump_demand_trace: true`). Replay with
`src/moe_eviction/experiments/evict_policies.py`, `belady_bound.py`, `belady_lookahead.py`,
`belady_flat_bound.py`, and train the imitation evictors with
`belady_imitation.py` / `belady_imitation_e9.py` (see the README; `make reproduce` runs them).
