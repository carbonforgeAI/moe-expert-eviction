# Claims and their evidence

This ledger maps the paper's empirical headline and appendix claims to their retained
evidence. Raw measurement outputs are in `results/`, recorded demand traces in `data/`,
the experiment log in `docs/EXPERIMENT_LOG.md`, and reproduction commands in
`README.md`. Rows marked "summary only" lack raw artifacts in this repository
and are scoped that way in the paper.
Unless noted, measurements are one L40S, batch-1 teacher-forced decode, frozen
OLMoE-1B-7B with a reuse-biased router, 8 restricted layers, pool capacity C=32, sixteen
held-out sequences of up to 384 tokens.

## Main text

| Claim | Number | Evidence |
|---|---|---|
| LRU is lossless but costs throughput | 49.0 → 38.3 tok/s (~22%), ΔNLL +0.000 | `results/e5a_frontier.jsonl` |
| No evaluated set-holding policy beats lossless LRU at native transfer cost without paying quality | ≥ +0.067 nats | `results/e5a_frontier.jsonl` |
| Cost model predicts the winner | predicted winner matches among six measured arms at τ×10 and τ×50; no full-ranking or universal absolute-error claim | `results/tau10.jsonl`, `results/tau50.jsonl`; copy and step events have different warm-up footprints, so stored compute residuals are not interpreted |
| τ varies fivefold across GPUs | 0.24 → 1.17 ms/expert | `docs/EXPERIMENT_LOG.md` § tau table; **summary only** for non-L40S GPUs |
| Farthest-reuse replacement halves LRU traffic at zero quality cost | 9.6 → 4.6 transfers/token (2.1×) | `results/e5a_belady_bound.json`, `src/moe_eviction/experiments/belady_bound.py` |
| Farthest-reuse schedule certified near-optimal | B\* ∈ [14,587, 15,087] (greedy is at most 3.5% above optimum) | `results/e10_flat_bound_eval.json`, `results/e10_flat_bound_perstream.json` |
| Greedy set-Belady matches exact optimum on checked small instances | 42,000 instances, including the served k/C ratio; this is a finite check, not a general proof | `src/moe_eviction/experiments/belady_optimality_check.py` (re-run; ~5 s) |
| Sixteen tokens of foresight recover 96% | k=16 → 96.0% held-out | `results/e9_lookahead.json` |
| Causal rungs recover ~27–33% | LFU-EMA 27%, trained scorer 32–33% held-out | `results/e6_evict_policies.json`, `results/e8_imitation.json`, `results/e9_imitation.json` |
| Learned controller dominated at thr ≤ 0.7 | thr 0.5: +0.076 nats at 12.3 ρ vs dwell +0.067 at 4.5 | `results/e5b_threshold_sweep.jsonl` |
| Controller low-traffic endpoints are high-loss, model-derived points at 50× | thr 0.8: +0.238 nats at 1.93 ρ; thr 0.9: +0.350 at 0.87 ρ | `results/e5b_threshold_sweep.jsonl`; Eq. 1 projection, not part of the six-arm physical slow-tier validation |
| μ short-horizon good, long-horizon bad | +0.030 at ~3-token refresh; +0.350 at dwell 73 | `docs/EXPERIMENT_LOG.md` § E5 |
| MMLU: learned below causal-recency baseline | 85.2% ± 1.3 vs 88.9%, 3 seeds, n=5000 | `docs/EXPERIMENT_LOG.md` § MMLU-5k; **aggregate summary only**, per-question outputs and heads absent |
| Router fine-tune moves no measured traffic | 32,050 vs 31,915; 10,963 vs 10,954 at trained layer | `results/e7_workingset_*.jsonl`, `results/e7_l8_*.jsonl` |
| Re-pricing the objective at measured τ does not reach the frontier (Appendix D) | price sweep e ∈ {0, .005, .02, .08} | `results/e12_eval_price*.jsonl` |

## Appendix

| Claim | Evidence |
|---|---|
| Pool-path NLL checks | `src/moe_eviction/experiments/offload_wallclock.py` (assert paths), `results/e5a_frontier.jsonl`; tolerance is 5e-3, not bit identity with the native kernel |
| Closed-loop eviction ladder incl. router-logit failure (46% worse) | `docs/EXPERIMENT_LOG.md` § E6; **aggregate summary only**, raw closed-loop log absent |
| E8 imitation rediscovers LFU (linear weights on frequency features) | `results/e8_imitation.json`, `results/e9_features.json` |
| E9 ablation: history-only 26%, probs-only 25%, combined 32%, GRU 33% | `results/e9_imitation.json` |
| Shuffle-label controls collapse | `results/e8_imitation.json`, `results/e9_imitation.json` (shuffle fields) |
| Per-stream sandwich: 14% of streams exactly optimal, median gap 2.9% | `results/e10_flat_bound_perstream.json` |
| Working-set router physical transfer table | `results/e7_workingset_*.jsonl`, `results/e7_l8_*.jsonl`; compared within each model lineage; reward-trained variants do not improve traffic, while SFT is 0.7% lower |
| Working-set reward improves its simulated training instrument | `docs/EXPERIMENT_LOG.md` § E7; **aggregate summary only**, raw simulator output absent |
| Disjoint-shard replication | `docs/EXPERIMENT_LOG.md` § protocol validation; **aggregate summary only**, raw shard logs absent |
| Warm-start/window ablation | `results/e5a_frontier.jsonl` |
| vLLM expert-cache context (35.1 eager tok/s) | public vLLM PR #37190 discussion; external reported context, not validation |

## Known gaps a reviewer should probe

- **E12 is one seed** per price point, matched steps/data to the shipped heads.
- **The working-set variant's protocol mismatch is real and stated**: its reward/metrics are
  rollout-based at one layer with C=16; ours are teacher-forced across 8 layers at C=32.
  The paper argues both readings; check that the caveat text still matches the numbers.
- **Estimated vs measured throughput**: farthest-reuse throughput and the plotted
  slowed-tier points are model-derived; the captions label them as such. The predicted
  winner among six arms was separately validated with physical 10×/50× runs.
- **Uncertainty is not established for Figure 1.** Its points are token-weighted
  aggregate NLL differences, while the retained per-sequence dispersion gives every
  sequence equal weight. The paper therefore omits error bars and makes no statistical-
  significance claim.
- **The farthest-reuse optimality argument is a sandwich, not a proof of greedy optimality.** Check
  that no sentence claims more than the bracket supports.
- Heavy artifacts (merged checkpoints, trained head weights, per-run logs) live on the
  original cluster and are not in this repo.
- **Replication, cross-GPU tau, MMLU, the closed-loop E6 run, and the working-set
  simulator have summary-only evidence** in the experiment log. The paper scopes these
  claims and does not imply raw artifacts are shipped.
