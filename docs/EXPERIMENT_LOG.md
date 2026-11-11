> Experiment log kept during the project, lightly edited for release. Experiment ids
> (E1-E12) are the ones used in `docs/CLAIMS.md`; E11 was a paper-only derivation and is
> not part of the released paper. Job numbers refer to the original cluster runs.

# Measured wall-clock frontier of expert-residency policies (E1-E5, corrected)

Date: 2026-08-25 (supersedes the same-day pre-referee version; dwell numbers before the
warm-start fix are invalid — see Corrections).

**Scope.** Batch-1, latency-bound, teacher-forced decode of a FROZEN OLMoE-1B-7B-0125
(+ ReMoE router, arXiv:2605.27081-style) on one L40S; 8 of 16 layers' experts held only in
pinned host RAM at C=32-of-64 per layer; pool refreshes are real synchronous H2D copies on
the decode critical path, timed with CUDA events. Pools reset per sequence (16 x <=384
OpenHermes); transfer rates therefore include a per-sequence warm-up and OVERSTATE
steady-state rates for long-context serving. At batch >= ~8 the per-step demand union
approaches all experts and the residency game degenerates — nothing here speaks to
batched serving.

## Protocol validation

- Base NLL 1.1052 vs the recorded batched-eval base 1.1048 (4e-4); in-run asserts force
  resident_pool / lru / dwell_1 to reproduce ceiling NLL.
- Copies run at 0.51 ms/expert (23 GiB/s) — full pinned-PCIe bandwidth; per-expert loop ==
  contiguous reference (bench_copy_paths); batched staging is slower. tau here means the
  measured per-expert copy cost of THIS path.
- Linear cost model ms/tok = compute + xf_rate x tau: its predicted winner matches the six
  measured arms at tau x{10,50} via physically repeated copies (real DMA traffic). Copy
  and step-event warm-up footprints differ, so these files do not support a universal
  absolute-error bound or subtraction of copy time to obtain a compute residual. The x10/x50 runs are a
  TAU SWEEP, not hardware emulation: x1 already IS pinned CPU RAM over PCIe4; x10 = 2.3
  GiB/s and x50 = 0.46 GiB/s bracket slow interconnects and un-tuned storage paths (a good
  Gen4 NVMe with GDS sits nearer x4-x8).
- Every ordering below replicates on a disjoint shard (skip 2200, job 10501530).
- Production caveat: copies here are deliberately unoverlapped. Systems that prefetch/overlap
  pay ~max(compute, copy), not the sum — overlap helps LRU most (its misses are small and
  frequent), so faster runtimes + overlap strengthen exact serving on fast links; the
  slow-tier orderings (copy >> compute) are overlap-robust.

## Corrected frontier (16x384, skip 2080; dNLL vs same-run resident_gpu; xf = steady transfers/token)

```text
policy               dNLL    +-SEM   tok/s   xf/tok  compute_ms
full residency      +0.000    -      49.0     0.00     20.4
LRU exact           +0.000   .001    38.3     9.59     21.2     (per-token admission)
BELADY exact bound  +0.000    -      42.5     4.58     21.2     (clairvoyant eviction, offline)
dwell16 window-x8   +0.067   .050    42.2     4.52     21.4
dwell64 window-x1   +0.085   .049    43.5     3.89     21.0
static_warm         +0.110   .050    44.0     3.39     21.0
learned thr 0.1     +0.030   .008    24.4    31.6      24.9
learned thr 0.5     +0.076   .012    33.8    12.3      23.4
learned thr 0.7     +0.172   .034    40.9     3.54     22.7
learned thr 0.9     +0.350   .151    43.6     0.87     22.5
```

SEMs are paired per-sequence (n=16): the heuristic band's internal ordering (dwell-wx8
variants vs static_warm, +-0.02-0.04 gaps at SEM ~0.05) is NOT resolved; the large gaps
(exact vs +0.07 band vs learned thr>=0.7) and all transfer counts (deterministic) are.

## Findings

1. **Exact serving is cheap on fast links and has 2.1x learnable headroom.** LRU holds
   ceiling quality at 18-20% throughput cost vs the matched resident_pool ceiling (a
   compute-substrate-dependent ratio: at production compute the ratio grows; with overlap
   it shrinks — quote xf=9.59 and tau, not the percentage). Offline Belady on the recorded
   demand trace needs only 4.58 xf/tok at the same zero quality loss (LRU/Belady = 2.11).
   "Nothing to learn" is FALSE in the strong sense: learned EVICTION inside exact serving
   could halve traffic quality-free. That, not termination, is the profitable learning
   target — and a demand predictor (mu) is its natural primitive.
2. **Warm-started windowed heuristics form a flat +0.07 band at ~4.5 xf/tok** (dwell-wx8,
   any k in 8..128) — the same transfer rate Belady-exact achieves at ZERO quality cost.
   Window decoupling (profiling window >> refresh cadence) beats window=dwell and
   static_warm on both shards; profiling length, not cadence, is what matters.
3. **The learned termination controller (seed-0 beta+mu heads, trained in simulation with
   cost-blind objectives) buys nothing at any tau <= 25.5 except the extreme-speed end.**
   Its threshold sweep traces a smooth curve strictly inside the heuristic-or-exact
   frontier for thr <= 0.7 (e.g. thr 0.5: quality equal to the heuristic band, 2.7x its
   transfers, plus ~2-4 ms/tok eager-mode controller cost — but it is dominated on
   transfers+quality alone, before charging the controller). Only thr 0.8/0.9 hold the
   sub-3.4 xf/tok regime (nothing else measured lives there), at +0.24-0.35 nats; at tau
   x50 thr 0.9 is the fastest restricted arm (37-22 tok/s across x10-x50). Two honesty
   notes: (a) this low-xf monopoly partly reflects the per-sequence warm-up our protocol
   charges heuristics every 384 tokens; (b) this evaluates ONE trained instance — the
   defensible reading is that a SWITCH-RATE-PROXY-trained controller does not transfer to
   physically priced serving, which is precisely the evaluation gap in proxy-only work
   (arXiv:2604.20156 reports switch rates only, moves no bytes, and reports no wall-clock).
4. **mu is a short-horizon demand tracker, not a long-horizon predictor**: near-exact pools
   when refreshed every ~3 tokens (+0.030, 4x better than recency at matched cadence), but
   +0.350 at mean dwell 73 where a 64-token recency profile gives +0.110.
5. **Selection is a (tau, quality-budget) lookup, demonstrated on the measured grid**:
   budget ~0 -> LRU at every tau (Belady bounds the learnable improvement); budget ~0.07 ->
   dwell-wx8 at x1 (42 tok/s) through x50 (7.3); budget ~0.35 -> learned thr 0.9 wins only
   at x50 (22.4 vs 8-9 for heuristics). Model-predicted winners match the measured winners
   at all three taus (the tau-sweep rows for pre-correction arms validated the mechanism;
   corrected-arm tier numbers are model-derived from measured xf and compute).

## Corrections vs the pre-referee version

- Dwell arms previously froze cold-start pools padded by ascending-expert-id tie-breaks
  (up to ~half of dwell_32's reported +0.245 was artifact). All dwell numbers here use the
  warm-start fix (refresh every token until the profiling window first fills).
- Previous "CPU-RAM/NVMe class" tier labels replaced by explicit effective bandwidths.
- Previous claim "faster production runtimes strengthen temporal extension on PCIe"
  withdrawn: with overlap the opposite holds on fast links (see Scope).
- Previous "learned dominated essentially everywhere" scoped per Finding 3.

Runs: E1 10499988; E4 10500747; tau sweep 10501037/10501038; corrected E5a 10501626,
threshold sweep E5b 10501529, replication E5c 10501530; Belady belady_bound.py on E5a's
demand_trace.npz. Figures: frontier_corrected.png, residency_layer8_seq0.png.

## Measured hardware tau table (bench_copy_paths, per-expert path, 12.6 MB bf16 expert)

```text
GPU                     link          tau (ms/expert)   eff GiB/s   where
H100 80GB HBM3          PCIe5             0.236           49.7      nibi
A100-SXM4-40GB          PCIe4 x16         0.478           24.5      cluster A
L40S                    PCIe4 x16         0.510           23.0      cluster A (harness node)
L4                      PCIe4 x8          1.030           11.4      GCP g2-standard-8
Quadro RTX 8000         PCIe3 x16         1.169           10.0      cluster A
```

Pending: A6000, H200, second A100. V100 dropped:
torch 2.12/cu13 ships no sm_70 kernels. Methodology fingerprint (per-expert loop ==
contiguous bandwidth; batched staging slower) reproduced on every platform measured.
Hardware alone spans ~5x in tau; the emulated x10/x50 sweep extends the validated axis to
storage-class costs. Policy orderings at any tier follow from the (validated) linear cost
model applied to measured xf rates; the L40S column is directly measured end-to-end.

## MMLU-5k final (3 seeds, jobs 10506741/42/43, 10506937, 10506745)

n=5000, matched 3.1% switch (dwell 32), base 48.4%: oracle 99.5% retention; causal-recency
88.9%; learned beta+mu 85.2% +- 1.3 SEM over seeds (87.3/85.7/82.8) — every seed below the
causal baseline, fourteen points under oracle. thr 0.5: 89.6/86.5/86.8. The 500-question
evaluations scattered seed retention across 82–96% (+-4.7pp/point at 95%): the legacy 96.2%
headline was a sampling artifact.

## E6 — scored eviction inside exact serving (job 10508769; offline: evict_policies.py)

Family-A victim scoring, quality exact by assert. Offline replay on the demand trace:
LFU-EMA 27,285 steady transfers (26.9% of the LRU->Belady gap recovered), naive mean
inter-arrival predictor 34,694 (WORSE than LRU: bursty demand, recency is the signal).
Closed-loop in the harness: lru 31,609 / lru_lfu 27,118 (27.2% recovery, matches offline
within 0.6%, copy 4.2 vs 4.9 ms/tok) / lru_logit 46,079 (46% WORSE — instantaneous router
logit is the wrong horizon, thrashes just-used experts). Belady floor 15,087. Pattern:
integrate-the-past signals help, one-shot signals hurt; ~73% of the gap remains for trained
imitation (Hawkeye/Parrot).

H200 (tamia, whole node): tau = 0.2363 ms/expert, 49.6 GiB/s — identical to nibi's H100
(0.2359) across two clusters; PCIe5 tier reproduces to 0.4%.

## E7 — GRPO working-set router models (Calanzone et al.) under physical measurement

Models: allenai/OLMoE-1B-7B-0125-Instruct baseline; SFT control and
prompt-conditioned checkpoint (GRPO with simulated LRU-hit-ratio reward at ONE middle
layer, c=16, DAPO + SFT-combined), both merged via peft merge_and_unload
(scripts/merge_workingset_adapters.py, SLURM CPU job). All runs ROUTER_STATE="" (their fine-tuned
routers must not be clobbered by the ReMoE state the run script defaults to). Protocol as
E5: wikitext 16x384 teacher-forced, L40S.

Full protocol (8 layers, C=32; jobs 10509319/21/22/23), LRU steady transfers @ base NLL:
  instruct 31,915 @ 1.382 | sft 31,686 @ 1.350 | pc 32,050 @ 1.397 | pc+tag 33,101 @ 1.545*
  (*tag NLL not comparable: the [CACHE_SIZE=16] tokens are scored.)
Matched operating point (restrict layer 8 only, C=16 — his exact trained scope; jobs
10510064/65/66), LRU transfers:
  instruct 10,954 | pc 10,963 | pc+tag 11,223 (+2.5%)

Verdict: no measured traffic reduction from the working-set training under either protocol
— not diluted across layers, absent at the trained layer and budget too. Controls behave
(sft ~ instruct). LFU-EMA eviction replicates ~15% traffic cut on all four variants
(quality exact) — that result is training-invariant. Layer-8 aside: a dwell-16 pool at
layer 8 alone costs almost nothing (~1.5k transfers, NLL -0.012 to -0.018 vs resident on
all three variants).

Caveats for fairness: his reward was computed on the model's own generations (rollouts),
ours is teacher-forced; his reported hit-ratio gains were measured in the same one-layer
simulator that provided the reward. The measurement says the simulator's gains do not
transfer to physically counted bytes — the same proxy-vs-priced gap the paper documents
for switch rate.

## E8 — first trained Belady imitation (offline, belady_imitation.py; adversarially audited)

Listwise victim scorer (linear + 2x64 MLP, torch, CPU ~4min) over strictly causal
access-history features (recency, EMA frequencies 0.90/0.95/0.99, uses-in-16/64,
inter-arrival mean/std, overdue ratio, ...), trained to imitate Belady's victim on seqs
0-11 (states collected under both LRU and Belady teacher replays, DAgger-style), closed
replay on held-out seqs 12-15 (model picks every victim online). Held-out steady
transfers: LRU 10,635 / LFU-EMA 9,011 (28.7% gap recovered) / linear 9,809 (14.6%) /
MLP 9,231 (24.9%) / Belady 4,985. Shuffled-label control collapses to 18,300 (-136%),
independent from-scratch simulator reproduces baselines exactly, all numbers
deterministic-reproduced twice; audit verdict PASS. Headline: the linear weights load
almost entirely on frequency features (evict-if-rare sign) — on access history alone,
imitation REDISCOVERS LFU and slightly trails the hand-built score. The remaining ~2/3
of the gap needs richer inputs (router logits, hidden state). Caveat: 4-seq holdout,
dominated by seq 14. In paper appendix "imitation ladder". Artifacts:

learned approaches", new appendix "Measuring a working-set fine-tuned model" with table;
E8 added to imitation-ladder appendix); main text re-fitted to exactly 4 pages.

## E9 — bounded-lookahead Belady + router-forecast features (adversarially audited, PASS)

Traces regenerated with full 64-way router softmax per token (fp16) in demand_trace.npz
(jobs 10512446 eval/16-seq — demand bit-identical to E5a; 10512447 train/48-seq disjoint,
skip 4000, 108,313 LRU transfers).

Lookahead (belady_lookahead.py; k-token truncated oracle, LRU tie-break beyond k;
independent reimplementation matched exactly): held-out gap recovered k=1 17.5% / 2 30.9%
/ 4 51.4% / 8 76.3% / 16 96.0% / 32 99.7%; full-16 within ~2pp. THE ORACLE'S ADVANTAGE IS
SHORT-HORIZON: 16 tokens of foresight ~ 96%.

Models (belady_imitation_e9.py; trained on 48-seq trace, val seqs 40-47, eval trace fully
unseen; baselines reproduce E8 bit-exactly): held-out 12-15 gap recovered — history-only
MLP 26.4% (up from E8's 24.9% with 3.4x data), probs-only 24.5%, combined MLP 32.0%
(selected by train-val), combined GRU 33.5% (reported, not selected); LFU-EMA 28.7%.
FIRST RUNG TO CLEAR THE HAND-BUILT RULE on the comparable holdout; on all 16 eval seqs
learned and LFU tie (25.3-25.4 vs 26.9%). Shuffle control -34.2%. Top features: history
frequency + prob_max_8/32, logprob_cur. Combined scorer ~ 2 tokens of equivalent
foresight. Audit: full bit-exact reruns, numeric causality corruption test, SHA-256
train/eval disjointness, LRU-imitating scorer reproduces LRU through the same replay.
paragraphs (v9). Headings de-blogged in v9; title under discussion.

## E10 — formal section + optimality certificates (adversarially reviewed)

Paper gains appendix "The two families, formally": exact-serving traffic as a clean
optimization (B* over schedules with S_t ⊆ R_t, |R_t| ≤ C; prefetch-elimination lemma),
PROPOSITION: B_flat ≤ B* ≤ B_greedy (flattened single-access MIN is a certified lower
bound — the naive "flattening + exchange" PROOF of greedy optimality is WRONG, referee
counterexample {a,b},{c,d},{a,b} C=3: flat-OPT 4 < set-OPT 5); lossy family formalized as
frontier Q_C(rho) with zero located at B*/T (masked top-k = true top-k when S_t ⊆ P_t,
absent ties); selection rule = argmin c_i + rho_i tau with crossings (c_j−c_i)/(rho_i−rho_j).
Certificates: (1) belady_optimality_check.py — exact DP over ALL schedules incl.
prefetching, 24,000 random instances (uniform/zipf/bursty), greedy == optimum on every
one; (2) belady_flat_bound.py — on the REAL traces the sandwich is NOT tight: eval full16
B* ∈ [14,512, 15,087] (gap 3.8%), held-out [4,795, 4,985]; train48 [49,963, 51,858]. So
reported Belady counts are certified within 4% of optimal and the 2.1x headroom is
conservative (true ratio in [2.1, 2.2]). Referee also caught: ragged sequences (≤384, not
16x384 — paper wording fixed), dwell-16 rho=4.52 sits strictly BELOW B*/T=4.58 (softened
"policy suboptimality" claim), fill-accounting footing, |P_t| ≤ C, tie caveats.

## E12 — tau-priced training of the beta/mu heads (the price sweep)

Discovery: in best_of_pool mode the objective's "switch" term = beta * soft set difference
= EXPERTS MOVED (the load term is a stub, hardwired 0). So the shipped heads (lambda 0.05
per expert moved) carried an implicit exchange rate e ~ 0.098 nats/ms — hand-set, ~2-20x
above measured frontier slopes. E12 retrained seed-0 heads with the switch slot priced at
e*tau, e in {0 (salvaged degenerate arm), 0.005, 0.02, 0.08}, everything else matched
(jobs 10525441, 10526761/62/63; evals 10526764, 10529177, 10530727, 10533761, e5b sweep).

Training response monotone in price: beta-fire {95, 91, 42, 14}%, soft moved-experts
{8.9, 8.4, 4.2, 1.6}, lm_loss {0.70, 0.70, 0.72, 0.81}. Harness (L40S, thresholds
.1-.9): e=0 collapses to ~per-token full-pool refresh (xf 66-73, dNLL +0.003-0.05 — near
ceiling quality, 7.5x LRU traffic); e=0.005 ~ free; e=0.02 smooth dial xf 62->4.8
(+0.004 -> +0.214); e=0.08 reproduces the shipped curve nearly point for point
(thr.5 +0.089@11.8 vs shipped +0.076@12.3; thr.9 +0.379@0.73 vs +0.350@0.87).

VERDICT: measured pricing makes the controller a well-behaved price-indexed family and
recovers the shipped heads as one price point — but NO price reaches the frontier: at
matched traffic, dwell/static/LRU/Belady dominate every arm (e.g. +0.16@4.0 vs dwell
+0.085@3.9, Belady 0@4.6). The deficit is the hazard estimator (mu), not the price —
empirically confirming the index-theory reading (rent sweeps the family; estimator sets
its distance from the frontier). One seed, matched steps/data. In paper: one sentence in
§3.3 + "Pricing the objective, measured" paragraph in app:math.
