"""Exhaustive check: demand-fetch farthest-reuse (set-Belady) vs. exact DP optimum.

Claim under test
----------------
For a demand stream of sets S_1..S_T (|S_t| = k experts demanded per step), over
schedules R_0 = empty, R_1..R_T with |R_t| <= C and S_t subseteq R_t (exactness),
the total number of insertions  sum_t |R_t \\ R_{t-1}|  is minimized by
DEMAND-FETCH FARTHEST-REUSE: on a miss, insert exactly the missing demanded
experts, and when over capacity evict the resident NOT in S_t whose next demand
is farthest in the future (never demanded again = +infinity; ties -> lowest id).
Prefetching (inserting non-demanded experts early) is allowed in the schedule
space searched by the DP and is claimed never to help.

Method
------
1. Exact optimum by dynamic programming over ALL states: every subset R of the
   N experts with |R| <= C, represented as bitmasks. Transitions R_prev ->
   R_next for every feasible R_next (S_t subseteq R_next, |R_next| <= C,
   R_next may contain ANY experts -- prefetching allowed), edge cost
   |R_next \\ R_prev|. Vectorized with numpy; per-demand-set transition data
   (feasible next states + full cost matrix) is cached across instances.
2. Greedy simulation of demand-fetch farthest-reuse with deterministic
   tie-breaking (lowest expert id).
3. Random instances: uniform demand sets at (N=6,C=3,k=2,T=10) and
   (N=8,C=4,k=3,T=12), plus skewed streams (zipfian expert popularity) and
   bursty streams (repeated-set bursts). Fixed seed. Low-ratio configs
   (N=8,C=4,k=1,T=12) and (N=8,C=6,k=2,T=12) match the served regime
   k/C = 0.25-0.375 (same three stream mixes).
4. Assert greedy cost == DP cost on every instance. On mismatch, print the full
   instance with both schedules and exit with a serialized counterexample.
"""

from __future__ import annotations

import bisect
import json
import sys
import time
from itertools import combinations

import numpy as np

SEED = 20260826


# --------------------------------------------------------------------------- #
# Exact DP over all states (bitmask subsets with |R| <= C)                    #
# --------------------------------------------------------------------------- #
class ExactDP:
    def __init__(self, N: int, C: int):
        self.N = N
        self.C = C
        self.full = (1 << N) - 1
        self.pc = np.array([bin(m).count("1") for m in range(1 << N)], dtype=np.int64)
        # All states: every subset of the N experts with |R| <= C (masks sorted).
        self.masks = np.array(
            [m for m in range(1 << N) if self.pc[m] <= C], dtype=np.int64
        )
        self.n_states = len(self.masks)
        self.state_index = {int(m): i for i, m in enumerate(self.masks)}
        # Cache per demand-set mask: (next-state indices, cost matrix).
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _step_data(self, s_mask: int) -> tuple[np.ndarray, np.ndarray]:
        d = self._cache.get(s_mask)
        if d is None:
            # Feasible next states: S_t subseteq R_next, |R_next| <= C.
            # R_next may contain ANY experts (prefetching allowed).
            nxt = self.masks[(self.masks & s_mask) == s_mask]
            nxt_idx = np.searchsorted(self.masks, nxt)
            # cost[i, j] = |R_next_i \ R_prev_j|
            cost = self.pc[(nxt[:, None] & ~self.masks[None, :]) & self.full]
            d = (nxt_idx, cost)
            self._cache[s_mask] = d
        return d

    def optimal_cost(self, s_masks: list[int]) -> int:
        INF = np.iinfo(np.int64).max // 4
        dp = np.full(self.n_states, INF, dtype=np.int64)
        dp[self.state_index[0]] = 0  # R_0 = empty
        for s in s_masks:
            nxt_idx, cost = self._step_data(s)
            vals = (cost + dp[None, :]).min(axis=1)
            dp = np.full(self.n_states, INF, dtype=np.int64)
            dp[nxt_idx] = vals
        return int(dp.min())

    def optimal_schedule(self, s_masks: list[int]) -> tuple[int, list[list[int]]]:
        """Same DP with backpointers; only used to print counterexamples."""
        INF = np.iinfo(np.int64).max // 4
        dp = np.full(self.n_states, INF, dtype=np.int64)
        dp[self.state_index[0]] = 0
        layers = []  # (nxt_idx, argmin over prev state index)
        for s in s_masks:
            nxt_idx, cost = self._step_data(s)
            tot = cost + dp[None, :]
            arg = tot.argmin(axis=1)
            vals = tot[np.arange(len(nxt_idx)), arg]
            layers.append((nxt_idx, arg))
            dp = np.full(self.n_states, INF, dtype=np.int64)
            dp[nxt_idx] = vals
        best_cost = int(dp.min())
        # Trace back.
        state = int(dp.argmin())
        rev = []
        for nxt_idx, arg in reversed(layers):
            pos = int(np.searchsorted(nxt_idx, state))
            rev.append(int(self.masks[state]))
            state = int(arg[pos])
        masks_seq = list(reversed(rev))
        return best_cost, [mask_to_list(m) for m in masks_seq]


def mask_to_list(m: int) -> list[int]:
    return [i for i in range(m.bit_length()) if (m >> i) & 1]


# --------------------------------------------------------------------------- #
# Greedy: demand-fetch farthest-reuse                                         #
# --------------------------------------------------------------------------- #
def greedy_farthest_reuse(
    N: int, C: int, sets: list[list[int]]
) -> tuple[int, list[list[int]]]:
    occ: list[list[int]] = [[] for _ in range(N)]
    for t, S in enumerate(sets):
        for e in S:
            occ[e].append(t)
    R: set[int] = set()
    cost = 0
    schedule: list[list[int]] = []
    for t, S in enumerate(sets):
        S_set = set(S)
        missing = S_set - R
        cost += len(missing)  # demand-fetch only: insert exactly the misses
        R |= missing
        while len(R) > C:
            # Evict resident not in S_t with farthest next demand;
            # never-demanded-again = +inf; ties -> lowest id (strict '>' keeps
            # the first, i.e. lowest, id among ties in sorted order).
            best_e, best_nu = -1, -1.0
            for e in sorted(R):
                if e in S_set:
                    continue
                i = bisect.bisect_right(occ[e], t)
                nu = occ[e][i] if i < len(occ[e]) else float("inf")
                if nu > best_nu:
                    best_e, best_nu = e, nu
            R.remove(best_e)
        schedule.append(sorted(R))
    return cost, schedule


# --------------------------------------------------------------------------- #
# Stream generators                                                           #
# --------------------------------------------------------------------------- #
def gen_uniform(rng: np.random.Generator, N: int, k: int, T: int) -> list[list[int]]:
    return [sorted(rng.choice(N, size=k, replace=False).tolist()) for _ in range(T)]


def gen_zipf(
    rng: np.random.Generator, N: int, k: int, T: int, a: float = 1.5
) -> list[list[int]]:
    w = 1.0 / np.arange(1, N + 1) ** a
    p = w / w.sum()
    return [sorted(rng.choice(N, size=k, replace=False, p=p).tolist()) for _ in range(T)]


def gen_bursty(rng: np.random.Generator, N: int, k: int, T: int) -> list[list[int]]:
    """Repeated-set bursts: pick a set, repeat it 1-4 times, repeat."""
    all_sets = list(combinations(range(N), k))
    out: list[list[int]] = []
    while len(out) < T:
        S = list(all_sets[rng.integers(len(all_sets))])
        for _ in range(int(rng.integers(1, 5))):
            if len(out) >= T:
                break
            out.append(sorted(S))
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
CONFIGS = [
    # (name, N, C, k, T, generator, count)
    ("uniform_N6_C3_k2_T10", 6, 3, 2, 10, gen_uniform, 15000),
    ("uniform_N8_C4_k3_T12", 8, 4, 3, 12, gen_uniform, 5000),
    ("zipf1.5_N6_C3_k2_T10", 6, 3, 2, 10, gen_zipf, 1000),
    ("zipf1.5_N8_C4_k3_T12", 8, 4, 3, 12, gen_zipf, 1000),
    ("bursty_N6_C3_k2_T10", 6, 3, 2, 10, gen_bursty, 1000),
    ("bursty_N8_C4_k3_T12", 8, 4, 3, 12, gen_bursty, 1000),
    # Low-ratio configs matching the served regime k/C = 0.25-0.375
    # (k=8 of 64 experts routed, C=32-ish resident: k/C = 0.25).
    ("uniform_N8_C4_k1_T12", 8, 4, 1, 12, gen_uniform, 3000),
    ("zipf1.5_N8_C4_k1_T12", 8, 4, 1, 12, gen_zipf, 3000),
    ("bursty_N8_C4_k1_T12", 8, 4, 1, 12, gen_bursty, 3000),
    ("uniform_N8_C6_k2_T12", 8, 6, 2, 12, gen_uniform, 3000),
    ("zipf1.5_N8_C6_k2_T12", 8, 6, 2, 12, gen_zipf, 3000),
    ("bursty_N8_C6_k2_T12", 8, 6, 2, 12, gen_bursty, 3000),
]


def main() -> int:
    rng = np.random.default_rng(SEED)
    solvers: dict[tuple[int, int], ExactDP] = {}
    total = 0
    t0 = time.time()
    for name, N, C, k, T, gen, count in CONFIGS:
        key = (N, C)
        if key not in solvers:
            solvers[key] = ExactDP(N, C)
        dp = solvers[key]
        for i in range(count):
            sets = gen(rng, N, k, T)
            s_masks = [sum(1 << e for e in S) for S in sets]
            opt = dp.optimal_cost(s_masks)
            g_cost, g_sched = greedy_farthest_reuse(N, C, sets)
            total += 1
            if g_cost != opt:
                opt_cost, opt_sched = dp.optimal_schedule(s_masks)
                inst = {
                    "config": name,
                    "instance_index": i,
                    "N": N,
                    "C": C,
                    "k": k,
                    "T": T,
                    "demand_sets": sets,
                    "greedy_cost": g_cost,
                    "greedy_schedule": g_sched,
                    "dp_cost": opt_cost,
                    "dp_schedule": opt_sched,
                }
                print("COUNTEREXAMPLE FOUND")
                print(json.dumps(inst, indent=2))
                return 1
        print(
            f"[{time.time() - t0:7.1f}s] {name}: {count} instances OK "
            f"(greedy == DP on all)"
        )
    print(f"ALL OK: {total} instances checked, greedy == DP optimum on every one.")
    print(f"Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
