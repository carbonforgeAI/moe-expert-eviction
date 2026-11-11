"""Causal eviction policies replayed on the recorded demand trace: how much of the
LRU-to-Belady gap does eviction skill recover, without touching routing?

Family-A policies only (routing untouched, quality exact by construction); the sole
decision is the victim on a miss. Replayed offline on demand_trace.npz:

  lru        evict the least-recently-touched resident       (the deployed baseline)
  lfu        evict the resident with the lowest exponentially-decayed use frequency
  reuse      evict the resident with the largest ESTIMATED time to next use, where the
             estimate is a causal per-(layer, expert) running mean of inter-arrival
             times (a statistical Belady imitator in the Hawkeye/Parrot tradition:
             predict reuse distance, evict the farthest)
  belady     evict the resident whose TRUE next use is farthest (oracle floor)

All policies share the harness's accounting (misses are 'initial' until the pool has
been full once) and the touch-first discipline (a currently-selected expert is never
the victim).

Usage: python -m moe_eviction.experiments.evict_policies <run_dir> [--capacity 32]
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def replay(demand: np.ndarray, capacity: int, policy: str,
           decay: float = 0.95) -> tuple[int, int]:
    T = demand.shape[0]
    next_use: dict[int, list[int]] = {}
    if policy == "belady":
        for t in range(T - 1, -1, -1):
            for e in demand[t]:
                next_use.setdefault(int(e), []).append(t)
    resident: OrderedDict[int, None] = OrderedDict()
    freq: dict[int, float] = {}
    last_seen: dict[int, int] = {}
    gaps: dict[int, tuple[float, int]] = {}  # expert -> (mean inter-arrival, count)
    steady = initial = 0
    full_once = False
    for t in range(T):
        needed = [int(e) for e in demand[t]]
        # causal statistics update (uses only the past, including this arrival)
        for e in needed:
            freq[e] = freq.get(e, 0.0) * 1.0 + 1.0
            if e in last_seen:
                gap = t - last_seen[e]
                m, c = gaps.get(e, (0.0, 0))
                gaps[e] = ((m * c + gap) / (c + 1), c + 1)
            last_seen[e] = t
        for e in freq:
            freq[e] *= decay
        if policy == "belady":
            for e in needed:
                uses = next_use.get(e)
                while uses and uses[-1] <= t:
                    uses.pop()
        for e in needed:
            if e in resident:
                resident.move_to_end(e)
        for e in needed:
            if e in resident:
                continue
            if len(resident) >= capacity:
                needed_set = set(needed)
                cands = [r for r in resident if r not in needed_set]
                if policy == "lru":
                    victim = cands[0]
                elif policy == "lfu":
                    victim = min(cands, key=lambda r: freq.get(r, 0.0))
                elif policy == "reuse":
                    def eta(r: int) -> float:
                        m, c = gaps.get(r, (float("inf"), 0))
                        exp_next = (last_seen.get(r, -10**9) + (m if c else float("inf")))
                        overdue = t - last_seen.get(r, -10**9)
                        # expected next use; if overdue beyond its mean gap, treat the
                        # remaining wait as another full mean gap (memoryless-ish)
                        return exp_next if exp_next > t else t + (m if c else float("inf")) - min(overdue, m if c else 0)
                    victim = max(cands, key=eta)
                elif policy == "belady":
                    def nxt(r: int) -> int:
                        uses = next_use.get(r)
                        return uses[-1] if uses else 10**9
                    victim = max(cands, key=nxt)
                else:
                    raise ValueError(policy)
                resident.pop(victim)
            resident[e] = None
            if full_once:
                steady += 1
            else:
                initial += 1
            if len(resident) >= capacity:
                full_once = True
    return steady, initial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--capacity", type=int, default=32)
    args = parser.parse_args()
    data = np.load(args.run_dir / "demand_trace.npz")
    keys = sorted(k for k in data.files if k.endswith("_demand"))
    out = {}
    for policy in ("lru", "lfu", "reuse", "belady"):
        s = i = 0
        for k in keys:
            a, b = replay(data[k], args.capacity, policy)
            s += a; i += b
        out[policy] = s
        print(f"{policy:>7s}: {s} steady transfers")
    lru, bel = out["lru"], out["belady"]
    for p in ("lfu", "reuse"):
        rec = (lru - out[p]) / (lru - bel) if lru > bel else 0.0
        print(f"{p:>7s} recovers {rec:.1%} of the LRU->Belady gap")
    (args.run_dir / "evict_policies.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
