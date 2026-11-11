"""Bounded-lookahead Belady: how much foresight does the eviction oracle need?

At each eviction the oracle knows true next-use distances only up to k future
tokens: a resident whose next use lies within k tokens of now has a KNOWN
distance; all other residents (next use farther than k, or never) are
indistinguishable "beyond k". Policy on a miss:

  - if any eviction candidates have no known use within k, evict the
    least-recently-used among those (the beyond-k set carries no oracle
    information, so we fall back to recency);
  - otherwise every candidate's next use is known: evict the known-farthest.

k=0 is plain LRU (nothing is ever known); k=infinity reproduces full Belady
exactly (never-used residents are "beyond k" and full Belady's max() tie-break
over the recency-ordered candidate list is precisely LRU-among-never-used).

Replay accounting is identical to evict_policies.py: misses count as 'initial'
until the pool has been full once, then 'steady'; touch-first discipline (a
currently-selected expert is never the victim); per-(layer, seq) streams
replayed independently and summed.

Usage:
  python -m moe_eviction.experiments.belady_lookahead <trace.npz> \
      [--capacity 32] [--out lookahead.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

KS = [1, 2, 4, 8, 16, 32, 64, 128]
HELDOUT_SEQS = {12, 13, 14, 15}


def replay_lookahead(demand: np.ndarray, capacity: int, k: float) -> tuple[int, int]:
    """Replay one (layer, seq) demand stream under bounded-lookahead Belady.

    k is the lookahead horizon in tokens (math.inf for full Belady,
    0 for plain LRU). Returns (steady, initial) miss counts.
    """
    T = demand.shape[0]
    # next-use index: per expert, use times in DESCENDING order so the next
    # future use is always uses[-1]; identical construction to evict_policies.
    next_use: dict[int, list[int]] = {}
    for t in range(T - 1, -1, -1):
        for e in demand[t]:
            next_use.setdefault(int(e), []).append(t)
    resident: OrderedDict[int, None] = OrderedDict()
    steady = initial = 0
    full_once = False
    for t in range(T):
        needed = [int(e) for e in demand[t]]
        # consume this step's uses so uses[-1] is strictly in the future
        for e in needed:
            uses = next_use.get(e)
            while uses and uses[-1] <= t:
                uses.pop()
        # touch-first: refresh recency of already-resident selected experts
        for e in needed:
            if e in resident:
                resident.move_to_end(e)
        for e in needed:
            if e in resident:
                continue
            if len(resident) >= capacity:
                needed_set = set(needed)
                # cands preserves resident order = recency order, LRU first
                cands = [r for r in resident if r not in needed_set]

                def nxt(r: int) -> int:
                    uses = next_use.get(r)
                    return uses[-1] if uses else 10**9

                # residents whose next use is NOT within k tokens: unknown
                unknown = [r for r in cands if nxt(r) - t > k]
                if unknown:
                    victim = unknown[0]  # LRU among the indistinguishable
                else:
                    victim = max(cands, key=nxt)  # known-farthest
                resident.pop(victim)
            resident[e] = None
            if full_once:
                steady += 1
            else:
                initial += 1
            if len(resident) >= capacity:
                full_once = True
    return steady, initial


def replay_lru(demand: np.ndarray, capacity: int) -> tuple[int, int]:
    """Plain LRU (the k=0 anchor), same accounting."""
    resident: OrderedDict[int, None] = OrderedDict()
    steady = initial = 0
    full_once = False
    for t in range(demand.shape[0]):
        needed = [int(e) for e in demand[t]]
        for e in needed:
            if e in resident:
                resident.move_to_end(e)
        for e in needed:
            if e in resident:
                continue
            if len(resident) >= capacity:
                needed_set = set(needed)
                cands = [r for r in resident if r not in needed_set]
                resident.pop(cands[0])
            resident[e] = None
            if full_once:
                steady += 1
            else:
                initial += 1
            if len(resident) >= capacity:
                full_once = True
    return steady, initial


def total_over(keys: list[str], data, capacity: int, k: float | None) -> int:
    """Sum steady transfers over the given demand keys. k=None means LRU."""
    s = 0
    for key in keys:
        if k is None:
            steady, _ = replay_lru(data[key], capacity)
        else:
            steady, _ = replay_lookahead(data[key], capacity, k)
        s += steady
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = np.load(args.trace)
    demand_keys = sorted(k for k in data.files if k.endswith("_demand"))

    def seq_of(key: str) -> int:
        m = re.search(r"_seq(\d+)_demand$", key)
        assert m, key
        return int(m.group(1))

    all_keys = demand_keys
    held_keys = [k for k in demand_keys if seq_of(k) in HELDOUT_SEQS]

    # anchors
    lru_full = total_over(all_keys, data, args.capacity, None)
    lru_held = total_over(held_keys, data, args.capacity, None)
    bel_full = total_over(all_keys, data, args.capacity, math.inf)
    bel_held = total_over(held_keys, data, args.capacity, math.inf)
    print(f"LRU   (k=0):   full16 {lru_full:6d}   heldout {lru_held:6d}")
    print(f"Belady(k=inf): full16 {bel_full:6d}   heldout {bel_held:6d}")

    def gap(lru: int, bel: int, x: int) -> float:
        return (lru - x) / (lru - bel) * 100.0 if lru > bel else 0.0

    rows = []
    rows.append({
        "k": 0,
        "transfers_full16": lru_full,
        "gap_recovered_full16_pct": 0.0,
        "transfers_heldout": lru_held,
        "gap_recovered_heldout_pct": 0.0,
    })
    for k in KS:
        tf = total_over(all_keys, data, args.capacity, k)
        th = total_over(held_keys, data, args.capacity, k)
        rows.append({
            "k": k,
            "transfers_full16": tf,
            "gap_recovered_full16_pct": round(gap(lru_full, bel_full, tf), 2),
            "transfers_heldout": th,
            "gap_recovered_heldout_pct": round(gap(lru_held, bel_held, th), 2),
        })
        print(f"k={k:>4d}: full16 {tf:6d} ({rows[-1]['gap_recovered_full16_pct']:6.2f}%)"
              f"   heldout {th:6d} ({rows[-1]['gap_recovered_heldout_pct']:6.2f}%)")
    rows.append({
        "k": "inf",
        "transfers_full16": bel_full,
        "gap_recovered_full16_pct": 100.0,
        "transfers_heldout": bel_held,
        "gap_recovered_heldout_pct": 100.0,
    })

    out = {
        "trace": str(args.trace),
        "capacity": args.capacity,
        "heldout_seqs": sorted(HELDOUT_SEQS),
        "anchors": {
            "lru_full16": lru_full, "belady_full16": bel_full,
            "lru_heldout": lru_held, "belady_heldout": bel_held,
        },
        "curve": rows,
    }
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
