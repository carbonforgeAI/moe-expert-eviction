"""Clairvoyant (Belady) eviction bound on the recorded exact demand trace.

Referee fix: "nothing to learn at fast PCIe" requires knowing how far LRU is from the
clairvoyant transfer minimum at ceiling quality. Reads demand_trace.npz (dumped by the
lru arm with eval.dump_demand_trace), replays per-layer per-sequence demand at capacity C
under (a) LRU and (b) Belady (evict the resident expert whose next use is farthest), with
the harness's accounting (misses are 'initial' until the pool has been full once).

Usage: python -m moe_eviction.experiments.belady_bound <run_dir> [--capacity 32]
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def replay(demand: np.ndarray, capacity: int, policy: str) -> tuple[int, int]:
    """(steady_transfers, initial_loads) for one layer/sequence demand [T, top_k]."""
    T = demand.shape[0]
    next_use: dict[int, list[int]] = {}
    for t in range(T - 1, -1, -1):
        for e in demand[t]:
            next_use.setdefault(int(e), []).append(t)
    resident: OrderedDict[int, None] = OrderedDict()
    steady = initial = 0
    full_once = False
    for t in range(T):
        needed = [int(e) for e in demand[t]]
        for e in needed:
            if e in resident:
                resident.move_to_end(e)
        for e in needed:
            uses = next_use.get(e)
            while uses and uses[-1] <= t:
                uses.pop()
        for e in needed:
            if e in resident:
                continue
            if len(resident) >= capacity:
                if policy == "lru":
                    victim = next(iter(resident))
                else:  # belady: farthest next use among non-needed residents
                    needed_set = set(needed)
                    victim, victim_next = None, -1
                    for r in resident:
                        if r in needed_set:
                            continue
                        uses = next_use.get(r)
                        nxt = uses[-1] if uses else T + 1
                        if nxt > victim_next:
                            victim, victim_next = r, nxt
                    if victim is None:
                        victim = next(iter(resident))
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
    keys = [k for k in data.files if k.endswith("_demand")]
    totals = {"lru": [0, 0], "belady": [0, 0]}
    for k in sorted(keys):
        demand = data[k]
        for policy in ("lru", "belady"):
            s, i = replay(demand, args.capacity, policy)
            totals[policy][0] += s
            totals[policy][1] += i
    out = {
        "capacity": args.capacity,
        "layer_seq_traces": len(keys),
        "lru_transfers": totals["lru"][0],
        "belady_transfers": totals["belady"][0],
        "lru_over_belady": round(totals["lru"][0] / max(totals["belady"][0], 1), 4),
        "initial_loads": totals["lru"][1],
    }
    print(json.dumps(out, indent=2))
    (args.run_dir / "belady_bound.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
