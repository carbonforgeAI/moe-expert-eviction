"""Flattened-MIN lower bound vs greedy set-Belady upper bound.

Greedy set-Belady (the 'belady' policy of evict_policies.py) is a feasible
schedule for the set-model paging problem (all 8 demanded experts must be
simultaneously resident at each token), hence an UPPER bound on set-OPT.

Flattened-MIN relaxes the simultaneity constraint: each token's 8 demanded
experts are expanded into 8 consecutive single-item accesses, and classical
Belady/MIN demand paging is run on that flattened stream (on a miss with a
full cache, evict the item whose next access is farthest; never = infinity).
Any feasible set-schedule induces a feasible single-item schedule with the
same insertions, so flattened-MIN is a LOWER bound on set-OPT. If the two
coincide, set-OPT is pinned exactly.

Classical MIN's optimal miss count on a FIXED stream does not depend on
tie-breaking (ties only occur among never-again items, which are
interchangeable), but the flattened stream itself depends on the within-token
order, so we try several orders (ascending id, descending id, seeded random
per token) and report each.

Accounting is identical to evict_policies.py / belady_lookahead.py: replay
per (layer, seq) stream independently; insertions count as 'initial' until
the pool has been full once (the insertion that first fills the pool is still
'initial'), then 'steady'; sums over all streams. Held-out = seqs 12-15.

Usage:
  python -m moe_eviction.experiments.belady_flat_bound <trace.npz> \
      [--capacity 32] [--out flat_bound.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

HELDOUT_SEQS = {12, 13, 14, 15}
INF = 10**9
ORDERS = ("asc", "desc", "random")
RANDOM_SEED = 0


def replay_set_belady(demand: np.ndarray, capacity: int) -> tuple[int, int]:
    """Greedy set-Belady, verbatim accounting from evict_policies.py 'belady'."""
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

                def nxt(r: int) -> int:
                    uses = next_use.get(r)
                    return uses[-1] if uses else INF

                victim = max(cands, key=nxt)
                resident.pop(victim)
            resident[e] = None
            if full_once:
                steady += 1
            else:
                initial += 1
            if len(resident) >= capacity:
                full_once = True
    return steady, initial


def flatten(demand: np.ndarray, order: str, seed: int = RANDOM_SEED) -> np.ndarray:
    """Expand each token's 8 demanded experts into 8 consecutive accesses."""
    if order == "asc":
        rows = np.sort(demand.astype(np.int64), axis=1)
    elif order == "desc":
        rows = np.sort(demand.astype(np.int64), axis=1)[:, ::-1]
    elif order == "random":
        rng = np.random.default_rng(seed)
        rows = demand.astype(np.int64).copy()
        for i in range(rows.shape[0]):
            rng.shuffle(rows[i])
    else:
        raise ValueError(order)
    return rows.reshape(-1)


def replay_flat_min(stream: np.ndarray, capacity: int) -> tuple[int, int]:
    """Classical single-item Belady/MIN on the flattened stream, same
    steady/initial convention (steady only after the cache first holds
    `capacity` distinct items)."""
    N = len(stream)
    # next_pos[i] = next index j > i with stream[j] == stream[i], else INF
    next_pos = np.full(N, INF, dtype=np.int64)
    last_seen: dict[int, int] = {}
    for i in range(N - 1, -1, -1):
        e = int(stream[i])
        next_pos[i] = last_seen.get(e, INF)
        last_seen[e] = i
    resident: dict[int, int] = {}  # item -> next access position (as of its last access)
    steady = initial = 0
    full_once = False
    for i in range(N):
        e = int(stream[i])
        if e in resident:
            resident[e] = int(next_pos[i])
            continue
        if len(resident) >= capacity:
            victim = max(resident, key=resident.__getitem__)
            del resident[victim]
        resident[e] = int(next_pos[i])
        if full_once:
            steady += 1
        else:
            initial += 1
        if len(resident) >= capacity:
            full_once = True
    return steady, initial


def seq_of(key: str) -> int:
    m = re.search(r"_seq(\d+)_demand$", key)
    assert m, key
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = np.load(args.trace)
    keys = sorted(k for k in data.files if k.endswith("_demand"))
    held_keys = [k for k in keys if seq_of(k) in HELDOUT_SEQS]
    n_seqs = len({seq_of(k) for k in keys})
    print(f"{args.trace}: {len(keys)} (layer,seq) streams, {n_seqs} seqs, C={args.capacity}")

    # 1) greedy set-Belady sanity check (must match evict_policies.py numbers)
    greedy_full = sum(replay_set_belady(data[k], args.capacity)[0] for k in keys)
    greedy_held = sum(replay_set_belady(data[k], args.capacity)[0] for k in held_keys)
    print(f"greedy set-Belady : full {greedy_full:6d}   heldout(12-15) {greedy_held:6d}")

    # 2) flattened-MIN per within-token order
    flat: dict[str, dict[str, int]] = {}
    for order in ORDERS:
        f_full = sum(
            replay_flat_min(flatten(data[k], order, RANDOM_SEED + j), args.capacity)[0]
            for j, k in enumerate(keys))
        f_held = sum(
            replay_flat_min(flatten(data[k], order, RANDOM_SEED + keys.index(k)),
                            args.capacity)[0]
            for k in held_keys)
        flat[order] = {"full": f_full, "heldout": f_held}
        gap_f = greedy_full - f_full
        gap_h = greedy_held - f_held
        print(f"flattened-MIN({order:>6s}): full {f_full:6d} (gap {gap_f:+d},"
              f" {100.0 * gap_f / greedy_full:.3f}%)"
              f"   heldout {f_held:6d} (gap {gap_h:+d},"
              f" {100.0 * gap_h / greedy_held if greedy_held else 0.0:.3f}%)")

    verdict = {o: flat[o]["full"] == greedy_full and flat[o]["heldout"] == greedy_held
               for o in ORDERS}
    print("lower == upper per order:", verdict)

    if args.out:
        args.out.write_text(json.dumps({
            "trace": str(args.trace),
            "capacity": args.capacity,
            "heldout_seqs": sorted(HELDOUT_SEQS),
            "greedy_set_belady": {"full": greedy_full, "heldout": greedy_held},
            "flattened_min": flat,
            "equal": verdict,
        }, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
