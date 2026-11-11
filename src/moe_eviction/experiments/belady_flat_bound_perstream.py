"""Per-(layer,seq)-stream sandwich: flat-MIN lower bound vs greedy set-Belady.

Companion to belady_flat_bound.py. The aggregate script sums steady-state
insertions over all (layer,seq) streams; here the sandwich is resolved PER
STREAM so we can see WHERE the lower and upper bounds coincide (greedy
provably optimal on that stream) and where the slack lives.

Per stream:
  upper  = greedy set-Belady steady insertions (verbatim accounting from
           evict_policies.py, via belady_flat_bound.replay_set_belady);
  lower  = max over the three within-token flattening orders (asc, desc,
           seeded random -- seed RANDOM_SEED + index of the key in the
           sorted key list, matching the aggregate script exactly) of
           classical Belady/MIN steady insertions on the flattened stream.
Each order is a valid lower bound on set-OPT for that stream, so the max is
too. tight <=> lower == upper <=> greedy is exactly optimal on the stream.

Reported per trace (and for eval also restricted to held-out seqs 12-15):
  (a) % of streams tight;
  (b) aggregate gap summed over non-tight streams only;
  (c) distribution of per-stream relative gap (gap/greedy): max, median;
  (e) mean stream length (tokens) of tight vs non-tight streams, to test
      whether slack grows with stream length.

Sanity: per-stream sums per order and for greedy must reproduce the
aggregate numbers of belady_flat_bound.py exactly; pass the aggregate JSONs
via --check-eval/--check-train to assert this before any stats are reported.

Usage:
  python -m moe_eviction.experiments.belady_flat_bound_perstream \
      --eval-trace e9/eval_trace.npz --train-trace e9/train_trace.npz \
      --check-eval e9/flat_bound_eval.json --check-train e9/flat_bound_train.json \
      --capacity 32 --out e9/flat_bound_perstream.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from moe_eviction.experiments.belady_flat_bound import (
    HELDOUT_SEQS,
    ORDERS,
    RANDOM_SEED,
    flatten,
    replay_flat_min,
    replay_set_belady,
    seq_of,
)


def layer_of(key: str) -> int:
    m = re.match(r"^layer(\d+)_", key)
    assert m, key
    return int(m.group(1))


def per_stream_rows(data: np.lib.npyio.NpzFile, capacity: int) -> list[dict]:
    keys = sorted(k for k in data.files if k.endswith("_demand"))
    rows = []
    for j, k in enumerate(keys):
        demand = data[k]
        greedy = replay_set_belady(demand, capacity)[0]
        flat = {
            order: replay_flat_min(
                flatten(demand, order, RANDOM_SEED + j), capacity
            )[0]
            for order in ORDERS
        }
        lower = max(flat.values())
        gap = greedy - lower
        assert gap >= 0, (k, greedy, flat)
        rows.append(
            {
                "key": k,
                "layer": layer_of(k),
                "seq": seq_of(k),
                "heldout": seq_of(k) in HELDOUT_SEQS,
                "T": int(demand.shape[0]),
                "greedy": greedy,
                "flat": flat,
                "lower": lower,
                "gap": gap,
                "rel_gap": (gap / greedy) if greedy else 0.0,
                "tight": gap == 0,
            }
        )
    return rows


def sanity_check(rows: list[dict], agg: dict, label: str) -> None:
    """Assert per-stream sums reproduce belady_flat_bound.py aggregates."""
    held = [r for r in rows if r["heldout"]]
    got_greedy = {
        "full": sum(r["greedy"] for r in rows),
        "heldout": sum(r["greedy"] for r in held),
    }
    assert got_greedy == agg["greedy_set_belady"], (
        label, "greedy", got_greedy, agg["greedy_set_belady"])
    for order in ORDERS:
        got = {
            "full": sum(r["flat"][order] for r in rows),
            "heldout": sum(r["flat"][order] for r in held),
        }
        assert got == agg["flattened_min"][order], (
            label, order, got, agg["flattened_min"][order])
    print(f"sanity OK [{label}]: per-stream sums == aggregate JSON "
          f"(greedy {got_greedy['full']}/{got_greedy['heldout']}, "
          f"all {len(ORDERS)} orders)")


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tight = [r for r in rows if r["tight"]]
    loose = [r for r in rows if not r["tight"]]
    rel = np.array([r["rel_gap"] for r in rows], dtype=float)
    lengths = np.array([r["T"] for r in rows], dtype=float)
    gaps = np.array([r["gap"] for r in rows], dtype=float)
    out = {
        "n_streams": n,
        "n_tight": len(tight),
        "pct_tight": 100.0 * len(tight) / n if n else 0.0,
        "greedy_sum": int(sum(r["greedy"] for r in rows)),
        "lower_sum": int(sum(r["lower"] for r in rows)),
        "gap_sum_nontight": int(sum(r["gap"] for r in loose)),
        "rel_gap_max": float(rel.max()) if n else 0.0,
        "rel_gap_median": float(np.median(rel)) if n else 0.0,
        "rel_gap_median_nontight": (
            float(np.median([r["rel_gap"] for r in loose])) if loose else 0.0),
        "mean_T_tight": (
            float(np.mean([r["T"] for r in tight])) if tight else None),
        "mean_T_nontight": (
            float(np.mean([r["T"] for r in loose])) if loose else None),
        "pearson_T_gap": (
            float(np.corrcoef(lengths, gaps)[0, 1])
            if n > 1 and gaps.std() > 0 and lengths.std() > 0 else None),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-trace", type=Path, required=True)
    parser.add_argument("--train-trace", type=Path, required=True)
    parser.add_argument("--check-eval", type=Path, default=None,
                        help="aggregate JSON from belady_flat_bound.py (eval)")
    parser.add_argument("--check-train", type=Path, default=None,
                        help="aggregate JSON from belady_flat_bound.py (train)")
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result: dict = {
        "capacity": args.capacity,
        "orders": list(ORDERS),
        "random_seed_base": RANDOM_SEED,
        "heldout_seqs": sorted(HELDOUT_SEQS),
        "traces": {},
    }
    all_rows: list[dict] = []
    for label, trace, check in (
        ("eval", args.eval_trace, args.check_eval),
        ("train", args.train_trace, args.check_train),
    ):
        data = np.load(trace)
        rows = per_stream_rows(data, args.capacity)
        print(f"{trace}: {len(rows)} (layer,seq) streams, C={args.capacity}")
        if check is not None:
            sanity_check(rows, json.loads(check.read_text()), label)
        entry = {
            "trace": str(trace),
            "summary": summarize(rows),
            "streams": rows,
        }
        if label == "eval":
            entry["summary_heldout"] = summarize(
                [r for r in rows if r["heldout"]])
        result["traces"][label] = entry
        all_rows.extend(rows)
        for name, summ in [("full", entry["summary"])] + (
            [("heldout", entry["summary_heldout"])] if label == "eval" else []
        ):
            print(f"  [{label}/{name}] tight {summ['n_tight']}/{summ['n_streams']}"
                  f" ({summ['pct_tight']:.1f}%)"
                  f"  gap(non-tight sum) {summ['gap_sum_nontight']}"
                  f"  rel gap max {summ['rel_gap_max']:.4f}"
                  f" median {summ['rel_gap_median']:.4f}"
                  f" median|nontight {summ['rel_gap_median_nontight']:.4f}"
                  f"  mean T tight/nontight "
                  f"{summ['mean_T_tight']}/{summ['mean_T_nontight']}")
    result["combined_summary"] = summarize(all_rows)
    s = result["combined_summary"]
    print(f"  [combined] tight {s['n_tight']}/{s['n_streams']}"
          f" ({s['pct_tight']:.1f}%)  gap(non-tight sum) {s['gap_sum_nontight']}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
