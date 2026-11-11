"""Microbenchmark: per-expert copy loop vs batched staging copy, at OLMoE expert geometry.

Resolves whether the harness's measured copy cost is bandwidth, per-copy overhead, or node
contention, and calibrates tau for THIS code path. Runs in ~2 minutes on one GPU.

Paths, for a batch of n experts (gate/up [I=1024, H=2048] bf16, down [H, I]):
  per_expert   3n separate slice copies pool[slot].copy_(slab[eid], non_blocking=True)  (v1)
  batched      CPU index_select into pinned staging, ONE H2D per projection, then
               device-side index_copy_ scatter into the pool slots                       (v2)
  contiguous   one H2D of n contiguous experts (pure-bandwidth reference, no scatter)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

E, I, H = 64, 1024, 2048
SHAPES = {"gate": (I, H), "up": (I, H), "down": (H, I)}


def _events(fn, iters=30, warmup=5):
    times = []
    for it in range(iters + warmup):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if it >= warmup:
            times.append(s.elapsed_time(e))
    return statistics.median(times), statistics.fmean(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(0)

    slabs = {n: torch.randn(E, *s, dtype=torch.bfloat16).pin_memory() for n, s in SHAPES.items()}
    pool = {n: torch.empty(32, *s, dtype=torch.bfloat16, device=device) for n, s in SHAPES.items()}
    stage_cpu = {n: torch.empty(32, *s, dtype=torch.bfloat16).pin_memory() for n, s in SHAPES.items()}
    stage_gpu = {n: torch.empty(32, *s, dtype=torch.bfloat16, device=device) for n, s in SHAPES.items()}
    bytes_per_expert = sum(2 * I * H * 2 for _ in range(1))  # gate+up+down = 3*I*H*2B
    bytes_per_expert = 3 * I * H * 2

    rows = []
    for n in (1, 2, 4, 8, 16, 32):
        eids = torch.randperm(E)[:n]
        slots = torch.randperm(32)[:n]
        eids_list, slots_list = eids.tolist(), slots.tolist()
        slots_gpu = slots.to(device)

        def per_expert():
            for eid, slot in zip(eids_list, slots_list):
                for name in SHAPES:
                    pool[name][slot].copy_(slabs[name][eid], non_blocking=True)

        def batched():
            for name in SHAPES:
                torch.index_select(slabs[name], 0, eids, out=stage_cpu[name][:n])
                stage_gpu[name][:n].copy_(stage_cpu[name][:n], non_blocking=True)
                pool[name].index_copy_(0, slots_gpu, stage_gpu[name][:n])

        def contiguous():
            for name in SHAPES:
                stage_gpu[name][:n].copy_(slabs[name][:n], non_blocking=True)

        for label, fn in (("per_expert", per_expert), ("batched", batched),
                          ("contiguous", contiguous)):
            med, mean = _events(fn)
            gib = n * bytes_per_expert / 2**30 / (med / 1000)
            rows.append({"path": label, "n_experts": n, "median_ms": round(med, 4),
                         "mean_ms": round(mean, 4), "eff_gib_per_s": round(gib, 2),
                         "ms_per_expert": round(med / n, 4)})
            print(f"{label:>11s} n={n:>2d}  {med:7.3f} ms  {med/n:6.3f} ms/expert  "
                  f"{gib:6.2f} GiB/s")

    meta = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "bytes_per_expert": bytes_per_expert, "rows": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(meta, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
