"""Residency plot: color-coded expert-residency trajectories over tokens, per layer.

Reads the residency trace dumped by offload_wallclock (--dump-trace): per restricted layer,
per token, the resident pool ids, the unrestricted top-k demand, and refresh events. Renders
the temporally extended structure following the concurrent study's boundary-plot design:
one horizontal band per expert, colored while resident, demand dots on top, and vertical
lines at refresh boundaries.

Usage:
    python -m moe_eviction.experiments.plot_residency <run_dir> [--layer 8] [--seq 0] [--max-tokens 384]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def plot(run_dir: Path, layer: int, seq: int, max_tokens: int, out: Path | None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(run_dir / "residency_trace.npz", allow_pickle=True)
    key = f"layer{layer}_seq{seq}"
    resident = data[f"{key}_resident"][:max_tokens]      # [T, C] expert ids
    demand = data[f"{key}_demand"][:max_tokens]          # [T, top_k] unrestricted top-k ids
    refreshes = data[f"{key}_refresh"][:max_tokens]      # [T] bool
    meta = json.loads((run_dir / "metadata.json").read_text())
    num_experts = int(data.get("num_experts", 64))
    T = resident.shape[0]

    fig, (ax, ax_miss) = plt.subplots(
        2, 1, figsize=(max(12, T * 0.03), 9), sharex=True,
        gridspec_kw={"height_ratios": [5, 1]},
    )

    grid = np.zeros((num_experts, T), dtype=np.int8)
    for t in range(T):
        ids = resident[t][resident[t] >= 0]                # -1 = cold-fill padding
        grid[ids, t] = 1                                   # 1 = resident
    for t in range(T):
        for e in demand[t]:
            grid[e, t] = 3 if grid[e, t] == 1 else 2      # 3 = resident+demanded, 2 = missed
    cmap = matplotlib.colors.ListedColormap(["#ffffff", "#c6dbef", "#d62728", "#08519c"])
    ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3,
              origin="lower")
    for t in np.flatnonzero(refreshes):
        ax.axvline(t, color="black", alpha=0.35, linewidth=0.7, linestyle="--")
    ax.set_ylabel("expert id")
    ax.set_title(
        f"Expert residency, layer {layer}, seq {seq} — {meta.get('gpu', '?')}, "
        f"C={meta.get('capacity')}, policy trace"
    )
    handles = [
        matplotlib.patches.Patch(color="#c6dbef", label="resident"),
        matplotlib.patches.Patch(color="#08519c", label="resident + demanded"),
        matplotlib.patches.Patch(color="#d62728", label="demanded, NOT resident (served in-pool)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    miss = np.array([np.mean([e not in set(resident[t]) for e in demand[t]]) for t in range(T)])
    ax_miss.plot(miss, color="#d62728", linewidth=0.8)
    ax_miss.set_ylabel("miss frac")
    ax_miss.set_xlabel("token position")
    ax_miss.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    out = out or run_dir / f"residency_layer{layer}_seq{seq}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    plot(args.run_dir, args.layer, args.seq, args.max_tokens, args.out)


if __name__ == "__main__":
    main()
