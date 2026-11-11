"""Render paper/figures/foresight.pdf: fraction of the LRU-to-Belady transfer
gap recovered by an oracle whose foresight is truncated to k tokens (E9;
regenerate the underlying numbers with belady_lookahead.py on
data/demand_trace_eval16.npz). Horizontal lines mark the causal rungs.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

RESULTS = Path(__file__).resolve().parent.parent / "results"
_curve = [p for p in json.loads((RESULTS / "e9_lookahead.json").read_text())["curve"]
          if p["k"] not in (0, None) and isinstance(p["k"], int)]
K = [p["k"] for p in _curve]
HELDOUT = [p["gap_recovered_heldout_pct"] for p in _curve]
FULL16 = [p["gap_recovered_full16_pct"] for p in _curve]
_e9 = json.loads((RESULTS / "e9_imitation.json").read_text())["heldout_12_15"]["gap_recovered_pct"]
LFU_PCT, GRU_PCT = _e9["lfu_ema"], _e9["combined_gru"]

fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.plot(K, HELDOUT, "o-", color="#0072B2", ms=3.2, lw=1.1, label="held-out sequences")
ax.plot(K, FULL16, "s--", color="#56B4E9", ms=2.8, lw=0.9, label="all sixteen")
ax.axhline(LFU_PCT, color="#009E73", lw=0.8, ls=":")
ax.annotate(f"decayed-frequency eviction ({LFU_PCT:.1f}%)", xy=(30, LFU_PCT), xytext=(0, -8.5),
            textcoords="offset points", fontsize=6.2, color="#00745a", ha="center")
ax.axhline(GRU_PCT, color="#CC79A7", lw=0.8, ls=":")
ax.annotate(f"trained scorer, router forecast ({GRU_PCT:.1f}%)", xy=(26, GRU_PCT), xytext=(0, 3),
            textcoords="offset points", fontsize=6.2, color="#a4577f", ha="center")
ax.set_xscale("log", base=2)
ax.set_xticks(K)
ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
ax.set_xlabel("foresight $k$ (tokens)")
ax.set_ylabel("% of LRU$\\to$Belady gap")
ax.set_ylim(0, 104)
ax.tick_params(length=2.5)
ax.legend(frameon=False, fontsize=6.5, loc="center right")
out = Path(__file__).resolve().parent.parent / "paper" / "figures" / "foresight.pdf"
fig.savefig(out)
print(f"wrote {out}")
