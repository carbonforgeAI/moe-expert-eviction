"""Render paper/figures/frontier.pdf: the measured quality-throughput frontier
at three transfer-cost tiers (Figure 1 of the paper).

Data are read from results/ (E5a frontier, E5b threshold sweep, E6/E9 replay
counts; Table 1 of the paper): per-policy excess NLL, steady transfers/token (rho),
and compute ms/token (c) on the L40S at native tau = 0.51 ms/expert.
Native-tier throughputs are the measured tok/s; slowed tiers are Eq. (1),
1000/(c + rho * tau), whose winning policy was validated against six
direct wall-clock arms at tau x10 and x50 (S3.1).

Usage:  python scripts/make_frontier_figure.py  (writes paper/figures/frontier.pdf)
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FULL = 5.5  # NeurIPS text width, inches

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

TAU_NATIVE = 0.51

EDGES = {"LFU-EMA (lossless)": "#D55E00",
         "imitation evictor (lossless)": "#CC79A7",
         "learned, thr.\\ .5": "#CC79A7"}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _rows(name):
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines() if line.strip()]


def load_policies():
    """Every plotted number is read from results/ (see docs/CLAIMS.md)."""
    e5a = _rows("e5a_frontier.jsonl")
    e5b = _rows("e5b_threshold_sweep.jsonl")
    base = {r["policy"]: r for r in e5a if r["policy"] in ("resident_gpu", "resident_pool", "lru")}
    base_b = {r["policy"]: r for r in e5b if r["policy"] == "resident_gpu"}
    T = base["lru"]["tokens_timed"]
    c_lru = base["lru"]["compute_ms_per_token"]

    def measured(r, ref):
        return (r["nll"] - ref["nll"], r["transfers"] / r["tokens_timed"],
                r["compute_ms_per_token"], r["tok_per_s"])

    def lossless(transfers):
        # replayed arms: same compute as LRU, throughput from Eq. (1)
        rho = transfers / T
        return (0.0, rho, c_lru, 1000.0 / (c_lru + rho * TAU_NATIVE))

    def pick(rows, **kw):
        (r,) = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
        return r

    gpu = base["resident_gpu"]
    evict = json.loads((RESULTS / "e6_evict_policies.json").read_text())
    e9 = json.loads((RESULTS / "e9_imitation.json").read_text())["full16"]
    # name: (dNLL, rho, compute_ms, native_tokps) + style (color, marker, size)
    # Okabe-Ito palette; lossless arms sit at dNLL = 0 by construction.
    data = {
        "full residency": measured(base["resident_pool"], gpu),
        "LRU (lossless)": measured(base["lru"], gpu),
        "LFU-EMA (lossless)": lossless(evict["lfu"]),
        "imitation evictor (lossless)": lossless(e9["combined_mlp"]),
        "farthest reuse (lossless)": lossless(evict["belady"]),
        "dwell-16": measured(pick(e5a, policy="dwell", dwell=16, window_mult=8), gpu),
        "dwell-64": measured(pick(e5a, policy="dwell", dwell=64, window_mult=1), gpu),
        "static": measured(pick(e5a, policy="static_warm"), gpu),
        "learned, thr.\\ .5": measured(pick(e5b, policy="learned", beta_threshold=0.5),
                                       base_b["resident_gpu"]),
        "learned, thr.\\ .9": measured(pick(e5b, policy="learned", beta_threshold=0.9),
                                       base_b["resident_gpu"]),
    }
    style = {
        "full residency": ("#000000", "*", 42),
        "LRU (lossless)": ("#D55E00", "s", 16),
        "LFU-EMA (lossless)": ("white", ">", 14),
        "imitation evictor (lossless)": ("white", "<", 14),
        "farthest reuse (lossless)": ("#E69F00", "*", 60),
        "dwell-16": ("#0072B2", "o", 16),
        "dwell-64": ("#56B4E9", "o", 16),
        "static": ("#009E73", "^", 18),
        "learned, thr.\\ .5": ("white", "D", 13),
        "learned, thr.\\ .9": ("#CC79A7", "D", 13),
    }
    return {k: (*data[k], *style[k]) for k in data}


POLICIES = load_policies()

TIERS = [(1, r"$\tau = 0.51$ ms (native)"),
         (10, r"$10\times$"),
         (50, r"$50\times$")]


XTICKS = [[35, 40, 45, 50], [10, 20, 40], [3, 10, 30]]


def tokps(policy, mult):
    _, rho, c, native, *_ = POLICIES[policy]
    if mult == 1:
        return native
    return 1000.0 / (c + rho * TAU_NATIVE * mult)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.26), sharey=True)
    for j, (ax, (mult, title)) in enumerate(zip(axes, TIERS)):
        ax.axhline(0.0, color="0.75", lw=0.6, zorder=0)
        for name, (d, rho, c, native, col, mk, ms) in POLICIES.items():
            x = tokps(name, mult)
            edge = EDGES.get(name, "black")
            ax.scatter([x], [d], s=ms, c=col, marker=mk,
                       edgecolors=edge, linewidths=0.6 if col == "white" else 0.4,
                       zorder=4 if col == "white" else 3)

        ax.set_xscale("log")
        ax.set_title(title, pad=3)
        ax.set_ylim(-0.030, 0.44)
        ax.tick_params(length=2.5)
        ax.set_xticks(XTICKS[j])
        ax.set_xticks([], minor=True)
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    axes[0].set_ylabel(r"$\Delta$NLL vs full residency")
    axes[1].set_xlabel("decode tokens/s (log scale)")
    # the one comparison the paper depends on: oracle eviction at LRU quality
    axes[0].annotate(r"oracle: $2.1\times$ fewer transfers, same quality",
                     xy=(42.4, 0.014), xytext=(33.2, 0.26), fontsize=6.2,
                     color="0.2", ha="left", va="top",
                     arrowprops=dict(arrowstyle="->", lw=0.6, color="0.45",
                                     connectionstyle="arc3,rad=-0.22"))

    import matplotlib.lines as mlines
    def handle(name):
        _, _, _, _, col, mk, ms = POLICIES[name]
        return mlines.Line2D([], [], color="none", marker=mk,
                             markersize=(ms ** 0.5) * 1.15, markerfacecolor=col,
                             markeredgecolor=EDGES.get(name, "black"),
                             markeredgewidth=0.5, label=name.replace("\\ ", " "))
    lossless = ["full residency", "LRU (lossless)", "LFU-EMA (lossless)",
                "imitation evictor (lossless)", "farthest reuse (lossless)"]
    lossy = ["dwell-16", "dwell-64", "static", "learned, thr.\\ .5",
             "learned, thr.\\ .9"]
    for ax_, names in ((axes[1], lossless), (axes[2], lossy)):
        leg = ax_.legend(handles=[handle(n) for n in names], loc="upper left",
                         fontsize=6.2, frameon=True, borderaxespad=0.15,
                         handletextpad=0.2, labelspacing=0.3, borderpad=0.35,
                         fancybox=False, framealpha=1.0)
        leg.get_frame().set_edgecolor("0.7")
        leg.get_frame().set_linewidth(0.5)
    out = ROOT / "paper" / "figures" / "frontier.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
