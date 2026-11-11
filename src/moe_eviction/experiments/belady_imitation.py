"""Supervised imitation of Belady (clairvoyant) eviction on the recorded demand trace.

Hawkeye/Parrot applied to expert offloading: replay the trace through the C=32 pool;
at every eviction event, compute STRICTLY CAUSAL features for each resident candidate
(features read only the access history of that (layer, sequence) up to the current
token) and label the candidate the Belady oracle would evict (labels ARE allowed to
read the future -- that is the oracle). Train (a) a linear scorer and (b) a small MLP
with a listwise softmax objective over the candidate set (all Belady-optimal
candidates count as positives, which handles "never used again" ties exactly).

Split is BY SEQUENCE: train on seqs 0-11, hold out seqs 12-15. Feature normalizers
are fit on training events only. Evaluation is CLOSED REPLAY on the held-out
sequences: the trained model picks every victim online; we count steady transfers
with the harness's accounting (misses are 'initial' until the pool has been full
once; a currently-selected expert is never the victim), and compare against LRU,
LFU-EMA and Belady recomputed on the same held-out subset.

Mandatory control: --shuffle-labels (on by default here via the control run) permutes
the optimal-victim mask within each event's candidate set before training the MLP;
its held-out gap-recovered must collapse toward ~0.

Usage:
  python -m moe_eviction.experiments.belady_imitation <run_dir> \
      [--capacity 32] [--out DIR] [--epochs 20] [--seed 0] \
      [--max-train-events 60000] [--shuffle-labels]

--shuffle-labels alone trains ONLY the shuffled-label MLP control; the default run
trains linear + MLP and ALSO retrains the shuffled control, reporting all three.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from moe_eviction.experiments.evict_policies import replay as baseline_replay

FEATURE_NAMES = [
    "dt_last_use",        # t - last_seen (tokens since candidate was demanded)
    "log1p_dt_last_use",
    "ema_use_0.90",       # exponentially decayed use frequency, decay 0.90/token
    "ema_use_0.95",
    "ema_use_0.99",
    "uses_last_16",       # demand count in the last 16 tokens
    "uses_last_64",
    "recency_rank",       # rank among candidates by last use (0=MRU .. 1=LRU)
    "gap_mean",           # mean past inter-arrival gap (t+1 if no gap observed yet)
    "gap_std",            # std of past inter-arrival gaps (0 if <2 gaps)
    "overdue_ratio",      # dt / (gap_mean + 1)
    "log1p_use_count",
    "layer_frac",         # layer id / 16
    "t_frac",             # t / 384 (max context length, a constant)
]
NUM_EXPERTS = 64
MAX_CTX = 384.0
NEVER = 10 ** 9


class FeatureTracker:
    """Strictly causal per-(layer, sequence) access statistics.

    observe(t, needed) must be called once per token BEFORE features are queried at
    that token; it folds in the current token's accesses (present, not future).
    """

    DECAYS = (0.90, 0.95, 0.99)

    def __init__(self, layer_id: int):
        self.layer_frac = layer_id / 16.0
        self.ema = [np.zeros(NUM_EXPERTS) for _ in self.DECAYS]
        self.last_seen = np.full(NUM_EXPERTS, -1, dtype=np.int64)
        self.use_count = np.zeros(NUM_EXPERTS, dtype=np.int64)
        self.gap_cnt = np.zeros(NUM_EXPERTS, dtype=np.int64)
        self.gap_sum = np.zeros(NUM_EXPERTS)
        self.gap_sumsq = np.zeros(NUM_EXPERTS)
        self.use_times: list[list[int]] = [[] for _ in range(NUM_EXPERTS)]

    def observe(self, t: int, needed: list[int]) -> None:
        for e in needed:
            for a in self.ema:
                a[e] += 1.0
            if self.last_seen[e] >= 0:
                gap = t - self.last_seen[e]
                self.gap_cnt[e] += 1
                self.gap_sum[e] += gap
                self.gap_sumsq[e] += gap * gap
            self.last_seen[e] = t
            self.use_count[e] += 1
            self.use_times[e].append(t)
        for a, d in zip(self.ema, self.DECAYS):
            a *= d

    def features(self, t: int, cands: list[int]) -> np.ndarray:
        n = len(cands)
        out = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
        last = self.last_seen[cands]
        order = np.argsort(last, kind="stable")  # oldest first
        rank = np.empty(n)
        rank[order] = np.linspace(1.0, 0.0, n) if n > 1 else 0.0
        for i, e in enumerate(cands):
            dt = float(t - last[i])
            times = self.use_times[e]
            u16 = len(times) - bisect_left(times, t - 15)
            u64 = len(times) - bisect_left(times, t - 63)
            if self.gap_cnt[e] > 0:
                gm = self.gap_sum[e] / self.gap_cnt[e]
                gv = max(self.gap_sumsq[e] / self.gap_cnt[e] - gm * gm, 0.0)
                gs = math.sqrt(gv) if self.gap_cnt[e] > 1 else 0.0
            else:
                gm, gs = float(t + 1), 0.0
            out[i] = (
                dt, math.log1p(dt),
                self.ema[0][e], self.ema[1][e], self.ema[2][e],
                u16, u64, rank[i],
                gm, gs, dt / (gm + 1.0),
                math.log1p(self.use_count[e]),
                self.layer_frac, t / MAX_CTX,
            )
        return out


def build_next_use(demand: np.ndarray) -> dict[int, list[int]]:
    next_use: dict[int, list[int]] = {}
    for t in range(demand.shape[0] - 1, -1, -1):
        for e in demand[t]:
            next_use.setdefault(int(e), []).append(t)
    return next_use


def collect_events(demand: np.ndarray, capacity: int, layer_id: int,
                   teacher: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Replay under `teacher` (lru|belady); at each eviction event emit
    (causal candidate features [M,F], Belady-optimal victim mask [M])."""
    T = demand.shape[0]
    next_use = build_next_use(demand)
    tracker = FeatureTracker(layer_id)
    resident: OrderedDict[int, None] = OrderedDict()
    events: list[tuple[np.ndarray, np.ndarray]] = []
    for t in range(T):
        needed = [int(e) for e in demand[t]]
        tracker.observe(t, needed)
        for e in needed:
            uses = next_use.get(e)
            while uses and uses[-1] <= t:
                uses.pop()
        for e in needed:
            if e in resident:
                resident.move_to_end(e)
        needed_set = set(needed)
        for e in needed:
            if e in resident:
                continue
            if len(resident) >= capacity:
                cands = [r for r in resident if r not in needed_set]
                nxts = np.array([next_use[r][-1] if next_use.get(r) else NEVER
                                 for r in cands], dtype=np.int64)
                opt = (nxts == nxts.max())
                events.append((tracker.features(t, cands), opt))
                if teacher == "lru":
                    victim = cands[0]
                elif teacher == "belady":
                    victim = cands[int(np.argmax(nxts))]
                else:
                    raise ValueError(teacher)
                resident.pop(victim)
            resident[e] = None
    return events


class Scorer(nn.Module):
    def __init__(self, dim: int, hidden: int | None):
        super().__init__()
        if hidden is None:
            self.net = nn.Linear(dim, 1, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pad_events(events, mu, sd):
    M = max(f.shape[0] for f, _ in events)
    F = events[0][0].shape[1]
    X = np.zeros((len(events), M, F), dtype=np.float32)
    valid = np.zeros((len(events), M), dtype=bool)
    opt = np.zeros((len(events), M), dtype=bool)
    for i, (f, o) in enumerate(events):
        m = f.shape[0]
        X[i, :m] = (f - mu) / sd
        valid[i, :m] = True
        opt[i, :m] = o
    return (torch.from_numpy(X), torch.from_numpy(valid), torch.from_numpy(opt))


def train_model(events, mu, sd, hidden, epochs, seed, lr=1e-3, batch=512):
    torch.manual_seed(seed)
    model = Scorer(len(FEATURE_NAMES), hidden)
    opt_ = torch.optim.Adam(model.parameters(), lr=lr)
    X, valid, opt = pad_events(events, mu, sd)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            s = model(X[idx])
            s = s.masked_fill(~valid[idx], -1e9)
            s_opt = s.masked_fill(~opt[idx], -1e9)
            loss = (torch.logsumexp(s, -1) - torch.logsumexp(s_opt, -1)).mean()
            opt_.zero_grad()
            loss.backward()
            opt_.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"    epoch {ep + 1:2d}/{epochs}  loss {tot / n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def learned_replay(demand: np.ndarray, capacity: int, layer_id: int,
                   model: Scorer, mu: np.ndarray, sd: np.ndarray) -> tuple[int, int]:
    """Closed replay: the trained model chooses every victim online."""
    T = demand.shape[0]
    tracker = FeatureTracker(layer_id)
    resident: OrderedDict[int, None] = OrderedDict()
    steady = initial = 0
    full_once = False
    for t in range(T):
        needed = [int(e) for e in demand[t]]
        tracker.observe(t, needed)
        for e in needed:
            if e in resident:
                resident.move_to_end(e)
        needed_set = set(needed)
        for e in needed:
            if e in resident:
                continue
            if len(resident) >= capacity:
                cands = [r for r in resident if r not in needed_set]
                x = (tracker.features(t, cands) - mu) / sd
                scores = model(torch.from_numpy(x.astype(np.float32)))
                resident.pop(cands[int(torch.argmax(scores))])
            resident[e] = None
            if full_once:
                steady += 1
            else:
                initial += 1
            if len(resident) >= capacity:
                full_once = True
    return steady, initial


def shuffle_event_labels(events, seed):
    rng = np.random.default_rng(seed)
    out = []
    for f, o in events:
        out.append((f, o[rng.permutation(len(o))]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[4]
                        / "belady_imitation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-events", type=int, default=60000)
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="train ONLY the shuffled-label control MLP")
    parser.add_argument("--holdout", type=int, nargs="+",
                        default=[12, 13, 14, 15])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.run_dir / "demand_trace.npz")
    keys = sorted(k for k in data.files if k.endswith("_demand"))

    def parse(k: str) -> tuple[int, int]:
        layer, seq, _ = k.split("_")
        return int(layer[len("layer"):]), int(seq[len("seq"):])

    holdout = set(args.holdout)
    train_keys = [k for k in keys if parse(k)[1] not in holdout]
    test_keys = [k for k in keys if parse(k)[1] in holdout]
    print(f"{len(train_keys)} train traces (layer,seq), {len(test_keys)} held-out")

    # 1) collect training events (causal features, Belady labels) under two
    #    teacher state distributions: LRU and Belady replays.
    events = []
    for k in train_keys:
        layer_id, _ = parse(k)
        for teacher in ("lru", "belady"):
            events.extend(collect_events(data[k], args.capacity, layer_id, teacher))
    print(f"{len(events)} training eviction events collected")
    if len(events) > args.max_train_events:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(events), args.max_train_events, replace=False)
        events = [events[i] for i in idx]
        print(f"subsampled to {len(events)} events")

    # 2) normalizers on TRAIN events only
    allf = np.concatenate([f for f, _ in events], axis=0)
    mu = allf.mean(axis=0)
    sd = allf.std(axis=0)
    sd[sd < 1e-8] = 1.0

    # 3) baselines on the held-out subset (closed replay, same accounting)
    base = {}
    for policy, name in (("lru", "lru"), ("lfu", "lfu_ema"), ("belady", "belady")):
        s = 0
        for k in test_keys:
            a, _ = baseline_replay(data[k], args.capacity, policy)
            s += a
        base[name] = s
        print(f"held-out {name:>8s}: {s} steady transfers")
    lru, bel = base["lru"], base["belady"]

    def gap_rec(x: int) -> float:
        return (lru - x) / (lru - bel) * 100.0 if lru > bel else 0.0

    def evaluate(model) -> int:
        s = 0
        for k in test_keys:
            layer_id, _ = parse(k)
            a, _ = learned_replay(data[k], args.capacity, layer_id, model, mu, sd)
            s += a
        return s

    args.out.mkdir(parents=True, exist_ok=True)
    results = {
        "capacity": args.capacity,
        "seed": args.seed,
        "epochs": args.epochs,
        "holdout_seqs": sorted(holdout),
        "n_train_events": len(events),
        "features": FEATURE_NAMES,
        "held_out": dict(base),
        "held_out_gap_recovered_pct": {
            "lfu_ema": round(gap_rec(base["lfu_ema"]), 2)},
    }

    if not args.shuffle_labels:
        print("training linear scorer...")
        linear = train_model(events, mu, sd, None, args.epochs, args.seed)
        print("training MLP scorer (2x64)...")
        mlp = train_model(events, mu, sd, 64, args.epochs, args.seed)
        n_lin = evaluate(linear)
        n_mlp = evaluate(mlp)
        results["held_out"]["learned_linear"] = n_lin
        results["held_out"]["learned_mlp"] = n_mlp
        results["held_out_gap_recovered_pct"]["learned_linear"] = round(gap_rec(n_lin), 2)
        results["held_out_gap_recovered_pct"]["learned_mlp"] = round(gap_rec(n_mlp), 2)
        print(f"held-out learned_linear: {n_lin}  ({gap_rec(n_lin):.1f}% gap recovered)")
        print(f"held-out learned_mlp:    {n_mlp}  ({gap_rec(n_mlp):.1f}% gap recovered)")

        w = linear.net.weight.detach().numpy().ravel()
        order = np.argsort(-np.abs(w))
        print("linear per-feature weights (on standardized features), |w| desc:")
        weights = {}
        for i in order:
            print(f"  {FEATURE_NAMES[i]:>18s}  w={w[i]:+.4f}")
            weights[FEATURE_NAMES[i]] = round(float(w[i]), 6)
        results["linear_weights_standardized"] = weights
        torch.save(linear.state_dict(), args.out / "linear.pt")
        torch.save(mlp.state_dict(), args.out / "mlp.pt")

    # 4) shuffled-label control (always run unless a clean run was requested
    #    with the flag, in which case ONLY this runs)
    print("training shuffled-label control MLP...")
    shuf = shuffle_event_labels(events, args.seed + 1)
    mlp_shuf = train_model(shuf, mu, sd, 64, args.epochs, args.seed)
    n_shuf = evaluate(mlp_shuf)
    results["held_out"]["mlp_shuffled_labels"] = n_shuf
    results["held_out_gap_recovered_pct"]["mlp_shuffled_labels"] = round(gap_rec(n_shuf), 2)
    print(f"held-out mlp_shuffled_labels: {n_shuf}  ({gap_rec(n_shuf):.1f}% gap recovered)")
    torch.save(mlp_shuf.state_dict(), args.out / "mlp_shuffled.pt")

    np.savez(args.out / "normalizer.npz", mu=mu, sd=sd)
    (args.out / "features.json").write_text(json.dumps(FEATURE_NAMES, indent=2))
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
