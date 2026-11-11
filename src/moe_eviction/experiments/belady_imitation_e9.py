"""E9: full-featured learned evictor -- Belady imitation with router-prob features.

Extends E8 (belady_imitation.py, read it first) in four ways:

1. TRAINING DATA: the 48-sequence train trace (disjoint text from the eval trace).
   Model selection (early stopping + best-model choice) uses events from the LAST 8
   train sequences (40-47); the eval trace is never touched during training or
   selection. FINAL evaluation is closed replay on the eval trace, reported both on
   held-out seqs 12-15 (comparable to E8) and on all 16 sequences. LRU / LFU-EMA /
   Belady are recomputed on both sets with the same replay machinery.

2. FEATURES: E8's 14 strictly-causal access-history features PLUS 10 per-candidate
   router-probability features computed from the recorded _probs [T,64] (the
   router's softmax at each token -- decision-time information, since the harness
   routes first and evicts after). All prob features read only rows <= t.

3. MODELS (all listwise softmax CE over resident candidates vs. the Belady-optimal
   victim set, states collected under both LRU and Belady teacher replays):
     (a) history-only MLP 2x64   (E8 replica on the bigger data)
     (b) probs-only MLP 2x64
     (c) combined MLP 2x64
     (d) combined GRU: per-candidate GRU(hidden 32) over the last 32 tokens of
         (router prob, was-in-top8) pairs, concatenated with the static features
         into a 2x64 head.

4. CONTROLS: shuffled-label combined MLP (must collapse), and a combined LINEAR
   model whose standardized |weights| are reported for interpretability.

Usage:
  python -m moe_eviction.experiments.belady_imitation_e9 \
      [--train-trace F] [--eval-trace F] [--out DIR] [--capacity 32] \
      [--epochs 20] [--seed 0] [--max-train-events 40000] [--gru-max-train-events 30000]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from moe_eviction.experiments.belady_imitation import (
    FEATURE_NAMES as HIST_FEATURES,
)
from moe_eviction.experiments.belady_imitation import (
    FeatureTracker,
    build_next_use,
)
from moe_eviction.experiments.evict_policies import replay as baseline_replay

NUM_EXPERTS = 64
NEVER = 10 ** 9
UNIFORM = 1.0 / NUM_EXPERTS
SEQ_LEN = 32  # GRU window

PROB_FEATURES = [
    "prob_cur",             # router prob for this expert at the current token
    "logprob_cur",          # log(prob + 1e-6)
    "prob_rank",            # rank of this expert in the current 64-way softmax (/63)
    "prob_ema_0.90",        # EMA of prob, decay 0.90/token
    "prob_ema_0.95",
    "prob_ema_0.99",
    "prob_trend",           # ema_0.90 - ema_0.99 (fast minus slow)
    "prob_max_8",           # max prob over the last 8 tokens (incl. current)
    "prob_max_32",
    "dt_prob_above_unif",   # tokens since prob > 1/64 (t+1 if never)
]
ALL_FEATURES = list(HIST_FEATURES) + PROB_FEATURES
HIST_COLS = list(range(len(HIST_FEATURES)))
PROB_COLS = list(range(len(HIST_FEATURES), len(ALL_FEATURES)))
ALL_COLS = list(range(len(ALL_FEATURES)))


# --------------------------------------------------------------------------- probs

def prob_precompute(P16: np.ndarray) -> dict[str, np.ndarray]:
    """Causal per-token derived arrays from the router softmax [T,64].

    Every row t of every output depends only on rows <= t of P (EMA/max/last-above
    are prefix computations), so indexing row t at decision time is strictly causal.
    """
    P = P16.astype(np.float32)
    T = P.shape[0]
    # rank of each expert within its row (0 = highest prob), stable
    rank = np.argsort(np.argsort(-P, axis=1, kind="stable"), axis=1).astype(np.float32)
    emas = {}
    for d in (0.90, 0.95, 0.99):
        e = np.empty_like(P)
        acc = np.zeros(NUM_EXPERTS, dtype=np.float32)
        for t in range(T):
            acc = d * acc + (1.0 - d) * P[t]
            e[t] = acc
        emas[d] = e
    def rollmax(w: int) -> np.ndarray:
        pad = np.zeros((w - 1, NUM_EXPERTS), dtype=np.float32)
        v = np.lib.stride_tricks.sliding_window_view(
            np.vstack([pad, P]), w, axis=0)
        return v.max(axis=-1)
    max8, max32 = rollmax(8), rollmax(32)
    last_above = np.empty((T, NUM_EXPERTS), dtype=np.int64)
    la = np.full(NUM_EXPERTS, -NEVER, dtype=np.int64)
    for t in range(T):
        la = np.where(P[t] > UNIFORM, t, la)
        last_above[t] = la
    return {"P": P, "rank": rank, "ema90": emas[0.90], "ema95": emas[0.95],
            "ema99": emas[0.99], "max8": max8, "max32": max32,
            "last_above": last_above}


def prob_features(pre: dict[str, np.ndarray], t: int, cands: list[int]) -> np.ndarray:
    c = np.asarray(cands)
    cur = pre["P"][t, c].astype(np.float64)
    dt_above = np.minimum(t - pre["last_above"][t, c], t + 1).astype(np.float64)
    return np.stack([
        cur,
        np.log(cur + 1e-6),
        pre["rank"][t, c] / 63.0,
        pre["ema90"][t, c], pre["ema95"][t, c], pre["ema99"][t, c],
        pre["ema90"][t, c] - pre["ema99"][t, c],
        pre["max8"][t, c], pre["max32"][t, c],
        dt_above,
    ], axis=1)


def gru_window(pre: dict[str, np.ndarray], top8: np.ndarray, t: int,
               cands: list[int]) -> np.ndarray:
    """[M, SEQ_LEN, 2] fp16: (router prob, was-in-top8) over tokens t-31..t,
    zero-padded on the left for t < 31. Rows <= t only: strictly causal."""
    lo = max(0, t - SEQ_LEN + 1)
    p = pre["P"][lo:t + 1][:, cands]                       # [w, M]
    k = top8[lo:t + 1][:, cands].astype(np.float32)        # [w, M]
    w = p.shape[0]
    seq = np.zeros((SEQ_LEN, len(cands), 2), dtype=np.float16)
    seq[SEQ_LEN - w:, :, 0] = p
    seq[SEQ_LEN - w:, :, 1] = k
    return seq.transpose(1, 0, 2)


# ---------------------------------------------------------------- event collection

def collect_events_e9(demand: np.ndarray, probs: np.ndarray, capacity: int,
                      layer_id: int, teacher: str,
                      keep_p: float = 1.0, rng: np.random.Generator | None = None,
                      count_only: bool = False):
    """Replay under `teacher` (lru|belady). At each eviction event (kept with prob
    keep_p) emit (static feats [M,24], gru seq [M,32,2] fp16, optimal mask [M]).
    count_only skips all feature work and returns the event count."""
    T = demand.shape[0]
    next_use = build_next_use(demand)
    if not count_only:
        tracker = FeatureTracker(layer_id)
        pre = prob_precompute(probs)
        top8 = np.zeros((T, NUM_EXPERTS), dtype=bool)
        top8[np.arange(T)[:, None], demand.astype(np.int64)] = True
    resident: OrderedDict[int, None] = OrderedDict()
    events = []
    n_events = 0
    for t in range(T):
        needed = [int(e) for e in demand[t]]
        if not count_only:
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
                n_events += 1
                if not count_only and (rng is None or rng.random() < keep_p):
                    feats = np.concatenate(
                        [tracker.features(t, cands), prob_features(pre, t, cands)],
                        axis=1)
                    opt = (nxts == nxts.max())
                    events.append((feats, gru_window(pre, top8, t, cands), opt))
                if teacher == "lru":
                    victim = cands[0]
                elif teacher == "belady":
                    victim = cands[int(np.argmax(nxts))]
                else:
                    raise ValueError(teacher)
                resident.pop(victim)
            resident[e] = None
    return n_events if count_only else events


# ------------------------------------------------------------------------- models

class MLPScorer(nn.Module):
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

    def forward(self, x: torch.Tensor, seq: torch.Tensor | None = None):
        return self.net(x).squeeze(-1)


class GRUScorer(nn.Module):
    def __init__(self, static_dim: int, gru_hidden: int = 32, head: int = 64):
        super().__init__()
        self.gru = nn.GRU(2, gru_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden + static_dim, head), nn.ReLU(),
            nn.Linear(head, head), nn.ReLU(),
            nn.Linear(head, 1),
        )

    def forward(self, x: torch.Tensor, seq: torch.Tensor):
        B, M = x.shape[0], x.shape[1]
        h = self.gru(seq.reshape(B * M, SEQ_LEN, 2))[1][-1]      # [B*M, H]
        z = torch.cat([h.view(B, M, -1), x], dim=-1)
        return self.head(z).squeeze(-1)


def pad_events(events, mu, sd, cols, with_seq: bool):
    M = max(f.shape[0] for f, _, _ in events)
    X = np.zeros((len(events), M, len(cols)), dtype=np.float32)
    S = (np.zeros((len(events), M, SEQ_LEN, 2), dtype=np.float16)
         if with_seq else None)
    valid = np.zeros((len(events), M), dtype=bool)
    opt = np.zeros((len(events), M), dtype=bool)
    for i, (f, s, o) in enumerate(events):
        m = f.shape[0]
        X[i, :m] = ((f - mu) / sd)[:, cols]
        if with_seq:
            S[i, :m] = s
        valid[i, :m] = True
        opt[i, :m] = o
    return (torch.from_numpy(X),
            torch.from_numpy(S) if with_seq else None,
            torch.from_numpy(valid), torch.from_numpy(opt))


def listwise_loss(scores, valid, opt):
    s = scores.masked_fill(~valid, -1e9)
    s_opt = s.masked_fill(~opt, -1e9)
    return torch.logsumexp(s, -1) - torch.logsumexp(s_opt, -1)


@torch.no_grad()
def dataset_loss(model, tensors, batch=1024) -> float:
    X, S, valid, opt = tensors
    tot, n = 0.0, X.shape[0]
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        seq = S[sl].float() if S is not None else None
        tot += listwise_loss(model(X[sl], seq), valid[sl], opt[sl]).sum().item()
    return tot / n


def train_scorer(name, train_ev, val_ev, mu, sd, cols, *, hidden=64,
                 use_seq=False, epochs=20, seed=0, lr=1e-3, batch=512,
                 patience=4):
    torch.manual_seed(seed)
    model = (GRUScorer(len(cols)) if use_seq else MLPScorer(len(cols), hidden))
    opt_ = torch.optim.Adam(model.parameters(), lr=lr)
    tr = pad_events(train_ev, mu, sd, cols, use_seq)
    va = pad_events(val_ev, mu, sd, cols, use_seq)
    X, S, valid, opt = tr
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    best_val, best_state, best_ep, bad = float("inf"), None, -1, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            seq = S[idx].float() if use_seq else None
            loss = listwise_loss(model(X[idx], seq), valid[idx], opt[idx]).mean()
            opt_.zero_grad()
            loss.backward()
            opt_.step()
            tot += loss.item() * len(idx)
        model.eval()
        vl = dataset_loss(model, va)
        print(f"    [{name}] epoch {ep + 1:2d}/{epochs}  "
              f"train {tot / n:.4f}  val {vl:.4f}")
        if vl < best_val - 1e-5:
            best_val, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    [{name}] early stop at epoch {ep + 1} "
                      f"(best {best_val:.4f} @ {best_ep + 1})")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


# ------------------------------------------------------------------------- replay

@torch.no_grad()
def learned_replay(demand, probs, capacity, layer_id, model, mu, sd, cols,
                   use_seq: bool) -> tuple[int, int]:
    """Closed replay on one (layer, seq): the trained model picks every victim
    online; harness accounting (initial until first full; touch-first)."""
    T = demand.shape[0]
    tracker = FeatureTracker(layer_id)
    pre = prob_precompute(probs)
    top8 = np.zeros((T, NUM_EXPERTS), dtype=bool)
    top8[np.arange(T)[:, None], demand.astype(np.int64)] = True
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
                feats = np.concatenate(
                    [tracker.features(t, cands), prob_features(pre, t, cands)],
                    axis=1)
                x = ((feats - mu) / sd)[:, cols].astype(np.float32)
                xt = torch.from_numpy(x).unsqueeze(0)
                seq = (torch.from_numpy(
                    gru_window(pre, top8, t, cands).astype(np.float32)
                ).unsqueeze(0) if use_seq else None)
                scores = model(xt, seq).squeeze(0)
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
    return [(f, s, o[rng.permutation(len(o))]) for f, s, o in events]


# --------------------------------------------------------------------------- main

def parse_key(k: str) -> tuple[int, int]:
    layer, seq, _ = k.split("_")
    return int(layer[len("layer"):]), int(seq[len("seq"):])


def main() -> None:
    base = Path(__file__).resolve().parents[4] / "e9"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace", type=Path, default=base / "train_trace.npz")
    parser.add_argument("--eval-trace", type=Path, default=base / "eval_trace.npz")
    parser.add_argument("--out", type=Path, default=base)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-events", type=int, default=40000)
    parser.add_argument("--gru-max-train-events", type=int, default=30000)
    parser.add_argument("--max-val-events", type=int, default=10000)
    parser.add_argument("--val-seqs", type=int, nargs="+",
                        default=list(range(40, 48)))
    args = parser.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = np.load(args.train_trace)
    eval_data = np.load(args.eval_trace)
    tkeys = sorted(k for k in train_data.files if k.endswith("_demand"))
    ekeys = sorted(k for k in eval_data.files if k.endswith("_demand"))
    val_set = set(args.val_seqs)
    fit_keys = [k for k in tkeys if parse_key(k)[1] not in val_set]
    val_keys = [k for k in tkeys if parse_key(k)[1] in val_set]
    print(f"train-trace: {len(fit_keys)} fit + {len(val_keys)} val (layer,seq); "
          f"eval-trace: {len(ekeys)}")

    def probs_of(data, k):
        return data[k.replace("_demand", "_probs")]

    # ---- pass 1: count eviction events, to set the keep probability ----------
    def count_group(keys):
        return sum(collect_events_e9(train_data[k], None, args.capacity,
                                     parse_key(k)[0], teacher, count_only=True)
                   for k in keys for teacher in ("lru", "belady"))

    n_fit_total = count_group(fit_keys)
    n_val_total = count_group(val_keys)
    print(f"eviction events available: fit {n_fit_total}, val {n_val_total} "
          f"(2 teachers)  [{time.time() - t0:.0f}s]")
    keep_fit = min(1.0, 1.08 * args.max_train_events / max(n_fit_total, 1))
    keep_val = min(1.0, 1.08 * args.max_val_events / max(n_val_total, 1))

    # ---- pass 2: collect features for kept events ----------------------------
    def collect_group(keys, keep_p, rng):
        ev = []
        for k in keys:
            layer_id, _ = parse_key(k)
            for teacher in ("lru", "belady"):
                ev.extend(collect_events_e9(
                    train_data[k], probs_of(train_data, k), args.capacity,
                    layer_id, teacher, keep_p=keep_p, rng=rng))
        return ev

    rng = np.random.default_rng(args.seed)
    fit_events = collect_group(fit_keys, keep_fit, rng)
    val_events = collect_group(val_keys, keep_val, rng)
    if len(fit_events) > args.max_train_events:
        idx = rng.choice(len(fit_events), args.max_train_events, replace=False)
        fit_events = [fit_events[i] for i in sorted(idx)]
    if len(val_events) > args.max_val_events:
        idx = rng.choice(len(val_events), args.max_val_events, replace=False)
        val_events = [val_events[i] for i in sorted(idx)]
    print(f"collected {len(fit_events)} fit / {len(val_events)} val events "
          f"[{time.time() - t0:.0f}s]")

    # normalizers on FIT events only
    allf = np.concatenate([f for f, _, _ in fit_events], axis=0)
    mu = allf.mean(axis=0)
    sd = allf.std(axis=0)
    sd[sd < 1e-8] = 1.0

    # ---- baselines on the eval trace (same replay machinery) -----------------
    heldout = {12, 13, 14, 15}
    ekeys_ho = [k for k in ekeys if parse_key(k)[1] in heldout]

    def run_baseline(policy, keys):
        return sum(baseline_replay(eval_data[k], args.capacity, policy)[0]
                   for k in keys)

    base_ho, base_full = {}, {}
    for policy, name in (("lru", "lru"), ("lfu", "lfu_ema"), ("belady", "belady")):
        base_ho[name] = run_baseline(policy, ekeys_ho)
        base_full[name] = run_baseline(policy, ekeys)
        print(f"{name:>8s}: held-out {base_ho[name]}  full16 {base_full[name]}")

    def gap(x, b):
        return round((b["lru"] - x) / (b["lru"] - b["belady"]) * 100.0, 2) \
            if b["lru"] > b["belady"] else 0.0

    # ---- train the four models (+ linear + shuffled control) -----------------
    specs = {
        "history_only_mlp": dict(cols=HIST_COLS, hidden=64, use_seq=False),
        "probs_only_mlp":   dict(cols=PROB_COLS, hidden=64, use_seq=False),
        "combined_mlp":     dict(cols=ALL_COLS, hidden=64, use_seq=False),
        "combined_gru":     dict(cols=ALL_COLS, hidden=None, use_seq=True),
    }
    models, val_losses = {}, {}
    for name, sp in specs.items():
        ev = fit_events
        if sp["use_seq"] and len(ev) > args.gru_max_train_events:
            idx = np.random.default_rng(args.seed + 7).choice(
                len(ev), args.gru_max_train_events, replace=False)
            ev = [ev[i] for i in sorted(idx)]
        print(f"training {name} on {len(ev)} events...")
        m, vl = train_scorer(name, ev, val_events, mu, sd, sp["cols"],
                             hidden=sp["hidden"], use_seq=sp["use_seq"],
                             epochs=args.epochs, seed=args.seed)
        models[name], val_losses[name] = m, vl
        print(f"  {name}: best val loss {vl:.4f}  [{time.time() - t0:.0f}s]")

    print("training combined linear (interpretability)...")
    linear, vl_lin = train_scorer("combined_linear", fit_events, val_events,
                                  mu, sd, ALL_COLS, hidden=None,
                                  epochs=args.epochs, seed=args.seed)
    w = linear.net.weight.detach().numpy().ravel()
    order = np.argsort(-np.abs(w))
    linear_weights = {ALL_FEATURES[i]: round(float(w[i]), 6) for i in order}
    print("combined linear |w| desc (standardized features):")
    for i in order[:12]:
        print(f"  {ALL_FEATURES[i]:>22s}  w={w[i]:+.4f}")

    print("training shuffled-label combined MLP control...")
    shuf_ev = shuffle_event_labels(fit_events, args.seed + 1)
    shuf_val = shuffle_event_labels(val_events, args.seed + 2)
    mlp_shuf, _ = train_scorer("combined_mlp_shuffled", shuf_ev, shuf_val,
                               mu, sd, ALL_COLS, hidden=64,
                               epochs=args.epochs, seed=args.seed)

    # ---- closed replay on the eval trace -------------------------------------
    def replay_group(model, cols, use_seq, keys):
        s = 0
        for k in keys:
            layer_id, _ = parse_key(k)
            a, _ = learned_replay(eval_data[k], probs_of(eval_data, k),
                                  args.capacity, layer_id, model, mu, sd,
                                  cols, use_seq)
            s += a
        return s

    learned_ho, learned_full = {}, {}
    for name, sp in specs.items():
        learned_ho[name] = replay_group(models[name], sp["cols"], sp["use_seq"],
                                        ekeys_ho)
        learned_full[name] = replay_group(models[name], sp["cols"],
                                          sp["use_seq"], ekeys)
        print(f"{name:>18s}: held-out {learned_ho[name]} "
              f"({gap(learned_ho[name], base_ho):.1f}%)  "
              f"full16 {learned_full[name]} "
              f"({gap(learned_full[name], base_full):.1f}%)  "
              f"[{time.time() - t0:.0f}s]")

    n_shuf = replay_group(mlp_shuf, ALL_COLS, False, ekeys_ho)
    print(f"shuffled-label control: held-out {n_shuf} "
          f"({gap(n_shuf, base_ho):.1f}%)")

    # best model chosen by TRAIN-trace validation loss (never by eval numbers)
    best_model = min(val_losses, key=val_losses.get)
    print(f"best model by val loss: {best_model}")

    args.out.mkdir(parents=True, exist_ok=True)
    results = {
        "capacity": args.capacity, "seed": args.seed, "epochs": args.epochs,
        "val_seqs": sorted(val_set),
        "n_fit_events": len(fit_events), "n_val_events": len(val_events),
        "n_fit_events_available": n_fit_total,
        "features": ALL_FEATURES,
        "val_losses": {k: round(v, 5) for k, v in val_losses.items()},
        "best_model_by_val_loss": best_model,
        "heldout_12_15": {
            **base_ho, **learned_ho, "combined_mlp_shuffled": n_shuf,
            "gap_recovered_pct": {
                "lfu_ema": gap(base_ho["lfu_ema"], base_ho),
                **{k: gap(v, base_ho) for k, v in learned_ho.items()},
                "combined_mlp_shuffled": gap(n_shuf, base_ho),
            },
        },
        "full16": {
            **base_full, **learned_full,
            "gap_recovered_pct": {
                "lfu_ema": gap(base_full["lfu_ema"], base_full),
                **{k: gap(v, base_full) for k, v in learned_full.items()},
            },
        },
        "combined_linear_val_loss": round(vl_lin, 5),
        "linear_weights_standardized": linear_weights,
        "runtime_sec": round(time.time() - t0, 1),
    }
    for name, m in models.items():
        torch.save(m.state_dict(), args.out / f"{name}.pt")
    torch.save(linear.state_dict(), args.out / "combined_linear.pt")
    torch.save(mlp_shuf.state_dict(), args.out / "combined_mlp_shuffled.pt")
    np.savez(args.out / "normalizer.npz", mu=mu, sd=sd)
    (args.out / "features.json").write_text(json.dumps(ALL_FEATURES, indent=2))
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out / 'results.json'}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
