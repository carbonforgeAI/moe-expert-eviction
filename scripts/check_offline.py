"""Compare freshly reproduced offline analyses with the recorded results/ files.

Run by `make reproduce`; exits non-zero if any reproduced number differs.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "reproduce"
REC = ROOT / "results"


def load(path):
    return json.loads(Path(path).read_text())


checks = []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol
    checks.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name:52s} reproduced {got:>10} recorded {want:>10}")


bound, rec_bound = load(OUT / "belady_bound.json"), load(REC / "e5a_belady_bound.json")
for k in ("lru_transfers", "belady_transfers", "lru_over_belady"):
    check(f"Belady bound: {k}", bound[k], rec_bound[k])

ev, rec_ev = load(OUT / "evict_policies.json"), load(REC / "e6_evict_policies.json")
for k in ("lru", "lfu", "reuse", "belady"):
    check(f"replayed eviction: {k} steady transfers", ev[k], rec_ev[k])

look = {p["k"]: p for p in load(OUT / "lookahead.json")["curve"]}
rec_look = {p["k"]: p for p in load(REC / "e9_lookahead.json")["curve"]}
for k in (1, 2, 4, 8, 16, 32):
    check(f"foresight k={k}: held-out gap recovered (%)",
          look[k]["gap_recovered_heldout_pct"], rec_look[k]["gap_recovered_heldout_pct"])

ps, rec_ps = load(OUT / "perstream.json"), load(REC / "e10_flat_bound_perstream.json")
for part in ("eval", "train"):
    for k, v in rec_ps["traces"][part]["summary"].items():
        if isinstance(v, (int, float)):
            check(f"per-stream bound ({part}): {k}", ps["traces"][part]["summary"][k], v, tol=1e-9)

e8, rec_e8 = load(OUT / "e8" / "results.json"), load(REC / "e8_imitation.json")
for k, v in rec_e8["held_out"].items():
    check(f"E8 imitation held-out: {k}", e8["held_out"][k], v)

if (OUT / "e9" / "results.json").exists():
    e9, rec_e9 = load(OUT / "e9" / "results.json"), load(REC / "e9_imitation.json")
    for k, v in rec_e9["heldout_12_15"].items():
        if isinstance(v, int):
            check(f"E9 imitation held-out: {k}", e9["heldout_12_15"][k], v)

print(f"\n{sum(checks)}/{len(checks)} reproduced numbers match the recorded results.")
sys.exit(0 if all(checks) else 1)
