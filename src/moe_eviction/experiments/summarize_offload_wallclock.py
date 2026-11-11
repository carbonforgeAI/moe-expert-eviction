"""Render offload_wallclock.jsonl as the E1 table: quality vs measured wall-clock.

Usage: python -m moe_eviction.experiments.summarize_offload_wallclock <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    for run_dir in sys.argv[1:]:
        run_dir = Path(run_dir)
        rows = [json.loads(l) for l in (run_dir / "offload_wallclock.jsonl").read_text().splitlines()]
        meta = json.loads((run_dir / "metadata.json").read_text())
        base = next((r for r in rows if r["policy"] == "resident_gpu"), None)
        print(f"\n=== {run_dir}  [{meta.get('gpu')}, {meta.get('model_id')}] ===")
        print(f"{'policy':>14} {'dwell':>6} {'NLL':>8} {'dNLL':>8} {'tok/s':>8} "
              f"{'ms/tok':>8} {'copy_ms':>8} {'xfer/tok':>9} {'MB/tok':>8}")
        for r in rows:
            dnll = r["nll"] - base["nll"] if base else float("nan")
            toks = max(r["tokens_timed"], 1)
            print(f"{r['policy']:>14} {str(r['dwell'] or '-'):>6} {r['nll']:8.4f} {dnll:+8.4f} "
                  f"{r['tok_per_s']:8.2f} {r['ms_per_token_mean']:8.2f} "
                  f"{r['copy_ms_per_token']:8.2f} {r['transfers']/toks:9.2f} "
                  f"{r['bytes_copied']/toks/2**20:8.2f}")


if __name__ == "__main__":
    main()
