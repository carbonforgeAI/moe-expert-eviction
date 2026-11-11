#!/bin/bash

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH is not set. Run this on a Mila login node." >&2
  exit 1
fi

mkdir -p "$SCRATCH/moe-eviction/logs"

sbatch \
  --output="$SCRATCH/moe-eviction/logs/%x-%j.out" \
  jobs/lock_deps.sh
