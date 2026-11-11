#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source jobs/_submit_helpers.sh

moe_submit_sbatch unkillable gpu:l40s:1 2 32G 00:45:00 jobs/run_train_learned_beta.sh
