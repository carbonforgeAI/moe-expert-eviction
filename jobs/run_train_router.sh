#!/bin/bash
#SBATCH --job-name=moe-train-router
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export MOE_CUDA_MODULE="${MOE_CUDA_MODULE:-cuda/12.2.2}"
source jobs/_mila_env.sh

if [[ ! -f uv.lock ]]; then
  echo "uv.lock is missing. Submit jobs/lock_deps.sh first." >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-$MOE_ROOT/runs/${SLURM_JOB_ID}/router_train_smoke}"
mkdir -p "$OUTPUT_DIR"

srun uv run --locked --extra ml python -m moe_eviction.models.train_router \
  --config "${CONFIG:-configs/router_train_smoke.yaml}" \
  --output-dir "$OUTPUT_DIR"
