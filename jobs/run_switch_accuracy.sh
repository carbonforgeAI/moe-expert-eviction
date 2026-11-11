#!/bin/bash
#SBATCH --job-name=moe-switch-accuracy
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MOE_CUDA_MODULE="${MOE_CUDA_MODULE:-cuda/12.2.2}"
source jobs/_mila_env.sh
if [[ ! -f uv.lock ]]; then echo "uv.lock missing; run lock_deps first" >&2; exit 2; fi
OUTPUT_DIR="${OUTPUT_DIR:-$MOE_ROOT/runs/${SLURM_JOB_ID}/switch_accuracy}"
mkdir -p "$OUTPUT_DIR"
CONFIG="${CONFIG:-configs/mmlu_switch_accuracy.yaml}"
ARGS=(--config "$CONFIG" --output-dir "$OUTPUT_DIR")
if [[ -n "${ROUTER_STATE:-}" ]]; then ARGS+=(--router-state "$ROUTER_STATE"); fi
# Trained heads from a best-of-pool training run (optional -- omit to run only vanilla + freeze rows).
if [[ -n "${LEARNED_BETA_STATE:-}" && -n "${POLICY_HEAD_STATE:-}" ]]; then
  ARGS+=(--learned-beta-state "$LEARNED_BETA_STATE" --policy-head-state "$POLICY_HEAD_STATE")
fi
srun uv run --locked --extra ml python -m moe_eviction.models.evaluate_switch_accuracy "${ARGS[@]}"
