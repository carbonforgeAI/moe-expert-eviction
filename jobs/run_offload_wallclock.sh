#!/bin/bash
#SBATCH --job-name=moe-offload-wallclock
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MOE_CUDA_MODULE="${MOE_CUDA_MODULE:-cuda/12.2.2}"
source jobs/_mila_env.sh
if [[ ! -f uv.lock ]]; then echo "uv.lock missing; run lock_deps first" >&2; exit 2; fi
OUTPUT_DIR="${OUTPUT_DIR:-$MOE_ROOT/runs/${SLURM_JOB_ID}/offload_wallclock}"
mkdir -p "$OUTPUT_DIR"
CONFIG="${CONFIG:-configs/offload_wallclock_e1.yaml}"
ARGS=(--config "$CONFIG" --output-dir "$OUTPUT_DIR")
# Default to the bundled ReMoE router so NLL rows are comparable to the recorded frontier.
ROUTER_STATE="${ROUTER_STATE:-artifacts/remoe_router_state.pt}"
if [[ -n "$ROUTER_STATE" ]]; then ARGS+=(--router-state "$ROUTER_STATE"); fi
if [[ -n "${LEARNED_BETA_STATE:-}" ]]; then ARGS+=(--learned-beta-state "$LEARNED_BETA_STATE"); fi
if [[ -n "${POLICY_HEAD_STATE:-}" ]]; then ARGS+=(--policy-head-state "$POLICY_HEAD_STATE"); fi
srun uv run --locked --extra ml python -m moe_eviction.experiments.offload_wallclock "${ARGS[@]}"
