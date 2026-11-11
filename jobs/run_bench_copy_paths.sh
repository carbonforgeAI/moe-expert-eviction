#!/bin/bash
#SBATCH --job-name=moe-bench-copy
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MOE_CUDA_MODULE="${MOE_CUDA_MODULE:-cuda/12.2.2}"
source jobs/_mila_env.sh
OUTPUT="${OUTPUT:-$MOE_ROOT/runs/${SLURM_JOB_ID}/bench_copy_paths.json}"
srun uv run --locked --extra ml python -m moe_eviction.experiments.bench_copy_paths --output "$OUTPUT"
