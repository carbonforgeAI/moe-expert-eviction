#!/bin/bash
#SBATCH --job-name=moe-lock
#SBATCH --partition=main-cpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source jobs/_mila_env.sh

echo "Generating uv.lock on a compute node."
srun uv lock
