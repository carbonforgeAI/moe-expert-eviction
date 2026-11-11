#!/bin/bash

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH is not set. This script must run on the Mila cluster." >&2
  exit 1
fi

export MOE_ROOT="${MOE_ROOT:-$SCRATCH/moe-eviction}"
export HF_HOME="$MOE_ROOT/hf"
export HF_DATASETS_CACHE="$MOE_ROOT/hf/datasets"
export TRANSFORMERS_CACHE="$MOE_ROOT/hf/transformers"
export UV_CACHE_DIR="$MOE_ROOT/uv-cache"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$MOE_ROOT/venv}"
export WANDB_DIR="$MOE_ROOT/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-}-p no:cacheprovider"

mkdir -p "$MOE_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$UV_CACHE_DIR" "$WANDB_DIR" "$MOE_ROOT/logs" "$MOE_ROOT/runs"

module purge
module load "${MOE_PYTHON_MODULE:-python/3.10}"
if [[ -n "${MOE_CUDA_MODULE:-}" ]]; then
  module load "$MOE_CUDA_MODULE"
fi

export UV_INSTALL_DIR="$MOE_ROOT/uv-bin"
export UV_PREFIX="$MOE_ROOT/uv-prefix"
export PATH="$UV_INSTALL_DIR:$UV_PREFIX/bin:$PATH"
if [[ "${MOE_ALLOW_UV_PYTHON_DOWNLOADS:-0}" == "1" ]]; then
  export UV_PYTHON_DOWNLOADS=automatic
else
  export UV_PYTHON_DOWNLOADS=never
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH; installing uv under $UV_PREFIX"
  python -m pip install --prefix "$UV_PREFIX" --cache-dir "$UV_CACHE_DIR/pip" uv
fi

if ! uv python find ">=3.10" >/dev/null 2>&1; then
  echo "No Python 3.10+ interpreter is visible to uv." >&2
  echo "Load an approved Mila Python 3.10+ module with MOE_PYTHON_MODULE, or set" >&2
  echo "MOE_ALLOW_UV_PYTHON_DOWNLOADS=1 to allow a scratch-local uv-managed Python." >&2
  exit 3
fi

echo "Date:                   $(date)"
echo "Hostname:               $(hostname)"
echo "SLURM_JOB_ID:           ${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID:    ${SLURM_ARRAY_TASK_ID:-none}"
echo "SLURM_RESTART_COUNT:    ${SLURM_RESTART_COUNT:-0}"
echo "MOE_ROOT:               $MOE_ROOT"
echo "HF_HOME:                $HF_HOME"
echo "UV_CACHE_DIR:           $UV_CACHE_DIR"
echo "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
echo "SLURM_TMPDIR:           ${SLURM_TMPDIR:-none}"
echo "CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES:-none}"
echo "UV_PYTHON_DOWNLOADS:    $UV_PYTHON_DOWNLOADS"
python --version || true
uv --version || true

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if command -v disk-quota >/dev/null 2>&1; then
  disk-quota || true
fi
