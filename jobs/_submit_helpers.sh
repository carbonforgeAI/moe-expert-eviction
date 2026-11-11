#!/bin/bash

moe_require_scratch() {
  if [[ -z "${SCRATCH:-}" ]]; then
    echo "SCRATCH is not set. Run this on a Mila login node." >&2
    exit 1
  fi

  mkdir -p "$SCRATCH/moe-eviction/logs"
}

moe_submit_sbatch() {
  local default_partition="$1"
  local default_gres="$2"
  local default_cpus="$3"
  local default_mem="$4"
  local default_time="$5"
  local script="$6"

  moe_require_scratch

  local args=(
    --partition="${MOE_SLURM_PARTITION:-$default_partition}"
    --cpus-per-task="${MOE_SLURM_CPUS:-$default_cpus}"
    --mem="${MOE_SLURM_MEM:-$default_mem}"
    --time="${MOE_SLURM_TIME:-$default_time}"
    --output="${MOE_SLURM_OUTPUT:-$SCRATCH/moe-eviction/logs/%x-%j.out}"
  )

  local gres="${MOE_SLURM_GRES:-$default_gres}"
  if [[ -n "$gres" ]]; then
    args+=(--gres="$gres")
  fi
  if [[ -n "${MOE_SLURM_DEPENDENCY:-}" ]]; then
    args+=(--dependency="$MOE_SLURM_DEPENDENCY")
  fi
  if [[ -n "${MOE_SLURM_JOB_NAME:-}" ]]; then
    args+=(--job-name="$MOE_SLURM_JOB_NAME")
  fi
  if [[ -n "${MOE_SLURM_ACCOUNT:-}" ]]; then
    args+=(--account="$MOE_SLURM_ACCOUNT")
  fi
  if [[ -n "${MOE_SLURM_QOS:-}" ]]; then
    args+=(--qos="$MOE_SLURM_QOS")
  fi

  sbatch "${args[@]}" "$script"
}
