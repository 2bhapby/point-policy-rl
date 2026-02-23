#!/bin/bash
# Sweep checkpoint evals and build success-rate curve.
# Usage example:
#   RUN_DIR=/abs/path/to/run \
#   EPISODES=10 CKPT_STRIDE=10000 \
#   sbatch -p debug point_policy_rl/experiments_rl/libero_object/slurm/sweep_eval_success_curve_rl_rel.sbatch

#SBATCH --job-name=pp-libob-evalsweep
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.out
#SBATCH --error=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero}"
CONDA_BASE="${CONDA_BASE:-/sjw_alinlab2/home/sanghyeok/miniconda3}"
CONDA_ENV="${CONDA_ENV:-libero_codex}"
RESFIT_ROOT="${RESFIT_ROOT:-/sjw_alinlab2/home/sanghyeok/residual-offpolicy-rl}"

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "RUN_DIR is required"
  exit 1
fi

EPISODES="${EPISODES:-10}"
CKPT_STRIDE="${CKPT_STRIDE:-10000}"
MIN_STEP="${MIN_STEP:-0}"
MAX_STEP="${MAX_STEP:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"

DEVICE="${DEVICE:-cuda}"
SUITE="${SUITE:-}"
TASK_NAME="${TASK_NAME:-}"
BENCHMARK_NAME="${BENCHMARK_NAME:-}"
TASK_ORDER_INDEX="${TASK_ORDER_INDEX:--1}"
STATE_COORD_MODE="${STATE_COORD_MODE:-}"
IMAGE_SIZE_OVERRIDE="${IMAGE_SIZE_OVERRIDE:-84}"

SAVE_VIDEO="${SAVE_VIDEO:-0}"
VIDEO_DIR="${VIDEO_DIR:-}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-256}"

OUTPUT_TAG="${OUTPUT_TAG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python point_policy_rl/sweep_eval_success_curve_rl_rel.py
  --run-dir "${RUN_DIR}"
  --episodes "${EPISODES}"
  --device "${DEVICE}"
  --resfit-root "${RESFIT_ROOT}"
  --ckpt-stride "${CKPT_STRIDE}"
  --min-step "${MIN_STEP}"
  --max-step "${MAX_STEP}"
  --timeout-sec "${TIMEOUT_SEC}"
  --image-size-override "${IMAGE_SIZE_OVERRIDE}"
)

if [[ -n "${SUITE}" ]]; then
  CMD+=(--suite "${SUITE}")
fi
if [[ -n "${TASK_NAME}" ]]; then
  CMD+=(--task-name "${TASK_NAME}")
fi
if [[ -n "${BENCHMARK_NAME}" ]]; then
  CMD+=(--benchmark-name "${BENCHMARK_NAME}")
fi
if [[ -n "${TASK_ORDER_INDEX}" ]]; then
  CMD+=(--task-order-index "${TASK_ORDER_INDEX}")
fi
if [[ -n "${STATE_COORD_MODE}" ]]; then
  CMD+=(--state-coord-mode "${STATE_COORD_MODE}")
fi
if [[ -n "${OUTPUT_TAG}" ]]; then
  CMD+=(--output-tag "${OUTPUT_TAG}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ "${SAVE_VIDEO}" == "1" ]]; then
  CMD+=(--save-video)
  CMD+=(--video-fps "${VIDEO_FPS}")
  CMD+=(--video-render-size "${VIDEO_RENDER_SIZE}")
  if [[ -n "${VIDEO_DIR}" ]]; then
    CMD+=(--video-dir "${VIDEO_DIR}")
  fi
fi

echo "[cmd] ${CMD[*]}"
"${CMD[@]}"

