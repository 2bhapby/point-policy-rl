#!/bin/bash
#SBATCH --job-name=pp-libob-rl-eval-rel
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
CKPT="${CKPT:-}"
EPISODES="${EPISODES:-10}"
TASK_NAME="${TASK_NAME:-pick_up_the_alphabet_soup_and_place_it_in_the_basket}"
SUITE="${SUITE:-libero_object_basefix_v1}"
BENCHMARK_NAME="${BENCHMARK_NAME:-}"
TASK_ORDER_INDEX="${TASK_ORDER_INDEX:--1}"
STATE_COORD_MODE="${STATE_COORD_MODE:-}"
IMAGE_SIZE_OVERRIDE="${IMAGE_SIZE_OVERRIDE:-84}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VIDEO_DIR="${VIDEO_DIR:-}"
VIDEO_TAG="${VIDEO_TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${CKPT}" ]]; then
  echo "CKPT is required"
  exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python point_policy_rl/eval_residual_td3_rl_rel.py
  --ckpt "${CKPT}"
  --suite "${SUITE}"
  --task-name "${TASK_NAME}"
  --episodes "${EPISODES}"
  --task-order-index "${TASK_ORDER_INDEX}"
  --image-size-override "${IMAGE_SIZE_OVERRIDE}"
)

if [[ -n "${BENCHMARK_NAME}" ]]; then
  CMD+=(--benchmark-name "${BENCHMARK_NAME}")
fi
if [[ -n "${STATE_COORD_MODE}" ]]; then
  CMD+=(--state-coord-mode "${STATE_COORD_MODE}")
fi
if [[ -n "${VIDEO_DIR}" ]]; then
  CMD+=(--video-dir "${VIDEO_DIR}")
fi
if [[ -n "${VIDEO_TAG}" ]]; then
  CMD+=(--video-tag "${VIDEO_TAG}")
fi
if [[ "${SAVE_VIDEO}" == "1" ]]; then
  CMD+=(--save-video)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"
