#!/bin/bash
#SBATCH --job-name=pp-libsp-rl-eval
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=point_policy_rl/experiments_rl/libero_spatial/slurm_logs/%x_%j.out
#SBATCH --error=point_policy_rl/experiments_rl/libero_spatial/slurm_logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero}"
CONDA_BASE="${CONDA_BASE:-/sjw_alinlab2/home/sanghyeok/miniconda3}"
CONDA_ENV="${CONDA_ENV:-libero_codex}"
CKPT="${CKPT:-}"
EPISODES="${EPISODES:-10}"
TASK_NAME="${TASK_NAME:-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
VIDEO_DIR="${VIDEO_DIR:-}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-256}"
VIDEO_TAG="${VIDEO_TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${CKPT}" ]]; then
  echo "CKPT is required"
  exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

VIDEO_ARGS=()
if [[ "${SAVE_VIDEO}" == "true" ]]; then
  VIDEO_ARGS+=(--save-video)
  VIDEO_ARGS+=(--video-fps "${VIDEO_FPS}")
  VIDEO_ARGS+=(--video-render-size "${VIDEO_RENDER_SIZE}")
  if [[ -n "${VIDEO_DIR}" ]]; then
    VIDEO_ARGS+=(--video-dir "${VIDEO_DIR}")
  fi
  if [[ -n "${VIDEO_TAG}" ]]; then
    VIDEO_ARGS+=(--video-tag "${VIDEO_TAG}")
  fi
fi

python point_policy_rl/eval_residual_td3_rl.py \
  --ckpt "${CKPT}" \
  --suite libero_spatial \
  --task-name "${TASK_NAME}" \
  --episodes "${EPISODES}" \
  "${VIDEO_ARGS[@]}" \
  ${EXTRA_ARGS}
