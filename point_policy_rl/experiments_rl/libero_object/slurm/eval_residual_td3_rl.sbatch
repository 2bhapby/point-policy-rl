#!/bin/bash
#SBATCH --job-name=pp-libob-rl-eval
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
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${CKPT}" ]]; then
  echo "CKPT is required"
  exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python point_policy_rl/eval_residual_td3_rl.py \
  --ckpt "${CKPT}" \
  --suite libero_object \
  --task-name "${TASK_NAME}" \
  --episodes "${EPISODES}" \
  ${EXTRA_ARGS}
