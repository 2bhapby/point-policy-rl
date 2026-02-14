#!/bin/bash
#SBATCH --job-name=pp-libob-rl
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.out
#SBATCH --error=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero}"
CONDA_BASE="${CONDA_BASE:-/sjw_alinlab2/home/sanghyeok/miniconda3}"
CONDA_ENV="${CONDA_ENV:-libero_codex}"
BC_WEIGHT="${BC_WEIGHT:-}"
TASK_NAME="${TASK_NAME:-pick_up_the_alphabet_soup_and_place_it_in_the_basket}"
RUN_NAME="${RUN_NAME:-debug_${SLURM_JOB_ID:-local}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${BC_WEIGHT}" ]]; then
  echo "BC_WEIGHT is required"
  exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python point_policy_rl/train_residual_td3_rl.py \
  --bc-weight "${BC_WEIGHT}" \
  --suite libero_object \
  --task-name "${TASK_NAME}" \
  --run-name "${RUN_NAME}" \
  --output-root point_policy/exp_local_rl \
  ${EXTRA_ARGS}
