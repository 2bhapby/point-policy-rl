#!/bin/bash
#SBATCH --job-name=pp-basechk
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.out
#SBATCH --error=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero}"
CONDA_BASE="${CONDA_BASE:-/sjw_alinlab2/home/sanghyeok/miniconda3}"
CONDA_ENV="${CONDA_ENV:-libero_codex}"

BC_WEIGHT="${BC_WEIGHT:-}"
SUITE="${SUITE:-libero_object_basefix_v1}"
TASK_NAME="${TASK_NAME:-KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it}"
NUM_STEPS="${NUM_STEPS:-5}"
ENV_MAX_EPISODE_LEN="${ENV_MAX_EPISODE_LEN:-3000}"
SEED="${SEED:-2}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${BC_WEIGHT}" ]]; then
  echo "BC_WEIGHT is required"
  exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python point_policy_rl/check_base_action_pipeline_rl.py \
  --bc-weight "${BC_WEIGHT}" \
  --suite "${SUITE}" \
  --task-name "${TASK_NAME}" \
  --num-steps "${NUM_STEPS}" \
  --env-max-episode-len "${ENV_MAX_EPISODE_LEN}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  ${EXTRA_ARGS}

