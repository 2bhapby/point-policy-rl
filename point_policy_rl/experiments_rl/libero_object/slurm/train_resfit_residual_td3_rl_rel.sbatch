#!/bin/bash
#SBATCH --job-name=pp-libob-resfit-rel
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.out
#SBATCH --error=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero}"
RESFIT_ROOT="${RESFIT_ROOT:-/sjw_alinlab2/home/sanghyeok/residual-offpolicy-rl}"
CONDA_BASE="${CONDA_BASE:-/sjw_alinlab2/home/sanghyeok/miniconda3}"
CONDA_ENV="${CONDA_ENV:-libero_codex}"
BC_WEIGHT="${BC_WEIGHT:-}"
TASK_NAME="${TASK_NAME:-pick_up_the_alphabet_soup_and_place_it_in_the_basket}"
RUN_NAME="${RUN_NAME:-resfit_rel_${SLURM_JOB_ID:-local}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-point-policy-residual-rl}"
WANDB_MODE="${WANDB_MODE:-online}"
ENV_MAX_EPISODE_LEN="${ENV_MAX_EPISODE_LEN:-3000}"
STATE_COORD_MODE="${STATE_COORD_MODE:-both_eef}"

if [[ -z "${BC_WEIGHT}" ]]; then
  echo "BC_WEIGHT is required"
  exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

WANDB_ARGS=""
if [[ "${WANDB_ENABLE}" == "1" ]]; then
  WANDB_ARGS="--wandb-enable --wandb-project ${WANDB_PROJECT} --wandb-mode ${WANDB_MODE}"
fi

python point_policy_rl/train_resfit_residual_td3_rl_rel.py \
  --bc-weight "${BC_WEIGHT}" \
  --suite libero_object \
  --task-name "${TASK_NAME}" \
  --state-coord-mode "${STATE_COORD_MODE}" \
  --env-max-episode-len "${ENV_MAX_EPISODE_LEN}" \
  --resfit-root "${RESFIT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --output-root point_policy/exp_local_rl_rel \
  ${WANDB_ARGS} \
  ${EXTRA_ARGS}
