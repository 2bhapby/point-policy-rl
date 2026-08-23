#!/usr/bin/env bash
#SBATCH --job-name=pp-rl-s3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.out
#SBATCH --error=point_policy_rl/experiments_rl/libero_object/slurm_logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sanghyeok/point-policy-rl}"
RESFIT_ROOT="${RESFIT_ROOT:-/home/sanghyeok/residual-offpolicy-rl}"
CONDA_BIN="${CONDA_BIN:-/home/sanghyeok/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-libero_codex}"

BC_WEIGHT="${BC_WEIGHT:-/home/sanghyeok/point-policy-rl/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt}"
OFFLINE_PKL="${OFFLINE_PKL:-/home/sanghyeok/point-policy-rl/point_policy/exp_local/offline_rlds/libero10_3_turn_on_the_stove_and_put_the_moka_pot_on_it_img50.pkl}"

SEED="${SEED:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-resfit_img_scene3_s${SEED}_${RUN_STAMP}}"

# User-facing overrides aligned with resfit_reproduce naming.
LEARNING_STARTS="${LEARNING_STARTS:-10000}"
CRITIC_WARMUP="${CRITIC_WARMUP:-10000}"
N_STEP="${N_STEP:-5}"
GAMMA="${GAMMA:-0.995}"
STDDEV="${STDDEV:-0.025}"
if [[ -n "${WARMUP_STEPS:-}" ]]; then
  WARMUP_STEPS="${WARMUP_STEPS}"
elif [[ "${LEARNING_STARTS}" -gt "${CRITIC_WARMUP}" ]]; then
  WARMUP_STEPS="${LEARNING_STARTS}"
else
  WARMUP_STEPS="${CRITIC_WARMUP}"
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${RESFIT_ROOT}" ]]; then
  echo "RESFIT_ROOT not found: ${RESFIT_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "CONDA_BIN not executable: ${CONDA_BIN}" >&2
  exit 1
fi
if [[ ! -f "${BC_WEIGHT}" ]]; then
  echo "BC_WEIGHT not found: ${BC_WEIGHT}" >&2
  exit 1
fi
if [[ ! -f "${OFFLINE_PKL}" ]]; then
  echo "OFFLINE_PKL not found: ${OFFLINE_PKL}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/point_policy_rl/experiments_rl/libero_object/slurm_logs"
cd "${REPO_ROOT}"

CONDA_NO_PLUGINS=true \
"${CONDA_BIN}" --no-plugins run --no-capture-output -n "${CONDA_ENV}" \
python point_policy_rl/train_resfit_residual_td3_rl_rel.py \
  --bc-weight "${BC_WEIGHT}" \
  --suite libero_object_basefix_v1 \
  --benchmark-name LIBERO_10 \
  --task-name KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --task-order-index -1 \
  --seed "${SEED}" \
  --device cuda \
  --resfit-root "${RESFIT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --output-root point_policy/exp_local_rl_rel \
  --steps 400000 \
  --warmup-steps "${WARMUP_STEPS}" \
  --batch-size 256 \
  --buffer-size 300000 \
  --n-step "${N_STEP}" \
  --gamma "${GAMMA}" \
  --sampling-strategy uniform \
  --num-updates-per-iteration 4 \
  --actor-updates-per-iteration 1 \
  --update-every-n-steps 1 \
  --stddev-max "${STDDEV}" \
  --stddev-min "${STDDEV}" \
  --stddev-step 300000 \
  --random-action-noise-scale 0.2 \
  --residual-action-scale 0.2 \
  --critic-target-tau 0.005 \
  --offline-source expert_demo \
  --offline-demo-root "${OFFLINE_PKL}" \
  --offline-max-demos 50 \
  --offline-base-action-mode demo_delta \
  --offline-fraction 0.5 \
  --offline-fraction-decay-steps 100000 \
  --residual-obs-mode image_only \
  --image-obs-key pixels1 \
  --state-coord-mode both_eef \
  --camera-key observation.images.agentview \
  --dummy-image-size 84 \
  --env-max-episode-len 3000 \
  --eval-every 5000 \
  --eval-episodes 2 \
  --save-every 10000 \
  --log-every 200 \
  --strict-resfit
