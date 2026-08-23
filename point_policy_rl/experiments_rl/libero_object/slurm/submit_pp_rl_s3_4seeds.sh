#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sanghyeok/point-policy-rl}"
SBATCH_FILE="${SBATCH_FILE:-${REPO_ROOT}/point_policy_rl/experiments_rl/libero_object/slurm/train_resfit_scene3_imgonly_seed.sbatch}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/point_policy_rl/experiments_rl/libero_object/slurm_logs}"

SEEDS="${SEEDS:-0 1 2 3}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"

if [[ ! -f "${SBATCH_FILE}" ]]; then
  echo "SBATCH_FILE not found: ${SBATCH_FILE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

for seed in ${SEEDS}; do
  run_name="resfit_img_scene3_s${seed}_${RUN_STAMP}"
  out_file="${LOG_DIR}/pp_rl_s3_s${seed}_%j.out"
  err_file="${LOG_DIR}/pp_rl_s3_s${seed}_%j.err"

  echo "Submitting seed=${seed} run_name=${run_name}"
  sbatch \
    --job-name=pp-rl-s3 \
    --output="${out_file}" \
    --error="${err_file}" \
    --export=ALL,SEED="${seed}",RUN_NAME="${run_name}",RUN_STAMP="${RUN_STAMP}" \
    "${SBATCH_FILE}"
done
