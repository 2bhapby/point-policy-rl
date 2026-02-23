# Point-Policy RL (`_rel`) Transfer Setup

This file documents the minimum setup to run the `_rel` residual RL pipeline on another server.

## 1) Clone Repositories

```bash
git clone git@github.com:2bhapby/point-policy-rl.git
cd point-policy-rl
git checkout feat/parallel-v1-delta-osc-scale
```

`_rel` pipeline also depends on the ResFiT codebase:

```bash
cd ..
git clone git@github.com:amazon-science/residual-offpolicy-rl.git
```

## 2) Python Environment

Recommended: Python 3.8 (same as current working runs).

Install core packages in your runtime env (example):

```bash
conda create -n libero_codex python=3.8 -y
conda activate libero_codex
pip install torch torchvision
pip install numpy opencv-python imageio matplotlib
pip install dm-control robosuite
pip install wandb
```

Then install project dependencies used by your current Point-Policy/LIBERO setup.

## 3) Required Paths

Set the following absolute paths in scripts or env variables:

- `REPO_ROOT`: path to this repo
- `RESFIT_ROOT`: path to `residual-offpolicy-rl`
- BC checkpoint path (`--bc_weight` in training configs / `run_meta.json`)
- Offline dataset path (`.pkl` / RLDS-like input)

## 4) Main `_rel` Entry Points

Training:

- `point_policy_rl/train_resfit_residual_td3_rl_rel.py`
- SLURM: `point_policy_rl/experiments_rl/libero_object/slurm/train_resfit_residual_td3_rl_rel.sbatch`

Evaluation:

- `point_policy_rl/eval_residual_td3_rl_rel.py`
- SLURM: `point_policy_rl/experiments_rl/libero_object/slurm/eval_residual_td3_rl_rel.sbatch`

Checkpoint sweep eval:

- `point_policy_rl/sweep_eval_success_curve_rl_rel.py`
- SLURM: `point_policy_rl/experiments_rl/libero_object/slurm/sweep_eval_success_curve_rl_rel.sbatch`

Merge per-step sweep shards:

- `point_policy_rl/merge_eval_curve_shards_rl_rel.py`

## 5) Minimal Smoke Commands

Single checkpoint eval:

```bash
python point_policy_rl/eval_residual_td3_rl_rel.py \
  --ckpt /ABS/PATH/TO/snapshot/100000.pt \
  --episodes 10 \
  --device cuda \
  --resfit-root /ABS/PATH/residual-offpolicy-rl \
  --suite libero_object_basefix_v1 \
  --task-name KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --benchmark-name LIBERO_10 \
  --state-coord-mode both_eef \
  --image-size-override 84
```

SLURM sweep (single step example):

```bash
sbatch -p background \
  --job-name=pp-eval-step100k \
  --export=ALL,\
RUN_DIR=/ABS/PATH/TO/RUN,\
EPISODES=50,\
CKPT_STRIDE=10000,\
MIN_STEP=100000,\
MAX_STEP=100000,\
TIMEOUT_SEC=7200,\
IMAGE_SIZE_OVERRIDE=84,\
SUITE=libero_object_basefix_v1,\
TASK_NAME=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it,\
BENCHMARK_NAME=LIBERO_10,\
TASK_ORDER_INDEX=-1,\
STATE_COORD_MODE=both_eef,\
OUTPUT_TAG=scene3_step100k,\
OUTPUT_DIR=/ABS/PATH/OUT \
  point_policy_rl/experiments_rl/libero_object/slurm/sweep_eval_success_curve_rl_rel.sbatch
```

## 6) Notes

- This commit includes compatibility patches in `_rel` training/eval path used in your recent runs.
- If cluster rendering differs, verify EGL/MuJoCo offscreen setup on the target server.
- Keep run outputs (`exp_local_rl*`, `visual_reports`) outside Git tracking.

