# Point-Policy RL (`_rel`) Transfer Setup (Copy-Paste)

목표: 다른 서버에서 바로 `clone -> artifact download -> eval/train` 순서로 실행.

## A. 이미 업로드된 Hugging Face 리포

- Models: `insagur/point-policy-rl-models`
- Datasets: `insagur/point-policy-rl-datasets`

위 리포는 `point_policy_rl/hf_artifacts_manifest.yaml` 기준 artifact가 올라가 있습니다.

## B. 다른 서버에서 한 번에 세팅 (복붙)

### 1) 코드 클론

```bash
cd $HOME
git clone git@github.com:2bhapby/point-policy-rl.git
cd point-policy-rl
git checkout feat/parallel-v1-delta-osc-scale

cd $HOME
git clone git@github.com:amazon-science/residual-offpolicy-rl.git
```

### 2) Python 환경

```bash
conda create -n libero_codex python=3.8 -y
conda activate libero_codex
pip install --upgrade pip
pip install torch torchvision
pip install numpy opencv-python imageio matplotlib pyyaml
pip install huggingface_hub wandb
```

`LIBERO/robosuite/MuJoCo`는 서버 환경에 맞게 별도 설치가 필요합니다.

### 3) HF 인증

```bash
export HF_TOKEN=hf_xxx_your_token
python -c "from huggingface_hub import HfApi; print(HfApi().whoami(token='$HF_TOKEN')['name'])"
```

### 4) Artifact 다운로드 (manifest 기준)

```bash
cd $HOME/point-policy-rl
export REPO_ROOT=$PWD

python point_policy_rl/hf_sync_artifacts.py \
  --mode download \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --repo-root "$REPO_ROOT" \
  --download-root "$REPO_ROOT"
```

### 5) 다운로드 검증

```bash
cd $HOME/point-policy-rl
python point_policy_rl/hf_sync_artifacts.py \
  --mode validate \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --repo-root "$PWD" \
  --strict
```

## C. 바로 실행 가능한 명령 (복붙)

### 1) 단일 eval (scene3, image-only residual ckpt)

```bash
cd $HOME/point-policy-rl
conda activate libero_codex

export RESFIT_ROOT=$HOME/residual-offpolicy-rl
export CKPT=$HOME/point-policy-rl/point_policy/exp_local_rl_rel/2026.02.20/residual_td3_libero_object_basefix_v1_rl/123809_resfit_img_scene3_pkl50_fix0220/snapshot/400000.pt

python point_policy_rl/eval_residual_td3_rl_rel.py \
  --ckpt "$CKPT" \
  --episodes 10 \
  --device cuda \
  --resfit-root "$RESFIT_ROOT" \
  --suite libero_object_basefix_v1 \
  --task-name KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --benchmark-name LIBERO_10 \
  --task-order-index -1 \
  --state-coord-mode both_eef \
  --image-size-override 84
```

### 2) SLURM eval (sbatch wrapper 사용)

```bash
cd $HOME/point-policy-rl
export REPO_ROOT=$PWD
export CONDA_BASE=$HOME/miniconda3
export CONDA_ENV=libero_codex
export CKPT=$HOME/point-policy-rl/point_policy/exp_local_rl_rel/2026.02.20/residual_td3_libero_object_basefix_v1_rl/123809_resfit_img_scene3_pkl50_fix0220/snapshot/400000.pt

sbatch -p background \
  --job-name=pp-eval-scene3 \
  --export=ALL,REPO_ROOT,CONDA_BASE,CONDA_ENV,CKPT,EPISODES=10,SUITE=libero_object_basefix_v1,BENCHMARK_NAME=LIBERO_10,TASK_ORDER_INDEX=-1,STATE_COORD_MODE=both_eef,TASK_NAME=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it,IMAGE_SIZE_OVERRIDE=84,SAVE_VIDEO=0 \
  point_policy_rl/experiments_rl/libero_object/slurm/eval_residual_td3_rl_rel.sbatch
```

### 3) SLURM train (template)

```bash
cd $HOME/point-policy-rl
export REPO_ROOT=$PWD
export RESFIT_ROOT=$HOME/residual-offpolicy-rl
export CONDA_BASE=$HOME/miniconda3
export CONDA_ENV=libero_codex
export BC_WEIGHT=$HOME/point-policy-rl/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt
export TASK_NAME=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
export RUN_NAME=resfit_rel_scene3_serverB
export EXTRA_ARGS='--suite libero_object_basefix_v1 --benchmark-name LIBERO_10 --task-order-index -1 --steps 100000 --offline-source expert_demo --offline-demo-root point_policy/exp_local/offline_rlds --offline-max-demos 50 --residual-obs-mode image_only --image-obs-key pixels1 --state-coord-mode both_eef --offline-fraction 0.5 --critic-target-tau 0.005 --residual-action-scale 0.2 --random-action-noise-scale 0.2'

sbatch -p background \
  --job-name=pp-train-scene3 \
  --export=ALL,REPO_ROOT,RESFIT_ROOT,CONDA_BASE,CONDA_ENV,BC_WEIGHT,TASK_NAME,RUN_NAME,EXTRA_ARGS \
  point_policy_rl/experiments_rl/libero_object/slurm/train_resfit_residual_td3_rl_rel.sbatch
```

## D. 재현 체크리스트

1. `python point_policy_rl/hf_sync_artifacts.py --mode validate ... --strict` 통과
2. eval 실행 시 `checkpoint_format` 로그 출력 확인
3. 결과 생성 확인:
`train_log.csv`, `eval_log.csv`, `run_meta.json`, `snapshot/latest.pt`

## E. 주의사항

1. 이 문서는 `_rel` 경로 기준입니다:
`train_resfit_residual_td3_rl_rel.py`, `eval_residual_td3_rl_rel.py`
2. 서버별로 `partition`, `--mem`, GPU 타입은 반드시 맞춰서 수정하세요.
3. EGL/MuJoCo 오프스크린 이슈가 있으면 렌더링 설정부터 먼저 점검하세요.
